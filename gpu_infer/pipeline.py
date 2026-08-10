"""The four pipelines, GPU-resident.

One-to-one with ``chachak.pipeline``: ``people_detect_first``, ``batch_people``,
``batch_detect`` and ``chain``, over one shared region back-end
(:func:`_infer_regions`) and one shared merge tail. What differs is only *where* the
synchronizations are, and there are two per frame at most:

* after stage 1 in the two-stage pipelines, because ``crop_bounds`` has to become host integers
  before a variable-extent slice can exist;
* at the end, where detections are compacted for the merge NMS and then leave for the host.

Faithfulness notes — places the original is surprising and this had to follow rather than tidy:

* **Detector predictions are normalized to their input, and that input may have been
  downscaled** to fit the engine's profile. The original converts them to frame pixels using the
  *original* frame size anyway (``pipeline.py:354``), which is correct precisely because
  normalized coordinates are invariant under the resize. So the fit is never undone.
* **``batch_people`` always tiles fractionally**, ignoring ``tiling.tile_size_px``
  (``pipeline.py:379`` calls ``tile_frame`` directly), while ``batch_detect`` honours it
  (``pipeline.py:74``). That is an inconsistency in the original, reproduced here deliberately —
  "fixing" it would change what an existing config detects.
* **The class-space remap is not applied here.** In the original it happens in the bundle's
  ``run_batch`` *after* the pipeline's own merge, and a multi-model bundle merges a second time
  afterwards. Doing it inside the pipeline would reorder those steps.
* **Chunking mirrors ``infer_in_chunks`` then ``TrtAdapter.predict``** — chunk the region list in
  order first, *then* group by shape within each chunk. Grouping globally and chunking afterwards
  would produce different batch compositions, and for the EfficientNMS archs a different batch
  composition can change the last bit of a convolution.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .config import GpuOptions


def _frame_size(frame) -> Tuple[int, int]:
    """``(width, height)`` of a CHW frame, as ``chachak.pipeline._frame_size`` returns it."""
    return int(frame.shape[2]), int(frame.shape[1])


def _inference_score_threshold(config) -> float:
    """``map_score_threshold`` when set, else ``score_threshold``. Mirrors ``chachak.config``."""
    if getattr(config, "map_score_threshold", None) is not None:
        return config.map_score_threshold
    return config.score_threshold


def _infer_regions(
    loaded,
    regions: Sequence,
    offsets,
    local_sizes,
    frame_w: int,
    frame_h: int,
    *,
    score_threshold: float,
    chunk_size: int,
    remap: bool = True,
):
    """Run one engine over same-frame regions. Returns padded ``([N, K, 6], [N, K])``.

    ``offsets``/``local_sizes`` are ``[N, 2]`` int64 tensors describing where each region sits and
    how big it is in frame pixels. With ``remap=False`` the detections are left normalized to
    their region — used for a full-frame stage 1, where the region *is* the frame and a remap
    would be an identity that still costs kernels.

    No synchronization happens in here.
    """
    import torch

    from . import crops_exact, tail

    if not regions:
        empty_dets = tail.empty_detections(device=offsets.device).reshape(0, 0, 6)
        return empty_dets, torch.zeros((0, 0), dtype=torch.bool, device=offsets.device)

    prepared, transforms = crops_exact.prepare_regions(regions, loaded.meta)
    engine = loaded.engine
    max_batch = engine.max_batch or 1
    effective = max(1, min(int(chunk_size), max_batch))

    count = len(regions)
    slot_boxes: List[Optional[object]] = [None] * count
    slot_scores: List[Optional[object]] = [None] * count
    slot_labels: List[Optional[object]] = [None] * count
    slot_valid: List[Optional[object]] = [None] * count

    # Chunk first, then group by shape inside the chunk -- see the module docstring.
    for start in range(0, count, effective):
        chunk = list(range(start, min(start + effective, count)))
        groups: Dict[tuple, List[int]] = {}
        for index in chunk:
            key = tuple(int(dim) for dim in prepared[index].shape[1:])
            groups.setdefault(key, []).append(index)

        for indices in groups.values():
            stacked = crops_exact.stack_group(prepared, indices)
            if len(indices) == 1:
                outputs = engine.submit(stacked)
                positions = [0]
            elif max_batch > 1:
                outputs = engine.submit(stacked)
                positions = list(range(len(indices)))
            else:
                # Batch-1 profile: one execution per region, pointing at each row in place.
                for local, index in enumerate(indices):
                    single = engine.submit_single(stacked, local)
                    valid = single.valid_mask()
                    slot_boxes[index] = single.boxes
                    slot_scores[index] = single.scores
                    slot_labels[index] = single.labels
                    slot_valid[index] = valid
                continue
            valid = outputs.valid_mask()
            for local, index in zip(positions, indices):
                slot_boxes[index] = outputs.boxes[local : local + 1]
                slot_scores[index] = outputs.scores[local : local + 1]
                slot_labels[index] = outputs.labels[local : local + 1]
                slot_valid[index] = valid[local : local + 1]

    boxes = torch.cat(slot_boxes, dim=0)
    scores = torch.cat(slot_scores, dim=0)
    labels = torch.cat(slot_labels, dim=0)
    valid = torch.cat(slot_valid, dim=0)

    dets, valid = tail.decode_regions(
        boxes,
        scores,
        labels,
        valid,
        tail.transforms_tensor(transforms, boxes.device, loaded.meta.box_coords),
        score_threshold=score_threshold,
        clip_boxes=loaded.meta.clip_boxes,
        box_coords=loaded.meta.box_coords,
    )
    if remap:
        dets, valid = tail.remap_regions_to_frame(
            dets, valid, offsets, local_sizes, frame_w, frame_h
        )
    return dets, valid


def _merge(dets, valid, frame_w: int, frame_h: int, nms_iou: float, options: GpuOptions):
    """Flatten, trim, compact and NMS-merge. **This is where a synchronization happens.**"""
    from . import nms, tail

    flat, flat_valid = tail.flatten_regions(dets, valid)
    overflow = tail.overflow_count(flat_valid, options.nms_topk)
    flat, flat_valid = tail.topk_trim(flat, flat_valid, options.nms_topk)
    compacted = tail.compact(flat, flat_valid)  # <- the sync
    merged, converged = nms.merge_detections(
        compacted, frame_w, frame_h, nms_iou,
        rounds=options.nms_rounds, strict=options.nms_strict,
    )
    nms.check_converged(
        converged, strict=options.nms_strict, rounds=options.nms_rounds, where="merge NMS"
    )
    dropped = int(overflow)
    if dropped:
        print(
            f"[gpu_infer] nms_topk={options.nms_topk} discarded {dropped} detection(s) "
            f"below the cap on this frame — raise it or set it to None to match "
            f"merge_predictions, which has no cap"
        )
    return merged


class GpuPipeline:
    """Base: subclasses implement :meth:`process_frame`."""

    name = "gpu_pipeline"

    def __init__(
        self,
        config,
        model,
        *,
        detector=None,
        options: Optional[GpuOptions] = None,
        device="cuda",
    ) -> None:
        self.config = config
        self.model = model
        self.detector = detector
        self.options = options or GpuOptions()
        self.device = device

    def process_frame(self, frame):
        """One CHW device frame -> a ``[M, 6]`` frame-normalized detection tensor.

        Runs on the package's dedicated non-default CUDA stream. That is not incidental:
        TensorRT inserts its own synchronizations around work submitted to the *default* stream,
        which would hand back exactly the overlap this package exists to create. Entering the
        stream here — around preprocessing, every engine call and the whole tail — also makes all
        ordering implicit, since one stream executes its work in order.
        """
        from .engine import inference_scope

        if not frame.is_cuda:
            return self._process_frame(frame)
        with inference_scope(frame.device, inputs=(frame,)):
            return self._process_frame(frame)

    def _process_frame(self, frame):
        """The actual work; subclasses implement this, not :meth:`process_frame`."""
        raise NotImplementedError

    def process_frames(self, frames: Sequence) -> List:
        """Per-frame detections. Frames are processed one at a time on purpose.

        Batching *frames* buys nothing for inference — a single 1080p frame already saturates
        the engine, and the regions within it are what get batched. Keeping the loop here also
        keeps the padded buffers sized by one frame's regions rather than the batch's.
        """
        return [self.process_frame(frame) for frame in frames]

    # -- shared stage-1 / stage-2 machinery ---------------------------------------

    def _detect_people_full_frame(self, frame) -> "object":
        """Person boxes as ``[P, 4]`` xyxy in frame pixels. **Synchronizes.**

        Mirrors ``chachak.detector.Detector.predict`` plus
        ``PeopleDetectFirstPipeline._person_boxes``: fit to the engine's input profile, run,
        keep the person class above threshold, and convert to frame pixels using the *original*
        frame extent.
        """
        import torch

        from . import geometry, loader, tail

        if self.detector is None:
            raise ValueError(f"{self.name} requires a detector")

        frame_w, frame_h = _frame_size(frame)
        fitted = loader.fit_to_input_profile(frame, self.detector.max_input_hw)
        fitted_w, fitted_h = _frame_size(fitted)
        threshold = self.config.detector.score_threshold

        dets, valid = _infer_regions(
            self.detector,
            [fitted],
            torch.zeros((1, 2), dtype=torch.int64, device=fitted.device),
            torch.tensor([[fitted_w, fitted_h]], dtype=torch.int64, device=fitted.device),
            fitted_w,
            fitted_h,
            score_threshold=threshold,
            chunk_size=self.config.detector.batch_size,
            remap=False,  # the region is the whole frame; normalized coords already match
        )
        person_id = self._person_class_id()
        valid = tail.filter_class(dets, valid, person_id, threshold)
        people = tail.compact(dets[0], valid[0])  # <- sync
        return geometry.xywhn_to_xyxy_batched(people[:, :4], frame_w, frame_h)

    def _person_class_id(self) -> int:
        from . import loader

        if getattr(self, "_cached_person_id", None) is None:
            self._cached_person_id = loader.resolve_person_class_id(
                self.detector.class_map,
                person_class_name=self.config.detector.person_class_name,
                person_class_id=self.config.detector.person_class_id,
            )
        return self._cached_person_id

    def _crop_infer_remap(self, frame, person_boxes):
        """Crop each person region, run the model, remap and merge. Mirrors
        ``Pipeline._crop_infer_remap`` including its deliberate absence of a size floor."""
        import torch

        from . import crops_exact, geometry, tail

        frame_w, frame_h = _frame_size(frame)
        detector_config = self.config.detector
        min_box_size = detector_config.min_box_size

        expanded = geometry.expand_boxes(
            person_boxes, detector_config.expand_ratio, frame_w, frame_h
        )
        if min_box_size > 0:
            expanded = geometry.grow_boxes_to_min_size(
                expanded, min_box_size, frame_w, frame_h
            )
        bounds = geometry.crop_bounds(expanded, frame_w, frame_h)

        regions, offsets, local_sizes = [], [], []
        for row in bounds.tolist():  # already-synced host integers
            region = crops_exact.slice_region(frame, row, min_box_size)
            if region is None:
                continue
            regions.append(region)
            offsets.append([row[0], row[1]])
            local_sizes.append([row[2] - row[0], row[3] - row[1]])

        if not regions:
            return tail.empty_detections(device=frame.device)

        dets, valid = _infer_regions(
            self.model,
            regions,
            torch.tensor(offsets, dtype=torch.int64, device=frame.device),
            torch.tensor(local_sizes, dtype=torch.int64, device=frame.device),
            frame_w,
            frame_h,
            score_threshold=_inference_score_threshold(self.config),
            chunk_size=self.config.infer_batch_size,
        )
        # min_box_size is deliberately NOT applied to these outputs: it gates which person
        # crops reach the model, not the model's own item-level boxes, which are routinely far
        # smaller than a person crop.
        return _merge(
            dets, valid, frame_w, frame_h, self.config.merge_nms_iou, self.options
        )


class PeopleDetectFirst(GpuPipeline):
    """Detect people on full frames, then crop-infer-remap."""

    name = "people_detect_first"

    def _process_frame(self, frame):
        return self._crop_infer_remap(frame, self._detect_people_full_frame(frame))


class BatchPeople(GpuPipeline):
    """Tile, detect people per tile, then crop-infer-remap on the original frame."""

    name = "batch_people"

    def _process_frame(self, frame):
        return self._crop_infer_remap(frame, self._detect_people_tiled(frame))

    def _detect_people_tiled(self, frame):
        """Person boxes from tiled detection. **Synchronizes.**

        Note this always tiles *fractionally*, ignoring ``tiling.tile_size_px`` — see the module
        docstring.
        """
        import torch

        from . import geometry, loader, nms, tail, tiling

        if self.detector is None:
            raise ValueError(f"{self.name} requires a detector")

        frame_w, frame_h = _frame_size(frame)
        config = self.config
        grid = tiling.plan_tiles_fractional(
            frame_h,
            frame_w,
            config.tiling.tile_width_pct / 100.0,
            config.tiling.tile_height_pct / 100.0,
            config.tiling.overlap,
        )
        tiles = [
            loader.fit_to_input_profile(tile, self.detector.max_input_hw)
            for tile in tiling.extract_tiles(frame, grid)
        ]
        threshold = config.detector.score_threshold

        dets, valid = _infer_regions(
            self.detector,
            tiles,
            tiling.offsets_tensor(grid, frame),
            tiling.local_sizes_tensor(grid, frame),
            frame_w,
            frame_h,
            score_threshold=threshold,
            chunk_size=config.detector.batch_size,
        )
        # The class filter runs before the merge, as Detector.predict does.
        valid = tail.filter_class(dets, valid, self._person_class_id(), threshold)
        merged = _merge(
            dets, valid, frame_w, frame_h, config.detector.nms_iou, self.options
        )
        return geometry.xywhn_to_xyxy_batched(merged[:, :4], frame_w, frame_h)


class BatchDetect(GpuPipeline):
    """Tile each frame, run the model per tile, remap and NMS-merge."""

    name = "batch_detect"

    def _process_frame(self, frame):
        from . import tiling

        frame_w, frame_h = _frame_size(frame)
        grid = tiling.plan_tiles(frame_h, frame_w, self.config.tiling)
        dets, valid = _infer_regions(
            self.model,
            tiling.extract_tiles(frame, grid),
            tiling.offsets_tensor(grid, frame),
            tiling.local_sizes_tensor(grid, frame),
            frame_w,
            frame_h,
            score_threshold=_inference_score_threshold(self.config),
            chunk_size=self.config.infer_batch_size,
        )
        return _merge(
            dets, valid, frame_w, frame_h, self.config.tiling.nms_iou, self.options
        )


class Chained(GpuPipeline):
    """Run several pipelines and merge their per-frame detections."""

    name = "chain"

    def __init__(self, config, model, *, pipelines=(), **kwargs) -> None:
        super().__init__(config, model, **kwargs)
        self.pipelines = list(pipelines)
        if self.pipelines:
            self.name = "chain[" + "+".join(p.name for p in self.pipelines) + "]"

    def _process_frame(self, frame):
        import torch

        from . import nms, tail

        frame_w, frame_h = _frame_size(frame)
        parts = [pipe.process_frame(frame) for pipe in self.pipelines]
        parts = [part for part in parts if part.shape[0] > 0]
        if not parts:
            return tail.empty_detections(device=frame.device)
        combined = torch.cat(parts, dim=0)
        merged, converged = nms.merge_detections(
            combined,
            frame_w,
            frame_h,
            self.config.merge_nms_iou,
            rounds=self.options.nms_rounds,
            strict=self.options.nms_strict,
        )
        nms.check_converged(
            converged,
            strict=self.options.nms_strict,
            rounds=self.options.nms_rounds,
            where="chain merge NMS",
        )
        return merged


_PIPELINES = {
    "people_detect_first": PeopleDetectFirst,
    "batch_people": BatchPeople,
    "batch_detect": BatchDetect,
}


def build(
    config,
    model,
    *,
    detector=None,
    extras: Sequence = (),
    options: Optional[GpuOptions] = None,
    eval_classes: Optional[Dict] = None,
    device="cuda",
):
    """Construct the pipeline ``config.pipeline`` names, mirroring ``chachak.registry``.

    Returns a :class:`MultiModel` wrapper, which is what applies the class-space remap and the
    cross-model merge — the steps the bundle's ``run_batch`` performs around the pipeline rather
        than inside it.
    """
    options = options or GpuOptions()
    name = config.pipeline

    def one(engine, pipeline_name):
        if pipeline_name not in _PIPELINES:
            raise ValueError(
                f"[gpu_infer] unsupported pipeline {pipeline_name!r}; this runtime implements "
                f"{sorted(_PIPELINES)} and 'chain' over them"
            )
        return _PIPELINES[pipeline_name](
            config, engine, detector=detector, options=options, device=device
        )

    def for_engine(engine):
        if name == "chain":
            if not config.chain:
                raise ValueError("[gpu_infer] chain pipeline requires a non-empty chain list")
            return Chained(
                config,
                engine,
                pipelines=[one(engine, child) for child in config.chain],
                detector=detector,
                options=options,
                device=device,
            )
        return one(engine, name)

    return MultiModel(
        config,
        [(for_engine(engine), engine) for engine in [model, *extras]],
        options=options,
        eval_classes=eval_classes,
    )


class MultiModel:
    """Runs every bundled model's pipeline and merges them, like the bundle's ``run_batch``.

    Keeps the original's ordering exactly: each pipeline merges its own regions first, *then* its
    detections are remapped onto the bundle's class space, *then* the models are merged with each
    other. A bundle with ``extra_models`` is only valid because such models were exported with
    disjoint class names, which is what makes that final merge meaningful.
    """

    def __init__(self, config, entries: Sequence, *, options: GpuOptions, eval_classes=None):
        self.config = config
        self.entries = list(entries)
        self.options = options
        self.eval_classes = eval_classes
        self._luts = None

    @property
    def name(self) -> str:
        return self.entries[0][0].name

    def _lut_for(self, engine, device):
        from . import tail

        if self._luts is None:
            self._luts = {}
        key = id(engine)
        if key not in self._luts:
            self._luts[key] = tail.build_class_lut(
                engine.class_map, self.eval_classes, device
            )
        return self._luts[key]

    def process_frame(self, frame):
        """Every model's detections for one frame, remapped and merged.

        Enters the package's non-default CUDA stream around the whole thing, for the reason
        :func:`gpu_infer.engine.inference_stream` documents. The nested entry inside each
        pipeline's own ``process_frame`` is the same stream, which is harmless.
        """
        from .engine import inference_scope

        if not frame.is_cuda:
            return self._process_frame(frame)
        with inference_scope(frame.device, inputs=(frame,)):
            return self._process_frame(frame)

    def _process_frame(self, frame):
        import torch

        from . import nms, tail

        frame_w, frame_h = _frame_size(frame)
        per_model = []
        for pipeline, engine in self.entries:
            dets = pipeline.process_frame(frame)
            lut = self._lut_for(engine, frame.device)
            if lut is not None and dets.shape[0]:
                valid = torch.ones(
                    (1, dets.shape[0]), dtype=torch.bool, device=dets.device
                )
                remapped, remapped_valid = tail.apply_class_lut(
                    dets.unsqueeze(0), valid, lut
                )
                dets = tail.compact(remapped[0], remapped_valid[0])
            per_model.append(dets)

        if len(per_model) == 1:
            return per_model[0]
        combined = [part for part in per_model if part.shape[0] > 0]
        if not combined:
            return tail.empty_detections(device=frame.device)
        merged, converged = nms.merge_detections(
            torch.cat(combined, dim=0),
            frame_w,
            frame_h,
            self.config.merge_nms_iou,
            rounds=self.options.nms_rounds,
            strict=self.options.nms_strict,
        )
        nms.check_converged(
            converged,
            strict=self.options.nms_strict,
            rounds=self.options.nms_rounds,
            where="multi-model merge NMS",
        )
        return merged

    def process_frames(self, frames: Sequence) -> List:
        return [self.process_frame(frame) for frame in frames]
