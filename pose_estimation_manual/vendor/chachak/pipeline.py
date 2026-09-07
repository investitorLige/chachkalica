"""Inference pipelines that wrap the trained detector.

Every pipeline reduces to one thing: :meth:`Pipeline.process_batch` takes a list
of CHW frames and returns one Friendy ``(N, 6)`` prediction tensor per frame, in
full-frame normalized coordinates. Pass its optional ``context`` and you also get
back the people the crops were taken from, plus a seventh column saying which of
them each detection was found on.

Three concrete pipelines, plus a chaining wrapper:

* ``batch_detect`` — tile each frame, run the model per tile, remap + NMS-merge.
* ``people_detect_first`` — detect people, crop, run the model per crop, remap.
* ``batch_people`` — tile, detect people per tile, then crop-infer-remap.
* ``chain`` — run several pipelines and merge their predictions.

The tiling front-end (:func:`_tile_infer`) and the crop→infer→remap back-end
(:meth:`Pipeline._crop_infer_remap`) are shared so the classes reuse each
other's logic without a heavyweight stage framework.

Trimmed relative to the original in ``chackalica_unified``: that version also
carried a ``Pipeline.run()`` which walked a Friendy eval dataloader, scored the
predictions with ``evaluate_detection``, and wrote a ``predictions.pt``. It was
the *only* reason this module imported from the training stack. Live camera
inference needs none of it — it calls ``process_batch`` per frame — so ``run()``
is gone and this module now depends on nothing outside ``vendor/``.
"""

from typing import Any, Dict, List, Optional

import torch

from .boxes import (
    PRED_WIDTH,
    crop_image,
    expand_box,
    grow_box_to_min_size,
    merge_predictions,
    remap_local_preds_to_frame,
    tile_frame,
    tile_frame_pixels,
    upscale_crop_to_min_size,
    xywhn_preds_to_xyxy,
)
from .formats import xyxy_to_xywhn
from .infer import infer_in_chunks


def _inference_score_threshold(config) -> float:
    if config.map_score_threshold is not None:
        return config.map_score_threshold
    return config.score_threshold


PARENT_NONE = -1
"""Seventh-column value for a detection that belongs to no person.

Every row a crop pipeline emits has a real parent, so this only shows up where a
pipeline that crops nothing contributes to the same merged tensor — a ``chain``
mixing ``batch_detect`` with a people pipeline.
"""


def _frame_size(image: torch.Tensor):
    """Return ``(width, height)`` of a CHW frame."""
    _, height, width = image.shape
    return width, height


def _with_parent_column(preds: torch.Tensor, person_idx: int) -> torch.Tensor:
    """Append the seventh column: which person these rows were detected on."""
    if preds.numel() == 0:
        return torch.zeros((0, PRED_WIDTH + 1), dtype=preds.dtype)
    parent = preds.new_full((preds.shape[0], 1), float(person_idx))
    return torch.cat([preds, parent], dim=1)


def _as_width(preds: torch.Tensor, width: int) -> torch.Tensor:
    """Pad ``preds`` out to ``width`` columns with :data:`PARENT_NONE`.

    Needed where tensors from pipelines that crop and pipelines that don't end up
    concatenated (a ``chain``), and for the empty tensor a frame with no crops
    produces — every frame in one batch has to agree on its width.
    """
    have = preds.shape[-1] if preds.ndim >= 2 else PRED_WIDTH
    if have >= width:
        return preds
    if preds.numel() == 0:
        return torch.zeros((0, width), dtype=preds.dtype)
    pad = preds.new_full((preds.shape[0], width - have), float(PARENT_NONE))
    return torch.cat([preds, pad], dim=1)


def _shift_parents(preds: torch.Tensor, offset: int) -> torch.Tensor:
    """Renumber the seventh column by ``offset``, leaving parentless rows alone."""
    if offset == 0 or preds.numel() == 0 or preds.shape[-1] <= PRED_WIDTH:
        return preds
    preds = preds.clone()
    parents = preds[:, PRED_WIDTH]
    preds[:, PRED_WIDTH] = torch.where(
        parents == float(PARENT_NONE), parents, parents + float(offset)
    )
    return preds


def _person_box_dicts(boxes: torch.Tensor, frame_w: int, frame_h: int) -> List[Dict[str, float]]:
    """Person boxes (xyxy pixels) -> the normalized dicts the rest of the app uses.

    The same ``{"cx","cy","w","h"}`` 0-1 shape a detection carries, so a consumer
    can hand these to the same geometry it already has (zone tests, drawing)
    without a second coordinate convention to get wrong.
    """
    if boxes is None or boxes.numel() == 0:
        return []
    xywhn = xyxy_to_xywhn(boxes[:, :4].float(), frame_w, frame_h)
    return [
        {"cx": float(row[0]), "cy": float(row[1]), "w": float(row[2]), "h": float(row[3])}
        for row in xywhn
    ]


def _tile_infer(adapter, image: torch.Tensor, config) -> List[torch.Tensor]:
    """Tile one frame, run ``adapter`` on the tiles, return remapped ``(N, 6)``.

    Shared by ``batch_detect`` (adapter = trained model). Returns a list of
    per-tile predictions already lifted into full-frame coordinates, ready for
    :func:`merge_predictions`.
    """
    frame_w, frame_h = _frame_size(image)
    if config.tiling.tile_size_px is not None:
        tiles = tile_frame_pixels(
            image, config.tiling.tile_size_px, config.tiling.overlap
        )
    else:
        tiles = tile_frame(
            image,
            config.tiling.tile_width_pct / 100.0,
            config.tiling.tile_height_pct / 100.0,
            config.tiling.overlap,
        )
    tile_images = [tile for tile, _, _ in tiles]
    preds = infer_in_chunks(
        adapter, tile_images, config.infer_batch_size, _inference_score_threshold(config)
    )
    remapped = []
    for (_, offset, (tile_w, tile_h)), tile_preds in zip(tiles, preds):
        remapped.append(
            remap_local_preds_to_frame(
                tile_preds.detach().cpu(), offset, tile_w, tile_h, frame_w, frame_h
            )
        )
    return remapped


class Pipeline:
    """Base pipeline: subclasses implement :meth:`process_batch`."""

    name = "pipeline"

    def __init__(self, model_adapter, device, config, detector=None, box_cache=None) -> None:
        self.model_adapter = model_adapter
        self.device = device
        self.config = config
        self.detector = detector
        # Optional dict-like cache of per-image person boxes, keyed by whatever
        # hashable identity a caller passes as `image_ids` (e.g. image_path).
        # None (the default) preserves the original always-run-the-detector
        # behavior exactly. Shared across pipeline instances/epochs by callers
        # that want the frozen detector's output reused instead of recomputed
        # (the detector never changes, so its output per image doesn't either).
        self.box_cache = box_cache

    def process_batch(
        self,
        images: List[torch.Tensor],
        targets: List[Dict[str, Any]],
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[torch.Tensor]:
        """Return one full-frame-normalized ``(N, 6)`` tensor per input frame.

        ``context``, when given, is a list of dicts one per image that the
        pipeline fills in with what it knew and would otherwise throw away:

        ``"person_boxes"``
            The people this frame's crops were taken from, as full-frame
            normalized ``{"cx","cy","w","h"}`` dicts. Empty for a pipeline that
            crops nothing.

        Asking for it also widens the returned tensor to ``(N, 7)``, the seventh
        column being the index into that frame's ``person_boxes`` of the person a
        row was detected on (``PARENT_NONE`` for a row that came from no person).
        That is how a caller can tell *whose* hard hat it is looking at — the
        person themselves is never a detection, so without this the association
        the crop pipeline already computed is unrecoverable downstream.

        ``None`` (the default) is the original behavior exactly: six columns, and
        nothing computed that isn't needed to produce them.
        """
        raise NotImplementedError

    # -- shared person-crop back-end (used by the two people pipelines) --
    def _person_boxes(
        self, images: List[torch.Tensor], image_ids: Optional[List[Any]] = None
    ) -> List[torch.Tensor]:
        """Per-frame person boxes as xyxy tensors in frame pixels.

        Implemented by the people pipelines (full-frame detection vs tiled
        detection). Not defined for tiling/chain pipelines.
        """
        raise NotImplementedError

    def crop_regions(
        self, images: List[torch.Tensor], image_ids: Optional[List[Any]] = None
    ) -> List[List[tuple]]:
        """Per-frame list of ``(crop_chw, (x0, y0), (crop_w, crop_h))`` regions.

        These are the person crops the pipeline feeds the trained model: each
        detected person box expanded by ``detector.expand_ratio`` and clipped to
        the frame. Shared by inference (:meth:`_crop_infer_remap`) and by
        training-time cropping (``friendy_chachkalica.preprocess.cropping.crop_batch``) so
        the model sees the exact same crops in both regimes. Zero-area crops are
        skipped. A crop narrower or shorter than ``detector.min_box_size`` is
        never dropped outright — a real (if distant/small) person must still
        reach the model — it's grown first using real neighboring frame pixels
        (:func:`grow_box_to_min_size`), and only bilinearly upscaled
        (:func:`upscale_crop_to_min_size`) if the frame itself is smaller than
        the floor. A frame with no person detections yields an empty list.
        """
        regions, _ = self._crop_regions_indexed(images, image_ids=image_ids)
        return [
            [(crop, offset, size) for _, crop, offset, size in frame_regions]
            for frame_regions in regions
        ]

    def _crop_regions_indexed(
        self, images: List[torch.Tensor], image_ids: Optional[List[Any]] = None
    ):
        """:meth:`crop_regions`, plus which person each crop came from.

        Returns ``(regions, person_boxes)`` where a region is
        ``(person_idx, crop_chw, (x0, y0), (crop_w, crop_h))`` and ``person_idx``
        indexes that frame's entry in ``person_boxes`` (xyxy, frame pixels).

        The index has to be carried explicitly rather than inferred from
        position: a zero-area crop is skipped below, so the crops of a frame are
        *not* one-per-person, and every crop after a skipped one would otherwise
        be attributed to the wrong person.
        """
        config = self.config
        min_box_size = config.detector.min_box_size
        person_boxes = self._person_boxes(images, image_ids=image_ids)
        regions: List[List[tuple]] = []
        for image, boxes in zip(images, person_boxes):
            frame_w, frame_h = _frame_size(image)
            frame_regions = []
            for person_idx, box in enumerate(boxes):
                expanded = expand_box(box, config.detector.expand_ratio, frame_w, frame_h)
                if min_box_size > 0:
                    expanded = grow_box_to_min_size(expanded, min_box_size, frame_w, frame_h)
                crop, offset, (crop_w, crop_h) = crop_image(image, expanded)
                if crop_w < 1 or crop_h < 1:
                    continue
                if min_box_size > 0 and (crop_w < min_box_size or crop_h < min_box_size):
                    crop, (crop_w, crop_h) = upscale_crop_to_min_size(crop, int(round(min_box_size)))
                frame_regions.append((person_idx, crop, offset, (crop_w, crop_h)))
            regions.append(frame_regions)
        return regions, person_boxes

    def _cached_person_boxes(self, images, image_ids, compute):
        """Return per-image person boxes, consulting/populating ``self.box_cache``.

        ``compute(subset_images)`` computes boxes for exactly the images that
        missed the cache, in order. With no cache configured, or no
        ``image_ids`` supplied (e.g. a caller outside a dataset context), this
        is just ``compute(images)`` — identical to the original always-run-the-
        detector behavior. An id of ``None`` for a given image (missing
        identity) is never cached under nor read from — it's computed fresh
        every call so unrelated frames can't collide on a shared ``None`` key.
        """
        if self.box_cache is None or image_ids is None:
            return compute(images)

        results: List[Any] = [None] * len(images)
        miss_indices = []
        for i, image_id in enumerate(image_ids):
            cached = self.box_cache.get(image_id) if image_id is not None else None
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)

        if miss_indices:
            computed = compute([images[i] for i in miss_indices])
            for local_i, global_i in enumerate(miss_indices):
                results[global_i] = computed[local_i]
                image_id = image_ids[global_i]
                if image_id is not None:
                    self.box_cache[image_id] = computed[local_i]
        return results

    def _crop_infer_remap(
        self,
        images: List[torch.Tensor],
        image_ids: Optional[List[Any]] = None,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[torch.Tensor]:
        """Crop each person region, run the trained model, remap and merge per frame.

        With ``context`` given, each frame's people are reported back through it
        and every output row carries the index of the person it was found on — see
        :meth:`Pipeline.process_batch`.
        """
        config = self.config
        crops: List[torch.Tensor] = []
        crop_frame_idx: List[int] = []
        crop_person_idx: List[int] = []
        crop_meta = []  # (offset_xy, (crop_w, crop_h))
        frame_sizes = [_frame_size(image) for image in images]

        regions_per_frame, person_boxes = self._crop_regions_indexed(
            images, image_ids=image_ids
        )
        for f_idx, regions in enumerate(regions_per_frame):
            for person_idx, crop, offset, (crop_w, crop_h) in regions:
                crops.append(crop)
                crop_frame_idx.append(f_idx)
                crop_person_idx.append(person_idx)
                crop_meta.append((offset, (crop_w, crop_h)))

        if context is not None:
            for f_idx, (frame_w, frame_h) in enumerate(frame_sizes):
                context[f_idx]["person_boxes"] = _person_box_dicts(
                    person_boxes[f_idx], frame_w, frame_h
                )

        crop_preds = infer_in_chunks(
            self.model_adapter, crops, config.infer_batch_size, _inference_score_threshold(config)
        )

        per_frame: List[List[torch.Tensor]] = [[] for _ in images]
        for c_idx, preds in enumerate(crop_preds):
            f_idx = crop_frame_idx[c_idx]
            offset, (crop_w, crop_h) = crop_meta[c_idx]
            frame_w, frame_h = frame_sizes[f_idx]
            preds = preds.detach().cpu()
            if context is not None:
                # Attached before the remap, not after, so it rides through the
                # coordinate lift and the NMS merge as one row — there is no
                # point downstream where the rows are still in crop order.
                preds = _with_parent_column(preds, crop_person_idx[c_idx])
            per_frame[f_idx].append(
                remap_local_preds_to_frame(
                    preds, offset, crop_w, crop_h, frame_w, frame_h
                )
            )

        # min_box_size is not applied here: it gates which *person crops* reach
        # the model (crop_regions), not the stage-2 model's own output boxes.
        # Those are item-level (e.g. a PPE item on the person), routinely far
        # smaller than a person crop, so reusing the same floor here would drop
        # nearly all real detections regardless of camera distance.
        width = PRED_WIDTH + 1 if context is not None else PRED_WIDTH
        return [
            _as_width(
                merge_predictions(per_frame[i], *frame_sizes[i], config.merge_nms_iou),
                width,
            )
            for i in range(len(images))
        ]

class BatchDetectPipeline(Pipeline):
    """Tile each frame, run the trained model per tile, remap + NMS-merge."""

    name = "batch_detect"

    def process_batch(self, images, targets, context=None):
        outputs = []
        for f_idx, image in enumerate(images):
            remapped = _tile_infer(self.model_adapter, image, self.config)
            merged = merge_predictions(
                remapped, *_frame_size(image), self.config.tiling.nms_iou
            )
            if context is not None:
                # Tiles are cut from the frame, not from people: this pipeline
                # crops nobody, so every row's parent is PARENT_NONE.
                context[f_idx]["person_boxes"] = []
                merged = _as_width(merged, PRED_WIDTH + 1)
            outputs.append(merged)
        return outputs


class PeopleDetectFirstPipeline(Pipeline):
    """Detect people on full frames, then crop-infer-remap the trained model."""

    name = "people_detect_first"

    def _person_boxes(self, images, image_ids=None):
        if self.detector is None:
            raise ValueError("people_detect_first requires a detector")

        def compute(subset_images):
            det_preds = self.detector.predict(subset_images)
            boxes = []
            for image, preds in zip(subset_images, det_preds):
                frame_w, frame_h = _frame_size(image)
                boxes.append(xywhn_preds_to_xyxy(preds.detach().cpu(), frame_w, frame_h))
            return boxes

        return self._cached_person_boxes(images, image_ids, compute)

    def process_batch(self, images, targets, context=None):
        image_ids = [target.get("image_path") for target in targets]
        return self._crop_infer_remap(images, image_ids=image_ids, context=context)


class BatchPeoplePipeline(Pipeline):
    """Tile, detect people per tile, then crop-infer-remap on the original frame."""

    name = "batch_people"

    def _person_boxes(self, images, image_ids=None):
        if self.detector is None:
            raise ValueError("batch_people requires a detector")
        config = self.config

        def compute(subset_images):
            boxes = []
            for image in subset_images:
                frame_w, frame_h = _frame_size(image)
                tiles = tile_frame(
                    image,
                    config.tiling.tile_width_pct / 100.0,
                    config.tiling.tile_height_pct / 100.0,
                    config.tiling.overlap,
                )
                tile_images = [tile for tile, _, _ in tiles]
                det_preds = self.detector.predict(tile_images)
                remapped = []
                for (_, offset, (tile_w, tile_h)), preds in zip(tiles, det_preds):
                    remapped.append(
                        remap_local_preds_to_frame(
                            preds.detach().cpu(), offset, tile_w, tile_h, frame_w, frame_h
                        )
                    )
                # Collapse duplicate person boxes from overlapping tiles before cropping.
                merged = merge_predictions(remapped, frame_w, frame_h, config.detector.nms_iou)
                boxes.append(xywhn_preds_to_xyxy(merged, frame_w, frame_h))
            return boxes

        return self._cached_person_boxes(images, image_ids, compute)

    def process_batch(self, images, targets, context=None):
        image_ids = [target.get("image_path") for target in targets]
        return self._crop_infer_remap(images, image_ids=image_ids, context=context)


class ChainedPipeline(Pipeline):
    """Run several pipelines and merge their per-frame predictions (stacking)."""

    name = "chain"

    def __init__(self, model_adapter, device, config, detector=None, pipelines=None, box_cache=None):
        super().__init__(model_adapter, device, config, detector=detector, box_cache=box_cache)
        self.pipelines = list(pipelines or [])
        if self.pipelines:
            self.name = "chain[" + "+".join(p.name for p in self.pipelines) + "]"

    def process_batch(self, images, targets, context=None):
        per_frame: List[List[torch.Tensor]] = [[] for _ in images]
        # Each member numbers its people from zero, so a chain with two people
        # pipelines would have two different person 0 per frame. Concatenate the
        # members' people and shift each member's indices past the ones already
        # collected, so an index means one person across the whole chain.
        people: List[List[Dict[str, float]]] = [[] for _ in images]
        for pipe in self.pipelines:
            member_context = [{} for _ in images] if context is not None else None
            outputs = pipe.process_batch(images, targets, context=member_context)
            for i, preds in enumerate(outputs):
                if context is not None:
                    preds = _shift_parents(preds, len(people[i]))
                    people[i].extend(member_context[i].get("person_boxes") or [])
                per_frame[i].append(preds)

        width = PRED_WIDTH + 1 if context is not None else PRED_WIDTH
        if context is not None:
            for i in range(len(images)):
                context[i]["person_boxes"] = people[i]
        return [
            _as_width(
                merge_predictions(
                    [_as_width(preds, width) for preds in per_frame[i]],
                    *_frame_size(images[i]),
                    self.config.merge_nms_iou,
                ),
                width,
            )
            for i in range(len(images))
        ]
