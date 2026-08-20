"""Inference/eval pipelines that wrap the trained detector.

Every pipeline reduces to: produce per-frame Friendy predictions ``(N, 6)`` in
full-frame normalized coordinates, then let the base class score them with
Friendy's ``evaluate_detection`` and serialize them exactly like
``eval_checkpoint.py`` (a ``predictions.pt`` of records + returned metrics).

Three concrete pipelines, plus a chaining wrapper:

* ``batch_detect`` — tile each frame, run the model per tile, remap + NMS-merge.
* ``people_detect_first`` — detect people, crop, run the model per crop, remap.
* ``batch_people`` — tile, detect people per tile, then crop-infer-remap.
* ``chain`` — run several pipelines and merge their predictions.

The tiling front-end (:func:`_tile_infer`) and the crop→infer→remap back-end
(:meth:`Pipeline._crop_infer_remap`) are shared so the classes reuse each
other's logic without a heavyweight stage framework.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    # Re-exported (not defined here) so bundle_export can reach it without
    # importing this module's torch — see chachak/config.py.
    from .config import _inference_score_threshold  # noqa: F401
    from ._friendy import EVAL_HARD_IMAGES_FRACTION, _to_builtin, _write_hard_images, evaluate_detection
    from .boxes import (
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
    from .infer import infer_in_chunks
except ImportError:  # run as a flat script
    from config import _inference_score_threshold  # noqa: F401
    from _friendy import EVAL_HARD_IMAGES_FRACTION, _to_builtin, _write_hard_images, evaluate_detection
    from boxes import (
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
    from infer import infer_in_chunks


def _frame_size(image: torch.Tensor):
    """Return ``(width, height)`` of a CHW frame."""
    _, height, width = image.shape
    return width, height


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
        self, images: List[torch.Tensor], targets: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """Return one full-frame-normalized ``(N, 6)`` tensor per input frame."""
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

        ``(crop_w, crop_h)`` is always the region's extent in *frame* pixels,
        which an upscale deliberately leaves alone — it is the geometry both
        callers remap by, so it can't track the tensor's own resized shape.
        """
        config = self.config
        min_box_size = config.detector.min_box_size
        person_boxes = self._person_boxes(images, image_ids=image_ids)
        regions: List[List[tuple]] = []
        for image, boxes in zip(images, person_boxes):
            frame_w, frame_h = _frame_size(image)
            frame_regions = []
            for box in boxes:
                expanded = expand_box(box, config.detector.expand_ratio, frame_w, frame_h)
                if min_box_size > 0:
                    expanded = grow_box_to_min_size(expanded, min_box_size, frame_w, frame_h)
                crop, offset, (crop_w, crop_h) = crop_image(image, expanded)
                if crop_w < 1 or crop_h < 1:
                    continue
                if min_box_size > 0 and (crop_w < min_box_size or crop_h < min_box_size):
                    # Only the tensor grows: (crop_w, crop_h) stays the region's
                    # size in *frame* pixels, which is what the caller remaps
                    # predictions and training targets against.
                    crop = upscale_crop_to_min_size(crop, int(round(min_box_size)))
                frame_regions.append((crop, offset, (crop_w, crop_h)))
            regions.append(frame_regions)
        return regions

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
        self, images: List[torch.Tensor], image_ids: Optional[List[Any]] = None
    ) -> List[torch.Tensor]:
        """Crop each person region, run the trained model, remap and merge per frame."""
        config = self.config
        crops: List[torch.Tensor] = []
        crop_frame_idx: List[int] = []
        crop_meta = []  # (offset_xy, (crop_w, crop_h))
        frame_sizes = [_frame_size(image) for image in images]

        for f_idx, regions in enumerate(self.crop_regions(images, image_ids=image_ids)):
            for crop, offset, (crop_w, crop_h) in regions:
                crops.append(crop)
                crop_frame_idx.append(f_idx)
                crop_meta.append((offset, (crop_w, crop_h)))

        crop_preds = infer_in_chunks(
            self.model_adapter, crops, config.infer_batch_size, _inference_score_threshold(config)
        )

        per_frame: List[List[torch.Tensor]] = [[] for _ in images]
        for c_idx, preds in enumerate(crop_preds):
            f_idx = crop_frame_idx[c_idx]
            offset, (crop_w, crop_h) = crop_meta[c_idx]
            frame_w, frame_h = frame_sizes[f_idx]
            per_frame[f_idx].append(
                remap_local_preds_to_frame(
                    preds.detach().cpu(), offset, crop_w, crop_h, frame_w, frame_h
                )
            )

        # min_box_size is not applied here: it gates which *person crops* reach
        # the model (crop_regions), not the stage-2 model's own output boxes.
        # Those are item-level (e.g. a PPE item on the person), routinely far
        # smaller than a person crop, so reusing the same floor here would drop
        # nearly all real detections regardless of camera distance.
        return [
            merge_predictions(per_frame[i], *frame_sizes[i], config.merge_nms_iou)
            for i in range(len(images))
        ]

    def run(
        self,
        loader,
        output_dir,
        *,
        num_classes: Optional[int] = None,
        prediction_classes: Optional[Dict[int, str]] = None,
        target_classes: Optional[Dict[int, str]] = None,
        eval_classes: Optional[Dict[int, str]] = None,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        """Run the pipeline over a Friendy eval dataloader and score the result."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records = []
        all_predictions = []
        all_targets = []
        started = time.perf_counter()
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = [image.to(self.device) for image in images]
            predictions = self.process_batch(images, targets)
            for target, prediction in zip(targets, predictions):
                prediction = prediction.detach().cpu()
                all_predictions.append(prediction)
                all_targets.append(target)
                orig_size = target.get("orig_size")
                if torch.is_tensor(orig_size):
                    orig_size = orig_size.detach().cpu().tolist()
                records.append(
                    {
                        "image_path": target.get("image_path"),
                        "label_path": target.get("label_path"),
                        "orig_size": orig_size,
                        "predictions": prediction,
                    }
                )
            print(
                f"[chachak] {self.name}: batch {batch_index} "
                f"frames={len(images)} total={len(records)}"
            )

        prediction_path = output_dir / "predictions.pt"
        torch.save(records, prediction_path)
        print(f"[chachak] Saved predictions: {prediction_path} records={len(records)}")

        if not compute_metrics:
            return {"prediction_path": prediction_path, "records": records,
                    "metrics": {"prediction_only": True,
                                "eval_seconds": round(time.perf_counter() - started, 3)}}

        metrics = evaluate_detection(
            all_predictions,
            all_targets,
            iou_thresholds=self.config.iou_thresholds,
            score_threshold=self.config.score_threshold,
            map_score_threshold=self.config.map_score_threshold,
            num_classes=num_classes,
            prediction_classes=prediction_classes,
            target_classes=target_classes,
            eval_classes=eval_classes,
        )
        metrics["eval_seconds"] = round(time.perf_counter() - started, 3)
        print(
            f"[chachak] {self.name} metrics: map50={metrics.get('map50')} "
            f"map50_95={metrics.get('map50_95')} precision={metrics.get('precision')} "
            f"recall={metrics.get('recall')}"
        )
        _write_hard_images(
            prediction_path,
            all_predictions,
            all_targets,
            records,
            config=None,
            prediction_classes=prediction_classes,
            target_classes=target_classes,
            eval_classes=eval_classes,
            score_threshold=self.config.score_threshold,
            top_k_fraction=EVAL_HARD_IMAGES_FRACTION,
        )
        return {
            "prediction_path": prediction_path,
            "records": records,
            "metrics": _to_builtin(metrics),
        }


class BatchDetectPipeline(Pipeline):
    """Tile each frame, run the trained model per tile, remap + NMS-merge."""

    name = "batch_detect"

    def process_batch(self, images, targets):
        outputs = []
        for image in images:
            remapped = _tile_infer(self.model_adapter, image, self.config)
            outputs.append(
                merge_predictions(remapped, *_frame_size(image), self.config.tiling.nms_iou)
            )
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

    def process_batch(self, images, targets):
        image_ids = [target.get("image_path") for target in targets]
        return self._crop_infer_remap(images, image_ids=image_ids)


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

    def process_batch(self, images, targets):
        image_ids = [target.get("image_path") for target in targets]
        return self._crop_infer_remap(images, image_ids=image_ids)


class ChainedPipeline(Pipeline):
    """Run several pipelines and merge their per-frame predictions (stacking)."""

    name = "chain"

    def __init__(self, model_adapter, device, config, detector=None, pipelines=None, box_cache=None):
        super().__init__(model_adapter, device, config, detector=detector, box_cache=box_cache)
        self.pipelines = list(pipelines or [])
        if self.pipelines:
            self.name = "chain[" + "+".join(p.name for p in self.pipelines) + "]"

    def process_batch(self, images, targets):
        per_frame: List[List[torch.Tensor]] = [[] for _ in images]
        for pipe in self.pipelines:
            for i, preds in enumerate(pipe.process_batch(images, targets)):
                per_frame[i].append(preds)
        return [
            merge_predictions(per_frame[i], *_frame_size(images[i]), self.config.merge_nms_iou)
            for i in range(len(images))
        ]
