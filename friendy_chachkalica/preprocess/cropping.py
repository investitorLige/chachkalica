"""Crop a training batch for person-crop pipelines (people_detect_first, batch_people).

When an experiment attaches a person-crop pipeline, the model is trained on the
same person crops it is validated/tested on: a person detector locates people in
each full frame, each person box is expanded and cropped, and the frame's
ground-truth boxes are re-mapped into every crop's local coordinates so each crop
becomes an independent training sample. Loss is then computed per crop inside the
adapter's ``training_step`` (the same differentiable path tiling uses).

The crop *windows* are produced by the chachak pipeline itself
(:meth:`chachak.pipeline.Pipeline.crop_regions`) — the exact code path used at
inference — so training and serving see identical crops. Re-mapping the
*targets* into those windows lives here because inference never needs it (the
eval pipeline predicts per crop and merges detections back to the frame; it does
not split labels).

Consequence worth noting: ground-truth objects that fall outside every detected
person crop are never presented to the model at train time. This mirrors
inference, where such objects are equally unreachable — the person detector's
recall is the ceiling for both regimes — but it does mean a person-crop pipeline
cannot learn objects the detector never frames.
"""

from typing import Any, Dict, Iterator, List, Tuple

import torch

try:
    from .window_targets import remap_target_to_window
except ImportError:  # run as a flat script
    from window_targets import remap_target_to_window

# A cropped box is kept only if at least this fraction of its original area falls
# inside the person crop. Matches tiling's floor (tiling._MIN_VISIBLE_FRACTION)
# so both train-time transforms treat edge slivers identically.
_MIN_VISIBLE_FRACTION = 0.1


def crop_batch_regions(
    images: List[torch.Tensor],
    targets: List[Dict[str, Any]],
    pipeline: Any,
) -> Iterator[Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Yield one tuple per surviving person-crop region in a batch:
    ``(crop_chw, crop_target, source_target, offset_xy, crop_size, frame_size)``.

    ``crop_target`` is the frame's target re-mapped into the crop's local
    coordinates (:func:`window_targets.remap_target_to_window`); ``source_target``
    is the original, un-remapped frame target (carries ``image_path`` etc.).
    ``offset_xy``/``crop_size``/``frame_size`` are pixel geometry in the *source*
    frame — the exact bookkeeping needed to later remap a crop's boxes back onto
    the full frame (see ``crop_cache.remap_crop_boxes_to_frame``). Frames with no
    detected people yield nothing; crops containing only a tiny clipped object
    sliver are skipped, matching :func:`remap_target_to_window`'s own rule.

    Shared by :func:`crop_batch` (training/eval, which only needs the flat
    image/target lists) and :func:`crop_cache.build_crop_cache` (which also needs
    the geometry, to write a manifest for later full-frame remapping).
    """
    image_ids = [target.get("image_path") for target in targets]
    with torch.no_grad():
        regions_per_frame = pipeline.crop_regions(images, image_ids=image_ids)

    for image, regions, target in zip(images, regions_per_frame, targets):
        frame_w, frame_h = int(image.shape[-1]), int(image.shape[-2])
        for crop, (x0, y0), (crop_w, crop_h) in regions:
            new_target = remap_target_to_window(
                target, x0, y0, crop_w, crop_h, _MIN_VISIBLE_FRACTION
            )
            if new_target is None:
                continue
            yield crop, new_target, target, (x0, y0), (crop_w, crop_h), (frame_w, frame_h)


def crop_batch(
    images: List[torch.Tensor],
    targets: List[Dict[str, Any]],
    pipeline: Any,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """Expand a batch of full frames into per-person-crop ``(image, target)`` samples.

    Person crops come from ``pipeline.crop_regions`` (run under ``no_grad``: it
    only forwards the frozen person detector, never the trained model), and each
    frame's targets are re-mapped per crop by
    :func:`window_targets.remap_target_to_window`. Frames with no detected people
    contribute nothing; crops containing only tiny clipped object slivers are
    ignored, while person crops with no ground-truth object are retained as
    background negatives. Returns flat lists suitable for feeding to an adapter's
    ``training_step`` (optionally re-chunked into micro-batches by the caller).
    """
    crop_images: List[torch.Tensor] = []
    crop_targets: List[Dict[str, Any]] = []
    for crop, new_target, _source_target, _offset, _crop_size, _frame_size in crop_batch_regions(
        images, targets, pipeline
    ):
        crop_images.append(crop)
        crop_targets.append(new_target)
    return crop_images, crop_targets
