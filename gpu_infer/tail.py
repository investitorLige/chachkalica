"""The postprocess tail, carrying padded tensors and a validity mask instead of compacting.

Everything downstream of an engine execution in ``chachak``/``trt_infer`` selects with a
boolean index — ``to_friendy_torch``'s ``boxes[keep]``, ``Detector.predict``'s ``preds[mask]``,
``remap_local_preds_to_frame``'s ``boxes[valid]``, ``_unpack_efficientnms``'s
``det_boxes[i, :n]``. Every one of those sizes its output from **device** data, so torch has to
synchronize to know how much memory to allocate. That is where this pipeline's ~14 stalls per
frame come from, not from the arithmetic.

So nothing here compacts. Detections stay in a fixed ``[N, K, 6]`` buffer — ``N`` regions
(crops or tiles), ``K`` the engine's static detection axis — with a parallel ``[N, K]`` boolean
``valid``. Dropping a detection means clearing its bit, which has a static shape and therefore
no synchronization. The single compaction happens once, in :func:`compact`, at the point where
results are leaving for the host anyway.

Two invariants that make the masked form bit-identical to the compacted one:

* Every operation here is **elementwise or per-row broadcast**, so computing it on a padded
  row cannot change a valid row's result. Garbage in a pad slot stays in that slot.
* The reference thresholds *before* the coordinate inverse. Doing it after, as the mask form
  must, gives the same numbers for the same rows — the threshold only selects, it does not
  feed the arithmetic.

Inherited rule, restated because it is easy to lose in a refactor: divisions by a scale or a
frame extent go through a **tensor**, never a python float. See
``trt_infer.postprocess_torch._divisors`` for the measurement — on CUDA the scalar form becomes
a multiply by the reciprocal and lands 1-2 ULP away, which is enough to move a borderline
detection across the score threshold.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

#: Column layout of the ``[..., 6]`` detection tensors, matching Friendy's prediction format
#: (``friendy_chachkalica.formats.FRIENDY_PREDICTION_COLUMNS``).
BOX_COLUMNS = slice(0, 4)
SCORE_COLUMN = 4
CLASS_COLUMN = 5

#: Column layout of the ``[N, 6]`` transform tensor :func:`transforms_tensor` builds.
_SCALE_X, _SCALE_Y, _PAD_X, _PAD_Y, _NORM_W, _NORM_H = range(6)


def mask_from_num_detections(num_detections, max_det: int):
    """``[N, K]`` validity from EfficientNMS's per-image detection counts.

    This is what replaces the host read at ``trt_infer/session.py:352``. That line does
    ``num_det.reshape(B, -1)[:, 0].cpu().tolist()`` so it can slice ``det_boxes[i, :n]``, and it
    is the one crossing in the original path that genuinely had to reach the host — the plugin
    zero-pads its tail and a score mask cannot stand in, because callers legitimately pass
    ``score_threshold=0.0`` (this repo's eval default is 0.001, so a zero-scored row is real).

    Comparing an ``arange`` against the count tensor answers the same question entirely on the
    device: row ``k`` of image ``i`` is real iff ``k < num_detections[i]``. Same result, no
    copy, no stall.
    """
    import torch

    counts = num_detections.reshape(num_detections.shape[0], -1)[:, 0]
    positions = torch.arange(max_det, device=counts.device).unsqueeze(0)
    return positions < counts.reshape(-1, 1).to(positions.dtype)


def transforms_tensor(transforms: Sequence, device, box_coords: str = "input_pixels"):
    """``[N, 6]`` float32 of ``(scale_x, scale_y, pad_x, pad_y, norm_w, norm_h)``.

    The batched stand-in for ``trt_infer.postprocess_torch._divisors``, built once per
    submission rather than once per region, and float32 for the same reason that function is:
    it matches numpy's NEP 50 weak promotion, which rounds a python-float scalar to the
    array's dtype before dividing.

    ``norm_*`` follows ``_divisors``' branch exactly — the original extent for
    ``input_pixels``, and ``1.0`` for ``input_normalized``, where boxes already arrive in
    ``[0, 1]`` over the model input.
    """
    import torch

    normalized = box_coords != "input_pixels"
    rows = [
        [
            float(transform.scale_x),
            float(transform.scale_y),
            float(transform.pad_x),
            float(transform.pad_y),
            1.0 if normalized else float(transform.orig_w),
            1.0 if normalized else float(transform.orig_h),
        ]
        for transform in transforms
    ]
    return torch.tensor(rows, dtype=torch.float32, device=device).reshape(-1, 6)


def decode_regions(
    boxes,
    scores,
    labels,
    valid,
    transforms,
    *,
    score_threshold: float,
    clip_boxes: bool = False,
    box_coords: str = "input_pixels",
) -> Tuple["object", "object"]:
    """Batched ``to_friendy_torch``: raw engine output -> ``[N, K, 6]`` + updated mask.

    ``boxes`` is ``[N, K, 4]`` xyxy in model-input pixels, ``scores``/``labels`` are
    ``[N, K]``, ``transforms`` is :func:`transforms_tensor`'s output. Returns detections
    normalized to **each region's own extent**, which is what
    :func:`remap_regions_to_frame` then lifts into the frame.

    Mirrors ``trt_infer.postprocess_torch.to_friendy_torch`` step for step, minus its two
    ``shape[0] == 0`` early returns — those read a device-dependent length, and a fully-masked
    batch is expressed by an all-false ``valid`` instead.
    """
    import torch

    boxes = boxes.to(torch.float32)
    scores = scores.to(torch.float32)
    valid = valid & (scores >= score_threshold)

    scale_x = transforms[:, _SCALE_X].reshape(-1, 1)
    scale_y = transforms[:, _SCALE_Y].reshape(-1, 1)
    pad_x = transforms[:, _PAD_X].reshape(-1, 1)
    pad_y = transforms[:, _PAD_Y].reshape(-1, 1)
    norm_w = transforms[:, _NORM_W].reshape(-1, 1)
    norm_h = transforms[:, _NORM_H].reshape(-1, 1)

    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    if box_coords == "input_pixels":
        # Invert preprocessing: model-input pixel -> the region's own pixel.
        x1, x2 = (x1 - pad_x) / scale_x, (x2 - pad_x) / scale_x
        y1, y2 = (y1 - pad_y) / scale_y, (y2 - pad_y) / scale_y
        if clip_boxes:
            x1, x2 = x1.clamp(min=0.0), x2.clamp(min=0.0)
            y1, y2 = y1.clamp(min=0.0), y2.clamp(min=0.0)
            x1, x2 = torch.minimum(x1, norm_w), torch.minimum(x2, norm_w)
            y1, y2 = torch.minimum(y1, norm_h), torch.minimum(y2, norm_h)

    width = x2 - x1
    height = y2 - y1
    dets = torch.stack(
        [
            (x1 + width / 2) / norm_w,
            (y1 + height / 2) / norm_h,
            width / norm_w,
            height / norm_h,
            scores,
            labels.to(torch.float32),
        ],
        dim=-1,
    )
    return dets, valid


def remap_regions_to_frame(
    dets,
    valid,
    offsets,
    local_sizes,
    frame_w,
    frame_h,
) -> Tuple["object", "object"]:
    """Batched ``chachak.boxes.remap_local_preds_to_frame``.

    ``dets`` is ``[N, K, 6]`` normalized to each region, ``offsets`` is ``[N, 2]`` pixel
    ``(x, y)``, ``local_sizes`` is ``[N, 2]`` pixel ``(w, h)``. Returns detections normalized
    to the full frame.

    The original's ``boxes[valid]`` drop of degenerate boxes (``x2 > x1 and y2 > y1`` after
    clipping) becomes a mask update — same boxes discarded, no synchronization to discard them.
    """
    import torch

    from . import geometry

    local_w = local_sizes[:, 0].reshape(-1, 1).to(dets.dtype)
    local_h = local_sizes[:, 1].reshape(-1, 1).to(dets.dtype)

    boxes = geometry.xywhn_to_xyxy_batched(dets[..., BOX_COLUMNS], local_w, local_h)
    offset_x = offsets[:, 0].reshape(-1, 1, 1).to(boxes.dtype)
    offset_y = offsets[:, 1].reshape(-1, 1, 1).to(boxes.dtype)
    boxes = boxes + torch.cat([offset_x, offset_y, offset_x, offset_y], dim=-1)

    boxes = geometry.clip_xyxy_batched(boxes, frame_w, frame_h)
    valid = valid & (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])

    xywhn = geometry.xyxy_to_xywhn_batched(boxes, frame_w, frame_h)
    return torch.cat([xywhn, dets[..., SCORE_COLUMN:]], dim=-1), valid


def build_class_lut(
    prediction_classes: Optional[Dict], eval_classes: Optional[Dict], device
):
    """An ``int64`` lookup table replacing ``metrics.remap_raw_predictions_to_eval_classes``.

    That function loops rows doing ``int(row[5].item())`` — one host synchronization *per
    detection* if the tensor is on the device. It is cheap today only because ``infer.py``
    hands it an already-host tensor, which makes it the tripwire for any attempt to keep the
    tail on the GPU. A table indexed by predicted class id, holding the eval id or ``-1`` for
    "this name is not in the eval space", does the same job in one gather.

    Returns ``None`` when no remap is needed (either map absent), matching the original's
    ``id_to_name is None`` passthrough. The dict coercions mirror ``_normalize_class_map``:
    ``{int(id): str(name)}``, and ``{str(name): int(id)}`` for the eval side, so duplicate
    names resolve last-wins exactly as they do there.
    """
    import torch

    if not prediction_classes or not eval_classes:
        return None
    id_to_name = {int(cid): str(name) for cid, name in prediction_classes.items()}
    name_to_id = {str(name): int(cid) for cid, name in eval_classes.items()}
    if not id_to_name:
        return None
    size = max(id_to_name) + 1
    lut = torch.full((size,), -1, dtype=torch.int64, device=device)
    for class_id, name in id_to_name.items():
        mapped = name_to_id.get(name)
        if mapped is not None:
            lut[class_id] = mapped
    return lut


def apply_class_lut(dets, valid, lut) -> Tuple["object", "object"]:
    """Rewrite the class column through ``lut``, clearing rows with no eval class.

    Predicted ids outside the table are dropped rather than clamped into it — the original
    drops them too, via ``id_to_name.get(...)`` returning ``None``. Clamping instead would
    quietly relabel an out-of-range detection as the last known class.
    """
    import torch

    if lut is None:
        return dets, valid
    predicted = dets[..., CLASS_COLUMN].to(torch.int64)
    in_range = (predicted >= 0) & (predicted < lut.shape[0])
    mapped = lut[predicted.clamp(0, lut.shape[0] - 1)]
    valid = valid & in_range & (mapped >= 0)
    dets = dets.clone()
    dets[..., CLASS_COLUMN] = mapped.to(dets.dtype)
    return dets, valid


def filter_class(dets, valid, class_id: int, score_threshold: float):
    """Keep one class above a threshold — batched ``chachak.detector.Detector.predict``.

    The original is ``(preds[:, 5].to(int64) == person_class_id) & (preds[:, 4] >= thr)``
    followed by ``preds[mask]``. Same predicate, mask instead of index.
    """
    import torch

    matches = dets[..., CLASS_COLUMN].to(torch.int64) == int(class_id)
    return valid & matches & (dets[..., SCORE_COLUMN] >= score_threshold)


def flatten_regions(dets, valid) -> Tuple["object", "object"]:
    """``[N, K, 6]`` + ``[N, K]`` -> ``[N*K, 6]`` + ``[N*K]``, for the per-frame merge.

    A reshape, so it is free and keeps the static shape. Which region a row came from stops
    mattering once its boxes are in frame coordinates — that is exactly what
    :func:`remap_regions_to_frame` established.
    """
    return dets.reshape(-1, dets.shape[-1]), valid.reshape(-1)


def concat_frames(det_list: Sequence, valid_list: Sequence) -> Tuple["object", "object"]:
    """Stack several regions' flattened detections for one frame.

    Used by ``chain``, which merges what several sub-pipelines produced for the same frame,
    and by the multi-model bundles whose ``extra_models`` each contribute detections.
    """
    import torch

    return torch.cat(list(det_list), dim=0), torch.cat(list(valid_list), dim=0)


def topk_trim(dets, valid, limit: Optional[int]) -> Tuple["object", "object"]:
    """Keep the ``limit`` highest-scoring valid detections, or everything if ``limit`` is None.

    ``None`` is the default because ``chachak.boxes.merge_predictions`` has **no** size cap, so
    any cap is a behaviour change. When set, invalid rows are pushed to ``-inf`` first so they
    can never displace a real detection, and the selection is a fixed-size ``topk`` rather than
    a sort-and-slice — same shape every call, so still no synchronization.
    """
    import torch

    if limit is None or dets.shape[0] <= limit:
        return dets, valid
    scores = torch.where(
        valid, dets[..., SCORE_COLUMN], torch.full_like(dets[..., SCORE_COLUMN], float("-inf"))
    )
    kept = torch.topk(scores, limit).indices
    return dets[kept], valid[kept]


def overflow_count(valid, limit: Optional[int]):
    """How many valid detections a :func:`topk_trim` at ``limit`` would discard.

    Returned as a **device** scalar so the caller can fold it into whatever synchronization it
    already performs, rather than adding one to find out. Reporting a silent truncation is the
    difference between "capped at 512" and "we quietly stopped at 512".
    """
    import torch

    if limit is None:
        return torch.zeros((), dtype=torch.int64, device=valid.device)
    return (valid.sum() - limit).clamp(min=0)


def compact(dets, valid):
    """``[M, 6]`` of just the valid rows — **the single synchronization point.**

    This is the one place a device-resident length becomes a host-side allocation, and it is
    deliberately the last thing that happens: everything above it runs on static shapes, so the
    host stays ahead of the device until results are leaving anyway.

    Rows keep their relative order, matching what a boolean index does in the original path.
    """
    return dets[valid]


def empty_detections(device=None, dtype=None):
    """``[0, 6]``, on ``device`` — the shape every caller expects for "nothing here".

    Takes the device from the caller rather than defaulting to CPU, for the reason
    ``chachak.boxes.merge_predictions`` documents at its own empty return: handing a
    device-side caller a bare CPU tensor is a silent device mismatch.
    """
    import torch

    return torch.zeros((0, 6), dtype=dtype or torch.float32, device=device)
