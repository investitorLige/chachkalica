"""Batched, per-row-bounds twins of the box geometry in ``chachak.boxes``.

Two things make these more than a vectorization exercise.

**Per-row bounds.** The scalar originals take ``frame_w``/``frame_h`` as python ints, so they
can only describe one frame — or one crop — at a time. Everything here accepts a tensor
instead, so a single call can clip N crops against N *different* extents. That is what lets the
tail process a whole batch's detections in one shot rather than looping per crop, and it is not
expressible in the originals at all.

**float64.** ``expand_box``, ``grow_box_to_min_size`` and ``crop_image`` all begin with
``float(v) for v in box_xyxy``, which promotes to a python float — i.e. **float64** — and do
every subsequent operation at that width before ``round()`` lands on an integer. A float32
tensor version computes different numbers, and since the result feeds an integer pixel slice, a
half-ULP difference can move a crop boundary by a whole pixel. So :func:`expand_boxes`,
:func:`grow_boxes_to_min_size` and :func:`crop_bounds` compute in float64 regardless of the
input dtype. float32 -> float64 is exact (every float32 is representable), so this is a
widening, never a change of value.

The conversions that mirror ``friendy_chachkalica.formats`` keep the input's dtype, because
those *are* float32 in the original — ``chachak.boxes`` calls them on model output.

One inherited rule worth restating: :func:`xyxy_to_xywhn_batched` divides by a **tensor**, as
``formats.xyxy_to_xywhn`` does with its ``new_tensor`` normalizer. On CUDA
``tensor / python_float`` lowers to a multiply by the reciprocal, which differs from true
division by 1-2 ULP; dividing by a real tensor selects the correctly-rounded kernel. Same
reasoning as ``trt_infer.postprocess_torch._divisors``. Do not "simplify" any division here
into a scalar.

This module never imports ``chachak.boxes`` or ``chachak._friendy``: the latter puts the repo
root on ``sys.path`` and imports the training package, which a vendored bundle must not do.
The originals are the *test* reference (``tests/test_geometry.py``), not a runtime dependency.
"""

from __future__ import annotations

from typing import Tuple, Union

Bound = Union[int, float, "object"]  # a python number or a broadcastable tensor


def _bound(value, reference, dtype=None):
    """Coerce a frame extent to a tensor on ``reference``'s device.

    A python number becomes a 0-dim tensor, which broadcasts against any batch shape. A
    tensor is cast and returned as-is, so a caller passing per-row extents of shape
    ``boxes.shape[:-1]`` gets per-row behaviour with no other code change.
    """
    import torch

    dtype = reference.dtype if dtype is None else dtype
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=dtype)
    return torch.tensor(float(value), dtype=dtype, device=reference.device)


def clip_xyxy_batched(boxes, frame_w: Bound, frame_h: Bound):
    """Clip absolute xyxy boxes to frame bounds. Twin of ``formats.clip_xyxy``.

    ``boxes`` is ``[..., 4]`` and the bounds broadcast against ``boxes.shape[:-1]``. The
    original's two-argument ``clamp(0, image_width)`` is ``min(max(x, 0), width)``, reproduced
    here as a ``clamp(min=0)`` followed by ``torch.minimum`` so the upper bound can be a
    tensor. Both forms are pure comparison and selection — no arithmetic, so no rounding to
    diverge on.
    """
    import torch

    if boxes.numel() == 0:
        return boxes.clone()
    boxes = boxes.clone()
    width = _bound(frame_w, boxes).unsqueeze(-1)
    height = _bound(frame_h, boxes).unsqueeze(-1)
    boxes[..., 0::2] = torch.minimum(boxes[..., 0::2].clamp(min=0), width)
    boxes[..., 1::2] = torch.minimum(boxes[..., 1::2].clamp(min=0), height)
    return boxes


def xyxy_to_xywh_batched(boxes):
    """Absolute xyxy -> absolute centre-xywh. Twin of ``formats.xyxy_to_xywh``.

    ``width / 2`` stays a python scalar deliberately: 1/2 is exactly representable, so the
    multiply-by-reciprocal lowering cannot round differently. Only divisions by a
    non-power-of-two need a tensor.
    """
    import torch

    if boxes.numel() == 0:
        return boxes.clone()
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    width = x2 - x1
    height = y2 - y1
    return torch.stack([x1 + width / 2, y1 + height / 2, width, height], dim=-1)


def xyxy_to_xywhn_batched(boxes, frame_w: Bound, frame_h: Bound):
    """Absolute xyxy -> frame-normalized centre-xywh. Twin of ``formats.xyxy_to_xywhn``.

    The normalizer is a tensor, matching the original's ``new_tensor`` — see the module
    docstring on why that is load-bearing rather than stylistic.
    """
    import torch

    if boxes.numel() == 0:
        return boxes.clone()
    xywh = xyxy_to_xywh_batched(boxes)
    width = _bound(frame_w, boxes)
    height = _bound(frame_h, boxes)
    # Stacking on the last axis covers both cases: four 0-dim bounds give (4,), which
    # broadcasts, and four per-row bounds give (..., 4), which is already aligned.
    normalizer = torch.stack([width, height, width, height], dim=-1)
    return xywh / normalizer


def xywhn_to_xyxy_batched(boxes, frame_w: Bound, frame_h: Bound):
    """Frame-normalized centre-xywh -> absolute xyxy. Twin of ``boxes._xywhn_to_xyxy_tensor``.

    The original multiplies by a python int; this multiplies by a tensor. Both are exact for
    any frame extent below 2**24, which every real frame is, and a multiply has no reciprocal
    lowering to worry about either way.
    """
    import torch

    if boxes.numel() == 0:
        return boxes.clone()
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    width = _bound(frame_w, boxes)
    height = _bound(frame_h, boxes)
    return torch.stack(
        [
            (cx - w / 2) * width,
            (cy - h / 2) * height,
            (cx + w / 2) * width,
            (cy + h / 2) * height,
        ],
        dim=-1,
    )


def expand_boxes(boxes, ratio: float, frame_w: Bound, frame_h: Bound):
    """Pad xyxy boxes outward by ``ratio`` of each side, clipped to frame. Twin of
    ``chachak.boxes.expand_box``.

    Computed in float64 — see the module docstring. ``ratio`` is compared against zero on the
    host, which is free: it is a config value, not device data.
    """
    import torch

    if boxes.numel() == 0:
        return boxes.to(torch.float64).clone()
    boxes = boxes.to(torch.float64)
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    if ratio > 0:
        dw = (x2 - x1) * ratio / 2.0
        dh = (y2 - y1) * ratio / 2.0
        x1, y1, x2, y2 = x1 - dw, y1 - dh, x2 + dw, y2 + dh
    width = _bound(frame_w, boxes, dtype=torch.float64)
    height = _bound(frame_h, boxes, dtype=torch.float64)
    return torch.stack(
        [
            x1.clamp(min=0.0),
            y1.clamp(min=0.0),
            torch.minimum(x2, width),
            torch.minimum(y2, height),
        ],
        dim=-1,
    )


def _grow_axis(lo, hi, total, min_size: float):
    """One axis of ``chachak.boxes.grow_box_to_min_size``, vectorized.

    The original's ``_grow`` applies **two sequential corrections** after re-centring, and the
    second reads the value the first produced:

    * ``if lo < 0: hi -= lo; lo = 0.0`` — note ``hi`` is adjusted using the *old* ``lo``.
    * ``if hi > total: lo -= hi - total; hi = float(total)`` — using the ``hi`` from above.

    Reproduced with two ordered ``torch.where`` pairs. Collapsing them into one masked update
    would drop the dependency and silently change any box that needs both corrections, which
    is every box on a frame narrower than ``min_size``.
    """
    import torch

    need = (hi - lo) < min_size
    centre = (lo + hi) / 2.0
    floor = torch.minimum(
        torch.tensor(float(min_size), dtype=total.dtype, device=total.device), total
    )
    half = floor / 2.0
    new_lo, new_hi = centre - half, centre + half

    below = new_lo < 0
    new_hi = torch.where(below, new_hi - new_lo, new_hi)
    new_lo = torch.where(below, torch.zeros_like(new_lo), new_lo)

    above = new_hi > total
    new_lo = torch.where(above, new_lo - (new_hi - total), new_lo)
    new_hi = torch.where(above, total.to(new_hi.dtype), new_hi)

    new_lo = new_lo.clamp(min=0.0)
    return torch.where(need, new_lo, lo), torch.where(need, new_hi, hi)


def grow_boxes_to_min_size(boxes, min_size: float, frame_w: Bound, frame_h: Bound):
    """Grow xyxy boxes outward, same centre, so neither side is below ``min_size``.

    Twin of ``chachak.boxes.grow_box_to_min_size``, float64. Prefers real frame pixels over
    padding, so a short side is widened using neighbouring image content and shifted back
    inside the frame if the centred window would overrun an edge. Only a frame itself smaller
    than ``min_size`` leaves a side short — the caller then upscales, which is
    ``crops_exact.upscale_to_min_size``'s job, not this one's.
    """
    import torch

    if boxes.numel() == 0:
        return boxes.to(torch.float64).clone()
    boxes = boxes.to(torch.float64)
    width = _bound(frame_w, boxes, dtype=torch.float64)
    height = _bound(frame_h, boxes, dtype=torch.float64)
    x1, x2 = _grow_axis(boxes[..., 0], boxes[..., 2], width, min_size)
    y1, y2 = _grow_axis(boxes[..., 1], boxes[..., 3], height, min_size)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def crop_bounds(boxes, frame_w: Bound, frame_h: Bound):
    """Integer pixel bounds for each xyxy box. Twin of ``chachak.boxes.crop_image``'s
    rounding, without taking the slice.

    Returns int64 ``[..., 4]`` of ``(ix1, iy1, ix2, iy2)`` — exactly the four numbers
    ``crop_image`` computes before ``image[:, iy1:iy2, ix1:ix2]``. Split out because those
    numbers are the one thing that genuinely has to reach the host: a slice of unknown extent
    needs concrete integers, and there is no tensor-bounded formulation of it.

    The clamp order follows the original exactly — ``min`` against the far edge first, then
    ``max`` against the near one, and ``ix2``/``iy2`` floored at ``ix1 + 1`` so a degenerate
    box still yields at least one pixel. ``torch.round`` is half-to-even, as python's
    ``round()`` is, so the two agree on ties.
    """
    import torch

    if boxes.numel() == 0:
        return torch.zeros(
            (*boxes.shape[:-1], 4), dtype=torch.int64, device=boxes.device
        )
    boxes = boxes.to(torch.float64)
    width = _bound(frame_w, boxes, dtype=torch.float64).to(torch.int64)
    height = _bound(frame_h, boxes, dtype=torch.float64).to(torch.int64)
    zero = torch.zeros((), dtype=torch.int64, device=boxes.device)

    rounded = torch.round(boxes).to(torch.int64)
    ix1 = torch.maximum(zero, torch.minimum(rounded[..., 0], width - 1))
    iy1 = torch.maximum(zero, torch.minimum(rounded[..., 1], height - 1))
    ix2 = torch.maximum(ix1 + 1, torch.minimum(rounded[..., 2], width))
    iy2 = torch.maximum(iy1 + 1, torch.minimum(rounded[..., 3], height))
    return torch.stack([ix1, iy1, ix2, iy2], dim=-1)


def crop_extents(bounds) -> Tuple["object", "object"]:
    """``(crop_w, crop_h)`` in frame pixels from :func:`crop_bounds` output.

    These are the extents predictions get remapped by, so they must stay the *region's* size
    in frame pixels even when the crop tensor is later upscaled — the same invariant
    ``chachak.boxes.upscale_crop_to_min_size`` documents.
    """
    return bounds[..., 2] - bounds[..., 0], bounds[..., 3] - bounds[..., 1]
