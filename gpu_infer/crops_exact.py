"""Per-region preprocessing that reuses ``trt_infer``'s verbatim resample.

This is the file that makes bit-parity hold, and it earns that by doing the *least* possible:
it slices regions out of a frame already in device memory and hands each one to
``trt_infer.preprocess_torch.preprocess_torch`` unchanged. No fused multi-crop kernel, no
re-derived geometry.

Why not one fused ``grid_sample``
--------------------------------

A single ``grid_sample`` could produce every crop already at model-input size from a
device-resident box tensor, and it would be a little faster. It would also **not be
bit-identical**, for four reasons worth writing down so this decision is not silently revisited:

* ``preprocess_torch._sample_coords_torch`` resolves source coordinates in **float64**;
  ``grid_sample``'s CUDA kernel does all coordinate arithmetic in float32. A ~1e-7 relative
  coordinate error is ~1e-5 absolute on a unit-scale pixel — the magnitude that module's own
  docstring already argues is enough to move a borderline detection across the threshold.
* The bilinear expression differs. The verbatim gather computes
  ``top = a(1-wx) + b*wx``, ``bot = c(1-wx) + d*wx``, ``out = top(1-wy) + bot*wy`` in that
  order with float32 weights; ``grid_sample`` computes a four-way area-weighted sum. Different
  expression tree, different rounding, even at identical coordinates.
* Boundary handling differs in kind — index clamping versus ``padding_mode``.
* ``upscale_to_min_size`` produces a **two-step** resample (region to floor, then floor to
  canvas). A fused grid does one. Double interpolation blurs differently, which is a
  larger-than-float divergence, not a rounding one.

The measured payoff for fusing was ~1-2% on this repo's batch-1 rfdetr stage 2, against an mAP
re-validation per architecture. Not worth it. The speedup this package delivers comes from
:mod:`gpu_infer.engine` removing the per-call synchronization, which costs no parity at all.

Grouping
--------

Regions whose preprocessed tensors come out the same shape can be submitted as one engine batch.
That is always true for ``resize_mode`` of ``square`` or ``letterbox`` (a fixed canvas regardless
of source size — every arch this repo exports), and can be false for ``none`` or
``longest_side``. The grouping mirrors ``trt_infer.adapter.TrtAdapter.predict:85-87`` exactly, so
a region can never end up in a batch with a differently-shaped one.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def upscale_to_min_size(crop, min_size: int):
    """Bilinearly upscale a CHW crop until neither side is under ``min_size``.

    Exact twin of ``chachak.boxes.upscale_crop_to_min_size`` — reimplemented rather than
    imported because ``chachak.boxes`` pulls ``chachak._friendy``, which puts the repo root on
    ``sys.path`` and imports the training package. A vendored bundle must not do that.

    Only engages when the source frame itself is narrower or shorter than the floor, because
    ``geometry.grow_boxes_to_min_size`` has already used every available real pixel. Interpolates
    rather than zero-pads so every output pixel stays real content, and scales both axes by one
    factor so the subject keeps its proportions.

    Returns the resized tensor only. The crop still covers the same region of the source frame,
    and callers must keep remapping with the *original* window size — normalized coordinates are
    invariant under this resize, absolute ones are not.
    """
    import torch.nn.functional as F

    _, height, width = crop.shape
    if height >= min_size and width >= min_size:
        return crop
    factor = max(min_size / height, min_size / width)
    target_h = max(min_size, int(round(height * factor)))
    target_w = max(min_size, int(round(width * factor)))
    return F.interpolate(
        crop.unsqueeze(0).float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def slice_region(frame, bounds, min_box_size: float = 0.0):
    """Cut one region out of a CHW frame, growing it to the floor if the frame is too small.

    ``bounds`` is ``(ix1, iy1, ix2, iy2)`` in host integers, as
    :func:`gpu_infer.geometry.crop_bounds` produces. The slice is a **view** — no pixels move —
    and only the upscale path, which is normally dead (``min_box_size`` defaults to ``0.0``),
    allocates.

    Mirrors ``chachak.pipeline.Pipeline.crop_regions:166-174``: the zero-area guard, then the
    floor check against the *frame-pixel* extent, then the upscale.
    """
    ix1, iy1, ix2, iy2 = (int(value) for value in bounds)
    crop = frame[:, iy1:iy2, ix1:ix2]
    crop_w, crop_h = ix2 - ix1, iy2 - iy1
    if crop_w < 1 or crop_h < 1:
        return None
    if min_box_size > 0 and (crop_w < min_box_size or crop_h < min_box_size):
        crop = upscale_to_min_size(crop, int(round(min_box_size)))
    return crop


def prepare_regions(regions: Sequence, meta) -> Tuple[List, List]:
    """Preprocess each region tensor. Returns ``(batched_tensors, transforms)``.

    Each entry of ``batched_tensors`` is the ``[1, 3, H, W]`` the graph expects, and each
    ``transforms`` entry is the matching :class:`~onnx_infer.preprocess.Transform` — the pair
    ``preprocess_torch`` returns, collected so the caller can group and submit them.

    ``preprocess_torch`` is imported here rather than at module scope because it imports torch
    at *its* module scope, which would leak into this package's import closure.
    """
    from trt_infer.preprocess_torch import preprocess_torch

    prepared, transforms = [], []
    for region in regions:
        batched, transform = preprocess_torch(region, meta)
        prepared.append(batched)
        transforms.append(transform)
    return prepared, transforms


def group_by_shape(prepared: Sequence) -> Dict[Tuple[int, ...], List[int]]:
    """Region indices bucketed by preprocessed shape, ready to stack per bucket.

    Same rule as ``TrtAdapter.predict``: group on ``batched.shape[1:]``, i.e. everything but the
    batch axis. Grouping only ever reduces engine calls; it never changes what a region sees.
    """
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for index, batched in enumerate(prepared):
        groups.setdefault(tuple(int(dim) for dim in batched.shape[1:]), []).append(index)
    return groups


def stack_group(prepared: Sequence, indices: Sequence[int]):
    """One contiguous ``[len(indices), 3, H, W]`` batch for a shape group.

    Contiguity is required, not incidental: :mod:`gpu_infer.engine` may hand TensorRT a pointer
    to row ``i`` of this buffer, which is only a valid single-image input if the rows are
    densely packed.
    """
    import torch

    if len(indices) == 1:
        return prepared[indices[0]].contiguous()
    return torch.cat([prepared[index] for index in indices], dim=0).contiguous()
