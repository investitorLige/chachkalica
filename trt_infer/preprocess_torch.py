"""Torch executor for ``onnx_infer.preprocess.plan_preprocess`` — the on-GPU twin
of that module's NumPy executor.

Why a second executor instead of making the first one duck-type over numpy and
torch: ``onnx_infer`` is deliberately torch-free and CPU-capable (exported bundles
ship plain ``onnxruntime`` and a CPU torch wheel, and ``--device cpu`` has to keep
working), while ``trt_infer`` already declares torch and a CUDA GPU as hard
dependencies. Keeping the torch code on this side preserves that split.

The two executors cannot drift on *geometry*: both consume the same
:class:`~onnx_infer.preprocess.PreprocessPlan`, and only ``plan_preprocess`` reads
``resize_mode`` or constructs a :class:`~onnx_infer.preprocess.Transform`. What
lives here is purely the arithmetic.

Running it on the GPU is the point. A person crop arrives from
``chachak.boxes.crop_image`` already in device memory; the NumPy executor forced a
device->host copy, a single-threaded fancy-index gather (~7M gathered elements
through ~8 temporaries — tens of milliseconds at a 960px canvas), and a host->device
copy back, all to feed an engine call measured in milliseconds.

``_resize_chw_torch`` is a **verbatim port** of ``onnx_infer.preprocess._resize_chw``'s
gather, not a call to ``F.interpolate``. The two compute the same bilinear formula,
but ``_sample_coords`` resolves source coordinates in float64 while torch's CUDA
bilinear kernel accumulates in float32 — a ~1e-5 (unit scale) / ~1e-2 (byte scale)
shift, enough to move a borderline detection across the score threshold. Both forms
are GPU-resident and cost tens of microseconds against a multi-millisecond engine
call, so the fused kernel would buy nothing worth re-validating mAP for. Keeping the
port exact is what lets an exported bundle reproduce the training repo's eval numbers
unchanged.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from onnx_infer.meta import ModelMeta
from onnx_infer.preprocess import Transform, plan_preprocess


def preprocess_torch(
    image_chw: torch.Tensor, meta: ModelMeta
) -> tuple[torch.Tensor, Transform]:
    """CHW float image -> the ``[1, 3, H, W]`` batch the graph expects, on its device.

    Mirrors ``onnx_infer.preprocess.preprocess`` step for step and returns the same
    :class:`Transform`. Device-agnostic — it runs correctly on a CPU tensor, which
    is what lets the numpy-vs-torch parity test run without a GPU.
    """
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(
            f"expected CHW image with 3 channels, got shape {tuple(image_chw.shape)}"
        )

    orig_h, orig_w = int(image_chw.shape[1]), int(image_chw.shape[2])
    plan = plan_preprocess(orig_h, orig_w, meta)

    # Person crops are non-contiguous slices of the frame; the gather below wants a
    # dense buffer, exactly as the numpy executor's np.ascontiguousarray does.
    img = image_chw.to(dtype=torch.float32).contiguous()

    if plan.flip_channels:
        img = img.flip(0)  # torch.flip materializes, so this is already contiguous

    if plan.scale_255:
        img = img * 255.0

    # 1. resize (a no-op when the plan's target equals the source size)
    resize_h, resize_w = plan.resize_hw
    img = _resize_chw_torch(img, resize_h, resize_w)

    # 2. normalize, before the pad so padded margins carry the raw pad_value
    if plan.mean is not None:
        mean, std = _norm_tensors(plan.mean, plan.std, str(img.device))
        img = (img - mean) / std

    # 3. pad bottom-right onto the plan's canvas
    canvas_h, canvas_w = plan.canvas_hw
    if (canvas_h, canvas_w) != (resize_h, resize_w):
        padded = torch.full(
            (3, canvas_h, canvas_w),
            plan.pad_value,
            dtype=torch.float32,
            device=img.device,
        )
        padded[:, :resize_h, :resize_w] = img
        img = padded

    return img.unsqueeze(0), plan.transform


def _resize_chw_torch(img: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Bilinear resize a CHW float image, ``align_corners=False`` (torch default).

    Operation-for-operation identical to ``onnx_infer.preprocess._resize_chw``:
    float64 coordinates, float32 weights, the same four-neighbour gather in the
    same order. ``index_select`` stands in for numpy's ``img[:, y0][:, :, x0]``.
    """
    _, in_h, in_w = img.shape
    if in_h == out_h and in_w == out_w:
        return img

    ys = _sample_coords_torch(out_h, in_h, img.device)
    xs = _sample_coords_torch(out_w, in_w, img.device)

    y0 = ys.floor().to(torch.int64)
    x0 = xs.floor().to(torch.int64)
    y1 = (y0 + 1).clamp(0, in_h - 1)
    x1 = (x0 + 1).clamp(0, in_w - 1)
    y0 = y0.clamp(0, in_h - 1)
    x0 = x0.clamp(0, in_w - 1)

    wy = (ys - ys.floor()).to(torch.float32)
    wx = (xs - xs.floor()).to(torch.float32)

    # Gather the four neighbours for every output pixel, per channel.
    rows_y0 = img.index_select(1, y0)
    rows_y1 = img.index_select(1, y1)
    top = rows_y0.index_select(2, x0) * (1 - wx) + rows_y0.index_select(2, x1) * wx
    bot = rows_y1.index_select(2, x0) * (1 - wx) + rows_y1.index_select(2, x1) * wx
    return top * (1 - wy).view(1, -1, 1) + bot * wy.view(1, -1, 1)


def _sample_coords_torch(out_size: int, in_size: int, device) -> torch.Tensor:
    """Source coordinates for each output index under half-pixel alignment.

    float64 throughout, matching ``onnx_infer.preprocess._sample_coords``. The
    tensors are one-dimensional and tiny, so the wider dtype costs nothing.
    """
    scale = in_size / out_size
    idx = torch.arange(out_size, dtype=torch.float64, device=device)
    coords = (idx + 0.5) * scale - 0.5
    return coords.clamp(0.0, float(in_size - 1))


@lru_cache(maxsize=32)
def _norm_tensors(
    mean: tuple[float, ...], std: tuple[float, ...], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcastable mean/std for a model, built once instead of per crop.

    Without this, every crop pays two 12-byte host->device copies. There is one
    distinct (mean, std) per loaded model, so the cache never grows. The returned
    tensors are read-only by contract — callers must not mutate them.
    """
    return (
        torch.tensor(mean, dtype=torch.float32, device=device).reshape(3, 1, 1),
        torch.tensor(std, dtype=torch.float32, device=device).reshape(3, 1, 1),
    )
