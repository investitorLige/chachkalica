"""Generic, meta-driven preprocessing (pure NumPy).

Turns a CHW float image (RGB, [0, 1] — the same tensor ``chachak`` feeds the
trained-model adapters) into the ``[1, 3, H, W]`` batch the ONNX graph expects,
and records the exact :class:`Transform` it applied so :mod:`postprocess` can
invert it uniformly for every architecture.

Split in two: :func:`plan_preprocess` decides *what* to do from the meta alone
(pure arithmetic on two ints), and :func:`preprocess` executes that plan in
NumPy. The split exists because ``trt_infer`` executes the same plan in torch, on
the GPU, to keep person crops from round-tripping through host memory — see
``trt_infer/preprocess_torch.py``. Only :func:`plan_preprocess` ever constructs a
:class:`Transform` and only it branches on ``resize_mode``, so the two executors
cannot drift on geometry no matter how their kernels differ.

Resizing uses an ``align_corners=False`` half-pixel bilinear that matches
``torch.nn.functional.interpolate``'s default, so a service-side resize stays in
parity with the training adapters. Archs where exact resize parity is critical
may instead bake the resize into the graph (``resize_mode: none``) — this module
supports both.

``meta.layout == "bgr"`` reverses the channel axis before any scaling/resize —
for checkpoints trained outside this repo's own RGB convention (e.g. a raw
Megvii YOLOX checkpoint, whose native preprocessing is BGR/letterbox/[0,255]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .meta import ModelMeta


@dataclass(frozen=True)
class Transform:
    """The resize+pad the service applied, so boxes can be mapped back.

    Boxes come out of the graph in its own input-pixel frame (resized + padded).
    Original pixel = (input_pixel - pad) / scale.
    """

    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int


@dataclass(frozen=True)
class PreprocessPlan:
    """Every decision :func:`preprocess` makes, resolved from the meta alone.

    Holding the decisions separately from the arithmetic lets a second executor
    (``trt_infer/preprocess_torch.py``, which runs on the GPU) apply exactly the
    same geometry with different kernels. Executors read these fields and never
    look at ``resize_mode`` again.
    """

    flip_channels: bool                 # meta.layout == "bgr"
    scale_255: bool                     # input_scale == "byte"
    resize_hw: tuple[int, int]          # resize target; == the source size means no resize
    mean: tuple[float, float, float] | None
    std: tuple[float, float, float] | None
    canvas_hw: tuple[int, int]          # final canvas after padding; == resize_hw means no pad
    pad_value: float
    transform: Transform


def plan_preprocess(orig_h: int, orig_w: int, meta: ModelMeta) -> PreprocessPlan:
    """Resolve the meta + a source size into a concrete :class:`PreprocessPlan`.

    The single place ``resize_mode`` is interpreted and the single place a
    :class:`Transform` is built.
    """
    spec = meta.input

    if spec.resize_mode == "none":
        resize_h, resize_w = orig_h, orig_w
    elif spec.resize_mode == "square":
        resize_h = resize_w = spec.size
    elif spec.resize_mode == "longest_side":
        longest = max(orig_h, orig_w)
        scale = spec.max_size / longest if longest > spec.max_size else 1.0
        resize_h = max(1, round(orig_h * scale))
        resize_w = max(1, round(orig_w * scale))
    elif spec.resize_mode == "letterbox":
        # Aspect-preserving: scale (up or down) so the longest side hits
        # spec.size exactly, then pad bottom-right to (size, size).
        scale = spec.size / max(orig_h, orig_w)
        resize_h = max(1, round(orig_h * scale))
        resize_w = max(1, round(orig_w * scale))
    else:  # pragma: no cover - validated in ModelMeta
        raise ValueError(f"unsupported resize_mode {spec.resize_mode!r}")

    # Padding is always bottom-right, so the box origin never moves and pad_x/pad_y
    # stay 0. The two pad mechanisms below therefore compose into a single canvas:
    # constant-padding to an intermediate size and then to a larger one leaves the
    # same pixels as padding once to the larger one.
    canvas_h, canvas_w = resize_h, resize_w
    if spec.multiple and spec.multiple > 1:
        canvas_h = _ceil_to_multiple(canvas_h, spec.multiple)
        canvas_w = _ceil_to_multiple(canvas_w, spec.multiple)
    if spec.resize_mode == "letterbox":
        # rfdetr and yolox both export multiple=0, so this never has to shrink a
        # multiple-of-N canvas. Fail loudly rather than silently cropping if some
        # future meta combines the two incompatibly.
        if canvas_h > spec.size or canvas_w > spec.size:
            raise ValueError(
                f"letterbox canvas {spec.size} is smaller than the multiple-of-"
                f"{spec.multiple} pad {(canvas_h, canvas_w)}; the two pad "
                f"mechanisms cannot both be satisfied"
            )
        canvas_h = canvas_w = spec.size

    return PreprocessPlan(
        flip_channels=meta.layout == "bgr",
        scale_255=spec.input_scale == "byte",
        resize_hw=(resize_h, resize_w),
        mean=tuple(meta.normalize.mean) if meta.normalize is not None else None,
        std=tuple(meta.normalize.std) if meta.normalize is not None else None,
        canvas_hw=(canvas_h, canvas_w),
        pad_value=spec.pad_value,
        # scale_x/scale_y as a ratio of the resize target is exact for every mode:
        # "none" gives orig/orig == 1.0, "square" gives size/orig.
        transform=Transform(
            resize_w / orig_w, resize_h / orig_h, 0, 0, orig_w, orig_h
        ),
    )


def preprocess(image_chw: np.ndarray, meta: ModelMeta) -> tuple[np.ndarray, Transform]:
    """NumPy executor for :func:`plan_preprocess`."""
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"expected CHW image with 3 channels, got shape {image_chw.shape}")

    orig_h, orig_w = int(image_chw.shape[1]), int(image_chw.shape[2])
    plan = plan_preprocess(orig_h, orig_w, meta)

    img = np.ascontiguousarray(image_chw, dtype=np.float32)

    if plan.flip_channels:
        img = np.ascontiguousarray(img[::-1, :, :])

    if plan.scale_255:
        img = img * 255.0

    # 1. resize (a no-op when the plan's target equals the source size)
    resize_h, resize_w = plan.resize_hw
    img = _resize_chw(img, resize_h, resize_w)

    # 2. normalize (elementwise — exact, matches the adapter's (x - mean) / std).
    #    Before the pad, so padded margins carry the raw pad_value.
    if plan.mean is not None:
        mean = np.asarray(plan.mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(plan.std, dtype=np.float32).reshape(3, 1, 1)
        img = (img - mean) / std

    # 3. pad bottom-right onto the plan's canvas
    canvas_h, canvas_w = plan.canvas_hw
    if (canvas_h, canvas_w) != (resize_h, resize_w):
        padded = np.full((3, canvas_h, canvas_w), plan.pad_value, dtype=np.float32)
        padded[:, :resize_h, :resize_w] = img
        img = padded

    return img[None].astype(np.float32, copy=False), plan.transform


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _resize_chw(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize a CHW float image, ``align_corners=False`` (torch default)."""
    _, in_h, in_w = img.shape
    if in_h == out_h and in_w == out_w:
        return img

    ys = _sample_coords(out_h, in_h)
    xs = _sample_coords(out_w, in_w)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y0 = np.clip(y0, 0, in_h - 1)
    x0 = np.clip(x0, 0, in_w - 1)

    wy = (ys - np.floor(ys)).astype(np.float32)
    wx = (xs - np.floor(xs)).astype(np.float32)

    # Gather the four neighbours for every output pixel, per channel.
    top = img[:, y0][:, :, x0] * (1 - wx)[None, None, :] + img[:, y0][:, :, x1] * wx[None, None, :]
    bot = img[:, y1][:, :, x0] * (1 - wx)[None, None, :] + img[:, y1][:, :, x1] * wx[None, None, :]
    out = top * (1 - wy)[None, :, None] + bot * wy[None, :, None]
    return out.astype(np.float32)


def _sample_coords(out_size: int, in_size: int) -> np.ndarray:
    """Source coordinates for each output index under half-pixel alignment."""
    scale = in_size / out_size
    idx = np.arange(out_size, dtype=np.float64)
    coords = (idx + 0.5) * scale - 0.5
    return np.clip(coords, 0.0, in_size - 1)
