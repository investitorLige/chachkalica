"""Generic, meta-driven preprocessing (pure NumPy).

Turns a CHW float image (RGB, [0, 1] — the same tensor ``chachak`` feeds the
trained-model adapters) into the ``[1, 3, H, W]`` batch the ONNX graph expects,
and records the exact :class:`Transform` it applied so :mod:`postprocess` can
invert it uniformly for every architecture.

Resizing uses an ``align_corners=False`` half-pixel bilinear that matches
``torch.nn.functional.interpolate``'s default, so a service-side resize stays in
parity with the training adapters. Archs where exact resize parity is critical
may instead bake the resize into the graph (``resize_mode: none``) — this module
supports both.
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


def preprocess(image_chw: np.ndarray, meta: ModelMeta) -> tuple[np.ndarray, Transform]:
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"expected CHW image with 3 channels, got shape {image_chw.shape}")

    orig_h, orig_w = int(image_chw.shape[1]), int(image_chw.shape[2])
    spec = meta.input
    img = np.ascontiguousarray(image_chw, dtype=np.float32)

    if spec.input_scale == "byte":
        img = img * 255.0

    # 1. resize
    if spec.resize_mode == "none":
        resized = img
        scale_x = scale_y = 1.0
    elif spec.resize_mode == "square":
        resized = _resize_chw(img, spec.size, spec.size)
        scale_x = spec.size / orig_w
        scale_y = spec.size / orig_h
    elif spec.resize_mode == "longest_side":
        longest = max(orig_h, orig_w)
        scale = spec.max_size / longest if longest > spec.max_size else 1.0
        new_h = max(1, round(orig_h * scale))
        new_w = max(1, round(orig_w * scale))
        resized = _resize_chw(img, new_h, new_w) if scale != 1.0 else img
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
    elif spec.resize_mode == "letterbox":
        # Aspect-preserving: scale (up or down) so the longest side hits
        # spec.size exactly, then pad bottom-right to (size, size) below.
        scale = spec.size / max(orig_h, orig_w)
        new_h = max(1, round(orig_h * scale))
        new_w = max(1, round(orig_w * scale))
        resized = _resize_chw(img, new_h, new_w)
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
    else:  # pragma: no cover - validated in ModelMeta
        raise ValueError(f"unsupported resize_mode {spec.resize_mode!r}")

    # 2. normalize (elementwise — exact, matches the adapter's (x - mean) / std)
    if meta.normalize is not None:
        mean = np.asarray(meta.normalize.mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(meta.normalize.std, dtype=np.float32).reshape(3, 1, 1)
        resized = (resized - mean) / std

    # 3. pad to a multiple, bottom-right (so box origin is unchanged: pad offset 0)
    pad_x = pad_y = 0
    if spec.multiple and spec.multiple > 1:
        _, rh, rw = resized.shape
        target_h = _ceil_to_multiple(rh, spec.multiple)
        target_w = _ceil_to_multiple(rw, spec.multiple)
        if target_h != rh or target_w != rw:
            padded = np.full((3, target_h, target_w), spec.pad_value, dtype=np.float32)
            padded[:, :rh, :rw] = resized
            resized = padded

    # 3b. letterbox pads to an exact (size, size) canvas, bottom-right — a
    # separate mechanism from the multiple-of-N pad above (rfdetr's export
    # sets multiple=0, so the two never overlap on the same call).
    if spec.resize_mode == "letterbox":
        _, rh, rw = resized.shape
        if rh != spec.size or rw != spec.size:
            padded = np.full((3, spec.size, spec.size), spec.pad_value, dtype=np.float32)
            padded[:, :rh, :rw] = resized
            resized = padded

    batched = resized[None].astype(np.float32, copy=False)
    return batched, Transform(scale_x, scale_y, pad_x, pad_y, orig_w, orig_h)


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
