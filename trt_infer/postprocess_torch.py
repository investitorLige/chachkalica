"""Torch twins of ``onnx_infer``'s output handling, so engine outputs can be turned
into Friendy predictions without leaving the GPU.

Same split rationale as :mod:`trt_infer.preprocess_torch`: ``onnx_infer`` stays
pure-NumPy and CPU-capable for exported ONNX bundles, and the torch versions live
here where torch is already a hard dependency.

:func:`to_friendy_torch` mirrors ``onnx_infer.postprocess.to_friendy`` operation for
operation. :func:`adapt_outputs_torch` covers the ``PassthroughHandler`` case that
every current arch uses, and deliberately declines (returns ``None``) for any handler
that overrides ``adapt_outputs`` — a future arch whose graph layout we don't control
would otherwise be silently mis-decoded. The caller falls back to the numpy path.
"""

from __future__ import annotations

import torch

from onnx_infer.arch.base import PassthroughHandler
from onnx_infer.postprocess import FRIENDY_COLUMNS
from onnx_infer.preprocess import Transform


def adapt_outputs_torch(handler, outputs: list) -> tuple | None:
    """Torch twin of ``PassthroughHandler.adapt_outputs``; ``None`` if unsupported.

    The identity check is exact: a subclass that doesn't override ``adapt_outputs``
    resolves to the very same function object. Returning ``None`` rather than
    guessing keeps a newly added non-passthrough arch *correct* (via the numpy
    fallback) while the accompanying registry test flags that it needs a twin.
    """
    if type(handler).adapt_outputs is not PassthroughHandler.adapt_outputs:
        return None

    if len(outputs) < 3:
        raise ValueError(
            f"{handler.name}: expected 3 graph outputs (boxes, scores, labels), "
            f"got {len(outputs)}"
        )
    boxes, scores, labels = outputs[0], outputs[1], outputs[2]
    return (
        boxes.to(torch.float32).reshape(-1, 4),
        scores.to(torch.float32).reshape(-1),
        # Truncates toward zero, exactly as numpy's .astype(np.int64) does.
        labels.reshape(-1).to(torch.int64),
    )


def to_friendy_torch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    transform: Transform,
    score_threshold: float,
    clip_boxes: bool = False,
    box_coords: str = "input_pixels",
) -> torch.Tensor:
    """boxes ``[N,4]`` xyxy, scores ``[N]``, labels ``[N]`` -> Friendy ``[M,6]``.

    Mirrors ``onnx_infer.postprocess.to_friendy`` step for step — same threshold-first
    ordering, same coordinate inverse, same column order — and returns a tensor on the
    inputs' device. See that function for what ``box_coords`` and ``clip_boxes`` mean;
    the semantics are defined there and must not diverge here.
    """
    boxes = boxes.to(torch.float32).reshape(-1, 4)
    scores = scores.to(torch.float32).reshape(-1)
    labels = labels.reshape(-1)

    if boxes.shape[0] == 0:
        return _empty(boxes.device)

    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.shape[0] == 0:
        return _empty(boxes.device)

    scale_x, scale_y, norm_w, norm_h = _divisors(transform, box_coords, boxes.device)

    boxes = boxes.clone()
    if box_coords == "input_pixels":
        # Invert preprocessing: input-pixel -> original-image pixel.
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform.pad_x) / scale_x
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform.pad_y) / scale_y
        if clip_boxes:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, transform.orig_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, transform.orig_h)
    # else: input_normalized — already [0,1] over the model input, norm_* are 1.0.
    # The clamp is to [0,1] there, which is the same bound in that frame (these
    # archs stretch, so the model input's frame is the original image's).
    elif clip_boxes:
        boxes = boxes.clamp(0.0, 1.0)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2

    xywhn = torch.stack(
        [x_center / norm_w, y_center / norm_h, width / norm_w, height / norm_h],
        dim=1,
    )
    return torch.cat(
        [xywhn, scores.reshape(-1, 1), labels.reshape(-1, 1).to(torch.float32)],
        dim=1,
    ).to(torch.float32)


def _divisors(transform: Transform, box_coords: str, device):
    """The four scalar divisors, as 0-dim float32 tensors, in one host->device copy.

    They have to be tensors, not python floats. On CUDA, torch lowers
    ``tensor / python_scalar`` into a multiply by the reciprocal, which differs from
    numpy's true division by 1-2 ULP — measured, and the divergence matches
    ``a * (1 / s)`` exactly. Dividing by a real tensor selects the correctly-rounded
    division kernel and reproduces numpy bit for bit. (CPU already agreed either way;
    routing both backends through the same form keeps them identical to each other
    too.) ``width / 2`` elsewhere in this module is safe as a plain scalar because
    1/2 is exactly representable, so the reciprocal form cannot round differently.

    Casting to float32 here matches numpy's NEP 50 weak promotion, which likewise
    rounds a python-float scalar to the array's dtype before dividing.
    """
    norm = (
        (transform.orig_w, transform.orig_h)
        if box_coords == "input_pixels"
        else (1.0, 1.0)
    )
    return torch.tensor(
        [transform.scale_x, transform.scale_y, *norm],
        dtype=torch.float32,
        device=device,
    ).unbind()


def _empty(device) -> torch.Tensor:
    return torch.zeros((0, len(FRIENDY_COLUMNS)), dtype=torch.float32, device=device)
