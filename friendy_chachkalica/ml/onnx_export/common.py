"""Shared helpers for the per-arch exporters: the meta.json builder and the
standard ``torch.onnx.export`` call for a wrapper that emits ``(boxes, scores,
labels)`` with dynamic input H/W.

Two output shapes are supported, picked by ``batch_aware``:

* Default (``batch_aware=False``): ``(boxes[N,4], scores[N], labels[N])`` — the
  batch axis is indexed away inside the wrapper, N is a *detection* count and is
  dynamic (real NMS/threshold output, as in yolox/retinanet/fasterrcnn's plain
  ONNX export). Only ever correct for a batch of 1.
* ``batch_aware=True``: ``(boxes[B,K,4], scores[B,K], labels[B,K])`` — the
  wrapper carries the real batch dim ``B`` through instead of indexing it away,
  and the per-image detection count ``K`` is a fixed constant baked in at export
  time (a static top-k, no data-dependent op). Used by the DETR-family
  exporters (ecdet/rtdetr/rfdetr), whose top-k is already static per image —
  see ``arch/ecdet.py`` for why that makes batching free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Keep in lockstep with onnx_infer.meta.SCHEMA_VERSION (the shared contract).
SCHEMA_VERSION = 1

# Contract-A output names, in order.
OUTPUT_NAMES = ["boxes", "scores", "labels"]
INPUT_NAME = "pixel_values"


def build_meta(
    *,
    arch: str,
    num_classes: int,
    class_map: dict,
    score_threshold: float,
    resize_mode: str = "none",
    size: Optional[int] = None,
    max_size: Optional[int] = None,
    multiple: int = 0,
    pad_value: float = 0.0,
    input_scale: str = "unit",
    normalize: Optional[dict] = None,
    clip_boxes: bool = False,
    box_coords: str = "input_pixels",
) -> dict:
    input_spec = {
        "resize_mode": resize_mode,
        "multiple": multiple,
        "pad_value": pad_value,
        "input_scale": input_scale,
    }
    if size is not None:
        input_spec["size"] = size
    if max_size is not None:
        input_spec["max_size"] = max_size

    return {
        "schema_version": SCHEMA_VERSION,
        "arch": arch,
        "num_classes": int(num_classes),
        "class_map": {int(k): str(v) for k, v in class_map.items()},
        "score_threshold": float(score_threshold),
        "input": input_spec,
        "normalize": normalize,
        "layout": "rgb",
        "box_coords": box_coords,
        "clip_boxes": bool(clip_boxes),
    }


def export_detection_wrapper(
    wrapper,
    onnx_path: str | Path,
    *,
    dummy_hw: tuple[int, int] = (640, 640),
    opset: int = 17,
    dynamic_input_hw: bool = True,
    batch_aware: bool = False,
) -> None:
    """``torch.onnx.export`` a wrapper emitting ``(boxes, scores, labels)``.

    ``batch_aware=False`` (default): ``forward(pixel_values[1,3,H,W]) ->
    (boxes[N,4], scores[N], labels[N])``, N a dynamic detection count.

    ``batch_aware=True``: ``forward(pixel_values[B,3,H,W]) -> (boxes[B,K,4],
    scores[B,K], labels[B,K])``, B the real (dynamic) batch dim and K a fixed
    per-image detection count baked in at trace time — see the module docstring.
    """
    import torch

    dummy = torch.rand(1, 3, dummy_hw[0], dummy_hw[1])
    input_axes = {0: "batch"}
    if dynamic_input_hw:
        input_axes.update({2: "height", 3: "width"})
    det_axis = {0: "batch"} if batch_aware else {0: "num_dets"}
    dynamic_axes = {
        INPUT_NAME: input_axes,
        "boxes": dict(det_axis),
        "scores": dict(det_axis),
        "labels": dict(det_axis),
    }
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=[INPUT_NAME],
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            # Legacy TorchScript exporter: torchvision detection models (and the
            # custom torchvision::nms op) have symbolics registered for it, and it
            # avoids the newer dynamo path's onnxscript dependency.
            dynamo=False,
        )
