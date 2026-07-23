"""Shared helpers for the per-arch exporters: the meta.json builder and the
standard ``torch.onnx.export`` call for a wrapper that emits ``(boxes, scores,
labels)`` with a dynamic detection count and dynamic input H/W.
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
) -> None:
    """``torch.onnx.export`` a wrapper ``forward(pixel_values[1,3,H,W]) ->
    (boxes[N,4], scores[N], labels[N])``."""
    import torch

    dummy = torch.rand(1, 3, dummy_hw[0], dummy_hw[1])
    input_axes = {0: "batch"}
    if dynamic_input_hw:
        input_axes.update({2: "height", 3: "width"})
    dynamic_axes = {
        INPUT_NAME: input_axes,
        "boxes": {0: "num_dets"},
        "scores": {0: "num_dets"},
        "labels": {0: "num_dets"},
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
