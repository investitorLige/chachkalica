"""Faster R-CNN ONNX exporter.

Like RetinaNet, torchvision's ``FasterRCNN`` carries its own preprocessing
(``GeneralizedRCNNTransform``: ImageNet normalize on [0,1] inputs + internal
resize) and postprocessing (RPN proposal NMS, RoIAlign, box-head decode +
score-threshold + per-class NMS + clip), mapping boxes back to the *input
image's* pixel frame. torchvision ships ONNX symbolics for ``roi_align`` and
``nms``/``batched_nms`` specifically so two-stage detectors like this one trace
through the legacy TorchScript exporter, so the wrapper is thin — feed a
``[1,3,H,W]`` float image in [0,1], get ``(boxes, scores, labels)`` in
input-pixel xyxy — and the service does no resize/normalize.

Unlike RetinaNet, the RPN + RoI heads make this graph's NMS/top-k data-dependent
in more than one place (proposal NMS feeding RoIAlign, then per-class NMS landing
inside ONNX ``If`` subgraphs). TensorRT can parse this graph but not *build* it —
Myelin rejects data-dependent shapes outside the top-level scope — so there is no
passthrough TRT path. ``trt_export/arch/fasterrcnn.py`` instead re-exports a
raw-output graph (fixed top-K RPN proposals + per-class boxes) and appends
``EfficientNMS_TRT``. This ONNX Runtime export is unaffected by any of that.
"""

from __future__ import annotations

from pathlib import Path

try:
    from ..common import build_meta, export_detection_wrapper
except ImportError:  # run as a flat script
    from common import build_meta, export_detection_wrapper


def export_fasterrcnn(adapter, *, num_classes, params, class_map, onnx_path: str | Path) -> dict:
    import torch
    from torch import nn

    class FasterRCNNExport(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            # torchvision detection models index their input like a list of
            # [C,H,W] images; a [1,3,H,W] tensor yields a single detection dict.
            detections = self.model(pixel_values)
            det = detections[0]
            # torchvision reserves label 0 for background and emits foreground
            # classes as 1..num_classes; shift back to the 0-indexed dataset ids
            # the meta class_map (and every other arch) uses.
            return det["boxes"], det["scores"], det["labels"] - 1

    model = adapter.model.eval()
    wrapper = FasterRCNNExport(model)
    export_detection_wrapper(wrapper, onnx_path)

    score_threshold = float(getattr(model.roi_heads, "score_thresh", 0.05))

    return build_meta(
        arch="fasterrcnn",
        num_classes=num_classes,
        class_map=class_map,
        score_threshold=score_threshold,
        resize_mode="none",
        input_scale="unit",
        normalize=None,  # folded into the graph (GeneralizedRCNNTransform)
    )
