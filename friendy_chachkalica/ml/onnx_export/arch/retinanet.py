"""RetinaNet ONNX exporter.

torchvision detection models carry their own preprocessing
(``GeneralizedRCNNTransform``: ImageNet normalize on [0,1] inputs + internal
resize) and postprocessing (decode + score-threshold + NMS + clip), and map the
returned boxes back to the *input image's* pixel frame. So the wrapper is thin —
feed a ``[1,3,H,W]`` float image in [0,1], get ``(boxes, scores, labels)`` in
input-pixel xyxy — and the service does no resize/normalize.

torchvision's score floor is 0.05 (``score_thresh`` default); we record that as
the meta default. Callers can raise the effective threshold at predict time.
"""

from __future__ import annotations

from pathlib import Path

try:
    from ..common import build_meta, export_detection_wrapper
except ImportError:  # run as a flat script
    from common import build_meta, export_detection_wrapper


def export_retinanet(adapter, *, num_classes, params, class_map, onnx_path: str | Path) -> dict:
    import torch
    from torch import nn

    class RetinaNetExport(nn.Module):
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

    wrapper = RetinaNetExport(adapter.model.eval())
    export_detection_wrapper(wrapper, onnx_path)

    return build_meta(
        arch="retinanet",
        num_classes=num_classes,
        class_map=class_map,
        score_threshold=0.05,  # torchvision's internal score_thresh floor
        resize_mode="none",
        input_scale="unit",
        normalize=None,  # folded into the graph (GeneralizedRCNNTransform)
    )
