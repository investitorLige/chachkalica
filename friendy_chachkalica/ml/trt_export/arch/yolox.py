"""YOLOX → TensorRT via EfficientNMS.

The standard YOLOX ONNX bakes ``score >= conf`` boolean masking + ``batched_nms``
(a data-dependent subgraph TensorRT can't compile). Here we re-export the same
decode up to **raw boxes + per-class scores** and let ``EfficientNMS_TRT`` do the
threshold + class-aware NMS, matching the vendored ``postprocess`` semantics:

  * boxes: cxcywh -> xyxy in input-pixel space (``box_coding=0``),
  * scores: ``obj * cls`` per class, already sigmoid'd (``score_activation=0``),
  * threshold = ``adapter.score_threshold``, IoU = ``adapter.nms_threshold``,
  * class-aware NMS (``class_agnostic=0``), no background class.

The resulting engine's boxes are still input-pixel xyxy, so the meta (Contract B)
is unchanged — ``box_coords: input_pixels``, ``clip_boxes: true``.
"""

from __future__ import annotations

from pathlib import Path

try:
    from ..efficientnms import append_efficientnms, export_raw_boxes_scores
except ImportError:  # run flat
    from trt_export.efficientnms import append_efficientnms, export_raw_boxes_scores  # type: ignore

# Upper bound on detections the plugin keeps. Comfortably above any real scene;
# the runtime slices to the actual ``num_detections``.
_MAX_OUTPUT_BOXES = 1024


def prep_yolox(adapter, meta: dict, out_onnx_path) -> None:
    import torch
    from torch import nn

    out_onnx_path = Path(out_onnx_path)
    n_classes = int(adapter.num_classes)

    class YOLOXRaw(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            decoded = self.model(pixel_values)  # [B, N, 5 + C], input-pixel cxcywh
            cx, cy, w, h = decoded[..., 0], decoded[..., 1], decoded[..., 2], decoded[..., 3]
            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)  # [B,N,4]
            obj = decoded[..., 4:5]  # [B,N,1]
            cls = decoded[..., 5 : 5 + n_classes]  # [B,N,C]
            scores = obj * cls  # [B,N,C], == vendored confidence obj*cls
            return boxes, scores  # [B,N,4], [B,N,C] — B is the real batch dim throughout

    wrapper = YOLOXRaw(adapter.model.eval())
    raw_path = out_onnx_path.with_suffix(".raw.onnx")
    export_raw_boxes_scores(wrapper, raw_path)
    append_efficientnms(
        raw_path,
        out_onnx_path,
        score_threshold=float(adapter.score_threshold),
        iou_threshold=float(adapter.nms_threshold),
        max_output_boxes=_MAX_OUTPUT_BOXES,
        box_coding=0,
        background_class=-1,
        score_activation=0,
        class_agnostic=0,
    )
