"""YOLOX ONNX exporter.

The vendored YOLOX model already decodes its heads in eval mode
(``decode_in_inference=True``): ``model(pixel_values)`` returns
``[1, N, 5 + num_classes]`` where each row is ``(cx, cy, w, h, obj, cls_0..)``,
with box centres/sizes in **input-pixel** space and ``obj``/``cls`` already
sigmoid'd (``vendor/yolox/models/yolo_head.py:186`` and ``decode_outputs``).

The arch-specific postprocess (``vendor/yolox/utils/boxes.py:32`` +
``adapters/yolox.py:yolox_detection_to_friendy``) is baked into the export
wrapper so Contract A holds:

  * cxcywh -> xyxy,
  * ``score = obj * max_cls`` and ``label = argmax_cls`` (matches the vendored
    ``conf_mask`` and the friendy ``confidence = obj * cls``),
  * confidence floor at ``conf_thre`` then class-aware ``batched_nms`` at
    ``nms_thre`` (identical ops to the vendored ``postprocess``).

Boxes come out in input-pixel xyxy; the service clips them to the original image
(``clip_boxes: true``, mirroring the adapter's ``clip_xyxy``). The service does
no resize/normalize — YOLOX eats the raw ``[0,1]`` image and pads to a multiple
of 32 (``resize_mode: none``, ``normalize: null``, ``multiple: 32``).
"""

from __future__ import annotations

from pathlib import Path

try:
    from ..common import build_meta, export_detection_wrapper
except ImportError:  # run as a flat script
    from common import build_meta, export_detection_wrapper


def export_yolox(adapter, *, num_classes, params, class_map, onnx_path: str | Path) -> dict:
    import torch
    import torchvision
    from torch import nn

    conf_thre = float(adapter.score_threshold)
    nms_thre = float(adapter.nms_threshold)
    n_classes = int(adapter.num_classes)

    class YOLOXExport(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            # eval-mode YOLOX returns decoded [1, N, 5 + C] in input pixels.
            decoded = self.model(pixel_values)[0]
            cx, cy, w, h = decoded[:, 0], decoded[:, 1], decoded[:, 2], decoded[:, 3]
            boxes = torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1
            )
            obj = decoded[:, 4]
            cls_conf, cls_pred = torch.max(decoded[:, 5 : 5 + n_classes], dim=1)
            scores = obj * cls_conf

            keep = scores >= conf_thre
            boxes, scores, labels = boxes[keep], scores[keep], cls_pred[keep]

            nms_idx = torchvision.ops.batched_nms(boxes, scores, labels, nms_thre)
            return boxes[nms_idx], scores[nms_idx], labels[nms_idx].to(torch.int64)

    wrapper = YOLOXExport(adapter.model.eval())
    export_detection_wrapper(wrapper, onnx_path)

    return build_meta(
        arch="yolox",
        num_classes=num_classes,
        class_map=class_map,
        score_threshold=conf_thre,  # the conf floor baked into the graph
        resize_mode="none",
        multiple=32,
        pad_value=0.0,
        input_scale="unit",  # the adapter feeds the raw [0,1] image (no *255)
        normalize=None,
        clip_boxes=True,  # adapter clips to the original image (clip_xyxy)
    )
