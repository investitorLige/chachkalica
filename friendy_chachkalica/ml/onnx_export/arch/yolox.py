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

Boxes come out in **canvas**-pixel xyxy — the adapter now letterboxes onto a
fixed square canvas (``YOLOXAdapter.input_max_size``/``input_size_multiple``,
mirroring RT-DETR's resize), so a canvas coordinate is not the same as an
original-image coordinate once padding is involved. The service therefore
inverts the same scale/pad-offset-0 it recorded for this image
(``onnx_infer/preprocess.py``'s ``"letterbox"`` resize_mode +
``onnx_infer/postprocess.py``'s ``input_pixels`` inverse — the same mechanism
RF-DETR already uses, see ``onnx_export/arch/rfdetr.py``) and then clips to the
original image bounds (``clip_boxes: true``, mirroring the adapter's
``clip_xyxy``). The pad value is YOLOX's own native gray (114 in 0-255 space,
``adapters.yolox.YOLOX_PAD_VALUE`` in this pipeline's ``[0,1]`` scale) — not
zero.
"""

from __future__ import annotations

from pathlib import Path

try:
    from ..common import build_meta, export_detection_wrapper
except ImportError:  # run as a flat script
    from common import build_meta, export_detection_wrapper

try:
    from ...adapters.yolox import YOLOX_PAD_VALUE, _make_divisible
except ImportError:
    from adapters.yolox import YOLOX_PAD_VALUE, _make_divisible


def export_yolox(adapter, *, num_classes, params, class_map, onnx_path: str | Path) -> dict:
    import torch
    import torchvision
    from torch import nn

    conf_thre = float(adapter.score_threshold)
    nms_thre = float(adapter.nms_threshold)
    n_classes = int(adapter.num_classes)
    # Same square canvas the adapter letterboxes to (YOLOXAdapter._fixed_canvas_size).
    canvas_size = _make_divisible(int(adapter.input_max_size), int(adapter.input_size_multiple))

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
    export_detection_wrapper(wrapper, onnx_path, dummy_hw=(canvas_size, canvas_size))

    return build_meta(
        arch="yolox",
        num_classes=num_classes,
        class_map=class_map,
        score_threshold=conf_thre,  # the conf floor baked into the graph
        resize_mode="letterbox",
        size=canvas_size,
        multiple=0,  # letterbox already pads to the exact square; no second pad
        pad_value=YOLOX_PAD_VALUE,
        input_scale="unit",  # the adapter feeds the raw [0,1] image (no *255)
        normalize=None,
        clip_boxes=True,  # predictions can land in the letterbox margin
        box_coords="input_pixels",  # canvas-pixel; service inverts scale/pad
    )
