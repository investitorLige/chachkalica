"""RF-DETR service handler.

The export wrapper (``onnx_export/arch/rfdetr.py``) bakes the full head math —
sigmoid + top-k selection + ``cxcywh -> xyxy`` + scale to canvas pixels +
background-class drop — into the graph and emits ``(boxes, scores, labels)``
where boxes are xyxy in the letterboxed canvas's pixel frame (``box_coords:
"input_pixels"`` in meta). So the handler is a straight passthrough;
``postprocess`` inverts the recorded letterbox scale (pad offset is always 0 —
padding is bottom-right) and clips to the original image bounds
(``clip_boxes: true``, since predictions can land in the padded margin). The
service replicates the adapter's input pipeline (letterbox resize to the
variant's resolution + ImageNet normalize) so the graph sees the same tensor.
"""

from .base import PassthroughHandler


class RFDetrHandler(PassthroughHandler):
    name = "rfdetr"
