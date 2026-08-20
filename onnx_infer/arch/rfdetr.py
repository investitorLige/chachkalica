"""RF-DETR service handler.

The export wrapper (``onnx_export/arch/rfdetr.py``) bakes the full head math —
sigmoid + top-k selection + ``cxcywh -> xyxy`` + scale to canvas pixels — into
the graph and emits ``(boxes, scores, labels)`` where boxes are xyxy in the
letterboxed canvas's pixel frame (``box_coords: "input_pixels"`` in meta). The
background class stays in the fixed-size output at a forced-low score rather
than being dropped in-graph (see the exporter's module docstring for why), so
this handler is a straight passthrough — the score-threshold filter every arch
already runs in ``postprocess`` drops it same as any other sub-threshold
detection. There is nothing arch-specific left to do in numpy.
``postprocess`` inverts the recorded letterbox scale (pad offset is always 0 —
padding is bottom-right) and clips to the original image bounds
(``clip_boxes: true``, since predictions can land in the padded margin). The
service replicates the adapter's input pipeline (letterbox resize to the
variant's resolution + ImageNet normalize) so the graph sees the same tensor.
"""

from .base import PassthroughHandler


class RFDetrHandler(PassthroughHandler):
    name = "rfdetr"
