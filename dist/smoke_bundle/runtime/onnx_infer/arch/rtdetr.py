"""RT-DETR service handler.

The export wrapper (``onnx_export/arch/rtdetr.py``) bakes sigmoid + top-k query
selection into the graph and emits ``(boxes, scores, labels)`` where boxes are
normalized ``[0,1]`` xyxy over the model input (``box_coords: "input_normalized"``
in meta). So the handler is a straight passthrough; ``postprocess`` maps the
normalized boxes to Friendy directly, with no resize/pad inverse — the adapter
stretches to a square canvas, so a fraction of the model input is the same
fraction of the original image. The service still replicates the adapter's input
pipeline (square stretch-resize, ImageNet normalize, no padding) so the graph
sees the same tensor.
"""

from .base import PassthroughHandler


class RTDetrHandler(PassthroughHandler):
    name = "rtdetr"
