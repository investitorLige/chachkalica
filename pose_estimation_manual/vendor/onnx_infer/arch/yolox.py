"""YOLOX service handler.

The export wrapper (``onnx_export/arch/yolox.py``) bakes decode + confidence
floor + class-aware NMS into the graph and emits ``(boxes, scores, labels)`` as
input-pixel xyxy directly — so the handler is a straight passthrough. The
service pads the raw ``[0,1]`` image to a multiple of 32, does no normalize, and
clips the mapped-back boxes to the original image (``clip_boxes: true`` in meta),
matching the adapter's ``clip_xyxy``.
"""

from .base import PassthroughHandler


class YOLOXHandler(PassthroughHandler):
    name = "yolox"
