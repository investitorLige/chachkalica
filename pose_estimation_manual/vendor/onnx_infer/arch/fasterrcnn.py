"""Faster R-CNN service handler.

The export wrapper returns the torchvision model's detections directly as
``(boxes, scores, labels)`` — boxes are already xyxy in the input image's pixel
frame (torchvision maps them back from its internal resize). So the handler is a
straight passthrough and the service does no resize/normalize (torchvision
normalizes internally; ``resize_mode: none``, ``normalize: null``).
"""

from .base import PassthroughHandler


class FasterRCNNHandler(PassthroughHandler):
    name = "fasterrcnn"
