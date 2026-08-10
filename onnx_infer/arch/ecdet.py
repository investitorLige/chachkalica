"""ECDet service handler.

Contract A: the exported graph already bakes the whole head — sigmoid over
``pred_logits``, a flattened top-k down to 300 queries, ``label = idx % C``,
``cxcywh -> xyxy`` — and emits ``(boxes, scores, labels)`` with the batch axis
indexed away. So this is a plain passthrough; there is nothing arch-specific left
to do in numpy.

Contract B: ``box_coords: "input_normalized"`` (the decoder's boxes are ``[0,1]``
over the model input, and the exporter deliberately does not scale them),
``resize_mode: "square"`` with the trained canvas as ``size`` — a *stretch*
resize, aspect ratio not preserved, which is upstream ECDet's own ``Resize
[640, 640]`` — ImageNet ``normalize``, ``input_scale: "unit"``, no padding
(``multiple: 0``), and ``clip_boxes: true`` because the torch path clamps too.

ECDet is NMS-free, so the graph carries no NMS node and TensorRT compiles it
directly: there is no ``trt_export/arch/ecdet.py`` prep, and its engines are
built batch-1 like rtdetr/rfdetr's.
"""

from __future__ import annotations

from .base import PassthroughHandler


class ECDetHandler(PassthroughHandler):
    name = "ecdet"
