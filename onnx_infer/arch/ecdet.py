"""ECDet service handler.

Contract A: the exported graph already bakes the whole head — sigmoid over
``pred_logits``, a per-batch-row flattened top-k down to 300 queries per image,
``label = idx % C``, ``cxcywh -> xyxy`` — and emits ``(boxes, scores, labels)``.
The ONNX runtime path here always calls with one image, so a batch axis of 1
reshapes away to the same ``(boxes[N,4], scores[N], labels[N])`` shape either
way; a plain passthrough handles both. There is nothing arch-specific left to
do in numpy.

Contract B: ``box_coords: "input_normalized"`` (the decoder's boxes are ``[0,1]``
over the model input, and the exporter deliberately does not scale them),
``resize_mode: "square"`` with the trained canvas as ``size`` — a *stretch*
resize, aspect ratio not preserved, which is upstream ECDet's own ``Resize
[640, 640]`` — ImageNet ``normalize``, ``input_scale: "unit"``, no padding
(``multiple: 0``), and ``clip_boxes: true`` because the torch path clamps too.

ECDet is NMS-free, so the graph carries no NMS node and TensorRT compiles it
directly: there is no ``trt_export/arch/ecdet.py`` prep. Unlike rtdetr/rfdetr's
description elsewhere in this codebase from before batching was wired up,
ecdet's (and rtdetr's/rfdetr's) engine is *not* architecturally stuck at
batch-1 — see ``onnx_export/arch/ecdet.py``'s module docstring and
``trt_export.arch.BATCH_AWARE_ARCHS``.
"""

from __future__ import annotations

from .base import PassthroughHandler


class ECDetHandler(PassthroughHandler):
    name = "ecdet"
