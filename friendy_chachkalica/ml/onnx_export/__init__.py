"""Export trained Friendy checkpoints to architecture-free ONNX artifacts.

Produces, next to a ``best.pt`` / ``last.pt`` checkpoint:

  * ``<name>.onnx``      — graph + weights (Contract A: input ``pixel_values``
    ``[1,3,H,W]``; outputs ``boxes``/``scores``/``labels``, decode + NMS baked in,
    boxes in input-pixel xyxy).
  * ``<name>.meta.json`` — Contract B: how to pre-process and interpret it.

Runs in the training environment (which already has every architecture's deps).
The ``onnx_infer`` service consumes the two files with only
onnxruntime / numpy / pillow — no training packages.

One file per arch under ``arch/``, dispatched through
:data:`registry.EXPORT_REGISTRY`, mirroring ``adapters/`` + ``registry.py``.
"""
