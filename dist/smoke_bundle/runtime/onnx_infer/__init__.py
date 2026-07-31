"""Architecture-free ONNX inference service.

Loads a trained detector exported to ONNX (graph + weights) plus a ``meta.json``
sidecar and runs it end-to-end — image in, Friendy ``(N, 6)`` predictions out —
**without importing any training architecture code** (``friendy_chachkalica``,
``transformers``, ``rfdetr``, ``ultralytics``, torchvision detection heads).

The core (``meta``/``session``/``preprocess``/``postprocess`` and the per-arch
handlers under ``arch/``) is pure NumPy + onnxruntime. Only :class:`OnnxAdapter`
lazily imports torch, so it can be a drop-in replacement for the trained-model
adapters wherever ``chachak`` already runs (which always has torch installed).

See ``onnx_infer/PLAN.md`` for the two contracts (the ONNX graph output and the
``meta.json`` schema) that decouple this service from the exporter.
"""

from .adapter import OnnxAdapter, load_onnx_adapter
from .meta import ModelMeta

__all__ = ["OnnxAdapter", "load_onnx_adapter", "ModelMeta"]
