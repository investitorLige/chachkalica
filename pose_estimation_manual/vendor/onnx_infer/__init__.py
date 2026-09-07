"""Architecture-free ONNX inference service.

Loads a trained detector exported to ONNX (graph + weights) plus a ``meta.json``
sidecar and runs it end-to-end — image in, Friendy ``(N, 6)`` predictions out —
**without importing any training architecture code** (``friendy_chachkalica``,
``transformers``, ``rfdetr``, ``ultralytics``, torchvision detection heads).

The core (``meta``/``session``/``preprocess``/``postprocess`` and the per-arch
handlers under ``arch/``) is pure NumPy + onnxruntime. Only :class:`OnnxAdapter`
lazily imports torch, so it can be a drop-in replacement for the trained-model
adapters wherever ``chachak`` already runs (which always has torch installed).

Two frozen contracts decouple this from whatever exported the model: **Contract
A**, the ONNX graph's output layout (see ``arch/``), and **Contract B**, the
``meta.json`` schema (see ``meta.py``). A third — Contract C, how the model
should be *run* — is this project's own addition and lives in ``engines/``,
not here, because it describes pipeline choice rather than the graph.
"""

from .adapter import OnnxAdapter, load_onnx_adapter
from .meta import ModelMeta

__all__ = ["OnnxAdapter", "load_onnx_adapter", "ModelMeta"]
