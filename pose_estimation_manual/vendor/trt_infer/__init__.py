"""TensorRT inference runtime — the GPU counterpart to ``onnx_infer``.

Loads a trained detector compiled to a TensorRT ``.engine`` (built from the
architecture-free ONNX by ``friendy_chachkalica/ml/trt_export``) plus the same
``meta.json`` sidecar, and runs it end-to-end — image in, Friendy ``(N, 6)``
predictions out — with **no training-architecture code**.

It deliberately reuses ``onnx_infer``'s pure-numpy core verbatim (``ModelMeta``,
``preprocess``, ``to_friendy``, and the per-arch output handlers): the TensorRT
engine is just a recompilation of the same ONNX graph, so both frozen contracts
(the graph I/O and the meta schema) are identical. Only the session layer differs
— a TensorRT execution context with torch-managed CUDA buffers instead of an
onnxruntime ``InferenceSession``.

Unlike ``onnx_infer`` (torch-free, CPU-capable), this runtime requires a CUDA GPU
and a matching TensorRT install, and torch is a hard dependency (it owns the
device memory and the CUDA stream).
"""

from .adapter import TrtAdapter, load_trt_adapter

__all__ = ["TrtAdapter", "load_trt_adapter"]
