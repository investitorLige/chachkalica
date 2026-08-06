"""TensorRT inference runtime — the GPU counterpart to ``onnx_infer``.

Loads a trained detector compiled to a TensorRT ``.engine`` (built from the
architecture-free ONNX by ``friendy_chachkalica/ml/trt_export``) plus the same
``meta.json`` sidecar, and runs it end-to-end — image in, Friendy ``(N, 6)``
predictions out — with **no training-architecture code**.

Both frozen contracts (the graph I/O and the meta schema) are shared with
``onnx_infer`` — a TensorRT engine is just a recompilation of the same ONNX graph —
so ``ModelMeta`` and the per-arch output handlers are reused verbatim, as is
``plan_preprocess``, which resolves the meta into the concrete resize/pad/normalize
geometry to apply.

What differs is where that geometry is *executed*. ``onnx_infer`` runs it in NumPy
on the host, because onnxruntime feeds are numpy arrays. Here it runs in torch on
the GPU (:mod:`trt_infer.preprocess_torch`, :mod:`trt_infer.postprocess_torch`), so
an image handed in as a CUDA tensor — a person crop is a GPU slice of its frame —
never round-trips through host memory on its way to the engine and back. The torch
executors are verbatim ports, asserted bit-identical to the numpy ones by
``tests/test_preprocess_torch.py`` and ``tests/test_postprocess_torch.py``; that is
what lets the two coexist without an exported bundle's detections moving.

Unlike ``onnx_infer`` (torch-free, CPU-capable), this runtime requires a CUDA GPU
and a matching TensorRT install, and torch is a hard dependency (it owns the
device memory and the CUDA stream).
"""

from .adapter import TrtAdapter, load_trt_adapter

__all__ = ["TrtAdapter", "load_trt_adapter"]
