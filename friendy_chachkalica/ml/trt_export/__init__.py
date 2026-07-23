"""Build TensorRT engines from the architecture-free ONNX artifacts.

This mirrors ``onnx_export`` one level down the deployment stack: instead of
tracing a checkpoint, it **compiles an already-exported ``.onnx`` graph** into a
serialized TensorRT engine. Because the ONNX graph already has decode / NMS /
top-k baked in (Contract A) and the ``meta.json`` (Contract B) already describes
pre-processing, the build is architecture-agnostic — every arch shares one build
path; only the optimization profile (allowed input H/W range) is meta-derived.

Produces, next to a ``<name>.engine``:

  * ``<name>.engine``      — the serialized TensorRT plan (graph + weights).
  * ``<name>.meta.json``   — a verbatim copy of the ONNX Contract-B sidecar, so a
    ``trt_infer`` runtime is self-contained (identical schema to the ONNX one).
  * ``<name>.engine.json`` — build provenance (precision used, TRT version, the
    optimization-profile shapes, source checkpoint).

IMPORTANT — this is NOT CPU-safe and NOT portable, unlike ``onnx_export``:
building requires a GPU with TensorRT installed, competes with active training
for that GPU, and the resulting engine is tied to the exact GPU + TensorRT
version it was built on. Rebuild on the target hardware before deploying.
"""
