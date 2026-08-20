"""GPU-resident inference for the chachak pipelines — an opt-in alternative to
``chachak.pipeline``, not a replacement for it.

Why this package exists
-----------------------

``trt_infer`` already keeps inference *pixels* in device memory end to end: its torch
pre/postprocess executors and ``TrtModel.run_torch`` never round-trip a crop through the
host. What it does not do is keep the *host* out of the way. ``TrtModel.run_torch`` ends
every engine execution with ``stream.synchronize()`` (``trt_infer/session.py:236``), which
drains the CUDA queue. A ``people_detect_first`` frame with six people therefore blocks the
host about fourteen times: once per engine call, once per ``[N,6]`` copy back, and once for
the EfficientNMS ``num_detections`` read. Frame *k+1* cannot start decoding while frame *k*'s
last crop is still in flight, and every host-side step — box geometry, the class filter, the
merge NMS, JSON serialization — lands on the critical path instead of hiding behind device
work.

This package runs the same pipelines with **one host synchronization per batch**. Two ideas
get it there:

* :mod:`gpu_infer.engine` submits engine work without synchronizing, and copies each
  execution's outputs into padded per-slot buffers with *stream-ordered* device-to-device
  copies. ``execute_i -> copy_i -> execute_{i+1}`` stays correctly ordered on one stream, so
  buffer reuse is safe without a host block (this is the async-safe form of ``run_torch``'s
  defensive ``.clone()``).
* :mod:`gpu_infer.tail` carries **fixed-size padded tensors plus a validity mask** instead of
  compacting. Every synchronization in the original path traces to a length that lives on the
  device — ``preds[mask]``, ``boxes[keep]``, ``det_boxes[i, :n]`` all size their output from
  device data, so torch must sync to allocate. A mask has a static shape, so it does not.

What it deliberately does not change
------------------------------------

**Detections are bit-identical to** ``chachak.pipeline``. The crop resample reuses
``trt_infer.preprocess_torch``'s verbatim float64-coordinate gather rather than a fused
``grid_sample``, because that port is what lets an exported bundle reproduce the training
repo's eval numbers; a fused kernel computes a different bilinear expression and would require
re-validating mAP per architecture. The box geometry twins in :mod:`gpu_infer.geometry` run in
**float64** for the same reason — ``chachak.boxes.expand_box`` and friends convert to python
floats, which *are* float64, so a float32 tensor version would not agree.

It also does not by itself reduce the number of engine calls. A passthrough-arch engine
(ecdet/rtdetr/rfdetr/dfine) still defaults to a batch-1 profile — see
``chachak.bundle_export.cli._batchable_in_trt`` / ``trt_export.arch.BATCH_AWARE_ARCHS`` — so N
person crops are still N executions unless the engine was actually built with a wider profile.
The DETR-family export wrappers are batch-aware now (``onnx_export/arch/{ecdet,rtdetr,rfdetr,dfine}.py``),
so that rebuild is no longer a separate change waiting to happen; it just has to be requested at
export time (the TRT export form's batch-size field, or ``chachak.bundle_export``'s batch
config). This package removes the synchronization and per-call host overhead *around* however
many engine calls that ends up being.

Layout
------

``config``      frozen ``GpuOptions`` (stdlib only)
``geometry``    batched, float64, per-row-bounds twins of the box conversions
``tiling``      device twins of the tile grids, for ``batch_detect`` / ``batch_people``
``crops_exact`` the verbatim per-crop resample, into a preallocated batch
``decode``      nvJPEG frame decode, with a PIL fallback per file
``engine``      ``AsyncEngine`` — ``TrtModel`` without the synchronize
``tail``        the padded/masked postprocess tail
``nms``         class-aware overlap NMS on device, exact greedy
``pipeline``    the four pipelines
``loader``      build a pipeline from a ``PipelineConfig`` + artifact paths

Nothing here is imported by the existing tree. Every module keeps ``torch``, ``torchvision``,
``tensorrt`` and every ``chachak`` / ``onnx_infer`` / ``trt_infer`` import **inside functions**
(the pattern in ``trt_infer/session.py:108``), so this package is importable with only the
standard library. That is what lets it be vendored into a bundle whose import closure is
checked in a bare subprocess, and what keeps ``torchvision`` genuinely optional.
"""

from __future__ import annotations

__all__ = [
    "GpuOptions",
    "build_gpu_pipeline",
    "load_frames",
    "GPU_PIPELINE_NAMES",
]

# The pipelines this package implements. A caller handed anything else must be told so
# rather than silently getting a different pipeline's answer.
GPU_PIPELINE_NAMES = ("people_detect_first", "batch_people", "batch_detect", "chain")


def __getattr__(name):
    """Resolve the public API lazily (PEP 562).

    Importing this package must not import torch — see the module docstring. Re-exporting
    eagerly at module scope would defeat that, so the names in ``__all__`` are bound on first
    access instead. ``GpuOptions`` comes from a stdlib-only module and could be eager, but is
    routed here too so the rule has no exceptions to remember.
    """
    if name == "GpuOptions":
        from .config import GpuOptions

        return GpuOptions
    if name == "build_gpu_pipeline":
        from .loader import build_gpu_pipeline

        return build_gpu_pipeline
    if name == "load_frames":
        from .decode import load_frames

        return load_frames
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
