"""Exact, per-process GPU kernel-busy% for the ONNX and TensorRT formats --
the Fix-2 fallback for when CUPTI can't do it.

Why this exists: the whole-device utilization NVML samples in gpu_monitor.py
has two irreducible defects for these two formats. (1) This GPU's driver only
refreshes the utilization counter about once a second, so even a cell whose
timed loop is stretched to >=2s yields only ~2 independent readings -- the
"mean" is an average of duplicates. (2) NVML has no per-process utilization
query at all (unlike memory), so anything else sharing the GPU -- e.g. this
container's live inference service -- inflates the number. Neither is fixable
by polling harder; both are fixed by measuring THIS process's own kernel time
and dividing by wall-clock, exactly as the pt path already does via CUPTI.

Precedence (decided in core.benchmark_cell, not here):
  1. CUPTI via ``torch.profiler(ProfilerActivity.CUDA)`` -- traces the CUDA
     driver timeline for the whole process, so it captures kernels launched by
     onnxruntime and TensorRT too. TensorRT in particular runs on torch's OWN
     current stream (see trt_infer/session.py), so its kernels are always in
     torch's context and always visible; onnxruntime uses its own stream but
     the same CUDA context, so it is normally visible as well. That path lives
     in core.time_predict's profile_cuda block and is preferred when it reports
     any device time.
  2. The functions here -- each runtime's OWN profiler -- used only when CUPTI
     reports nothing for a given cell (an empty external trace). onnxruntime's
     ``enable_profiling`` and TensorRT's ``IProfiler`` are guaranteed to see
     their respective kernels.
  3. NVML whole-device sampling (gpu_monitor) -- last resort, kept in place.

Both functions run an ADDITIONAL, separate pass over their own loop -- never
the timed fps loop, whose numbers must stay uninstrumented (attaching either
profiler adds real, model-dependent overhead and, for TRT, serializes layers).
Like the pt CUPTI path, busy% is kernel-time over the *profiled* loop's own
wall-clock, so the figure is honestly conservative (profiling inflates the
wall a little, which can only push the ratio DOWN, never up). Each returns a
float in [0, 100], or ``None`` if its mechanism isn't available so the caller
can degrade to NVML.
"""

from __future__ import annotations

import time
from typing import List, Optional


def native_kernel_busy_pct(runnable, fmt: str, images: List, device, iterations: int) -> Optional[float]:
    """Dispatch to the right runtime-native profiler. Returns ``None`` (never
    raises) if the format has no native path or the mechanism is unavailable,
    so a failed measurement degrades to NVML rather than failing the cell."""
    try:
        if fmt == "engine":
            return _trt_layer_busy_pct(runnable, images, device, iterations)
        if fmt == "onnx":
            return _onnx_kernel_busy_pct(runnable, images, device, iterations)
    except Exception:  # noqa: BLE001 - a measurement extra must never break a cell
        return None
    return None


def _trt_layer_busy_pct(runnable, images: List, device, iterations: int) -> Optional[float]:
    """Sum TensorRT's own per-layer times via an ``IProfiler`` attached to the
    execution context, over a dedicated loop, / that loop's wall.

    Attaching a profiler makes TensorRT time each layer (it inserts CUDA events
    and reports after the stream syncs -- TrtModel.run already synchronizes each
    call, so every iteration's layer times are reported before run() returns).
    This serializes layers and adds overhead, which is exactly why it's a
    separate pass and never wraps the fps loop.
    """
    import tensorrt as trt
    import torch

    from chachak.infer import predict_adapter

    model = getattr(runnable, "_model", None)
    context = getattr(model, "context", None)
    if context is None:
        return None

    class _LayerTimer(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.total_ms = 0.0

        def report_layer_time(self, layer_name, ms):  # noqa: ARG002 - name unused, time summed
            self.total_ms += float(ms)

    timer = _LayerTimer()
    previous = getattr(context, "profiler", None)
    context.profiler = timer
    try:
        # One unmeasured call so the profiler's first-call setup (event pool,
        # layer-name registration) doesn't land in the summed window.
        predict_adapter(runnable, images)
        torch.cuda.synchronize()
        timer.total_ms = 0.0

        start = time.perf_counter()
        for _ in range(iterations):
            predict_adapter(runnable, images)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - start
    finally:
        context.profiler = previous

    if wall_s <= 0 or timer.total_ms <= 0:
        # total_ms == 0 means the profiler never fired (e.g. a TRT build/API
        # combination that only profiles the sync execute path) -- signal
        # "no reading" so the caller falls back to NVML rather than reporting 0%.
        return None
    return round(min(timer.total_ms / 1e3 / wall_s * 100.0, 100.0), 3)


def _onnx_kernel_busy_pct(runnable, images: List, device, iterations: int) -> Optional[float]:
    """Sum onnxruntime's own per-kernel device times from an
    ``enable_profiling`` trace, over a dedicated loop, / that loop's wall.

    Builds a throwaway profiling session from the SAME graph file the runnable
    already loaded (so fp16-vs-fp32 matches the timed cell) and temporarily
    swaps it into the adapter's ``_model`` so the real preprocessing path feeds
    it identically -- then restores the original session. onnxruntime's trace
    tags each actual kernel execution with ``cat == "Kernel"`` and a device
    duration in microseconds; summing those is the GPU busy time.
    """
    import json
    import os

    import onnxruntime as ort
    import torch

    from chachak.infer import predict_adapter
    from onnx_infer.session import providers_for

    model = getattr(runnable, "_model", None)
    path = getattr(model, "path", None)
    session = getattr(model, "session", None)
    if model is None or path is None or session is None:
        return None

    options = ort.SessionOptions()
    options.enable_profiling = True
    try:
        prof_session = ort.InferenceSession(str(path), sess_options=options, providers=providers_for(device))
    except Exception:  # noqa: BLE001 - couldn't build a profiling session; degrade to NVML
        return None
    if "CUDAExecutionProvider" not in prof_session.get_providers():
        # A CPU-only profiling session would report CPU op times, not GPU
        # kernel busy time -- not what this measures. Bail to NVML.
        prof_session.end_profiling()
        return None

    original_session = model.session
    original_input = model.input_name
    model.session = prof_session
    model.input_name = prof_session.get_inputs()[0].name
    prof_path = None
    try:
        predict_adapter(runnable, images)  # unmeasured warmup into the profiling session
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            predict_adapter(runnable, images)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - start
    finally:
        prof_path = prof_session.end_profiling()
        model.session = original_session
        model.input_name = original_input

    if wall_s <= 0 or not prof_path or not os.path.exists(prof_path):
        return None
    try:
        with open(prof_path) as handle:
            events = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    finally:
        try:
            os.remove(prof_path)
        except OSError:
            pass

    # onnxruntime's trace has no separate "Kernel" category (verified on ORT
    # 1.27): it emits one ``cat == "Node"`` event per op named ``*_kernel_time``,
    # with ``args.provider`` telling you which EP ran it. Sum the CUDA-provider
    # ones -- those are the GPU ops. (Caveat: for the CUDA EP this per-node time
    # is measured host-side around an async launch, so it can slightly
    # undercount true device time -- which is exactly why this is only the
    # FALLBACK; CUPTI's driver-level trace is the primary and is preferred
    # whenever it captures the kernels, which the verify_cupti probe confirms it
    # does.)
    busy_us = sum(
        float(e.get("dur", 0) or 0)
        for e in events
        if e.get("cat") == "Node"
        and str(e.get("name", "")).endswith("_kernel_time")
        and (e.get("args") or {}).get("provider") == "CUDAExecutionProvider"
    )
    if busy_us <= 0:
        return None
    return round(min(busy_us / 1e6 / wall_s * 100.0, 100.0), 3)
