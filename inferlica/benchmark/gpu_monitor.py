"""Background GPU utilization/memory sampler used while timing a benchmark cell.

Wraps ``pynvml`` (the ``nvidia-ml-py`` package). Degrades gracefully to an
"unavailable" :class:`GpuSample` — never raises out of the ``with`` block —
when ``pynvml`` isn't installed, there's no NVIDIA GPU, or ``nvmlInit()``
fails for any other reason (e.g. driver/permission issues). This is the
expected outcome when benchmarking a CPU-only execution provider (this repo's
onnxruntime is CPU-only here; see project memory on onnxruntime-gpu being
blocked), not a bug to work around.

Memory readings are **per-process** where the environment actually supports it
(falls back to whole-device ``nvmlDeviceGetMemoryInfo`` otherwise) --
``nvmlDeviceGetMemoryInfo`` alone reports every process's usage on the GPU, so
any other container/process sharing it would silently leak into "this cell's"
number. ``nvmlDeviceGetComputeRunningProcesses`` is the per-process API, but
merely calling it successfully is NOT enough to trust it: in this repo's own
trainer container, a `docker exec` session running real, synchronized CUDA
kernels never appeared in its own GPU's process list, while two unrelated
processes on the same GPU did -- silently reporting 0 MB used for every
sample in that case would be worse than the whole-device fallback. So
support is verified with a real allocation once per process
(``_detect_per_process_support``) rather than assumed from the call not
raising. Utilization has no equivalent per-process API in standard NVML (that
needs DCGM/nsight), so ``gpu_util_pct_*`` stays whole-device -- a real,
unavoidable caveat if something else is actively using the GPU during a
sweep.

Per-process accounting alone isn't enough within a single long sweep, though:
this process's OWN memory from earlier cells lingers too (PyTorch's caching
allocator and onnxruntime/TensorRT sessions don't return CUDA memory to the
driver just because a cell function returned). ``read_current_mem_mb`` is
exposed so a caller can snapshot a baseline right before each cell's timed
region (after clearing what it can -- see ``benchmark_cell`` in ``core.py``)
and report peak-minus-baseline instead of a raw, ever-growing absolute peak.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GpuSample:
    peak_mem_mb: Optional[float] = None
    mean_util_pct: Optional[float] = None
    peak_util_pct: Optional[float] = None
    available: bool = False
    error: Optional[str] = None


@dataclass
class _Samples:
    util_pct: List[float] = field(default_factory=list)
    mem_used_mb: List[float] = field(default_factory=list)


# Computed once per process (real allocation probe is not free) and reused by
# every GpuMonitor instance and read_current_mem_mb call after the first.
_per_process_supported: Optional[bool] = None


def _detect_per_process_support(pynvml_mod, handle) -> bool:
    """Real-allocation probe: does nvmlDeviceGetComputeRunningProcesses
    actually attribute *this* process's own memory, or does it just silently
    omit us (observed in this repo's trainer container -- see module
    docstring)? Any probe failure (no torch, no CUDA, driver quirk) means
    "don't trust it" -- caller falls back to whole-device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        pid = os.getpid()
        probe = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")  # ~8MB
        torch.cuda.synchronize()
        try:
            return any(
                proc.pid == pid and (proc.usedGpuMemory or 0) > 0
                for proc in pynvml_mod.nvmlDeviceGetComputeRunningProcesses(handle)
            )
        finally:
            del probe
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - any probe failure means "don't trust it"
        return False


def _per_process_support(pynvml_mod, handle) -> bool:
    global _per_process_supported
    if _per_process_supported is None:
        _per_process_supported = _detect_per_process_support(pynvml_mod, handle)
    return _per_process_supported


def _read_mem_mb(pynvml_mod, handle, pid: int, per_process: bool) -> float:
    """One memory reading: this process's own usage if ``per_process``, else
    whole-device. Falls back to whole-device for a single sample if the
    per-process query errors (rare; keeps a bad sample from raising)."""
    if per_process:
        try:
            for proc in pynvml_mod.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid == pid:
                    return float(proc.usedGpuMemory or 0) / (1024 * 1024)
            return 0.0  # this process currently holds no attributed allocation
        except Exception:  # noqa: BLE001 - fall through to whole-device this once
            pass
    mem = pynvml_mod.nvmlDeviceGetMemoryInfo(handle)
    return float(mem.used) / (1024 * 1024)


def read_current_mem_mb(device_index: int = 0) -> Optional[float]:
    """One-shot snapshot of this process's current GPU memory (per-process
    where supported, else whole-device). Returns ``None`` if pynvml/NVML is
    unavailable. Meant as a pre-cell baseline -- see module docstring."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception:  # noqa: BLE001 - covers pynvml.NVMLError + friends
        return None
    try:
        per_process = _per_process_support(pynvml, handle)
        return _read_mem_mb(pynvml, handle, os.getpid(), per_process)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - shutdown failures are never fatal
            pass


class GpuMonitor:
    """Context manager: samples GPU utilization/memory on a background thread
    for the duration of the ``with`` block.

        with GpuMonitor(device_index=0, interval_s=0.05) as mon:
            ...timed region...
        sample = mon.result
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.05):
        self.device_index = device_index
        self.interval_s = max(0.001, interval_s)
        self._handle = None
        self._pynvml = None
        self._pid = os.getpid()
        self._per_process_mem = False
        self._init_error: Optional[str] = None
        self._samples = _Samples()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "GpuMonitor":
        if self._try_init():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - shutdown failures are never fatal
                pass

    def _try_init(self) -> bool:
        try:
            import pynvml
        except ImportError as exc:
            self._init_error = f"pynvml not installed: {exc}"
            return False

        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:  # noqa: BLE001 - covers pynvml.NVMLError + friends
            self._init_error = f"nvmlInit/GetHandleByIndex failed: {exc}"
            return False

        self._pynvml = pynvml
        self._per_process_mem = _per_process_support(pynvml, self._handle)
        return True

    def _sample_loop(self) -> None:
        while True:
            self._sample_once()
            if self._stop_event.wait(self.interval_s):
                self._sample_once()
                return

    def _sample_once(self) -> None:
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            self._samples.util_pct.append(float(util.gpu))
        except Exception:  # noqa: BLE001 - a single failed sample must not kill the thread
            pass
        try:
            self._samples.mem_used_mb.append(
                _read_mem_mb(self._pynvml, self._handle, self._pid, self._per_process_mem)
            )
        except Exception:  # noqa: BLE001 - a single failed sample must not kill the thread
            pass

    @property
    def result(self) -> GpuSample:
        if self._pynvml is None:
            return GpuSample(available=False, error=self._init_error)

        util = self._samples.util_pct
        mem = self._samples.mem_used_mb
        return GpuSample(
            peak_mem_mb=max(mem) if mem else None,
            mean_util_pct=(sum(util) / len(util)) if util else None,
            peak_util_pct=max(util) if util else None,
            available=True,
            error=None,
        )
