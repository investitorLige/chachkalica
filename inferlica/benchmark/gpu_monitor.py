"""Background GPU utilization/memory sampler used while timing a benchmark cell.

Wraps ``pynvml`` (the ``nvidia-ml-py`` package). Degrades gracefully to an
"unavailable" :class:`GpuSample` — never raises out of the ``with`` block —
when ``pynvml`` isn't installed, there's no NVIDIA GPU, or ``nvmlInit()``
fails for any other reason (e.g. driver/permission issues). This is the
expected outcome when benchmarking a CPU-only execution provider (this repo's
onnxruntime is CPU-only here; see project memory on onnxruntime-gpu being
blocked), not a bug to work around.
"""

from __future__ import annotations

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
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._samples.util_pct.append(float(util.gpu))
            self._samples.mem_used_mb.append(float(mem.used) / (1024 * 1024))
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
