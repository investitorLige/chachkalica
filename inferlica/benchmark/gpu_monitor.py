"""Background GPU utilization/memory sampler used while timing a benchmark cell.

Wraps ``pynvml`` (the ``nvidia-ml-py`` package). Degrades gracefully to an
"unavailable" :class:`GpuSample` — never raises out of the ``with`` block —
when ``pynvml`` isn't installed, there's no NVIDIA GPU, or ``nvmlInit()``
fails for any other reason (e.g. driver/permission issues).

Memory readings use the best of three modes, picked once per process by a
real-allocation probe (``_detect_mem_mode``):

  - ``"process"``  -- ``nvmlDeviceGetComputeRunningProcesses`` correctly
    attributes *this* process's own memory under its own ``os.getpid()``.
    Immune to both cross-cell accumulation and other processes sharing the
    GPU.
  - ``"residual"`` -- the process-list API works and lists every OTHER
    process's usage, but never lists an entry matching this process's own
    ``os.getpid()`` (observed in this repo's trainer container: a
    `docker exec` session running real, synchronized CUDA kernels never
    appeared in its own GPU's process list under its own pid, while two
    unrelated processes did). This process's own usage is inferred as
    whole-device-used minus the sum of every OTHER listed process.
  - ``"device"``   -- the process-list API isn't available at all. Pure
    whole-device ``nvmlDeviceGetMemoryInfo``; contaminated by anything else
    on the GPU.

Mode is verified with a real ~8MB allocation rather than assumed from the API
call not raising -- merely calling nvmlDeviceGetComputeRunningProcesses
without error does NOT mean it sees this process under its own pid (see
above).

**"residual" mode has its own real bug, found via user pushback on
near-universal 0MB readings**: this process's own CUDA activity does get
listed by ``nvmlDeviceGetComputeRunningProcesses`` -- just under a PID that
does NOT match ``os.getpid()`` at all (confirmed directly: allocating 800MB
made a brand-new entry appear under an unrelated-looking pid; this is not a
PID-namespace nesting artifact -- ``/proc/self/status``'s ``NSpid`` shows only
one value here, so there's no nesting to explain it away). Since "residual"
mode's ``others`` sum only excludes entries matching ``os.getpid()``, it was
inadvertently including THIS process's own entry as if it were external, and
then subtracting it back out -- so any allocation this process made was
subtracted from the total right alongside itself, netting to ~zero
regardless of how much was actually allocated. ``snapshot_process_mem_mb`` /
``own_usage_delta_mb`` below sidestep this entirely: instead of trying to
identify "our" pid (broken here), they diff two full per-pid snapshots and
sum whichever pids grew, however they're labeled. A genuinely external,
otherwise-idle tenant contributes ~0 growth across one short measurement
window, so growth in the diff is attributable to whatever ran in that window
-- correct regardless of which (possibly unrecognized) pid the driver
happens to file it under.

Utilization has no per-process or residual-subtraction equivalent in standard
NVML (that needs DCGM/nsight): ``gpu_util_pct_*`` is always whole-device, a
real, unavoidable caveat if something else is actively using the GPU during a
sweep.

Even the best mode isn't enough within a single long sweep on its own, though:
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
from typing import Dict, List, Optional


@dataclass
class GpuSample:
    peak_mem_mb: Optional[float] = None
    mean_util_pct: Optional[float] = None
    peak_util_pct: Optional[float] = None
    available: bool = False
    error: Optional[str] = None
    # Trust context, added for the bundle benchmark: a latency number measured
    # while the card is power- or thermally-capped is not the number the same
    # bundle produces on an unthrottled card, and without these fields there is
    # no way to tell the two apart after the fact. Purely additive -- every
    # existing caller reads only the five fields above.
    mean_power_w: Optional[float] = None
    peak_power_w: Optional[float] = None
    mean_sm_clock_mhz: Optional[float] = None
    throttle_reasons: List[str] = field(default_factory=list)


@dataclass
class _Samples:
    util_pct: List[float] = field(default_factory=list)
    mem_used_mb: List[float] = field(default_factory=list)
    power_w: List[float] = field(default_factory=list)
    sm_clock_mhz: List[float] = field(default_factory=list)
    throttle_reasons: set = field(default_factory=set)


# Computed once per process (the probe allocates real GPU memory, not free)
# and reused by every GpuMonitor instance and read_current_mem_mb call after
# the first. One of "process" | "residual" | "device".
_mem_mode: Optional[str] = None


def _detect_mem_mode(pynvml_mod, handle) -> str:
    """Real-allocation probe deciding which memory-reading mode to trust.
    Any probe failure (no torch, no CUDA, driver quirk) falls back a tier."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "device"
        try:
            procs_before = pynvml_mod.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception:  # noqa: BLE001 - process-list API unavailable at all
            return "device"

        pid = os.getpid()
        probe = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")  # ~8MB
        torch.cuda.synchronize()
        try:
            procs = pynvml_mod.nvmlDeviceGetComputeRunningProcesses(handle)
            if any(proc.pid == pid and (proc.usedGpuMemory or 0) > 0 for proc in procs):
                return "process"
            return "residual"  # API works (procs_before didn't raise) but omits us
        finally:
            del probe
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - any other probe failure: safest fallback
        return "device"


def _mem_mode_for(pynvml_mod, handle) -> str:
    global _mem_mode
    if _mem_mode is None:
        _mem_mode = _detect_mem_mode(pynvml_mod, handle)
    return _mem_mode


def _read_mem_mb(pynvml_mod, handle, pid: int, mode: str) -> float:
    """One memory reading in the given mode. Falls back a tier for a single
    sample if the process-list query errors (rare; keeps one bad sample from
    raising rather than killing the sampling thread)."""
    if mode in ("process", "residual"):
        try:
            procs = pynvml_mod.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception:  # noqa: BLE001 - fall through to whole-device this once
            procs = None
        if procs is not None:
            if mode == "process":
                for proc in procs:
                    if proc.pid == pid:
                        return float(proc.usedGpuMemory or 0) / (1024 * 1024)
                return 0.0  # this process currently holds no attributed allocation
            total = pynvml_mod.nvmlDeviceGetMemoryInfo(handle).used
            others = sum(float(proc.usedGpuMemory or 0) for proc in procs if proc.pid != pid)
            return max(0.0, float(total) - others) / (1024 * 1024)
    mem = pynvml_mod.nvmlDeviceGetMemoryInfo(handle)
    return float(mem.used) / (1024 * 1024)


def snapshot_process_mem_mb(device_index: int = 0) -> Optional[Dict[int, float]]:
    """Full per-pid GPU memory snapshot (MB), keyed by whatever pid the
    driver reports (which may not match any real pid this process recognizes
    as its own -- see module docstring). Falls back to a single sentinel key
    (``-1``) mapped to the whole-device total if the process-list API isn't
    available at all, so ``own_usage_delta_mb`` still degrades to a plain
    whole-device before/after delta in that case. Returns ``None`` if
    pynvml/NVML itself is unavailable.
    """
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
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            return {int(p.pid): float(p.usedGpuMemory or 0) / (1024 * 1024) for p in procs}
        except Exception:  # noqa: BLE001 - process-list API unavailable at all
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {-1: float(mem.used) / (1024 * 1024)}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - shutdown failures are never fatal
            pass


def own_usage_delta_mb(
    before: Optional[Dict[int, float]], after: Optional[Dict[int, float]]
) -> Optional[float]:
    """Infer this process's own memory growth between two snapshots without
    needing to identify which pid is "ours" (broken in this environment --
    see module docstring). Sums positive growth across every pid seen in
    either snapshot: a genuinely external, otherwise-idle tenant contributes
    ~0 growth across one short window, so growth in the diff is attributable
    to whatever ran between the two snapshots, regardless of which pid the
    driver happened to file it under."""
    if before is None or after is None:
        return None
    pids = set(before) | set(after)
    return sum(max(0.0, after.get(pid, 0.0) - before.get(pid, 0.0)) for pid in pids)


def read_current_mem_mb(device_index: int = 0) -> Optional[float]:
    """One-shot snapshot of this process's current GPU memory (best mode
    available -- see module docstring). Returns ``None`` if pynvml/NVML is
    unavailable. Meant as a pre-cell baseline."""
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
        mode = _mem_mode_for(pynvml, handle)
        return _read_mem_mb(pynvml, handle, os.getpid(), mode)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - shutdown failures are never fatal
            pass


# Bit -> name for nvmlDeviceGetCurrentClocksThrottleReasons. Looked up by name on
# the module rather than hardcoded: the constant set grows between driver releases,
# and an unknown bit is better reported as its hex value than dropped.
_THROTTLE_BITS = (
    ("nvmlClocksThrottleReasonGpuIdle", "gpu-idle"),
    ("nvmlClocksThrottleReasonApplicationsClocksSetting", "app-clocks-setting"),
    ("nvmlClocksThrottleReasonSwPowerCap", "sw-power-cap"),
    ("nvmlClocksThrottleReasonHwSlowdown", "hw-slowdown"),
    ("nvmlClocksThrottleReasonSyncBoost", "sync-boost"),
    ("nvmlClocksThrottleReasonSwThermalSlowdown", "sw-thermal"),
    ("nvmlClocksThrottleReasonHwThermalSlowdown", "hw-thermal"),
    ("nvmlClocksThrottleReasonHwPowerBrakeSlowdown", "hw-power-brake"),
    ("nvmlClocksThrottleReasonDisplayClockSetting", "display-clock-setting"),
)


def _throttle_names(pynvml_mod, bits: int) -> List[str]:
    """Human-readable throttle reasons for a NVML reason bitmask.

    ``gpu-idle`` is dropped: it is set whenever the card has nothing to do, which
    during a benchmark's own idle gaps is normal and would flag every run.
    """
    names = []
    matched = 0
    for const_name, label in _THROTTLE_BITS:
        bit = getattr(pynvml_mod, const_name, None)
        if bit is None:
            continue
        matched |= int(bit)
        if bits & int(bit) and label != "gpu-idle":
            names.append(label)
    leftover = bits & ~matched
    if leftover:
        names.append(f"unknown-0x{leftover:x}")
    return names


def _first(*getters, transform=None):
    """First getter that returns a value, optionally transformed. ``None`` if none do."""
    for getter in getters:
        try:
            value = getter()
        except Exception:  # noqa: BLE001 - an absent or unsupported NVML getter
            continue
        if value is None:
            continue
        try:
            return transform(value) if transform else value
        except (TypeError, ValueError):
            continue
    return None


def _decode(value):
    """NVML string getters return ``bytes`` on some wheel versions, ``str`` on others."""
    return value.decode() if isinstance(value, bytes) else value


def device_context(device_index: int = 0) -> dict:
    """Static device facts plus a census of who else is on the GPU right now.

    The census is the point: this repo has already published a benchmark sweep that
    ran against a live camera-inference workload holding ~13.7 GB, and the numbers
    were not comparable with the idle baseline they were merged next to. A run that
    records its co-tenants can be labelled contended instead of being quietly wrong.
    Returns ``{"available": False, ...}`` rather than raising when NVML is absent.
    """
    try:
        import pynvml
    except ImportError as exc:
        return {"available": False, "error": f"pynvml not installed: {exc}"}
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception as exc:  # noqa: BLE001 - covers pynvml.NVMLError + friends
        return {"available": False, "error": f"nvmlInit failed: {exc}"}

    def _try(fn, default=None):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - any single fact may be unsupported
            return default

    try:
        name = _decode(_try(lambda: pynvml.nvmlDeviceGetName(handle)))
        mem = _try(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
        cotenants = []
        for proc in _try(lambda: pynvml.nvmlDeviceGetComputeRunningProcesses(handle), []) or []:
            if int(proc.pid) == os.getpid():
                continue
            cotenants.append({
                "pid": int(proc.pid),
                "mem_mb": round(float(proc.usedGpuMemory or 0) / (1024 * 1024), 1),
            })
        return {
            "available": True,
            "name": name,
            "driver": _decode(_try(pynvml.nvmlSystemGetDriverVersion)),
            "total_mem_mb": round(float(mem.total) / (1024 * 1024), 1) if mem else None,
            "used_mem_mb": round(float(mem.used) / (1024 * 1024), 1) if mem else None,
            # nvidia-ml-py drops and renames getters between versions: the "enforced"
            # variant is absent on the wheel in this image, so fall back rather than
            # leaving the report's power headroom permanently blank.
            "power_limit_w": _first(
                lambda: pynvml.nvmlDeviceGetEnforcedPowerManagementLimit(handle),
                lambda: pynvml.nvmlDeviceGetPowerManagementLimit(handle),
                transform=lambda mw: round(float(mw) / 1000.0, 1),
            ),
            "max_sm_clock_mhz": _try(
                lambda: float(pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM))
            ),
            "utilization_pct": _try(
                lambda: float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            ),
            "cotenants": cotenants,
        }
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
        self._mem_mode = "device"
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
        self._mem_mode = _mem_mode_for(pynvml, self._handle)
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
                _read_mem_mb(self._pynvml, self._handle, self._pid, self._mem_mode)
            )
        except Exception:  # noqa: BLE001 - a single failed sample must not kill the thread
            pass
        # Trust context. Each read is guarded separately: these APIs are the most
        # likely of the lot to be missing on a given driver/wheel pair, and a
        # missing clock reading must not cost the caller its utilization samples.
        try:
            self._samples.power_w.append(
                float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            self._samples.sm_clock_mhz.append(
                float(self._pynvml.nvmlDeviceGetClockInfo(
                    self._handle, self._pynvml.NVML_CLOCK_SM))
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            bits = self._pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._handle)
            self._samples.throttle_reasons.update(_throttle_names(self._pynvml, bits))
        except Exception:  # noqa: BLE001
            pass

    @property
    def result(self) -> GpuSample:
        if self._pynvml is None:
            return GpuSample(available=False, error=self._init_error)

        util = self._samples.util_pct
        mem = self._samples.mem_used_mb
        power = self._samples.power_w
        clocks = self._samples.sm_clock_mhz
        return GpuSample(
            peak_mem_mb=max(mem) if mem else None,
            mean_util_pct=(sum(util) / len(util)) if util else None,
            peak_util_pct=max(util) if util else None,
            available=True,
            error=None,
            mean_power_w=(sum(power) / len(power)) if power else None,
            peak_power_w=max(power) if power else None,
            mean_sm_clock_mhz=(sum(clocks) / len(clocks)) if clocks else None,
            throttle_reasons=sorted(self._samples.throttle_reasons),
        )
