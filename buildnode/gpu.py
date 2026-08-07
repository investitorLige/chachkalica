"""What this node is: GPU identity, driver/CUDA/TensorRT versions, free memory.

This is the whole point of the ``/health`` endpoint. An engine is only valid on
the GPU + TensorRT version that built it, so before an operator dispatches a
build they need to know exactly what they are dispatching it to — and after a
bundle comes back, the recorded identity is what says which machine it will run
on. Everything here degrades to ``None`` rather than raising: a node with a
missing driver should report itself as unusable, not fail to answer.
"""

from typing import Any, Dict, Optional


def tensorrt_version() -> Optional[str]:
    try:
        import tensorrt as trt

        return str(trt.__version__)
    except Exception:  # noqa: BLE001 - absent/broken TensorRT is a reportable state
        return None


def _nvml() -> Optional[Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:  # noqa: BLE001 - no driver, no GPU, or no nvidia-ml-py
        return None


def _text(value: Any) -> Optional[str]:
    """NVML returns ``str`` on modern nvidia-ml-py and ``bytes`` on older ones."""
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def _cuda_driver_version(pynvml) -> Optional[str]:
    try:
        raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
    except Exception:  # noqa: BLE001
        return None
    # NVML packs it as 1000 * major + 10 * minor.
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def describe() -> Dict[str, Any]:
    """Identity + capability for ``/health``. Never raises."""
    info: Dict[str, Any] = {
        "gpu_name": None,
        "gpu_count": 0,
        "compute_capability": None,
        "driver_version": None,
        "cuda_version": None,
        "total_mem_mb": None,
        "free_mem_mb": None,
        "tensorrt_version": tensorrt_version(),
    }

    pynvml = _nvml()
    if pynvml is None:
        return info

    try:
        info["gpu_count"] = int(pynvml.nvmlDeviceGetCount())
        info["driver_version"] = _text(pynvml.nvmlSystemGetDriverVersion())
        info["cuda_version"] = _cuda_driver_version(pynvml)
        if info["gpu_count"]:
            # Device 0 only. A build takes one GPU and TensorRT picks the current
            # device; reporting a second card would imply a scheduling story this
            # node does not have.
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info["gpu_name"] = _text(pynvml.nvmlDeviceGetName(handle))
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            info["compute_capability"] = f"{major}.{minor}"
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["total_mem_mb"] = int(memory.total) // (1024 * 1024)
            info["free_mem_mb"] = int(memory.free) // (1024 * 1024)
    except Exception:  # noqa: BLE001 - a partial report beats no report
        pass

    return info


def usable() -> bool:
    """Whether this node can actually build: a visible GPU and a TensorRT import."""
    info = describe()
    return bool(info["gpu_count"]) and info["tensorrt_version"] is not None
