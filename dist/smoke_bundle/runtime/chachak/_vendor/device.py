from __future__ import annotations

import torch


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda":
        _ensure_cuda_available(device_name)
    return device


def _ensure_cuda_available(device_name: str) -> None:
    if torch.cuda.is_available():
        return

    diagnostics = [
        f"torch_version={torch.__version__}",
        f"torch_cuda_build={torch.version.cuda}",
        f"cuda_device_count={torch.cuda.device_count()}",
    ]
    raise RuntimeError(
        f"Requested device {device_name!r}, but PyTorch cannot use CUDA. "
        f"Diagnostics: {', '.join(diagnostics)}. "
        "On the server, verify nvidia-smi works, run inside a GPU-enabled "
        "container/session, and install a CUDA-enabled torch/torchvision build."
    )
