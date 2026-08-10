"""Frame decoding, on the GPU where possible.

After the engine work in :mod:`gpu_infer.engine` removes the per-call stalls, decode is the
largest remaining host cost — PIL measured 44 of 54 ms/frame at 4K. ``torchvision.io.decode_jpeg``
with ``device="cuda"`` runs it on nvJPEG instead, and the frame is *born* in device memory, so the
host-to-device copy of the raw pixels disappears along with the decode.

Three things this is careful about.

**It is not bit-identical to PIL, and the gap is bigger than "rounding".** Measured on this
repo's PPE frames (1080x810, 4:2:0): the median per-pixel difference is **0** and the mean is
sub-LSB, so most of the image agrees exactly — but the tail is heavy. About **12% of pixels
differ by >= 1/255**, **0.6% by >= 8/255**, and the maximum is near **64/255**, concentrated on
sharp edges, which is the signature of different chroma upsampling rather than a different
rounding mode. On one of four sample frames that moved a detection box by ~11% of the frame.

An earlier version of this docstring said "a few least significant bits". That was wrong, and
the measurement is why ``gpu_decode`` now defaults to **off**: a package whose claim is
byte-identical detections cannot enable a different input by default. GPU decode is a deliberate
throughput trade, not a free win, and it is the *only* thing that has to be given up for exact
output — everything else here is bit-identical either way.

**It always falls back rather than failing.** Non-JPEG suffixes, a missing torchvision, and the
JPEG variants nvJPEG rejects (progressive, CMYK, 12-bit, arithmetic-coded) all route to the same
PIL path ``chachak.preview._load_image_tensor`` uses. A bundle without torchvision installed
runs correctly at PIL speed.

**The divide is by a tensor.** ``uint8 -> float32`` is exact, but ``tensor / 255.0`` on CUDA
lowers to a multiply by the reciprocal, and 1/255 is not representable in float32 — a handful of
pixels come back one ULP off. Dividing by a 0-dim tensor selects the correctly-rounded kernel.
This is the same reasoning as ``chachak/preview.py:50`` and
``trt_infer.postprocess_torch._divisors``, and it is why the fallback path is *also* exact rather
than merely close.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

#: Suffixes nvJPEG can attempt. Everything else goes straight to PIL.
_JPEG_SUFFIXES = {".jpg", ".jpeg"}

#: Set once so a fallback is reported to the operator a single time, not per frame.
_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[gpu_infer] {message}")


def _scale(device):
    """The 255 divisor as a 0-dim tensor. See the module docstring on why it is not a float."""
    import torch

    return torch.tensor(255.0, dtype=torch.float32, device=device)


def _load_with_pil(path: Path, device):
    """PIL decode, matching ``chachak.preview._load_image_tensor`` operation for operation.

    Reimplemented rather than imported: ``chachak.preview`` imports ``chachak.infer``, which
    imports ``chachak._friendy`` and thereby the training package. The transfer happens *before*
    the permute and scale so the bus carries uint8 (a quarter of the bytes) and the per-pixel
    divide lands on the device.
    """
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    # .copy(): PIL exposes a read-only buffer and torch.from_numpy needs a writable one.
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).to(device)
    return tensor.permute(2, 0, 1).float() / _scale(tensor.device)


def _decode_jpeg_cuda(paths: Sequence[Path], device):
    """nvJPEG decode of a batch of JPEGs. Returns CHW float [0,1] tensors, or ``None``.

    ``None`` means "this batch could not be decoded on the device" — a missing torchvision, an
    older torchvision whose ``decode_jpeg`` does not take a list, or a file nvJPEG rejects. The
    caller then falls back per file rather than failing.
    """
    import torch

    try:
        from torchvision.io import decode_jpeg
    except Exception:
        _warn_once("no-torchvision", "torchvision unavailable; decoding with PIL")
        return None

    try:
        buffers = [
            torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
            for path in paths
        ]
        decoded = decode_jpeg(buffers, device=device)
    except Exception as error:  # nvJPEG rejects some valid JPEGs; PIL handles them
        _warn_once(
            "nvjpeg-reject",
            f"nvJPEG could not decode a frame ({type(error).__name__}: {error}); "
            f"falling back to PIL for the affected files",
        )
        return None

    if not isinstance(decoded, (list, tuple)):
        decoded = [decoded]
    scale = _scale(device)
    return [frame.float() / scale for frame in decoded]


def load_frames(
    paths: Sequence, device, *, gpu_decode: bool = False
) -> List["object"]:
    """Load frames as CHW float32 ``[0, 1]`` RGB tensors on ``device``.

    JPEGs go through nvJPEG as one batch when ``gpu_decode`` is set; anything else, and anything
    that batch fails on, goes through PIL individually. Order always matches ``paths``.
    """
    paths = [Path(path) for path in paths]
    if not paths:
        return []

    if not gpu_decode:
        return [_load_with_pil(path, device) for path in paths]

    jpeg_indices = [
        index
        for index, path in enumerate(paths)
        if path.suffix.lower() in _JPEG_SUFFIXES
    ]
    frames: List[Optional["object"]] = [None] * len(paths)

    if jpeg_indices:
        decoded = _decode_jpeg_cuda([paths[index] for index in jpeg_indices], device)
        if decoded is not None:
            for slot, frame in zip(jpeg_indices, decoded):
                frames[slot] = frame

    for index, path in enumerate(paths):
        if frames[index] is None:
            frames[index] = _load_with_pil(path, device)
    return frames


def describe_decoder(gpu_decode: bool) -> str:
    """One line naming the decoder actually in use, for the operator log.

    Worth printing: "GPU decode" that silently fell back to PIL is the difference between a
    number that reproduces and one that does not, and between exact output and near-exact.
    """
    if not gpu_decode:
        return "decode: PIL (host) — byte-identical to the reference path"
    try:
        import torchvision  # noqa: F401
    except Exception:
        return "decode: PIL (host) — torchvision unavailable, so nvJPEG was not used"
    return (
        "decode: nvJPEG (device) for .jpg/.jpeg, PIL for everything else — NOT byte-identical "
        "to PIL (median delta 0, but ~12% of pixels differ by >=1/255 on 4:2:0 content)"
    )


def frame_size(frame) -> Tuple[int, int]:
    """``(width, height)`` of a CHW frame, matching ``chachak.pipeline._frame_size``."""
    return int(frame.shape[2]), int(frame.shape[1])
