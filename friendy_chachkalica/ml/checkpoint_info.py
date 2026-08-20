"""Cheap, read-only inspection of a training checkpoint.

Answers "what input size will this checkpoint export at" without doing an
actual export — no model build, no forward pass, no file write, just the
``torch.load`` every exporter already does first (see ``ml/onnx_export/cli.py``
/ ``ml/trt_export/cli.py``) plus the same per-arch size math each exporter
applies before tracing anything (``adapters/rtdetr.py``'s, ``adapters/dfine.py``'s, and
``adapters/ecdet.py``'s ``_ceil_to_multiple``,
``adapters/yolox.py``'s ``_make_divisible``, ``adapters/rfdetr.py``'s
``resolution`` field). Reusing those helpers keeps the size math defined once.

Faster R-CNN and RetinaNet export at ``resize_mode="none"`` (see
``onnx_export/arch/fasterrcnn.py`` / ``retinanet.py``) — genuinely variable
input, no single trained size — so :func:`resolve_trained_size` returns
``None`` for them.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch

# Adapter constructor defaults, duplicated here as fallbacks for a checkpoint
# whose params dict didn't override them (see build_rtdetr/YOLOXAdapter's
# input_max_size=640/input_size_multiple=32, RFDETRAdapter.resolution=560).
_RTDETR_YOLOX_DEFAULT_MAX_SIZE = 640
_RTDETR_YOLOX_DEFAULT_MULTIPLE = 32
_RFDETR_DEFAULT_RESOLUTION = 560


def _int_or_default(value, default: int) -> int:
    """``int(value)``, or ``default`` when ``value`` is unset (``None``).

    Not the same as ``value or default`` — a checkpoint that explicitly set
    ``input_max_size: 0`` (resizing disabled) must stay ``0``, not fall back to
    the adapter default, matching ``RTDETRAdapter._fixed_canvas_size``'s own
    ``self.input_max_size is None or self.input_max_size <= 0`` check.
    """
    return default if value is None else int(value)


def resolve_trained_size(arch: str, params: dict) -> Optional[Tuple[int, int]]:
    """The square ``(H, W)`` a checkpoint of this ``arch`` exports at, or ``None``
    when the arch has no single fixed size (Faster R-CNN, RetinaNet)."""
    if arch == "rtdetr":
        from .adapters.rtdetr import _ceil_to_multiple

        max_size = _int_or_default(params.get("input_max_size"), _RTDETR_YOLOX_DEFAULT_MAX_SIZE)
        multiple = _int_or_default(params.get("input_size_multiple"), _RTDETR_YOLOX_DEFAULT_MULTIPLE)
        if max_size <= 0:
            return None  # resizing disabled -- no fixed canvas (ONNX export itself refuses this)
        side = _ceil_to_multiple(max_size, multiple)
        return (side, side)

    if arch == "dfine":
        # Same geometry contract as rtdetr (DFineAdapter is a near-verbatim
        # clone): input_max_size/input_size_multiple, dynamically resized, no
        # anchors baked at build time (unlike ecdet's ECTransformer).
        from .adapters.dfine import _ceil_to_multiple

        max_size = _int_or_default(params.get("input_max_size"), _RTDETR_YOLOX_DEFAULT_MAX_SIZE)
        multiple = _int_or_default(params.get("input_size_multiple"), _RTDETR_YOLOX_DEFAULT_MULTIPLE)
        if max_size <= 0:
            return None
        side = _ceil_to_multiple(max_size, multiple)
        return (side, side)

    if arch == "ecdet":
        from .adapters.ecdet import (
            ECDET_NATIVE_SIZE,
            ECDET_SIZE_MULTIPLE,
            _ceil_to_multiple,
        )

        max_size = _int_or_default(params.get("input_max_size"), ECDET_NATIVE_SIZE)
        multiple = _int_or_default(params.get("input_size_multiple"), ECDET_SIZE_MULTIPLE)
        if max_size <= 0:
            return None
        side = _ceil_to_multiple(max_size, multiple)
        return (side, side)

    if arch == "yolox":
        from .adapters.yolox import _make_divisible

        max_size = _int_or_default(params.get("input_max_size"), _RTDETR_YOLOX_DEFAULT_MAX_SIZE)
        multiple = _int_or_default(params.get("input_size_multiple"), _RTDETR_YOLOX_DEFAULT_MULTIPLE)
        if max_size <= 0:
            return None
        side = _make_divisible(max_size, multiple)
        return (side, side)

    if arch == "rfdetr":
        resolution = _int_or_default(params.get("resolution"), _RFDETR_DEFAULT_RESOLUTION)
        return (resolution, resolution)

    return None  # fasterrcnn / retinanet: resize_mode="none", genuinely variable input


def inspect_checkpoint(checkpoint_path: str | Path) -> dict:
    """``{"arch", "trained_size", "fp16_trusted", "batch_aware"}`` for a
    checkpoint, without exporting anything.

    ``trained_size`` is ``[H, W]`` or ``None`` (see :func:`resolve_trained_size`).

    ``fp16_trusted`` mirrors ``trt_export.arch.is_fp16_trusted`` — False for an
    arch whose fp16 engine is known not to reproduce its fp32 output (currently
    ecdet). It rides along here because the caller that needs it is the export UI:
    ``precision="auto"`` already floors an untrusted arch to fp32 inside
    ``build_engine``, but a caller that names fp16 *explicitly* bypasses that
    floor, and the Django TRT export form does exactly that. Surfacing the flag
    lets the form default to fp32 and say why, without teaching the Django app any
    arch names of its own.

    ``batch_aware`` mirrors ``trt_export.arch.is_batch_aware`` — the same idea,
    for the export form's batch-size field: only a batch-aware arch's engine
    decodes a submission wider than 1 correctly, so the form only offers a
    variable/fixed>1 batch profile when this is True.

    ``auto_precision``/``auto_cast_backend`` are what ``precision="auto"`` would
    actually build here, from ``trt_export.arch.resolve_auto_precision``. They are
    not always ``fp16 if fp16_trusted else fp32``: an arch whose blanket-cast fp16
    is untrusted can still have a clean fp16 engine through ModelOpt AutoCast
    (dfine does), and whether that route exists depends on nvidia-modelopt being
    installed *on this machine*. The export form defaults its precision select
    from this rather than re-deriving the policy in Django.
    """
    from .trt_export.arch import is_batch_aware, is_fp16_trusted, resolve_auto_precision
    from .trt_export.modelopt_cast import modelopt_version

    state = torch.load(checkpoint_path, map_location="cpu")
    arch = state["model_name"]
    params = dict((state.get("model_config") or {}).get("params") or {})
    size = resolve_trained_size(arch, params)
    auto_precision, auto_backend = resolve_auto_precision(
        arch, modelopt_available=modelopt_version() is not None
    )
    return {
        "arch": arch,
        "trained_size": list(size) if size else None,
        "fp16_trusted": bool(is_fp16_trusted(arch)),
        "batch_aware": bool(is_batch_aware(arch)),
        "auto_precision": auto_precision,
        "auto_cast_backend": auto_backend,
    }
