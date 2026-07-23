"""Export-function registry — one exporter per architecture, keyed by the same
model name as ``friendy_chachkalica.registry.MODEL_REGISTRY``.

Each exporter has signature::

    export(adapter, *, num_classes, params, class_map, onnx_path) -> dict

It writes the ``.onnx`` at ``onnx_path`` and returns the ``meta.json`` dict
(Contract B). The service-side handler in ``onnx_infer/arch/<name>.py`` is the
matching half — the two must agree on Contracts A/B for that arch.
"""

from __future__ import annotations

try:
    from .arch.fasterrcnn import export_fasterrcnn
    from .arch.retinanet import export_retinanet
    from .arch.rfdetr import export_rfdetr
    from .arch.rtdetr import export_rtdetr
    from .arch.yolox import export_yolox
except ImportError:  # run as a flat script
    from arch.fasterrcnn import export_fasterrcnn
    from arch.retinanet import export_retinanet
    from arch.rfdetr import export_rfdetr
    from arch.rtdetr import export_rtdetr
    from arch.yolox import export_yolox

EXPORT_REGISTRY = {
    "fasterrcnn": export_fasterrcnn,
    "retinanet": export_retinanet,
    "yolox": export_yolox,
    "rtdetr": export_rtdetr,
    "rfdetr": export_rfdetr,
}


def get_exporter(model_name: str):
    try:
        return EXPORT_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(EXPORT_REGISTRY)) or "(none)"
        raise ValueError(
            f"No ONNX exporter for arch {model_name!r}. Registered: {available}"
        ) from exc
