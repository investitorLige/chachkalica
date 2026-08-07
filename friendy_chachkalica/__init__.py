"""friendy_chachkalica — the torch training/eval toolkit and its model registry.

Attribute access is lazy (PEP 562). Every name below still resolves exactly as it
did, on first use instead of at import. The reason is the subpackages: importing
this module eagerly built every adapter (torchvision, transformers, rfdetr), which
meant ``friendy_chachkalica.ml.trt_export.builder`` — a torch-free TensorRT
compile helper — could not be reached without the entire training stack. The slim
build node (``buildnode/``) reaches exactly that subset and nothing else.
"""

import importlib

# name -> submodule it lives in.
_LAZY = {
    "ExperimentConfig": ".config",
    "DatasetConfig": ".config",
    "ModelConfig": ".config",
    "load_config": ".config",
    "FRIENDY_PREDICTION_COLUMNS": ".formats",
    "clip_xyxy": ".formats",
    "xywhn_to_xyxy": ".formats",
    "xyxy_prediction_to_friendy": ".formats",
    "xyxy_to_xywh": ".formats",
    "xyxy_to_xywhn": ".formats",
    "RetinaNetAdapter": ".ml.adapters.retinanet",
    "build_retinanet": ".ml.adapters.retinanet",
    "RTDETRAdapter": ".ml.adapters.rtdetr",
    "build_rtdetr": ".ml.adapters.rtdetr",
    "YOLOXAdapter": ".ml.adapters.yolox",
    "build_yolox": ".ml.adapters.yolox",
    "MODEL_REGISTRY": ".registry",
    "build_model": ".registry",
    "train_from_config": ".ml.train",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value  # resolve once; later lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "train_from_config",
    "FRIENDY_PREDICTION_COLUMNS",
    "clip_xyxy",
    "xywhn_to_xyxy",
    "xyxy_prediction_to_friendy",
    "xyxy_to_xywh",
    "xyxy_to_xywhn",
    "load_config",
    "ModelConfig",
    "ExperimentConfig",
    "DatasetConfig",
    "MODEL_REGISTRY",
    "RTDETRAdapter",
    "RetinaNetAdapter",
    "YOLOXAdapter",
    "build_model",
    "build_retinanet",
    "build_rtdetr",
    "build_yolox",
]
