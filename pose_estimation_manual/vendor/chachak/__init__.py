"""chachak — stackable inference pipelines over an exported detector.

Each pipeline decides *how a frame is presented to the model* before detection:
whole-frame, cut into overlapping tiles, cropped around detected people, or
several of those stacked and merged. That choice matters as much as the model
itself — a model trained on person crops has to be run on person crops.

Public API::

    build_pipeline(config, adapter, device, detector=..., box_cache=...)
    InferenceConfig / TilingConfig / DetectorConfig
    Detector, load_checkpoint_adapter

Every pipeline exposes ``process_batch(images, targets)``: a list of CHW frames
in, one Friendy ``(N, 6)`` tensor of ``[cx, cy, w, h, confidence, class_id]``
per frame out, normalized to the full frame.

Pass its optional ``context`` and each frame's people come back too, along with a
seventh column saying which of them a detection was found on — see
:meth:`pipeline.Pipeline.process_batch`.
"""

from importlib import import_module
from typing import Any

from .config import (
    DETECTOR_PIPELINES,
    PIPELINE_NAMES,
    DetectorConfig,
    InferenceConfig,
    TilingConfig,
)

__all__ = [
    "DETECTOR_PIPELINES",
    "PIPELINE_NAMES",
    "PIPELINE_REGISTRY",
    "BatchDetectPipeline",
    "BatchPeoplePipeline",
    "ChainedPipeline",
    "Detector",
    "DetectorConfig",
    "InferenceConfig",
    "PeopleDetectFirstPipeline",
    "Pipeline",
    "TilingConfig",
    "as_class_map",
    "build_pipeline",
    "infer_in_chunks",
    "load_checkpoint_adapter",
    "predict_adapter",
]

# ``config`` is pure dataclasses and has no business needing torch, but everything
# below it does (detector.py, infer.py, pipeline.py all `import torch`). The web
# and camera-workers containers import ``vendor.chachak.config`` directly (for
# these dataclasses only) and deliberately carry no torch install — see
# Dockerfile.web. A package's __init__ always runs before any of its submodules
# import, so eagerly importing the torch-dependent names here would drag torch
# into that import even though config.py itself never touches it. Resolving them
# lazily, via PEP 562 module __getattr__, means torch is only imported by the
# process that actually asks for ``Detector``/``build_pipeline``/etc. — the GPU
# worker, through vendor/chachak/detector.py's ``load_detector``.
_LAZY = {
    "Detector": ".detector",
    "as_class_map": ".infer",
    "infer_in_chunks": ".infer",
    "load_checkpoint_adapter": ".infer",
    "predict_adapter": ".infer",
    "BatchDetectPipeline": ".pipeline",
    "BatchPeoplePipeline": ".pipeline",
    "ChainedPipeline": ".pipeline",
    "PeopleDetectFirstPipeline": ".pipeline",
    "Pipeline": ".pipeline",
    "PIPELINE_REGISTRY": ".registry",
    "build_pipeline": ".registry",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
