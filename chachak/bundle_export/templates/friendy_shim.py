"""Bundle-local stand-in for ``chachak/_friendy.py`` — DO NOT EDIT.

The real shim imports the whole ``friendy_chachkalica`` training toolkit
(torchvision detection heads, transformers, rfdetr, the dataset loader). An
exported bundle runs already-exported ONNX/TensorRT artifacts, so it carries only
the self-contained pieces chachak's *inference* path actually imports, vendored
verbatim under ``_vendor/``:

* ``formats``     — box-format conversions         (used by ``boxes.py``)
* ``metrics``     — ``evaluate_detection``, the by-name class remap, and the
                    ``EVAL_HARD_IMAGES_FRACTION`` constant (imported by
                    ``pipeline.py``; the remap is what lets several bundled
                    models be merged into one class space)
* ``postprocess`` — class-aware NMS                (used by ``metrics``)
* ``device``      — ``resolve_device``             (used by ``infer.py``)

Everything else is a stub that raises with an explanation if called. They exist so
the verbatim-copied chachak modules — which import these names at module level —
load unchanged; the inference path never reaches them. That covers the training
toolkit (model building, dataset loaders, experiment configs) and the handful of
helpers ``Pipeline.run()`` uses only in its batch-eval tail (serializing the
result, writing the worst-images artifact) — ``friendy_chachkalica/ml/val.py``
and ``ml/train.py``, both of which pull in the whole trainer at module level.

Regenerate by re-exporting the bundle.
"""

from ._vendor import formats  # noqa: F401  (re-exported: `from ._friendy import formats`)
from ._vendor.device import resolve_device
from ._vendor.formats import (
    FRIENDY_PREDICTION_COLUMNS,
    clip_xyxy,
    xywhn_to_xyxy,
    xyxy_to_xywh,
    xyxy_to_xywhn,
)
from ._vendor.metrics import (
    EVAL_HARD_IMAGES_FRACTION,
    evaluate_detection,
    remap_raw_predictions_to_eval_classes,
)


def _training_only(name: str):
    """A placeholder for a training-toolkit symbol the bundle does not ship."""

    def stub(*_args, **_kwargs):
        raise RuntimeError(
            f"{name!r} belongs to the friendy_chachkalica training toolkit, which is "
            f"not part of an exported bundle. Bundles run the exported ONNX/TensorRT "
            f"artifacts through onnx_infer/trt_infer — they never rebuild an "
            f"architecture from a .pt checkpoint. Use the training repo for {name!r}."
        )

    stub.__name__ = name
    return stub


# Reached only from `Pipeline.run()` (batch eval over a Friendy dataloader), which
# a bundle never runs — `infer.py` drives `process_batch` directly.
_to_builtin = _training_only("_to_builtin")
_write_yaml = _training_only("_write_yaml")
_write_hard_images = _training_only("_write_hard_images")

build_model = _training_only("build_model")
build_eval_dataloader = _training_only("build_eval_dataloader")
detection_collate_fn = _training_only("detection_collate_fn")
DatasetConfig = _training_only("DatasetConfig")
EvaluationConfig = _training_only("EvaluationConfig")
ExperimentConfig = _training_only("ExperimentConfig")
ModelConfig = _training_only("ModelConfig")
TrainingConfig = _training_only("TrainingConfig")

__all__ = [
    "formats",
    "FRIENDY_PREDICTION_COLUMNS",
    "clip_xyxy",
    "xywhn_to_xyxy",
    "xyxy_to_xywh",
    "xyxy_to_xywhn",
    "EVAL_HARD_IMAGES_FRACTION",
    "evaluate_detection",
    "remap_raw_predictions_to_eval_classes",
    "resolve_device",
    "_to_builtin",
    "_write_yaml",
    "_write_hard_images",
    "build_model",
    "build_eval_dataloader",
    "detection_collate_fn",
    "DatasetConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
]
