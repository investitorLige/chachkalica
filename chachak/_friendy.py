"""Bridge to the sibling ``friendy_chachkalica`` toolkit.

``friendy_chachkalica`` is a proper package (it has ``__init__.py`` and uses
relative imports) living next to this one. ``chachak`` reuses its adapters, bbox
helpers, dataset loader, metrics, and eval serialization rather than
reimplementing them.

We import it under its ``friendy_chachkalica.`` namespace by putting the *repo
root* on ``sys.path`` — deliberately NOT friendy's own directory. Both projects
have modules named ``config`` and ``registry``; namespacing friendy avoids that
collision and lets friendy's relative imports resolve. The rest of ``chachak``
imports friendy symbols from this single place::

    from ._friendy import formats, build_model, evaluate_detection
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from friendy_chachkalica import formats  # noqa: E402
from friendy_chachkalica.formats import (  # noqa: E402
    FRIENDY_PREDICTION_COLUMNS,
    clip_xyxy,
    xywhn_to_xyxy,
    xyxy_to_xywh,
    xyxy_to_xywhn,
)
from friendy_chachkalica.registry import build_model  # noqa: E402
from friendy_chachkalica.data import (  # noqa: E402
    build_eval_dataloader,
    detection_collate_fn,
)
from friendy_chachkalica.metrics import (  # noqa: E402
    evaluate_detection,
    remap_raw_predictions_to_eval_classes,
)
from friendy_chachkalica.device import resolve_device  # noqa: E402
from friendy_chachkalica.config import (  # noqa: E402
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from friendy_chachkalica.val import _to_builtin, _write_yaml  # noqa: E402

__all__ = [
    "formats",
    "FRIENDY_PREDICTION_COLUMNS",
    "clip_xyxy",
    "xywhn_to_xyxy",
    "xyxy_to_xywh",
    "xyxy_to_xywhn",
    "build_model",
    "build_eval_dataloader",
    "detection_collate_fn",
    "evaluate_detection",
    "remap_raw_predictions_to_eval_classes",
    "resolve_device",
    "DatasetConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "_to_builtin",
    "_write_yaml",
]
