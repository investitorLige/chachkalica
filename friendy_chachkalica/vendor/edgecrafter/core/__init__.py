"""Trimmed twin of upstream ``engine/core/__init__.py``.

Upstream also re-exports ``BaseConfig`` and ``YAMLConfig`` from ``_config.py`` /
``yaml_config.py``, which are not vendored (they pull in SummaryWriter,
GradScaler and DataLoader for a solver we do not use). Everything the model
modules actually import — ``register`` — plus the config helpers the adapter
needs to stand in for ``YAMLConfig`` is here.
"""

from .workspace import GLOBAL_CONFIG, create, register
from .yaml_utils import load_config, merge_config, merge_dict

__all__ = [
    "GLOBAL_CONFIG",
    "create",
    "register",
    "load_config",
    "merge_config",
    "merge_dict",
]
