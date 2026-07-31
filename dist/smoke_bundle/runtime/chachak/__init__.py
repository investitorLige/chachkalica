"""chachak (exported-bundle subset) — DO NOT EDIT.

The same pipeline stack as the training repo's ``chachak`` package, minus
``run.py``: its batch-eval entrypoint builds a Friendy dataloader, which needs
the training toolkit. The bundle's entrypoint is ``infer.py`` at the bundle root
— it loads images directly and drives ``Pipeline.process_batch``.

Regenerate by re-exporting the bundle.
"""

from .config import PipelineConfig, load_pipeline_config, pipeline_config_from_dict
from .pipeline import (
    BatchDetectPipeline,
    BatchPeoplePipeline,
    ChainedPipeline,
    PeopleDetectFirstPipeline,
    Pipeline,
)
from .registry import PIPELINE_REGISTRY, build_pipeline

__all__ = [
    "PipelineConfig",
    "load_pipeline_config",
    "pipeline_config_from_dict",
    "PIPELINE_REGISTRY",
    "build_pipeline",
    "Pipeline",
    "BatchDetectPipeline",
    "PeopleDetectFirstPipeline",
    "BatchPeoplePipeline",
    "ChainedPipeline",
]
