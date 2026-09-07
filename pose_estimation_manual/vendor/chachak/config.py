"""Config for chachak pipelines: frozen dataclasses, built in code.

The original in ``chackalica_unified`` carried a ``PipelineConfig`` describing a
whole *dataset eval run* — model checkpoint, image/label directories, output
directory, class map, IoU threshold sweeps — loaded from a YAML file. None of
that applies here: this project's config source is the Django admin, one row per
camera, and the thing being configured is a live stream rather than a dataset.

So :class:`InferenceConfig` carries exactly the fields ``pipeline.py`` and
``registry.py`` actually read, and nothing else. ``TilingConfig`` and
``DetectorConfig`` are kept verbatim, because the pipelines reach into their
attributes by name and their defaults encode real tuning knowledge.

See ``cameras.models.Camera.build_inference_config`` for the layering that fills
these in: per-camera field → the engine's Contract-C ``recommended`` block →
the defaults below.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# The names ``registry.build_pipeline`` can actually construct.
PIPELINE_NAMES = ("batch_detect", "people_detect_first", "batch_people", "chain")

# "raw" is not one of them: it means "no pipeline at all — hand the whole frame
# straight to the model", and a caller seeing it short-circuits to
# ``adapter.predict`` instead of building anything. It is still a legal value for
# :class:`InferenceConfig`, so that "how is this camera configured" has a single
# total answer and callers never juggle a None config.
RAW = "raw"
ALL_PIPELINES = (RAW, *PIPELINE_NAMES)

# Pipelines that need a second model (a person detector) loaded alongside the
# main one. Checked by callers before building, so a misconfiguration surfaces as
# a clear error instead of an AttributeError deep inside a pipeline.
DETECTOR_PIPELINES = ("people_detect_first", "batch_people")


@dataclass(frozen=True)
class TilingConfig:
    tile_size_px: Optional[int] = None
    tile_width_pct: float = 50.0
    tile_height_pct: float = 50.0
    overlap: float = 0.2
    nms_iou: float = 0.5


@dataclass(frozen=True)
class DetectorConfig:
    score_threshold: float = 0.5
    person_class_name: Optional[str] = "person"
    person_class_id: Optional[int] = None
    expand_ratio: float = 0.0
    nms_iou: float = 0.5
    min_box_size: float = 0.0
    batch_size: int = 4


@dataclass(frozen=True)
class InferenceConfig:
    """One camera's pipeline configuration, as the pipelines read it."""

    pipeline: str
    chain: List[str] = field(default_factory=list)
    infer_batch_size: int = 4
    score_threshold: float = 0.5
    # An eval-era vestige that ``pipeline._inference_score_threshold`` still
    # consults: when set it wins over ``score_threshold``. Live inference has one
    # threshold, so this stays None — kept only so the pipelines are unmodified.
    map_score_threshold: Optional[float] = None
    merge_nms_iou: float = 0.5
    tiling: TilingConfig = field(default_factory=TilingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def validate(self) -> None:
        """Raise ``ValueError`` on a config no pipeline could run.

        Called at build time (per camera, on the GPU worker's reconcile) so a bad
        admin edit reports itself against that camera instead of throwing from
        inside a tiling loop several frames later.
        """
        if self.pipeline not in ALL_PIPELINES:
            raise ValueError(
                f"Unknown pipeline {self.pipeline!r}. "
                f"Available: {', '.join(ALL_PIPELINES)}."
            )
        if self.infer_batch_size < 1:
            raise ValueError("infer_batch_size must be at least 1")

        # Structural checks — only meaningful for the pipeline actually selected.
        if self.pipeline == "chain":
            if not self.chain:
                raise ValueError("The 'chain' pipeline requires a non-empty chain list.")
            for child in self.chain:
                if child not in PIPELINE_NAMES or child == "chain":
                    raise ValueError(f"Invalid chain member: {child!r}")

        # Range checks — applied whatever the pipeline, including 'raw'. These
        # values come from a form, and a nonsensical one is wrong whether or not it
        # happens to be used today: skipping the check for a raw camera would let
        # overlap=1.5 sit there quietly until someone switched that camera to
        # tiling, and then fail in the GPU worker rather than in the admin.
        if not 0.0 <= self.tiling.overlap < 1.0:
            raise ValueError("tiling.overlap must be in [0, 1)")
        if self.tiling.tile_size_px is not None and self.tiling.tile_size_px <= 0:
            raise ValueError("tiling.tile_size_px must be greater than 0")
        for label, value in (
            ("tile_width_pct", self.tiling.tile_width_pct),
            ("tile_height_pct", self.tiling.tile_height_pct),
        ):
            if not 0.0 < value <= 100.0:
                raise ValueError(f"tiling.{label} must be in (0, 100]")
        if self.tiling.overlap and self.tiling.nms_iou is not None:
            if not 0.0 <= self.tiling.nms_iou <= 1.0:
                raise ValueError("tiling.nms_iou must be in [0, 1]")
        if not 0.0 <= self.merge_nms_iou <= 1.0:
            raise ValueError("merge_nms_iou must be in [0, 1]")
        if self.detector.expand_ratio < 0:
            raise ValueError("detector.expand_ratio must be at least 0")
        if self.detector.min_box_size < 0:
            raise ValueError("detector.min_box_size must be at least 0")

    def is_raw(self) -> bool:
        """Whether this config means "skip chachak, predict on the whole frame"."""
        return self.pipeline == RAW

    def needs_detector(self) -> bool:
        """Whether running this config requires a person-detector engine."""
        if self.pipeline in DETECTOR_PIPELINES:
            return True
        return self.pipeline == "chain" and any(
            child in DETECTOR_PIPELINES for child in self.chain
        )
