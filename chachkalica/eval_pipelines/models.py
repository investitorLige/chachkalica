"""Chachak pipeline evals, and the "Eval Pipelines" admin section.

This app owns nothing but the eval-comparison corner of the admin: a concrete
:class:`PipelineEvalRun` (one run of a chachak pipeline against a dataset) plus a
:class:`BaseEval` proxy that re-homes the existing :class:`training.EvalRun` under
this section so the base and pipeline evals sit side by side.

A ``PipelineEvalRun`` deliberately mirrors ``EvalRun``: the admin action writes a
chachak request YAML and enqueues a job that drives the trainer service's
``/pipeline`` endpoint, then ingests the resulting metrics back here. The extra
fields describe *which* pipeline and its detector/tiling knobs.
"""

from django.core.validators import MinValueValidator
from django.db import models

from fleet.models import Annotator, Dataset
from training import pipelines
from training.models import EvalRun, ExperimentDataset, TrainedModel, default_iou_thresholds


class PipelineEvalRun(models.Model):
    """One evaluation of a :class:`TrainedModel` through a chachak pipeline."""

    # Status vocabulary matches EvalRun so the shared status badge/analytics work.
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    STATUS_CHOICES = [
        (CREATED, "created"),
        (QUEUED, "queued"),
        (RUNNING, "running"),
        (OK, "ok"),
        (ERROR, "error"),
    ]

    # Reuse the label-source vocabulary from ExperimentDataset (as EvalRun does).
    SOURCE = ExperimentDataset.SOURCE
    ANNOTATOR = ExperimentDataset.ANNOTATOR
    EXPLICIT = ExperimentDataset.EXPLICIT
    NONE = ExperimentDataset.NONE

    # Pipeline vocabulary lives in ``training.pipelines`` so the Experiment and
    # this model share one definition (kept in sync with chachak.config).
    BATCH_DETECT = pipelines.BATCH_DETECT
    PEOPLE_DETECT_FIRST = pipelines.PEOPLE_DETECT_FIRST
    BATCH_PEOPLE = pipelines.BATCH_PEOPLE
    CHAIN = pipelines.CHAIN
    PIPELINE_CHOICES = pipelines.PIPELINE_CHOICES
    # Pipelines that require a person detector checkpoint.
    DETECTOR_PIPELINES = pipelines.DETECTOR_PIPELINES

    trained_model = models.ForeignKey(
        TrainedModel, on_delete=models.CASCADE, related_name="pipeline_eval_runs"
    )
    # Extra models combined with `trained_model` into one merged evaluation —
    # see TrainedModelAdmin.evaluate. Empty for an ordinary single-model eval.
    combined_models = models.ManyToManyField(TrainedModel, blank=True, related_name="+")
    dataset = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name="+")
    label_source = models.CharField(
        max_length=16,
        choices=ExperimentDataset.LABEL_SOURCE_CHOICES,
        default=ExperimentDataset.SOURCE,
    )
    annotator = models.ForeignKey(
        Annotator, on_delete=models.PROTECT, null=True, blank=True, related_name="+"
    )
    explicit_labels_path = models.CharField(max_length=1024, blank=True)
    map_score_threshold = models.FloatField(
        default=0.001,
        help_text="Minimum prediction confidence kept for AP/mAP. Keep this low so mAP can "
                  "sweep the precision-recall curve.",
    )
    score_threshold = models.FloatField(
        default=0.25,
        help_text="Operating-point confidence threshold used for precision, recall, F1, "
                  "prediction counts, and the confusion matrix.",
    )

    pipeline = models.CharField(max_length=32, choices=PIPELINE_CHOICES, default=BATCH_DETECT)
    detector_checkpoint = models.CharField(
        max_length=1024, blank=True,
        help_text="Person-detector checkpoint; required for people_detect_first / batch_people.",
    )
    detector_expand_ratio = models.FloatField(
        default=0.10,
        validators=[MinValueValidator(0.0)],
        verbose_name="Person-box expand ratio",
        help_text="Grow each detected person box by this fraction before cropping "
                  "(0.10 = +10% on width and height, 5% per side; 0 crops the box "
                  "exactly). Match the value the model was trained with. Only used by "
                  "people_detect_first / batch_people.",
    )
    detector_min_box_size = models.FloatField(
        null=True, blank=True,
        verbose_name="Person-crop minimum size (px)",
        help_text="Grow person crops smaller than this many pixels rather than "
                  "dropping them. Blank = chachak's default. Only used by "
                  "people_detect_first. Match what the model was trained with.",
    )
    tile_size_px = models.PositiveIntegerField(
        null=True, blank=True,
        verbose_name="Tile size (pixels)",
        help_text="Fixed square tile size; overrides the tile width/height "
                  "percentages. Blank = percentage tiling. Set this to the "
                  "resolution the model was trained at to preserve its native "
                  "pixel scale.",
    )
    tile_width_pct = models.FloatField(
        null=True, blank=True,
        help_text="Tile width as a percent (0–100] of each image's width.",
    )
    tile_height_pct = models.FloatField(
        null=True, blank=True,
        help_text="Tile height as a percent (0–100] of each image's height.",
    )
    overlap = models.FloatField(null=True, blank=True)
    merge_nms_iou = models.FloatField(
        verbose_name="merge_nms_iou_threshold",
        null=True, blank=True,
        help_text="Class-aware NMS IoU used to merge predictions across tiles/crops. "
                  "Blank = chachak's default.",
    )
    chain = models.JSONField(
        default=list, blank=True,
        help_text="Ordered pipeline names for the 'chain' pipeline.",
    )

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=CREATED)
    request_yaml_path = models.CharField(max_length=1024, blank=True)
    output_dir = models.CharField(max_length=1024, blank=True)
    metrics = models.JSONField(null=True, blank=True)
    last_error = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "All pipeline eval"
        verbose_name_plural = "All pipeline evals"

    def __str__(self) -> str:
        return (
            f"Pipeline Eval #{self.pk} — {self._models_label()} "
            f"[{self.pipeline}] on {self.dataset.name}"
        )

    def _models_label(self) -> str:
        names = [self.trained_model.name, *(m.name for m in self.combined_models.all())]
        return " + ".join(names)

    @property
    def is_combined(self) -> bool:
        return self.pk is not None and self.combined_models.exists()

    def all_models(self) -> list:
        return [self.trained_model, *self.combined_models.all()]

    def metric(self, key: str):
        return self.metrics.get(key) if isinstance(self.metrics, dict) else None

    def default_iou_thresholds(self):
        return default_iou_thresholds()


class BaseEval(EvalRun):
    """Proxy of :class:`training.EvalRun`, re-homed under "Eval Pipelines".

    Defined in this app, it inherits ``app_label="eval_pipelines"`` so the admin
    lists it in this section (renamed "Base Eval") while sharing EvalRun's table.
    """

    class Meta:
        proxy = True
        verbose_name = "Base Eval"
        verbose_name_plural = "Base Eval"


# Pipeline value stamped on base (raw-model) evals in the combined view, so a
# plain EvalRun sits in "All pipeline evals" beside the chachak-pipeline runs.
BASE_PIPELINE = "base"


class CombinedEval(models.Model):
    """Read-only union of :class:`PipelineEvalRun` + :class:`training.EvalRun`.

    Backs the single "All pipeline evals" list so base (raw-model) evals and
    chachak-pipeline evals sit in one changelist and can be Analyzed/compared
    together. It maps a Postgres view (``eval_pipelines_combined_eval``, created
    in the migration) that ``UNION ALL``s the two tables into one shape; base
    rows report ``pipeline='base'``. ``managed=False`` — Django never creates or
    writes this table, and the admin exposes it read-only.

    ``id`` is a text key (``"pe-<pk>"`` / ``"be-<pk>"``) so the two integer pk
    spaces never collide; ``kind`` + ``orig_id`` let actions route a selected row
    back to its real :class:`PipelineEvalRun` / :class:`EvalRun`.
    """

    PIPELINE = "pipeline"
    BASE = "base"

    id = models.CharField(primary_key=True, max_length=32)
    orig_id = models.IntegerField()
    kind = models.CharField(max_length=16)
    trained_model = models.ForeignKey(
        TrainedModel, on_delete=models.DO_NOTHING, db_constraint=False, related_name="+"
    )
    dataset = models.ForeignKey(
        Dataset, on_delete=models.DO_NOTHING, db_constraint=False, related_name="+"
    )
    pipeline = models.CharField(max_length=32)
    status = models.CharField(max_length=16)
    metrics = models.JSONField(null=True, blank=True)
    score_threshold = models.FloatField(null=True)
    output_dir = models.CharField(max_length=1024, blank=True)
    request_yaml_path = models.CharField(max_length=1024, blank=True)
    created_at = models.DateTimeField()
    finished_at = models.DateTimeField(null=True)

    class Meta:
        managed = False
        db_table = "eval_pipelines_combined_eval"
        ordering = ["-created_at"]
        verbose_name = "All pipeline eval"
        verbose_name_plural = "All pipeline evals"

    def __str__(self) -> str:
        return (
            f"{'Pipeline Eval' if self.kind == self.PIPELINE else 'Eval'} #{self.orig_id} — "
            f"{self.trained_model.name} [{self.pipeline}] on {self.dataset.name}"
        )

    def metric(self, key: str):
        return self.metrics.get(key) if isinstance(self.metrics, dict) else None


def _pipeline_manager(pipeline_value: str) -> models.Manager:
    """Manager that scopes a proxy to a single ``pipeline`` value.

    Each per-pipeline proxy below shares ``PipelineEvalRun``'s table; the manager
    filters it down so the proxy's admin list shows only that pipeline's runs.
    """

    class _Manager(models.Manager):
        def get_queryset(self):
            return super().get_queryset().filter(pipeline=pipeline_value)

    return _Manager()


# Per-pipeline proxies — one admin list per pipeline type. Each shares
# ``PipelineEvalRun``'s table but its manager scopes the queryset to a single
# pipeline, so a run created by the "Evaluate…" action appears under the list
# matching the pipeline chosen there.
class BatchDetectEval(PipelineEvalRun):
    objects = _pipeline_manager(PipelineEvalRun.BATCH_DETECT)

    class Meta:
        proxy = True
        verbose_name = "Batch detect eval"
        verbose_name_plural = "Batch detect"


class PeopleDetectFirstEval(PipelineEvalRun):
    objects = _pipeline_manager(PipelineEvalRun.PEOPLE_DETECT_FIRST)

    class Meta:
        proxy = True
        verbose_name = "People detect first eval"
        verbose_name_plural = "People detect first"


class BatchPeopleEval(PipelineEvalRun):
    objects = _pipeline_manager(PipelineEvalRun.BATCH_PEOPLE)

    class Meta:
        proxy = True
        verbose_name = "Batch people eval"
        verbose_name_plural = "Batch people"


class ChainEval(PipelineEvalRun):
    objects = _pipeline_manager(PipelineEvalRun.CHAIN)

    class Meta:
        proxy = True
        verbose_name = "Chain eval"
        verbose_name_plural = "Chain"
