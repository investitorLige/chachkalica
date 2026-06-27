"""Training & eval configs as first-class models.

These mirror the YAML schema consumed by the ``friendy_mercury`` trainer
(``/home/luka/workspace/friendy_mercury``): an :class:`Experiment` is one
``ExperimentConfig`` minus the dataset/model lists, which live as the
:class:`ExperimentDataset` and :class:`ExperimentModel` child rows. There is no
dataset *splitting* — friendy_mercury assigns whole datasets to roles, so one
Django :class:`~fleet.models.Dataset` maps to exactly one YAML dataset entry and
a run only picks roles + architectures.

This module is the config surface only. Generating the YAML lives in
``training.services.config_gen``; actually *executing* a run against the
friendy_mercury service (and ingesting metrics) is a later phase — a
:class:`TrainingRun` here records the generated config and its eventual status.
"""

from django.db import models

from fleet.models import Annotator, Dataset

# Defaults lifted from friendy_mercury's configs/config_reference.yaml so the
# generated YAML matches a hand-written one field-for-field.
DEFAULT_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def default_iou_thresholds() -> list[float]:
    return list(DEFAULT_IOU_THRESHOLDS)


class TrainingSettings(models.Model):
    """Singleton row holding training-wide defaults (mirrors FleetSettings)."""

    configs_root = models.CharField(
        max_length=512, default="data/training/configs",
        help_text="Where generated experiment YAMLs are written "
                  "(relative to the project root or absolute).",
    )
    runs_root = models.CharField(
        max_length=512, default="data/training/runs",
        help_text="Shared output root; each run's output_dir is runs_root/<name>-<id>.",
    )
    default_device = models.CharField(
        max_length=16,
        choices=[("auto", "auto"), ("cuda", "cuda"), ("cpu", "cpu")],
        default="auto",
    )
    service_base_url = models.CharField(
        max_length=512, default="http://localhost:8200",
        help_text="Base URL of the friendy_mercury training service "
                  "(unused until the execution phase).",
    )

    class Meta:
        verbose_name = "Training settings"
        verbose_name_plural = "Training settings"

    def __str__(self) -> str:
        return "Training settings"

    def save(self, *args, **kwargs):
        self.pk = 1  # enforce a single row
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):  # never delete the singleton
        pass

    @classmethod
    def load(cls) -> "TrainingSettings":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class Experiment(models.Model):
    """A reusable training+eval configuration (one friendy_mercury experiment).

    The dataset roster (train/val/test) and the model architectures to compare
    live as child rows; everything else — training hyperparameters and the
    evaluation settings — are columns here.
    """

    OPTIMIZER_CHOICES = [("adamw", "adamw"), ("adam", "adam"), ("sgd", "sgd")]
    SCHEDULER_CHOICES = [
        ("none", "none"),
        ("step", "step"),
        ("multistep", "multistep"),
        ("cosine", "cosine"),
        ("exponential", "exponential"),
    ]
    DEVICE_CHOICES = [("auto", "auto"), ("cuda", "cuda"), ("cpu", "cpu")]

    name = models.CharField(
        max_length=128, unique=True,
        help_text="Experiment name; also the YAML 'name' and output dir prefix.",
    )
    description = models.TextField(blank=True)

    # --- training ---
    epochs = models.PositiveIntegerField(default=100)
    batch_size = models.PositiveIntegerField(default=4)
    num_workers = models.PositiveIntegerField(default=4)
    device = models.CharField(max_length=16, choices=DEVICE_CHOICES, default="auto")
    seed = models.IntegerField(default=42)
    amp = models.BooleanField(default=False, help_text="Automatic mixed precision.")
    gradient_clip_norm = models.FloatField(null=True, blank=True)

    optimizer_name = models.CharField(max_length=16, choices=OPTIMIZER_CHOICES, default="adamw")
    lr = models.FloatField(default=0.0001)
    weight_decay = models.FloatField(default=0.0001)
    optimizer_params = models.JSONField(
        default=dict, blank=True,
        help_text="Extra kwargs passed to the torch optimizer (e.g. momentum for sgd).",
    )

    scheduler_name = models.CharField(max_length=16, choices=SCHEDULER_CHOICES, default="none")
    scheduler_params = models.JSONField(
        default=dict, blank=True,
        help_text="Scheduler kwargs, e.g. {\"step_size\": 30, \"gamma\": 0.1}.",
    )

    # --- evaluation ---
    eval_batch_size = models.PositiveIntegerField(default=4)
    eval_num_workers = models.PositiveIntegerField(default=4)
    eval_score_threshold = models.FloatField(default=0.001)
    iou_thresholds = models.JSONField(default=default_iou_thresholds, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class ExperimentDataset(models.Model):
    """One dataset assigned a role in an experiment (a YAML dataset entry).

    ``train`` may repeat (it is friendy_mercury's comparison axis); ``val`` and
    ``test`` are at most one each. ``label_source`` picks which on-disk labels
    folder feeds this dataset for this run.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ROLE_CHOICES = [(TRAIN, "train"), (VAL, "val"), (TEST, "test")]

    SOURCE = "source"        # data/source/<name>/labels
    ANNOTATOR = "annotator"  # data/target/<name>/<annotator>
    EXPLICIT = "explicit"    # an arbitrary absolute path
    LABEL_SOURCE_CHOICES = [
        (SOURCE, "source labels"),
        (ANNOTATOR, "annotator output"),
        (EXPLICIT, "explicit path"),
    ]

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name="datasets")
    dataset = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name="+")
    role = models.CharField(max_length=8, choices=ROLE_CHOICES, default=TRAIN)
    label_source = models.CharField(max_length=16, choices=LABEL_SOURCE_CHOICES, default=SOURCE)
    annotator = models.ForeignKey(
        Annotator, on_delete=models.PROTECT, null=True, blank=True, related_name="+",
        help_text="Required when label source is 'annotator output'.",
    )
    explicit_labels_path = models.CharField(
        max_length=1024, blank=True,
        help_text="Required when label source is 'explicit path'.",
    )
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order", "id"]

    def __str__(self) -> str:
        return f"{self.dataset.name} [{self.role}]"

    def clean(self):
        from django.core.exceptions import ValidationError

        if self.label_source == self.ANNOTATOR and self.annotator_id is None:
            raise ValidationError({"annotator": "Pick an annotator for 'annotator output'."})
        if self.label_source == self.EXPLICIT and not self.explicit_labels_path.strip():
            raise ValidationError({"explicit_labels_path": "Provide a path for 'explicit path'."})


class ExperimentModel(models.Model):
    """One model architecture to train within an experiment (a YAML model entry).

    Every model is paired with every train dataset, so listing several here
    fans out into several friendy_mercury runs.
    """

    RETINANET = "retinanet"
    YOLOX = "yolox"
    RTDETR = "rtdetr"
    ARCH_CHOICES = [(RETINANET, "retinanet"), (YOLOX, "yolox"), (RTDETR, "rtdetr")]

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name="models")
    arch = models.CharField(max_length=32, choices=ARCH_CHOICES, default=RETINANET)
    num_classes = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Leave blank for 'auto' (resolved per train dataset).",
    )
    params = models.JSONField(
        default=dict, blank=True,
        help_text="Architecture kwargs, e.g. {\"variant\": \"resnet50_fpn_v2\", "
                  "\"weights_backbone\": \"DEFAULT\"}.",
    )
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order", "id"]

    def __str__(self) -> str:
        return self.arch


class TrainingRun(models.Model):
    """One generated/executed run of an experiment.

    Phase 1 only generates the config and records it. Execution against the
    friendy_mercury service (status transitions, results ingest) is a later
    phase; the status/timestamp/results fields are here to receive it.
    """

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

    experiment = models.ForeignKey(
        Experiment, on_delete=models.SET_NULL, null=True, blank=True, related_name="runs"
    )
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=CREATED)
    config_yaml_path = models.CharField(max_length=1024, blank=True)
    output_dir = models.CharField(max_length=1024, blank=True)
    last_error = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    results = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        name = self.experiment.name if self.experiment else "(deleted experiment)"
        return f"Run #{self.pk} — {name}"


class RunResult(models.Model):
    """One friendy_mercury internal run (a train-dataset × model pairing).

    A single :class:`TrainingRun` fans out into several of these — friendy_mercury
    pairs every train dataset with every model. Rows are populated by
    ``training.services.ingest`` after a run finishes, by joining the trainer's
    ``results.yaml`` (training) with ``val_results.yaml``/``test_results.yaml``
    (metrics), keyed by the internal ``run_name``.
    """

    run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE, related_name="run_results")
    run_name = models.CharField(max_length=255)
    run_index = models.IntegerField(null=True, blank=True)
    model_arch = models.CharField(max_length=32, blank=True)
    train_dataset_name = models.CharField(max_length=255, blank=True)

    best_epoch = models.IntegerField(null=True, blank=True)
    best_loss = models.FloatField(null=True, blank=True)
    run_dir = models.CharField(max_length=1024, blank=True)
    best_checkpoint = models.CharField(max_length=1024, blank=True)
    last_checkpoint = models.CharField(max_length=1024, blank=True)

    val_metrics = models.JSONField(null=True, blank=True)
    test_metrics = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["run_index", "id"]
        unique_together = [("run", "run_name")]

    def __str__(self) -> str:
        return self.run_name

    @property
    def primary_metrics(self) -> dict | None:
        """Best metrics to surface: test if present, else val."""
        return self.test_metrics or self.val_metrics

    def metric(self, key: str):
        m = self.primary_metrics
        return m.get(key) if isinstance(m, dict) else None


class TrainedModel(models.Model):
    """A catalogued trained model — the 'Models' tab.

    Usually promoted from a :class:`RunResult` (its ``best.pt``), capturing the
    architecture, checkpoint path, class space, and a metrics snapshot so it can
    later be evaluated against new datasets independently of the training run.
    """

    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    STAGE_CHOICES = [
        (DEV, "dev"),
        (STAGING, "staging"),
        (PRODUCTION, "production"),
        (ARCHIVED, "archived"),
    ]

    name = models.CharField(max_length=128, unique=True)
    description = models.TextField(blank=True)
    stage = models.CharField(max_length=16, choices=STAGE_CHOICES, default=DEV)

    arch = models.CharField(max_length=32)
    checkpoint_path = models.CharField(max_length=1024)
    num_classes = models.PositiveIntegerField(null=True, blank=True)
    classes = models.JSONField(
        default=list, blank=True,
        help_text="Class names (ordered) the model predicts; needed to build eval configs.",
    )
    metrics = models.JSONField(
        null=True, blank=True, help_text="Metrics snapshot at promotion time.",
    )

    source_run_result = models.ForeignKey(
        RunResult, on_delete=models.SET_NULL, null=True, blank=True, related_name="trained_models",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class EvalRun(models.Model):
    """One standalone evaluation of a :class:`TrainedModel` against a dataset.

    Mirrors :class:`TrainingRun`: the admin action writes an eval request YAML
    and enqueues a job that drives the trainer service's /eval endpoint, then
    ingests the resulting metrics back here.
    """

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

    # Reuse the label-source vocabulary from ExperimentDataset.
    SOURCE = ExperimentDataset.SOURCE
    ANNOTATOR = ExperimentDataset.ANNOTATOR
    EXPLICIT = ExperimentDataset.EXPLICIT

    trained_model = models.ForeignKey(
        TrainedModel, on_delete=models.CASCADE, related_name="eval_runs"
    )
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

    def __str__(self) -> str:
        return f"Eval #{self.pk} — {self.trained_model.name} on {self.dataset.name}"

    def metric(self, key: str):
        return self.metrics.get(key) if isinstance(self.metrics, dict) else None
