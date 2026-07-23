"""Training & eval configs as first-class models.

These mirror the YAML schema consumed by the ``friendy_chachkalica`` trainer
(``/home/luka/workspace/chachkalica/friendy_chachkalica``): an :class:`Experiment` is one
``ExperimentConfig`` minus the dataset/model lists, which live as the
:class:`ExperimentDataset` and :class:`ExperimentModel` child rows. There is no
dataset *splitting* — friendy_chachkalica assigns whole datasets to roles, so one
Django :class:`~fleet.models.Dataset` maps to exactly one YAML dataset entry and
a run only picks roles + architectures.

This module is the config surface only. Generating the YAML lives in
``training.services.config_gen``; actually *executing* a run against the
friendy_chachkalica service (and ingesting metrics) is a later phase — a
:class:`TrainingRun` here records the generated config and its eventual status.
"""

from django.core.validators import MinValueValidator
from django.db import models

from fleet.models import Annotator, Dataset
from training import pipelines

# Defaults lifted from friendy_chachkalica's configs/config_reference.yaml so the
# generated YAML matches a hand-written one field-for-field.
DEFAULT_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


DEFAULT_PERSON_DETECTOR_CHECKPOINT = "models/people/best_ckpt.engine"


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
        help_text="Base URL of the friendy_chachkalica training service "
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
    """A reusable training+eval configuration (one friendy_chachkalica experiment).

    The dataset roster (train/val/test) and the model architectures to compare
    live as child rows; everything else — training hyperparameters and the
    evaluation settings — are columns here.
    """

    OPTIMIZER_CHOICES = [("adamw", "adamw"), ("adam", "adam"), ("sgd", "sgd")]
    # Values are what the trainer's training.best_metric accepts; a "+" joins
    # metrics whose AVERAGE is tracked (config_gen splits on it).
    BEST_METRIC_CHOICES = [
        ("map50", "map50"),
        ("map50_95", "map50_95"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "f1"),
        ("f1+map50", "f1+map50 (average of both)"),
    ]
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
    gradient_clip_norm = models.FloatField(
        null=True, blank=True, default=1.0,
        help_text="Max gradient L2 norm (clipping). Stabilizes transformer detectors "
                  "(RT-DETR / RF-DETR) against diverging to NaN loss; lower toward 0.1 "
                  "if a model still diverges on epoch 1. Blank = no clipping.",
    )
    early_stopping_patience = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Stop a model's training after this many scored epochs with no "
                  "best-metric improvement (the best checkpoint is kept). Blank = "
                  "train all epochs. Needs a val dataset.",
    )
    best_metric = models.CharField(
        max_length=32, choices=BEST_METRIC_CHOICES, default="map50",
        help_text="Validation metric that selects each run's best checkpoint and "
                  "drives early stopping. precision/recall/f1 are computed at the "
                  "eval score threshold; f1+map50 tracks the average of both.",
    )
    val_interval = models.PositiveIntegerField(
        default=1,
        help_text="Run validation every N epochs (the final epoch always validates). "
                  "1 = every epoch. Early-stopping patience counts scored epochs only, "
                  "so patience 3 with interval 5 spans 15 train epochs.",
    )

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
    eval_score_threshold = models.FloatField(
        default=0.25,
        help_text="Operating-point confidence threshold used for precision, recall, and F1 "
                  "during per-epoch validation. map50/map50_95 are unaffected — they're always "
                  "computed at a fixed low threshold (0.001) so mAP can sweep the full "
                  "precision-recall curve.",
    )
    eval_operating_nms_threshold = models.FloatField(
        verbose_name="operating_nms_iou_threshold",
        null=True, blank=True,
        help_text="Class-aware NMS IoU applied to val/test precision/recall/F1 and the "
                  "confusion matrix only — mAP always stays NMS-free. Dedupes the NMS-free "
                  "DETR models; effectively a no-op for YOLOX (its predictions are already "
                  "NMS'd at 0.45). A model's own NMS threshold overrides this. Blank = off.",
    )
    iou_thresholds = models.JSONField(default=default_iou_thresholds, blank=True)

    # --- train / eval pipeline (chachak) ---
    # When set, the model is trained, validated, and tested through this chachak
    # pipeline — one consistent image representation across all three phases: the
    # train loop applies the same transform (tiling, or person-cropping via the
    # detector) as eval. Blank = plain full-frame training/eval.
    pipeline = models.CharField(
        max_length=32, choices=pipelines.PIPELINE_CHOICES, blank=True,
        help_text="Run train, val, and test through this chachak pipeline. Training "
                  "applies the same transform as eval (tile the frame, or crop around "
                  "detected people). Blank = plain full-frame training and eval.",
    )
    detector_checkpoint = models.CharField(
        max_length=1024, blank=True, default=DEFAULT_PERSON_DETECTOR_CHECKPOINT,
        help_text="Person-detector checkpoint (.pt, .onnx, or .engine); required for people_detect_first / "
                  "batch_people (and any chain that includes them). Runs at train, "
                  "val, and test: it selects the person crops the model is trained "
                  "and evaluated on. Objects outside every detected person crop are "
                  "not seen during training. TensorRT person engines automatically resize "
                  "oversized frames while preserving aspect ratio; the default engine has a "
                  "1280-pixel maximum side.",
    )
    detector_expand_ratio = models.FloatField(
        default=0.10,
        validators=[MinValueValidator(0.0)],
        verbose_name="Person-box expand ratio",
        help_text="Grow each detected person box by this fraction before cropping, to "
                  "avoid clipping objects that sit at or just past the person's edge "
                  "(e.g. a hard hat above the head). 0.10 = +10% on width and height "
                  "(5% per side); 0 crops the detector box exactly. Applied identically "
                  "at train, val, and test. Larger values pull in more context but risk "
                  "dragging neighbouring people/clutter back into the crop. Only used by "
                  "people_detect_first / batch_people.",
    )
    detector_min_box_size = models.FloatField(
        default=224.0,
        validators=[MinValueValidator(0.0)],
        verbose_name="Person-crop minimum size (px)",
        help_text="If a detected person's crop (after expand ratio) is narrower or "
                  "shorter than this many pixels, grow it — using real neighboring "
                  "frame pixels first, falling back to zero-padding only if the frame "
                  "itself is smaller than the floor — rather than dropping the person "
                  "outright; small/distant CCTV subjects must still reach the model. "
                  "0 disables the floor. Only applied for people_detect_first — "
                  "batch_people already starts from fixed-size tiles so its person crops "
                  "don't shrink to the same degenerate sizes. Introduced after run "
                  "PPE_v0.4_ppl_first-29's "
                  "RT-DETR run crashed on epoch 1 batch 1 with 'selected index k out of "
                  "range': RT-DETR's encoder does topk(num_queries) over its last feature "
                  "map, which has (padded_size / 32)^2 tokens, so a crop must be large "
                  "enough to out-token whatever num_queries the model uses. The default "
                  "224 gives >= (224/32)^2 = 49 tokens (a conservative square-crop floor; "
                  "real person crops are usually taller than wide, so the true count is "
                  "typically higher) — comfortably above the 25-query default this app "
                  "now injects for rtdetr models on this pipeline (see config_gen.model_entry). "
                  "Raise both together if you deliberately increase num_queries in a "
                  "model's params.",
    )
    tile_size_px = models.PositiveIntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1)],
        verbose_name="Tile size (pixels)",
        help_text=(
            "Fixed square source-image tile size used by batch_detect. Set this to the "
            "'trained @' resolution shown beside the selected pretrained weights when "
            "you want to preserve that model's native pixel scale. For example, 560 "
            "produces 560×560 tiles. When set, this overrides Tile width % and Tile "
            "height %. Full-size windows are shifted flush to the right/bottom edge "
            "where possible; only an image smaller than this size is zero-padded on "
            "the right/bottom, with no stretching or upscaling. The saved value is used "
            "consistently for training, validation, and pipeline inference. For RF-DETR, "
            "keep the model Input resolution equal to this value. If an experiment "
            "contains models with different native resolutions, choose one common tile "
            "size deliberately or use separate experiments. Blank keeps the legacy "
            "percentage-based tiling behavior."
        ),
    )
    tile_width_pct = models.FloatField(
        null=True, blank=True,
        help_text="Tile width as a percent (0–100] of each image's width. Blank = "
                  "chachak's default.",
    )
    tile_height_pct = models.FloatField(
        null=True, blank=True,
        help_text="Tile height as a percent (0–100] of each image's height. Blank = "
                  "chachak's default.",
    )
    overlap = models.FloatField(
        null=True, blank=True,
        help_text="Fraction (0–1) by which adjacent tiles overlap. Blank = default.",
    )
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

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class ExperimentDataset(models.Model):
    """One dataset assigned a role in an experiment (a YAML dataset entry).

    ``train`` may repeat (it is friendy_chachkalica's comparison axis); ``val`` and
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
    NONE = "none"            # prediction-only run; no ground truth
    LABEL_SOURCE_CHOICES = [
        (SOURCE, "source labels"),
        (ANNOTATOR, "annotator output"),
        (EXPLICIT, "explicit path"),
        (NONE, "no label source (prediction only)"),
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

    # --- train-time augmentations (train role only) ---
    # Each checkbox enables one augmentation; its fraction is how much of the
    # dataset gets that augmentation per epoch (0-1). Emitted into the YAML as
    # the dataset's `augmentation` block, consumed by friendy_chachkalica.
    aug_hflip = models.BooleanField(
        default=False, verbose_name="hflip",
        help_text="Randomly mirror images (and their boxes) horizontally during "
                  "training. Applied on the fly, replacing the original for that "
                  "epoch — the dataset does not get bigger. Train datasets only.",
    )
    aug_hflip_fraction = models.FloatField(
        default=0.5, verbose_name="hflip fraction",
        help_text="Chance (0-1) each image is flipped in a given epoch; 0.5 means "
                  "about half the images, a different random half every epoch.",
    )
    aug_scale_crop = models.BooleanField(
        default=False, verbose_name="scale+crop",
        help_text="Randomly crop a 60-100% window and scale it back up; boxes are "
                  "clipped/dropped at crop edges. Applied on the fly, replacing the "
                  "original for that epoch — the dataset does not get bigger. "
                  "Train datasets only.",
    )
    aug_scale_crop_fraction = models.FloatField(
        default=0.5, verbose_name="scale+crop fraction",
        help_text="Chance (0-1) each image is scale-cropped in a given epoch; 0.5 "
                  "means about half the images, a different random half every epoch.",
    )

    class Meta:
        ordering = ["id"]  # the order rows were added

    def __str__(self) -> str:
        return f"{self.dataset.name} [{self.role}]"

    def clean(self):
        from django.core.exceptions import ValidationError

        if self.label_source == self.ANNOTATOR and self.annotator_id is None:
            raise ValidationError({"annotator": "Pick an annotator for 'annotator output'."})
        if self.label_source == self.EXPLICIT and not self.explicit_labels_path.strip():
            raise ValidationError({"explicit_labels_path": "Provide a path for 'explicit path'."})

        if self.role != self.TRAIN and (self.aug_hflip or self.aug_scale_crop):
            field = "aug_hflip" if self.aug_hflip else "aug_scale_crop"
            raise ValidationError(
                {field: "Augmentations only apply to train datasets — untick it or "
                        "set the role to train."}
            )
        for enabled, fraction_field in [
            (self.aug_hflip, "aug_hflip_fraction"),
            (self.aug_scale_crop, "aug_scale_crop_fraction"),
        ]:
            fraction = getattr(self, fraction_field)
            if enabled and not (fraction is not None and 0 < fraction <= 1):
                raise ValidationError(
                    {fraction_field: "Enter a fraction between 0 (exclusive) and 1."}
                )


class ExperimentModel(models.Model):
    """One model architecture to train within an experiment (a YAML model entry).

    Every model is paired with every train dataset, so listing several here
    fans out into several friendy_chachkalica runs.
    """

    RETINANET = "retinanet"
    YOLOX = "yolox"
    RTDETR = "rtdetr"
    RFDETR = "rfdetr"
    FASTERRCNN = "fasterrcnn"
    ARCH_CHOICES = [
        (RETINANET, "retinanet"),
        (YOLOX, "yolox"),
        (RTDETR, "rtdetr"),
        (RFDETR, "rfdetr"),
        (FASTERRCNN, "fasterrcnn"),
    ]

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name="models")
    arch = models.CharField(max_length=32, choices=ARCH_CHOICES, default=RETINANET)
    num_classes = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Leave empty for auto — resolved per train dataset from its "
                  "classes.txt. Only set this to override the class count.",
    )
    pretrained = models.BooleanField(
        default=False,
        help_text="Start from the architecture's published (COCO) pretrained weights "
                  "instead of random init — adds weights=true to the generated config. "
                  "An explicit 'weights' in params overrides this.",
    )
    params = models.JSONField(
        default=dict, blank=True,
        help_text="Architecture kwargs, e.g. {\"variant\": \"resnet50_fpn_v2\", "
                  "\"weights_backbone\": \"DEFAULT\"}.",
    )

    class Meta:
        ordering = ["id"]  # the order rows were added

    def __str__(self) -> str:
        return self.arch


class TrainingRun(models.Model):
    """One generated/executed run of an experiment.

    Phase 1 only generates the config and records it. Execution against the
    friendy_chachkalica service (status transitions, results ingest) is a later
    phase; the status/timestamp/results fields are here to receive it.
    """

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    OK = "ok"
    ERROR = "error"
    STATUS_CHOICES = [
        (CREATED, "created"),
        (QUEUED, "queued"),
        (RUNNING, "running"),
        (PAUSED, "paused"),
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
    """One friendy_chachkalica internal run (a train-dataset × model pairing).

    A single :class:`TrainingRun` fans out into several of these — friendy_chachkalica
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

    # The trainer's error string for a model that failed to train (results.yaml
    # entries with an "error" key). Empty for successful runs.
    error = models.TextField(blank=True, default="")

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
    NONE = ExperimentDataset.NONE

    trained_model = models.ForeignKey(
        TrainedModel, on_delete=models.CASCADE, related_name="eval_runs"
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
        return f"Eval #{self.pk} — {self._models_label()} on {self.dataset.name}"

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
