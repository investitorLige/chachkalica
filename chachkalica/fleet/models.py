"""Fleet state, moved out of the gitignored YAMLs into the database.

The old tooling kept two YAML stores in lockstep — ``fleet.local.yaml`` (the
*live* fleet) and ``register.yaml`` (the *durable* roster of every annotator
ever added, so a removed one could be restored). Both collapse into a single
``Annotator`` table here: ``status`` distinguishes live (``active``) from
removed-but-restorable (``retired``), and a full purge just deletes the row.

Secrets (``password``/``token``) live as plain columns — the same threat model
as the gitignored YAMLs, on a local/trusted Postgres. They are kept out of the
admin list/detail views.
"""

import re
import secrets

from django.db import models

from training import pipelines

# Project title format, kept byte-for-byte compatible with the old tooling so
# the idempotent "skip if a project with this title exists" check still matches
# projects the CLI created. The separator is an em dash with surrounding spaces.
PROJECT_TITLE_SEP = " — "


def project_title(dataset_name: str, username: str) -> str:
    return f"{dataset_name}{PROJECT_TITLE_SEP}{username}"


class FleetSettings(models.Model):
    """Singleton row holding fleet-wide defaults (was the top of fleet.local.yaml)."""

    base_port = models.PositiveIntegerField(
        default=8081, help_text="First host port; new annotators take the next free one."
    )
    image_name = models.CharField(
        max_length=255, default="heartexlabs/label-studio:latest"
    )
    source_dir = models.CharField(
        max_length=512, default="data/source",
        help_text="Shared read-only source mount (relative to the project root or absolute).",
    )
    target_dir = models.CharField(
        max_length=512, default="data/target",
        help_text="Shared target mount where per-image txts + COCO are written.",
    )
    videos_dir = models.CharField(
        max_length=512, default="data/videos",
        help_text="Directory holding raw video files (imported or downloaded).",
    )
    vlm_videos_dir = models.CharField(
        max_length=512, default="data/vlm_videos",
        help_text="Directory holding VLM video files (uploaded, imported, or downloaded). "
                  "Deliberately separate from videos_dir: the VLM section owns its own "
                  "library, so adding a clip there never disturbs the detection Videos tab.",
    )
    marketing_videos_dir = models.CharField(
        max_length=512, default="data/marketing/videos",
        help_text="Directory holding Marketing Studio's video files. Deliberately "
                  "separate from videos_dir: the studio owns its own library, and "
                  "its own <dir>/inferred output folder, so a render there can "
                  "never collide with an Inferred video from the Videos tab.",
    )
    marketing_bundles_dir = models.CharField(
        max_length=512, default="data/marketing/bundles",
        help_text="Where Marketing Studio looks for infer bundles. Deliberately "
                  "separate from TrainingSettings.bundles_root: the studio ships "
                  "its own curated set. Must live on the mount the trainer "
                  "container sees (anything under data/ is), because a bundle's "
                  "model reaches /predict_image as an absolute path.",
    )
    webhook_url = models.CharField(
        max_length=512, default="http://host.docker.internal:9000",
        help_text="Base URL each container POSTs annotation events to (the /hook receiver).",
    )

    class Meta:
        verbose_name = "Fleet settings"
        verbose_name_plural = "Fleet settings"

    def __str__(self) -> str:
        return "Fleet settings"

    def save(self, *args, **kwargs):
        self.pk = 1  # enforce a single row
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):  # never delete the singleton
        pass

    @classmethod
    def load(cls) -> "FleetSettings":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class Annotator(models.Model):
    """One annotator == one isolated Label Studio container.

    Fields mirror the old per-annotator record. Identity-derived fields
    (container/volume/email) and secrets are auto-filled on first save, and the
    port is reserved from the first gap at/above ``FleetSettings.base_port`` —
    reserving across *all* rows (active and retired) so re-provisioning a
    retired annotator reclaims its original port.
    """

    ACTIVE = "active"
    RETIRED = "retired"
    STATUS_CHOICES = [(ACTIVE, "active"), (RETIRED, "retired")]

    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(blank=True)
    password = models.CharField(max_length=255, blank=True)  # secret
    token = models.CharField(max_length=255, blank=True)     # secret
    port = models.PositiveIntegerField(unique=True, null=True, blank=True)
    container_name = models.CharField(max_length=255, unique=True, blank=True)
    volume_name = models.CharField(max_length=255, unique=True, blank=True)
    ls_url = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=ACTIVE)

    # Outcome of the last enqueued job (provisioning/removal), for admin display.
    last_action = models.CharField(max_length=64, blank=True)
    last_status = models.CharField(max_length=32, blank=True)
    last_error = models.TextField(blank=True)
    last_run_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["username"]

    def __str__(self) -> str:
        return self.username

    @classmethod
    def reserve_port(cls) -> int:
        """First free port at/above base_port, across active AND retired rows."""
        base = FleetSettings.load().base_port
        used = set(cls.objects.exclude(port__isnull=True).values_list("port", flat=True))
        port = base
        while port in used:
            port += 1
        return port

    def save(self, *args, **kwargs):
        if not self.email:
            self.email = f"{self.username}@labelers.local"
        if not self.password:
            self.password = secrets.token_urlsafe(16)
        if not self.token:
            self.token = secrets.token_hex(20)
        if not self.container_name:
            self.container_name = f"label-studio-{self.username}"
        if not self.volume_name:
            self.volume_name = f"label-studio-{self.username}-data"
        if self.port is None:
            self.port = self.reserve_port()
        if not self.ls_url:
            self.ls_url = f"http://localhost:{self.port}"
        super().save(*args, **kwargs)


class Dataset(models.Model):
    """A labelable dataset. Source/target stay on disk or in a bucket — only
    the name (and cloud bucket root) live here, never image or label data.

    Paths are derived: source is ``<source_dir>/<name>`` and per-annotator
    output is ``<target_dir>/<name>/<username>/``. The labels and labeling
    tools come from the on-disk ``classes.txt``, not the database.

    ``name`` is deliberately *not* unique at the database level: two rows may
    point at the same on-disk directory as long as they're never visible on
    the same admin front at once (main admin plus at most one row per section
    — enforced in ``DatasetAdminForm``, not here) — that's what lets the same
    already-imported dataset get its own independent row per project admin.
    """

    LOCAL = "local"
    CLOUD = "cloud"
    STORAGE_CHOICES = [(LOCAL, "local"), (CLOUD, "cloud")]

    name = models.CharField(
        max_length=255,
        help_text="Directory name under the source root (e.g. dataset1).",
    )
    storage_type = models.CharField(max_length=16, choices=STORAGE_CHOICES, default=LOCAL)
    storage_root = models.CharField(
        max_length=1024, blank=True, help_text="Cloud bucket root (required when storage is cloud)."
    )
    has_labels = models.BooleanField(
        default=False,
        help_text="A labels/ folder was detected in the source dir; its contents are "
                  "imported as predictions when projects are created.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Project(models.Model):
    """A Label Studio project: one dataset set up for one annotator.

    Persisting ``ls_project_id`` is the improvement over the old tooling, which
    re-looked-up projects by the ``"<dataset> — <username>"`` title on every
    sync. The title is kept for display and as an import-time fallback.
    """

    annotator = models.ForeignKey(Annotator, on_delete=models.CASCADE, related_name="projects")
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="projects")
    ls_project_id = models.PositiveIntegerField(null=True, blank=True)
    webhook_id = models.PositiveIntegerField(null=True, blank=True)
    title = models.CharField(max_length=512, blank=True)

    last_status = models.CharField(max_length=32, blank=True)
    last_error = models.TextField(blank=True)
    last_run_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("annotator", "dataset")]
        ordering = ["dataset__name", "annotator__username"]

    def __str__(self) -> str:
        return self.title or project_title(self.dataset.name, self.annotator.username)

    def save(self, *args, **kwargs):
        if not self.title and self.dataset_id and self.annotator_id:
            self.title = project_title(self.dataset.name, self.annotator.username)
        super().save(*args, **kwargs)


class AnnotationTag(models.Model):
    """One extra thing an annotator records beyond the geometry.

    Boxes and polygons answer *where* and *what class*; a tag answers everything
    else — is this frame indoors, is this box occluded, how far away is it. Two
    scopes, and the difference is how often the question is asked: a
    ``frame``-wide tag is answered once for the whole image, a ``region``
    (box-wide) tag once for every region drawn in it.

    Tags hang off the ``Dataset`` rather than off a ``Project`` because they
    describe the labeling job, not one annotator's copy of it: every annotator
    set up on the dataset must be answering the same questions or their output
    can't be compared. They are compiled into the project's labeling interface
    at setup time (``lsapi.build_label_config``); editing them afterwards needs
    an explicit push to the projects that already exist.

    A tag's ``name`` is not just a key: Label Studio labels a ``perRegion``
    control in the region details panel with its ``name``, so it is also what a
    box-wide tag's annotator reads on screen.

    Nothing here reaches the YOLO ``.txt``/COCO export — that format has no
    place to put an attribute — so tag values live in Label Studio and come out
    through its own JSON export.
    """

    FRAME = "frame"
    REGION = "region"
    SCOPE_CHOICES = [
        (FRAME, "frame-wide — asked once per image"),
        (REGION, "box-wide — asked for every region"),
    ]

    CHECKBOX = "checkbox"
    RADIO = "radio"
    DROPDOWN = "dropdown"
    MULTISELECT = "multiselect"
    RATING = "rating"
    TEXT = "text"
    WIDGET_CHOICES = [
        (CHECKBOX, "checkboxes (pick any)"),
        (RADIO, "radio buttons (pick one)"),
        (DROPDOWN, "dropdown (pick one)"),
        (MULTISELECT, "multi-select dropdown (pick any)"),
        (RATING, "star rating"),
        (TEXT, "free text"),
    ]

    #: Widget name regex — a Label Studio control name, and by extension the
    #: JSON key its answers arrive under.
    NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*$"

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="tags")
    scope = models.CharField(max_length=16, choices=SCOPE_CHOICES, default=FRAME)
    name = models.CharField(
        max_length=64,
        help_text="Control name, e.g. occlusion_level. Box-wide tags show this "
                  "name to the annotator in the region panel.",
    )
    widget = models.CharField(max_length=16, choices=WIDGET_CHOICES, default=RADIO)
    choices = models.JSONField(
        default=list, blank=True,
        help_text="The options, for the checkbox/radio/dropdown widgets.",
    )
    max_rating = models.PositiveSmallIntegerField(
        default=5, help_text="Number of stars, for the rating widget.",
    )
    required = models.BooleanField(
        default=False,
        help_text="Label Studio refuses to submit until this one is answered.",
    )
    order = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["dataset", "scope", "order", "pk"]
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "name"], name="fleet_annotationtag_dataset_name_unique",
            ),
        ]
        verbose_name = "annotation tag"
        verbose_name_plural = "annotation tags"

    def __str__(self) -> str:
        return f"{self.name} ({self.get_scope_display().split(' —')[0]})"

    def clean(self):
        # Imported here rather than at module scope: the service layer imports
        # models, so a top-level import back the other way would be a cycle.
        from django.core.exceptions import ValidationError
        from fleet.services import lsapi

        errors = {}
        name = (self.name or "").strip()
        if not re.match(self.NAME_PATTERN, name):
            errors["name"] = (
                "Start with a letter; letters, digits, '_' and '-' only (no spaces)."
            )
        elif name in lsapi.RESERVED_CONTROL_NAMES:
            errors["name"] = (
                f"{name!r} is already the name of a drawing control in the generated "
                "config — Label Studio would merge the two. Pick another name."
            )
        if self.widget in lsapi.CHOICE_WIDGETS and not self.cleaned_choices():
            errors["choices"] = f"A {self.get_widget_display()} tag needs at least one option."
        if self.widget == self.RATING and not 1 <= (self.max_rating or 0) <= 10:
            errors["max_rating"] = "Between 1 and 10 stars."
        if errors:
            raise ValidationError(errors)

    def cleaned_choices(self) -> list[str]:
        """The choice list with blanks dropped and duplicates collapsed, in order."""
        out: list[str] = []
        for raw in self.choices or []:
            value = str(raw).strip()
            if value and value not in out:
                out.append(value)
        return out

    def to_spec(self) -> dict:
        """The ORM-free dict the label-config renderer consumes."""
        return {
            "scope": self.scope,
            "name": (self.name or "").strip(),
            "widget": self.widget,
            "choices": self.cleaned_choices(),
            "max_rating": self.max_rating,
            "required": self.required,
        }


class GroundingSamRun(models.Model):
    """One "Generate labels with Grounding SAM" job against one dataset.

    Unlike Annotator/Project (a handful of status columns on a domain row),
    this exists purely to track an in-flight job's progress — the worker
    updates ``images_processed``/``images_labeled``/``detections_written`` as
    it works through the dataset's batches, so the admin list can be refreshed
    to watch a run move along instead of only learning queued/ok/error.
    """

    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    STATUS_CHOICES = [(QUEUED, "queued"), (RUNNING, "running"), (OK, "ok"), (ERROR, "error")]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="grounding_sam_runs")
    confidence = models.FloatField()
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=QUEUED)
    error = models.TextField(blank=True)

    images_total = models.PositiveIntegerField(default=0)
    images_processed = models.PositiveIntegerField(default=0)
    images_labeled = models.PositiveIntegerField(default=0)
    detections_written = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.dataset.name} — {self.status}"


class DatasetInferenceRun(models.Model):
    """One "Run model inference…" over a dataset: a model applied image by image
    to a dataset's frames, timed.

    The dataset-shaped sibling of :class:`videos.models.InferenceJob`, and it
    exists because the two answer different questions. A video run answers *what
    does this model see?* and its product is a watchable file. This one answers
    *how fast is this model, on real frames, through the pipeline it will be
    served with?* — so its product is the timing distribution, and the boxes it
    stores per image are what makes that number believable rather than the point
    of the run.

    Any of the three model formats can run here, which is the other reason it
    exists: the trained-models tab's "Preview model on selected dataset…" viewer
    only ever runs a catalogued ``.pt``, while a deployment ships an ``.onnx``, a
    TensorRT ``.engine`` or a whole bundle, and those are the artifacts whose
    speed anyone actually asks about. ``model_source`` picks between them exactly
    as an ``InferenceJob`` does (see
    :data:`training.services.inference_form.MODEL_SOURCE_CHOICES`), and the
    pipeline half of the row is the same field set carried by ``InferenceJob``,
    ``cameras.CameraInference`` and ``eval_pipelines.PipelineEvalRun`` — a model
    configured for a video can be pointed at a dataset unchanged.

    Two things are deliberately *not* here. There is no accuracy: scoring
    predictions against labels is what the Evaluate action and its mAP already
    do, and duplicating it would be a second, worse number for the same
    question. And nothing annotated is written to disk: the report draws the
    stored boxes over the dataset's own images on demand, so a run over ten
    thousand frames costs ten thousand small rows rather than a second copy of
    the dataset.
    """

    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    STATUS_CHOICES = [
        (QUEUED, "queued"),
        (RUNNING, "running"),
        (OK, "ok"),
        (ERROR, "error"),
        (CANCEL_REQUESTED, "cancel requested"),
        (CANCELLED, "cancelled"),
    ]

    #: Statuses after which nothing more will be written — the report stops
    #: polling and renders straight from the database.
    TERMINAL_STATUSES = frozenset({OK, ERROR, CANCELLED})

    # Same three values as InferenceJob/CameraInference declare against their own
    # column; the shared spelling the forms read lives in
    # training.services.inference_form.
    TRAINED = "trained"
    EXPORTED = "exported"
    BUNDLE = "bundle"
    MODEL_SOURCE_CHOICES = [
        (TRAINED, "trained model (.pt checkpoint)"),
        (EXPORTED, "exported artifact (ONNX / TensorRT)"),
        (BUNDLE, "infer bundle (self-contained pipeline directory)"),
    ]

    # "raw" is not a chachak pipeline — it means "no pipeline, feed the model the
    # whole image". Same meaning as InferenceJob.RAW.
    RAW = "raw"
    PIPELINE_CHOICES = [
        (RAW, "raw — whole image straight to the model"),
        *pipelines.PIPELINE_CHOICES,
    ]

    dataset = models.ForeignKey(
        Dataset, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="inference_runs",
        help_text="For admin linking only — the name snapshot below is authoritative.",
    )
    # Snapshotted for the same reason the VLM dataset runs snapshot theirs: a run
    # is a measurement of a directory on disk at a moment, and it stays readable
    # after its Dataset row is deleted or renamed.
    dataset_name_snapshot = models.CharField(max_length=255, blank=True)

    model_source = models.CharField(
        max_length=16, choices=MODEL_SOURCE_CHOICES, default=TRAINED,
        help_text="Where the model comes from: the trained-models catalogue, an "
                  "exported artifact in the export output directory, or a "
                  "self-contained infer bundle under the bundle root.",
    )
    trained_model = models.ForeignKey(
        "training.TrainedModel", null=True, blank=True,
        on_delete=models.SET_NULL, related_name="dataset_inference_runs",
        help_text="Set when model_source is 'trained'.",
    )
    # Kept as text beside the FK so a finished run still says what it measured
    # after the catalogue entry, artifact or bundle behind it is gone.
    model_label_snapshot = models.CharField(max_length=512, blank=True)
    artifact_path = models.CharField(
        max_length=1024, blank=True,
        help_text="Path of the exported .onnx/.engine relative to the export output "
                  "directory. Set when model_source is 'exported'.",
    )
    bundle_path = models.CharField(
        max_length=1024, blank=True,
        help_text="Path of the bundle directory relative to the bundle root. Set "
                  "when model_source is 'bundle'; the bundle's manifest supplies "
                  "the model, the detector and the whole pipeline geometry.",
    )
    score_threshold = models.FloatField(
        default=0.5, help_text="Detections below this confidence are dropped.",
    )

    pipeline = models.CharField(
        max_length=32, choices=PIPELINE_CHOICES, default=RAW,
        help_text="How each image is presented to the model: whole ('raw'), tiled, "
                  "or cropped around detected people.",
    )
    detector_checkpoint = models.CharField(
        max_length=1024, blank=True,
        help_text="Person-detector checkpoint; required for people_detect_first / "
                  "batch_people (and any chain including them).",
    )
    detector_expand_ratio = models.FloatField(
        null=True, blank=True,
        verbose_name="Person-box expand ratio",
        help_text="Grow each detected person box by this fraction before cropping. "
                  "Blank = chachak's default. Match what the model was trained with.",
    )
    detector_min_box_size = models.FloatField(
        null=True, blank=True,
        verbose_name="Person-crop minimum size (px)",
        help_text="Grow person crops smaller than this many pixels rather than "
                  "dropping them. Blank = chachak's default. Used by "
                  "people_detect_first / batch_people.",
    )
    tile_size_px = models.PositiveIntegerField(
        null=True, blank=True,
        verbose_name="Tile size (pixels)",
        help_text="Fixed square tile size for batch_detect; overrides the tile "
                  "width/height percentages. Blank = percentage tiling.",
    )
    tile_width_pct = models.FloatField(
        null=True, blank=True,
        help_text="Tile width as a percent (0–100] of each image's width. "
                  "Blank = chachak's default.",
    )
    tile_height_pct = models.FloatField(
        null=True, blank=True,
        help_text="Tile height as a percent (0–100] of each image's height. "
                  "Blank = chachak's default.",
    )
    overlap = models.FloatField(
        null=True, blank=True,
        help_text="Fraction (0–1) by which adjacent tiles overlap. Blank = default.",
    )
    merge_nms_iou = models.FloatField(
        null=True, blank=True,
        verbose_name="merge_nms_iou_threshold",
        help_text="Class-aware NMS IoU used to merge predictions across tiles/crops. "
                  "Blank = chachak's default. Match what the model was trained with.",
    )
    chain = models.JSONField(
        default=list, blank=True,
        help_text="Ordered pipeline names for the 'chain' pipeline.",
    )

    image_limit = models.PositiveIntegerField(
        default=200,
        help_text="Cap on images to run (0 = all). The subset is taken at an even "
                  "stride so it spans the whole dataset rather than its first N files.",
    )
    warmup = models.PositiveIntegerField(
        default=3,
        help_text="Untimed calls on the first image before the timed loop starts. "
                  "Never sensibly zero: the first call for a model loads it (and "
                  "deserializes a TensorRT plan), which is not a frame time.",
    )

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=QUEUED)
    last_error = models.TextField(blank=True)

    images_total = models.PositiveIntegerField(
        null=True, blank=True, help_text="How many images this run will go through.",
    )
    images_processed = models.PositiveIntegerField(default=0)
    # One unreadable image or one backend hiccup must not void a long run, so
    # per-image failures are recorded on their row and counted here.
    errors_count = models.PositiveIntegerField(default=0)
    detections_total = models.PositiveIntegerField(default=0)

    timings = models.JSONField(
        default=dict, blank=True,
        help_text="What the run measured — see fleet.services.dataset_inference."
                  "summarize: per-image latency percentiles for the model itself "
                  "and for the whole round trip, plus stage shares when the "
                  "trainer reports them.",
    )
    # Denormalized from ``timings`` as the run goes, purely so the changelist is
    # legible without opening a report.
    fps = models.FloatField(
        null=True, blank=True,
        help_text="Images per second the model itself sustained (1000 / mean "
                  "model latency), warmup excluded.",
    )
    p50_ms = models.FloatField(
        null=True, blank=True, help_text="Median model latency per image.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "dataset inference run"
        verbose_name_plural = "Dataset inference runs"

    def __str__(self) -> str:
        return (f"{self.dataset_name_snapshot or 'dataset'} — "
                f"{self.model_label_snapshot or 'model'} [{self.pipeline}]")

    def is_terminal(self) -> bool:
        return self.status in self.TERMINAL_STATUSES

    def model_label(self) -> str:
        """Human name of the model this run measures, whichever source it came from."""
        if self.model_label_snapshot:
            return self.model_label_snapshot
        if self.model_source == self.EXPORTED:
            return self.artifact_path or "(no artifact)"
        if self.model_source == self.BUNDLE:
            return self.bundle_path or "(no bundle)"
        return self.trained_model.name if self.trained_model_id else "(deleted model)"

    def sync_bundle(self) -> dict | None:
        """Re-derive the pipeline fields from this run's bundle, if it has one.

        Bundle-sourced rows own no geometry of their own — the manifest does; see
        :meth:`videos.models.InferenceJob.sync_bundle`, whose contract this
        shares (and whose service does the work for both).
        """
        if self.model_source != self.BUNDLE:
            return None
        from training.services import bundles

        return bundles.apply_defaults(self, self.bundle_path)

    def model_checkpoint(self) -> str:
        """Absolute path of the model artifact to run.

        Same resolution, and the same out-of-root refusals, as
        :meth:`videos.models.InferenceJob.model_checkpoint` — which is what lets
        both rows be fed to ``videos.services.inference.build_predict_payload``.
        Raises ``ValueError`` when the run's model is missing or unusable.
        """
        if self.model_source == self.EXPORTED:
            from training.services import exports

            return str(exports.resolve(self.artifact_path))

        if self.model_source == self.BUNDLE:
            from training.services import bundles

            return str(bundles.model_artifact(self.bundle_path))

        if not self.trained_model_id:
            raise ValueError("Dataset inference run has no trained model.")
        return str(self.trained_model.resolved_checkpoint_path())


class DatasetInferenceResult(models.Model):
    """One image's detections, and what it cost.

    ``boxes`` is the model's own output for the image (normalized centre-xywh
    dicts, exactly as ``/predict_image`` returns them), kept so the report can
    draw them over the source image later without paying for the GPU again —
    the same reason the VLM dataset runs keep each answer's raw text.

    Two latencies, because they are two different facts and only one of them is
    the model's: ``model_ms`` is what the trainer measured around its own
    inference call, ``request_ms`` what the worker measured around the whole HTTP
    round trip. An older trainer service reports no timing of its own, in which
    case ``model_ms`` is null and the report says so rather than quietly passing
    the round trip off as the model's speed.

    ``seq`` is the cursor the report pages on, not ``pk`` — same reasoning as
    :class:`vlm.models.VlmDatasetResult`.
    """

    run = models.ForeignKey(
        DatasetInferenceRun, on_delete=models.CASCADE, related_name="results",
    )
    seq = models.PositiveIntegerField(help_text="Monotonic within the run, from 1.")
    image_filename = models.CharField(max_length=512)

    boxes = models.JSONField(
        default=list, blank=True,
        help_text="The model's detections for this image, as /predict_image returns them.",
    )
    detections = models.PositiveIntegerField(default=0)

    model_ms = models.FloatField(
        null=True, blank=True,
        help_text="Trainer-side inference time for this image (read, preprocess, "
                  "forward, postprocess). Null when the trainer reports no timing.",
    )
    request_ms = models.FloatField(
        null=True, blank=True,
        help_text="Whole round trip measured by the worker, HTTP included.",
    )
    stages = models.JSONField(
        default=dict, blank=True,
        help_text="Trainer-side per-stage split of model_ms, when it reports one.",
    )
    error = models.TextField(
        blank=True, help_text="Why this one image produced no detections, if it didn't.",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["run", "seq"]
        constraints = [
            models.UniqueConstraint(
                fields=["run", "seq"], name="fleet_dataset_inference_result_run_seq_unique",
            ),
        ]
        indexes = [models.Index(fields=["run", "seq"])]
        verbose_name = "dataset inference result"
        verbose_name_plural = "dataset inference results"

    def __str__(self) -> str:
        return f"#{self.seq} {self.image_filename}"
