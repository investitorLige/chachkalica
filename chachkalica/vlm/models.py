"""VLM catalogue, video library, and run/alert history.

The detection side of this project trains and exports models; nothing here is
trained. A :class:`VlmModel` row is purely a *configuration* — which pretrained
vision-language model to load, and the prompt to ask of every sampled frame.

Six tables, three of which are registered as admin tabs:

* :class:`VlmModel`  — the **Models** tab.
* :class:`VlmVideo`  — the **Videos** tab. Like :class:`videos.models.Video`,
  only metadata lives in the database; the bytes stay on disk, here under
  ``vlm_videos_root()`` so the VLM library is genuinely separate from the
  detection one.
* :class:`VlmRun`    — one "Run VLM Inference…" execution over a video. Not a
  tab: reached from the action's redirect and from the past-runs panel on a
  video's page.
* :class:`VlmAlert`  — one VLM output for one video frame. Every answer is an
  alert, unfiltered.
* :class:`VlmDatasetRun`    — the **Dataset runs** tab: one "Run on a dataset…"
  execution over a labeled (or unlabeled) dataset on disk.
* :class:`VlmDatasetResult` — one answer for one dataset image, plus how it
  scored against that image's label file.

Both run tables snapshot the model configuration they ran with rather than
reading it back through the FK, so editing or deleting a ``VlmModel`` later
cannot rewrite what a finished run actually did — the same reasoning as
``training.TrainedModel.pipeline_metadata``.
"""

from pathlib import Path

from django.db import models

from fleet.models import Dataset
from fleet.services.paths import vlm_videos_root
from vlm.weights_catalog import BACKEND_CHOICES, FAMILY_CHOICES, TRANSFORMERS


class VlmModel(models.Model):
    """A pretrained vision-language model, configured and ready to run."""

    QUANTIZATION_CHOICES = [
        ("", "none (bf16/fp16)"),
        ("8bit", "8-bit"),
        ("4bit", "4-bit"),
    ]

    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)

    backend = models.CharField(
        max_length=16, choices=BACKEND_CHOICES, default=TRANSFORMERS,
        help_text="Where the model runs. Ollama is not wired up yet.",
    )
    family = models.CharField(
        max_length=32, choices=FAMILY_CHOICES,
        help_text="Which adapter loads it — picks the weights offered below.",
    )
    weights = models.CharField(
        max_length=512,
        help_text="Hugging Face repo id (transformers) or model tag (ollama). "
                  "huggingface.co is unreachable from this network: the repo's "
                  "snapshot must already be present in data/hf_cache/hub.",
    )
    variant = models.CharField(
        max_length=32, blank=True, help_text="e.g. 3B, 7B — display only.",
    )

    prompt = models.TextField(
        help_text="Sent with every sampled frame. Every response becomes an alert, "
                  "so a prompt that always answers will produce an alert per frame.",
    )
    max_new_tokens = models.PositiveIntegerField(
        default=128, help_text="Caps the answer length — and so the per-frame latency.",
    )
    quantization = models.CharField(
        max_length=16, choices=QUANTIZATION_CHOICES, blank=True, default="",
        help_text="transformers backend only; ignored by ollama.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "name"]
        verbose_name = "VLM model"
        verbose_name_plural = "Models"

    def __str__(self) -> str:
        return self.name

    def config(self) -> dict:
        """The payload the backend needs to build this model's adapter.

        Kept as one method so ``VlmRun.model_config_snapshot`` and the live
        ``/infer`` calls can never drift apart.
        """
        return {
            "backend": self.backend,
            "family": self.family,
            "weights": self.weights,
            "quantization": self.quantization,
            "max_new_tokens": self.max_new_tokens,
        }


class VlmVideo(models.Model):
    """A video file under the VLM videos root.

    Same shape as :class:`videos.models.Video` — metadata in the database, bytes
    on disk — with three ways to create a row: upload one through the browser,
    import a file already in the folder, or paste a link a worker downloads.
    The probed geometry fields are filled once the file is actually present;
    ``source_fps`` is what turns a requested frames-per-second into a stride.
    """

    PENDING = "pending"
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"
    STATUS_CHOICES = [
        (PENDING, "pending"),
        (DOWNLOADING, "downloading"),
        (READY, "ready"),
        (ERROR, "error"),
    ]

    name = models.CharField(
        max_length=255, unique=True,
        help_text="Display name (defaults to the file name without extension).",
    )
    filename = models.CharField(
        max_length=512, blank=True,
        help_text="File name under the VLM videos root.",
    )
    source_url = models.URLField(
        max_length=1024, blank=True,
        help_text="Link the video was downloaded from (empty for uploads/imports).",
    )
    quality = models.CharField(
        max_length=16, blank=True,
        help_text="Requested max height for downloads (e.g. 1080), or blank for best.",
    )
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=READY)
    last_error = models.TextField(blank=True)

    duration_seconds = models.FloatField(null=True, blank=True)
    source_fps = models.FloatField(null=True, blank=True)
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "name"]
        verbose_name = "VLM video"
        verbose_name_plural = "Videos"

    def __str__(self) -> str:
        return self.name

    def path(self) -> Path:
        """Absolute path to the video file on disk."""
        return vlm_videos_root() / self.filename

    def exists(self) -> bool:
        return bool(self.filename) and self.path().is_file()


class VlmRun(models.Model):
    """One "Run VLM Inference…" execution over a video.

    Everything needed to read the run back is snapshotted here, so the page can
    be reopened long after the ``VlmModel`` it used was edited or deleted.
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

    #: Statuses after which nothing more will be written — the live page stops
    #: polling and renders straight from the database.
    TERMINAL_STATUSES = frozenset({OK, ERROR, CANCELLED})

    video = models.ForeignKey(VlmVideo, on_delete=models.CASCADE, related_name="runs")
    vlm_model = models.ForeignKey(
        VlmModel, on_delete=models.SET_NULL, null=True, blank=True, related_name="runs",
        help_text="For admin linking/filtering only — the snapshots below are authoritative.",
    )

    model_config_snapshot = models.JSONField(default=dict, blank=True)
    prompt_snapshot = models.TextField(blank=True)
    model_label_snapshot = models.CharField(max_length=255, blank=True)

    fps = models.FloatField(help_text="Requested sampling rate, in frames per second.")
    frame_stride = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Resolved from the video's own fps once the run starts.",
    )

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=QUEUED)
    last_error = models.TextField(blank=True)

    frames_total = models.PositiveIntegerField(
        null=True, blank=True, help_text="Estimated number of frames that will be sampled.",
    )
    frames_processed = models.PositiveIntegerField(default=0)
    # Denormalized so list views and the poll endpoint never COUNT() the alerts.
    alerts_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "VLM run"
        verbose_name_plural = "VLM runs"

    def __str__(self) -> str:
        return f"{self.video.name} — {self.model_label_snapshot or 'VLM'} @ {self.fps} fps"

    def is_terminal(self) -> bool:
        return self.status in self.TERMINAL_STATUSES


class VlmAlert(models.Model):
    """One thing the VLM said about one sampled frame.

    ``seq`` is the cursor the live page pages on, not ``pk`` — it stays
    meaningful per run and starts at 1, so the client can seed it from what was
    server-rendered and never replay history after a refresh.
    """

    run = models.ForeignKey(VlmRun, on_delete=models.CASCADE, related_name="alerts")
    seq = models.PositiveIntegerField(help_text="Monotonic within the run, from 1.")
    frame_index = models.PositiveIntegerField(
        help_text="Index into the decoded frame stream, not the sampled one.",
    )
    video_timestamp_seconds = models.FloatField(
        help_text="Where in the video this frame came from.",
    )
    text = models.TextField(help_text="Raw model output — unfiltered.")
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    # Wall-clock write time; the gap against video_timestamp_seconds is the lag.
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["run", "seq"]
        constraints = [
            models.UniqueConstraint(fields=["run", "seq"], name="vlm_alert_run_seq_unique"),
        ]
        indexes = [models.Index(fields=["run", "seq"])]
        verbose_name = "VLM alert"
        verbose_name_plural = "VLM alerts"

    def __str__(self) -> str:
        return f"#{self.seq} @ {self.video_timestamp_seconds:.1f}s"


class VlmDatasetRun(models.Model):
    """One "Run on a dataset…" execution: a VLM asked about every image of a dataset.

    The sibling of :class:`VlmRun`, over a folder instead of a video, and it
    differs from it in three ways that are deliberate rather than incidental:

    * **It is a tab.** A video run hangs off its video's page; a dataset run has
      no such home, and comparing two prompts over the same dataset is the whole
      point of having one, so the runs need a list of their own.
    * **Nobody has to watch it.** A video run pauses when its live page is
      closed, because a VLM frame costs GPU time and a closed tab is watching
      nothing. A dataset run is a measurement you start and come back to, so it
      keeps going; the Stop button is the only way to end it early.
    * **It can be scored.** When ``grade`` is set, every image's answer is
      compared against its YOLO label file and the run carries the resulting
      metrics — see :mod:`vlm.services.grading`.

    ``classes_snapshot`` freezes the ``classes.txt`` the run was scored against,
    and ``alias_map``/``negation_aware`` freeze the grading configuration. All
    three are re-readable and re-writable afterwards: grading is a pure function
    over the stored answers, so a run can be re-graded with better aliases
    without paying for the GPU a second time.
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

    #: Statuses after which nothing more will be written — the page stops
    #: polling and renders straight from the database.
    TERMINAL_STATUSES = frozenset({OK, ERROR, CANCELLED})

    dataset = models.ForeignKey(
        Dataset, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="vlm_dataset_runs",
        help_text="For admin linking only — the snapshots below are authoritative.",
    )
    dataset_name_snapshot = models.CharField(max_length=255, blank=True)

    vlm_model = models.ForeignKey(
        VlmModel, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="dataset_runs",
        help_text="For admin linking/filtering only — the snapshots below are authoritative.",
    )
    model_config_snapshot = models.JSONField(default=dict, blank=True)
    prompt_snapshot = models.TextField(blank=True)
    model_label_snapshot = models.CharField(max_length=255, blank=True)

    image_limit = models.PositiveIntegerField(
        default=0,
        help_text="Cap on images to run (0 = all). The subset is taken at an even "
                  "stride so it spans the whole dataset rather than its first N files.",
    )

    grade = models.BooleanField(
        default=False,
        help_text="Compare every answer against the image's YOLO label file and "
                  "report accuracy. Needs the dataset's labels/ folder.",
    )
    negation_aware = models.BooleanField(
        default=True,
        help_text="Read a negated mention (“no helmet”) as absent rather than "
                  "present. Without this, question-shaped prompts score inverted.",
    )
    classes_snapshot = models.JSONField(
        default=list, blank=True,
        help_text="The classes.txt this run was scored against, in file order.",
    )
    alias_map = models.JSONField(
        default=dict, blank=True,
        help_text="Extra terms per class, {class: [alias, …]}. The class's own "
                  "name is always matched.",
    )
    metrics = models.JSONField(
        default=dict, blank=True,
        help_text="Scores from the last grading pass (vlm.services.grading.metrics).",
    )

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=QUEUED)
    last_error = models.TextField(blank=True)

    images_total = models.PositiveIntegerField(
        null=True, blank=True, help_text="How many images this run will ask about.",
    )
    images_processed = models.PositiveIntegerField(default=0)
    # A single unreadable image or one backend hiccup must not void a long run,
    # so per-image failures are recorded and counted rather than raised.
    errors_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "dataset run"
        verbose_name_plural = "Dataset runs"

    def __str__(self) -> str:
        return (f"{self.dataset_name_snapshot or 'dataset'} — "
                f"{self.model_label_snapshot or 'VLM'}")

    def is_terminal(self) -> bool:
        return self.status in self.TERMINAL_STATUSES

    def accuracy(self):
        """Exact-match accuracy from the last grading pass, or None."""
        return (self.metrics or {}).get("exact_match_accuracy")


class VlmDatasetResult(models.Model):
    """One image's answer, and how it scored.

    The raw ``text`` is the part that cost GPU time; ``gt_classes``,
    ``predicted_classes`` and ``matched`` are the cheap, derived part, rewritten
    in place by every re-grade. ``has_label`` distinguishes *no label file*
    (unlabeled — left out of the scoring) from *an empty label file* (nothing is
    present — a real negative), which
    :func:`vlm.services.grading.metrics` relies on.

    ``seq`` is the cursor the page pages on, not ``pk`` — same reasoning as
    :class:`VlmAlert`.
    """

    run = models.ForeignKey(
        VlmDatasetRun, on_delete=models.CASCADE, related_name="results",
    )
    seq = models.PositiveIntegerField(help_text="Monotonic within the run, from 1.")
    image_filename = models.CharField(max_length=512)

    text = models.TextField(blank=True, help_text="Raw model output — unfiltered.")
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    error = models.TextField(
        blank=True, help_text="Why this one image produced no answer, if it didn't.",
    )

    has_label = models.BooleanField(
        default=False, help_text="A label file exists for this image (possibly empty).",
    )
    gt_classes = models.JSONField(default=list, blank=True)
    predicted_classes = models.JSONField(default=list, blank=True)
    matched = models.BooleanField(
        null=True, blank=True,
        help_text="Answer named exactly the labeled classes. None when ungraded.",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["run", "seq"]
        constraints = [
            models.UniqueConstraint(
                fields=["run", "seq"], name="vlm_dataset_result_run_seq_unique",
            ),
        ]
        indexes = [models.Index(fields=["run", "seq"])]
        verbose_name = "dataset result"
        verbose_name_plural = "dataset results"

    def __str__(self) -> str:
        return f"#{self.seq} {self.image_filename}"
