from pathlib import Path

from django.db import models

from fleet.services.paths import videos_root
from training import pipelines


class Video(models.Model):
    """A single video file living under the videos root (``data/videos`` by default).

    Like :class:`fleet.models.Dataset`, only metadata lives in the database — the
    bytes stay on disk at ``<videos_root>/<filename>``. A row is created either by
    importing an existing file already in the folder, or by pasting a link that a
    background job downloads into the folder (``status`` tracks that download).
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
        help_text="File name under the videos root. Filled in by the download job.",
    )
    source_url = models.URLField(
        max_length=1024, blank=True,
        help_text="Link the video was downloaded from (empty for imported files).",
    )
    quality = models.CharField(
        max_length=16, blank=True,
        help_text="Requested max height for downloads (e.g. 1080), or blank for best.",
    )
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=READY)
    last_error = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "name"]

    def __str__(self) -> str:
        return self.name

    def path(self) -> Path:
        """Absolute path to the video file on disk."""
        return videos_root() / self.filename

    def exists(self) -> bool:
        return bool(self.filename) and self.path().is_file()


class FrameExtractionJob(models.Model):
    """One "Extract frames…" run on a video, and the dataset it produced.

    A dedicated row per run (rather than fields on ``Video``) since a video can
    reasonably be re-extracted with different parameters, and ``Video.status``
    already means "download state." Mirrors the training app's per-run models:
    flip to ``running``, do the work, record ``ok``/``error``.
    """

    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    STATUS_CHOICES = [
        (QUEUED, "queued"),
        (RUNNING, "running"),
        (OK, "ok"),
        (ERROR, "error"),
    ]

    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="extraction_jobs")
    dataset_name = models.CharField(
        max_length=255,
        help_text="Directory name the frames + classes.txt are written under (source root).",
    )
    every_n_frames = models.PositiveIntegerField(default=30, help_text="Save every Nth frame.")
    max_frames_per_video = models.PositiveIntegerField(
        null=True, blank=True, help_text="Optional cap on frames saved.",
    )
    image_ext = models.CharField(max_length=8, default=".jpg")
    classes_text = models.TextField(
        blank=True, help_text="One class per line; written verbatim to classes.txt.",
    )
    only_with_people = models.BooleanField(
        default=False,
        help_text="Discard frames with no person detected by the people-detector model.",
    )
    similarity_threshold = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Perceptual-hash Hamming distance (0-64) at or below which a frame is "
                  "dropped as a near-duplicate of the previously kept frame; blank disables dedup.",
    )

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=QUEUED)
    last_error = models.TextField(blank=True)
    frames_extracted = models.PositiveIntegerField(null=True, blank=True)
    frames_dropped_similar = models.PositiveIntegerField(null=True, blank=True)
    frames_dropped_no_person = models.PositiveIntegerField(null=True, blank=True)
    dataset = models.ForeignKey(
        "fleet.Dataset", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="extraction_jobs",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.video.name} → {self.dataset_name}"


class InferenceJob(models.Model):
    """One "Run model inference…" run: a model applied frame-by-frame to a video,
    with detected boxes burned onto an annotated output video.

    Mirrors :class:`FrameExtractionJob` — a dedicated row per run (a video can
    reasonably be re-run against different models/thresholds), status lifecycle
    ``queued -> running -> ok``/``error``. The output video lives under
    ``<videos_root>/inferred/`` (see ``videos.services.inference.output_dir``),
    same "DB stores metadata, bytes stay on disk" convention as ``Video``.

    The model to run comes from one of three places (``model_source``): a
    catalogued :class:`training.TrainedModel` (its ``.pt`` checkpoint), an
    exported ``.onnx`` / ``.engine`` artifact sitting in the export output
    directory, addressed by ``artifact_path`` relative to it (see
    ``training.services.exports``), or an **infer bundle** under the bundle root,
    addressed by ``bundle_path`` (see ``training.services.bundles``). Neither
    exported artifacts nor bundles have a DB row of their own, hence
    ``trained_model`` is nullable.

    The remaining fields describe *how* frames reach the model: ``pipeline`` is
    ``"raw"`` (whole frame straight to the model) or a chachak pipeline that
    tiles the frame and/or crops around detected people, with that pipeline's
    detector/tiling knobs beside it. Mirrors
    :class:`eval_pipelines.models.PipelineEvalRun`'s field set, minus the
    dataset/metrics half.

    For a bundle those knobs are not the operator's to choose: a bundle ships the
    geometry its weights were tuned with, so the admin renders them read-only and
    :meth:`sync_bundle` re-derives them from the manifest on save.
    """

    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    STATUS_CHOICES = [
        (QUEUED, "queued"),
        (RUNNING, "running"),
        (OK, "ok"),
        (ERROR, "error"),
    ]

    TRAINED = "trained"
    EXPORTED = "exported"
    BUNDLE = "bundle"
    MODEL_SOURCE_CHOICES = [
        (TRAINED, "trained model (.pt checkpoint)"),
        (EXPORTED, "exported artifact (ONNX / TensorRT)"),
        (BUNDLE, "infer bundle (self-contained pipeline directory)"),
    ]

    # "raw" is not a chachak pipeline — it means "no pipeline, feed the model the
    # whole frame", which is what this action did before pipelines were offered.
    RAW = "raw"
    PIPELINE_CHOICES = [
        (RAW, "raw — whole frame straight to the model"),
        *pipelines.PIPELINE_CHOICES,
    ]

    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="inference_jobs")
    model_source = models.CharField(
        max_length=16, choices=MODEL_SOURCE_CHOICES, default=TRAINED,
        help_text="Where the model comes from: the trained-models catalogue, an "
                  "exported artifact in the export output directory, or a "
                  "self-contained infer bundle under the bundle root.",
    )
    trained_model = models.ForeignKey(
        "training.TrainedModel", null=True, blank=True,
        on_delete=models.CASCADE, related_name="video_inference_jobs",
        help_text="Set when model_source is 'trained'.",
    )
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
        help_text="How each frame is presented to the model: whole ('raw'), tiled, "
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
                  "dropping them. Blank = chachak's default. Only used by "
                  "people_detect_first.",
    )
    tile_size_px = models.PositiveIntegerField(
        null=True, blank=True,
        verbose_name="Tile size (pixels)",
        help_text="Fixed square tile size for batch_detect; overrides the tile "
                  "width/height percentages. Blank = percentage tiling.",
    )
    tile_width_pct = models.FloatField(
        null=True, blank=True,
        help_text="Tile width as a percent (0–100] of each frame's width. "
                  "Blank = chachak's default.",
    )
    tile_height_pct = models.FloatField(
        null=True, blank=True,
        help_text="Tile height as a percent (0–100] of each frame's height. "
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
    frame_stride = models.PositiveIntegerField(
        default=1,
        help_text="Run inference every Nth frame; frames in between reuse the last "
                  "inferred boxes. Higher = faster, less temporally precise.",
    )

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=QUEUED)
    last_error = models.TextField(blank=True)
    output_filename = models.CharField(
        max_length=512, blank=True,
        help_text="File name under <videos_root>/inferred/ holding the annotated video.",
    )
    frames_total = models.PositiveIntegerField(null=True, blank=True)
    frames_processed = models.PositiveIntegerField(
        null=True, blank=True, help_text="Number of frames actually run through the model.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Inferred video"
        verbose_name_plural = "Inferred videos"

    def __str__(self) -> str:
        return f"{self.video.name} → {self.model_label()} [{self.pipeline}]"

    def model_label(self) -> str:
        """Human name of the model this job runs, whichever source it came from."""
        if self.model_source == self.EXPORTED:
            return self.artifact_path or "(no artifact)"
        if self.model_source == self.BUNDLE:
            return self.bundle_path or "(no bundle)"
        return self.trained_model.name if self.trained_model_id else "(deleted model)"

    def sync_bundle(self) -> dict | None:
        """Re-derive the pipeline fields from this job's bundle, if it has one.

        Bundle-sourced rows own no geometry of their own — see the class
        docstring. Returns the applied metadata, or ``None`` for the other two
        sources. Raises ``ValueError`` (``bundles.BundleError``) when the bundle
        can't be read.
        """
        if self.model_source != self.BUNDLE:
            return None
        from training.services import bundles

        return bundles.apply_defaults(self, self.bundle_path)

    def model_checkpoint(self) -> str:
        """Absolute path of the model artifact to run.

        Trained models carry a checkpoint path that may be project-relative;
        exported artifacts resolve against the export output directory and
        bundles against the bundle root (both reject anything escaping their
        root). Raises ``ValueError`` when the job's model is missing or unusable.
        """
        from django.conf import settings

        if self.model_source == self.EXPORTED:
            from training.services import exports

            return str(exports.resolve(self.artifact_path))

        if self.model_source == self.BUNDLE:
            from training.services import bundles

            return str(bundles.model_artifact(self.bundle_path))

        if not self.trained_model_id:
            raise ValueError("Inference job has no trained model.")
        raw = (self.trained_model.checkpoint_path or "").strip()
        if not raw:
            raise ValueError(f"{self.trained_model.name}: no checkpoint path to run.")
        path = Path(raw)
        return str(path if path.is_absolute() else Path(settings.BASE_DIR) / path)

    def output_path(self) -> Path:
        from videos.services.inference import output_dir

        return output_dir() / self.output_filename

    def output_exists(self) -> bool:
        return bool(self.output_filename) and self.output_path().is_file()
