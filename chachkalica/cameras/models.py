from pathlib import Path

from django.core.exceptions import ValidationError
from django.db import models

from training import pipelines
from training.models import DEFAULT_PERSON_DETECTOR_CHECKPOINT

_RTSP_SCHEMES = ("rtsp://", "rtsps://")


def validate_rtsp_url(value: str) -> None:
    if not value.lower().startswith(_RTSP_SCHEMES):
        raise ValidationError("Must be an rtsp:// or rtsps:// URL.")


class Camera(models.Model):
    """An IP camera reachable over RTSP.

    Only the connection details live in the database — frames are pulled live
    from the camera on demand (via ``cameras.services.rtsp``), nothing is
    stored on disk yet. ``status`` reflects the last time the stream was
    opened successfully, refreshed on every save.
    """

    UNKNOWN = "unknown"
    ONLINE = "online"
    OFFLINE = "offline"
    STATUS_CHOICES = [
        (UNKNOWN, "unknown"),
        (ONLINE, "online"),
        (OFFLINE, "offline"),
    ]

    name = models.CharField(max_length=255, unique=True, help_text="Display name.")
    rtsp_url = models.CharField(
        max_length=1024,
        validators=[validate_rtsp_url],
        help_text="Full stream URL, e.g. rtsp://user:pass@10.10.10.24:554/Streaming/Channels/101",
    )
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=UNKNOWN)
    last_error = models.TextField(blank=True)
    last_checked_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class CameraInference(models.Model):
    """Live model inference for one camera: which model to run, and how often.

    A persistent *configuration* rather than a run — unlike
    :class:`videos.models.InferenceJob` (one row per finished video), this row
    describes an ongoing loop that the ``infer_cameras`` worker keeps alive for
    as long as ``enabled`` is set. The status fields at the bottom are that
    loop reporting back, the same way ``Camera.status`` is ``stream_cameras``
    reporting back.

    The model/pipeline field set is deliberately the same one carried by
    ``InferenceJob`` and ``eval_pipelines.PipelineEvalRun``, so a model
    configured for a video or an eval can be pointed at a camera unchanged.

    Frames never come from the camera directly: the loop reads whatever
    ``stream_cameras`` last pushed to Redis, so enabling this adds no second
    RTSP session against the camera's own connection limit. See
    ``docs/camera-live-inference.md``.
    """

    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    STATUS_CHOICES = [
        (IDLE, "idle"),
        (RUNNING, "running"),
        (ERROR, "error"),
    ]

    TRAINED = "trained"
    EXPORTED = "exported"
    MODEL_SOURCE_CHOICES = [
        (TRAINED, "trained model (.pt checkpoint)"),
        (EXPORTED, "exported artifact (ONNX / TensorRT)"),
    ]

    # "raw" is not a chachak pipeline — it means "no pipeline, feed the model the
    # whole frame". Same meaning as InferenceJob.RAW.
    RAW = "raw"
    PIPELINE_CHOICES = [
        (RAW, "raw — whole frame straight to the model"),
        *pipelines.PIPELINE_CHOICES,
    ]

    camera = models.OneToOneField(
        Camera, on_delete=models.CASCADE, related_name="inference",
        help_text="The camera whose live frames this model runs on.",
    )
    enabled = models.BooleanField(
        default=False,
        help_text="Run this model continuously over the camera's live frames. "
                  "The raw preview keeps working either way.",
    )
    target_fps = models.FloatField(
        default=2.0,
        help_text="How often the detection boxes refresh, in times per second. The "
                  "annotated video itself stays as smooth as the live preview — "
                  "only the boxes are limited by this. Caps GPU load; an upper "
                  "bound, not a guarantee. Above ~6-7 it has no effect, since the "
                  "camera preview supplies no new frames faster than that.",
    )

    model_source = models.CharField(
        max_length=16, choices=MODEL_SOURCE_CHOICES, default=TRAINED,
        help_text="Where the model comes from: the trained-models catalogue, or an "
                  "exported artifact in the export output directory.",
    )
    trained_model = models.ForeignKey(
        "training.TrainedModel", null=True, blank=True,
        on_delete=models.CASCADE, related_name="camera_inferences",
        help_text="Set when model_source is 'trained'.",
    )
    artifact_path = models.CharField(
        max_length=1024, blank=True,
        help_text="Path of the exported .onnx/.engine relative to the export output "
                  "directory. Set when model_source is 'exported'.",
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
        max_length=1024, blank=True, default=DEFAULT_PERSON_DETECTOR_CHECKPOINT,
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

    # ── written by the infer_cameras worker, read-only in the admin ──
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=IDLE)
    last_error = models.TextField(blank=True)
    last_inference_at = models.DateTimeField(null=True, blank=True)
    last_latency_ms = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Round-trip of the most recent /predict_image call, including the "
                  "frame write and any wait for the trainer's GPU lock.",
    )
    last_box_count = models.PositiveIntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Live inference"
        verbose_name_plural = "Live inference"

    def __str__(self) -> str:
        return f"{self.camera.name} → {self.model_label()} [{self.pipeline}]"

    def model_label(self) -> str:
        """Human name of the model this config runs, whichever source it came from."""
        if self.model_source == self.EXPORTED:
            return self.artifact_path or "(no artifact)"
        return self.trained_model.name if self.trained_model_id else "(no model)"

    def model_checkpoint(self) -> str:
        """Absolute path of the model artifact to run.

        Mirrors :meth:`videos.models.InferenceJob.model_checkpoint`: trained
        models carry a possibly project-relative checkpoint path, exported
        artifacts resolve against the export output directory (which also
        rejects anything escaping it). Raises ``ValueError`` when the config's
        model is missing or unusable.
        """
        from django.conf import settings

        if self.model_source == self.EXPORTED:
            from training.services import exports

            if not (self.artifact_path or "").strip():
                raise ValueError("No exported artifact selected.")
            return str(exports.resolve(self.artifact_path))

        if not self.trained_model_id:
            raise ValueError("Live inference has no trained model selected.")
        raw = (self.trained_model.checkpoint_path or "").strip()
        if not raw:
            raise ValueError(f"{self.trained_model.name}: no checkpoint path to run.")
        path = Path(raw)
        return str(path if path.is_absolute() else Path(settings.BASE_DIR) / path)

    def interval_seconds(self) -> float:
        """Minimum seconds between two inference calls, from :attr:`target_fps`."""
        fps = self.target_fps or 0
        return 1.0 / fps if fps > 0 else 0.0

    def worker_fingerprint(self) -> tuple:
        """Everything a worker process bakes in at start time.

        The ``infer_cameras`` reconcile loop restarts a camera's worker when this
        changes, so editing the model/pipeline/threshold in the admin takes
        effect without restarting the whole command — the same contract
        ``stream_cameras`` has with ``Camera.rtsp_url``.
        """
        return (
            self.model_source, self.trained_model_id, self.artifact_path,
            self.score_threshold, self.pipeline, self.detector_checkpoint,
            self.detector_expand_ratio, self.detector_min_box_size,
            self.tile_size_px, self.tile_width_pct, self.tile_height_pct,
            self.overlap, self.merge_nms_iou, tuple(self.chain or []),
            self.target_fps,
        )
