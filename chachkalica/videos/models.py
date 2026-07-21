from pathlib import Path

from django.db import models

from fleet.services.paths import videos_root


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
