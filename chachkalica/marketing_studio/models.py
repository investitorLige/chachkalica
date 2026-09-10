"""Marketing Studio's own tables.

A deliberate near-duplicate of the marketing half of :mod:`videos.models`, not a
set of proxies over it. The studio is its own surface: its own video library
under ``FleetSettings.marketing_videos_dir``, its own bundle set under
``marketing_bundles_dir``, its own render log and its own presets. Nothing here
is shared with the Videos tab except *code* — the style system, the render loop,
the downloader and the byte-range streamer are stateless and get imported (see
:mod:`videos.services.render_style` and :mod:`videos.services.inference`).

The one structural simplification against ``videos.InferenceJob``: a render here
always runs an **infer bundle**. There is no trained-checkpoint or exported-
artifact source, so ``model_source``/``trained_model``/``artifact_path`` are
gone, and with them the cross-app FK into ``training`` — which is why this app's
initial migration depends on nothing outside itself.
"""

from pathlib import Path

from django.db import models

from marketing_studio.services.paths import marketing_videos_root
from training import pipelines


class Video(models.Model):
    """A clip in the studio's library, living under the studio's video root.

    Same "DB holds metadata, bytes stay on disk" convention as
    :class:`videos.models.Video`, and the same two ways in: import a file
    already sitting in the folder, or paste a link a background job downloads.
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
        help_text="File name under the studio's video root. Filled in by the download job.",
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
        return marketing_videos_root() / self.filename

    def exists(self) -> bool:
        return bool(self.filename) and self.path().is_file()


class RenderPreset(models.Model):
    """A saved marketing look, by name.

    Same shape and purpose as :class:`videos.models.RenderPreset`, on its own
    table: a look saved here is invisible on the Videos tab and vice versa. That
    is the point of a separate section, but it does mean the same preset name can
    exist on both sides holding different styles.

    Holds the *look* only, deliberately — not the bundle. That comes from the
    bundle's own manifest (see ``training.services.bundles``), and a preset that
    overrode it would quietly undo that convention.
    """

    name = models.CharField(
        max_length=120, unique=True,
        help_text="What this look is called in the preset dropdown.",
    )
    style = models.JSONField(
        default=dict,
        help_text="The style dict (see videos.services.render_style).",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "Render preset"
        verbose_name_plural = "Render presets"

    def __str__(self) -> str:
        return self.name

    def summary(self) -> str:
        """The couple of choices that identify a look at a glance — same pair the
        Inferred videos list shows."""
        from videos.services import render_style

        style = render_style.normalize(self.style)
        return f"{style['box_style']} · {style['palette']}"


class Render(models.Model):
    """One marketing render: a bundle applied frame-by-frame to a clip, with
    styled boxes burned onto an annotated output video.

    Status lifecycle ``queued -> running -> ok``/``error``; the output lives
    under ``<marketing_videos_dir>/inferred/``, which is a *different* directory
    from the Videos tab's — see :func:`marketing_studio.services.paths.output_root`
    for why that separation is load-bearing rather than tidy.

    The model always comes from an infer bundle under the studio's bundle root,
    addressed by ``bundle_path``. Bundles have no DB row of their own; the
    manifest is the record. That also means the pipeline geometry below is not
    the operator's to choose — a bundle ships the geometry its weights were tuned
    with, so the admin renders those fields read-only and :meth:`sync_bundle`
    re-derives them from the manifest.

    ``render_style`` is the whole point of this table rather than an option on
    it: every row here is a look. Kept as one JSON blob rather than thirty
    columns because nothing queries an individual knob — the row is the record of
    a look, read back whole by the renderer and by the form when a run is
    duplicated.
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

    # "raw" is not a chachak pipeline — it means "no pipeline, feed the model the
    # whole frame". Kept in the choices so a bundle that declares it renders a
    # label rather than a bare value.
    RAW = "raw"
    PIPELINE_CHOICES = [
        (RAW, "raw — whole frame straight to the model"),
        *pipelines.PIPELINE_CHOICES,
    ]

    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="renders")
    bundle_path = models.CharField(
        max_length=1024,
        help_text="Path of the bundle directory relative to the studio's bundle "
                  "root. The bundle's manifest supplies the model, the detector "
                  "and the whole pipeline geometry.",
    )
    score_threshold = models.FloatField(
        default=0.5, help_text="Detections below this confidence are dropped.",
    )
    frame_stride = models.PositiveIntegerField(
        default=1,
        help_text="Run inference every Nth frame; frames in between reuse the last "
                  "inferred boxes. Higher = faster, less temporally precise.",
    )
    render_style = models.JSONField(
        default=dict, blank=True,
        help_text="Look of the burned-in overlay (see videos.services.render_style).",
    )

    # ------------------------------------------------ synced from the manifest
    pipeline = models.CharField(
        max_length=32, choices=PIPELINE_CHOICES, default=RAW,
        help_text="How each frame is presented to the model: whole ('raw'), tiled, "
                  "or cropped around detected people.",
    )
    detector_checkpoint = models.CharField(max_length=1024, blank=True)
    detector_expand_ratio = models.FloatField(
        null=True, blank=True, verbose_name="Person-box expand ratio",
    )
    detector_min_box_size = models.FloatField(
        null=True, blank=True, verbose_name="Person-crop minimum size (px)",
    )
    tile_size_px = models.PositiveIntegerField(
        null=True, blank=True, verbose_name="Tile size (pixels)",
    )
    tile_width_pct = models.FloatField(null=True, blank=True)
    tile_height_pct = models.FloatField(null=True, blank=True)
    overlap = models.FloatField(null=True, blank=True)
    merge_nms_iou = models.FloatField(
        null=True, blank=True, verbose_name="merge_nms_iou_threshold",
    )
    chain = models.JSONField(
        default=list, blank=True,
        help_text="Ordered pipeline names for the 'chain' pipeline.",
    )

    # ------------------------------------------------------------- lifecycle
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=QUEUED)
    last_error = models.TextField(blank=True)
    output_filename = models.CharField(
        max_length=512, blank=True,
        help_text="File name under <marketing_videos_dir>/inferred/ holding the render.",
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
        """Human name of the bundle this render runs. Named for the duck type
        ``videos.services.inference`` and the shared player template expect."""
        return self.bundle_path or "(no bundle)"

    def sync_bundle(self) -> dict:
        """Re-derive the pipeline fields from this render's bundle.

        Bundle-sourced rows own no geometry of their own — see the class
        docstring. Raises ``bundles.BundleError`` when the bundle can't be read.
        """
        from marketing_studio.services import paths
        from training.services import bundles

        return bundles.apply_defaults(self, self.bundle_path, ts=paths.bundle_settings())

    def model_checkpoint(self) -> str:
        """Absolute path of the model artifact to run — part of the duck type
        ``videos.services.inference.build_predict_payload`` consumes."""
        from marketing_studio.services import paths
        from training.services import bundles

        return str(bundles.model_artifact(self.bundle_path, ts=paths.bundle_settings()))

    def output_path(self) -> Path:
        from marketing_studio.services.paths import output_root
        from videos.services.inference import output_dir

        return output_dir(output_root()) / self.output_filename

    def output_exists(self) -> bool:
        return bool(self.output_filename) and self.output_path().is_file()
