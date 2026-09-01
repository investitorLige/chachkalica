"""One row per bundle benchmark run.

Deliberately shaped like :class:`training.models.ExportRun`: a queued/running/ok/error
status, the timestamp triple, the input parameters, a ``results`` JSON blob for
everything the harness measured, and a handful of *denormalized* headline columns so
the changelist is readable without opening a report.

Bytes stay on disk (the harness writes ``benchmark.json`` into ``output_dir`` on the
shared data mount); the database holds metadata and the parsed result. A bundle itself
has no DB row at all — it is addressed by its path under
:func:`training.services.bundles.bundles_root`, which is what lets one arrive from
another machine and still describe itself completely.
"""

from pathlib import Path

from django.db import models

from training.services import config_gen

# Where a run's output_dir is created. Under runs_root because that is already the
# shared, trainer-visible output tree; dot-prefixed subdirectory names are not needed
# here (unlike under bundles_root, which is scanned for manifests).
BENCHMARK_SUBDIR = "benchmarks"


class BundleBenchmark(models.Model):
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    STATUS_CHOICES = [
        (QUEUED, "queued"), (RUNNING, "running"), (OK, "ok"), (ERROR, "error"),
    ]

    # -- what was measured ----------------------------------------------------
    bundle_path = models.CharField(
        max_length=1024,
        help_text="Bundle directory, relative to the bundle root in Training settings.",
    )
    bundle_name = models.CharField(max_length=255, blank=True)
    fmt = models.CharField(max_length=16, blank=True, help_text="onnx or engine.")

    dataset = models.ForeignKey(
        "fleet.Dataset", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="bundle_benchmarks",
        help_text="Frames come from this dataset's source images unless a path is given below.",
    )
    images_path = models.CharField(
        max_length=1024, blank=True,
        help_text="Explicit image file or directory. Overrides the dataset above.",
    )
    max_images = models.PositiveIntegerField(
        default=24,
        help_text="Frames used, sorted then truncated so two runs on the same source "
                  "see the same frames. 0 means all of them.",
    )

    # -- how it was measured --------------------------------------------------
    warmup = models.PositiveIntegerField(
        default=30, help_text="Untimed inference calls before each cell. Never zero: an "
                              "engine's first call carries deserialization.",
    )
    calls = models.PositiveIntegerField(
        default=200, help_text="Timed inference calls per cell, grown if needed to span "
                               "the minimum duration.",
    )
    min_duration_s = models.FloatField(
        default=2.0,
        help_text="Floor on the throughput loop's wall time. NVML's utilization counter "
                  "refreshes about once a second, so a shorter loop reports noise.",
    )
    batch_sizes = models.CharField(
        max_length=64, default="1",
        help_text="Comma-separated. Each is capped by the artifacts' own batch profile.",
    )
    concurrency = models.CharField(
        max_length=64, default="1",
        help_text="Comma-separated worker-thread counts. Each worker loads its own copy "
                  "of the bundle, so more streams cost more GPU memory.",
    )
    measure_stages = models.BooleanField(
        default=True,
        help_text="Add a per-stage pass (preprocess / person / in-between / model / "
                  "postprocess). Synchronized at each boundary, so it runs separately "
                  "from the timing passes.",
    )
    skip_bare_model = models.BooleanField(
        default=False,
        help_text="Skip the independent polygraphy/onnxruntime reading of the model "
                  "artifact alone.",
    )
    profile_nsys = models.BooleanField(
        "Nsight kernel profile",
        default=False,
        help_text="Add a final sequential pass that re-runs one cell under Nsight "
                  "Systems, giving a per-kernel table and a .nsys-rep to open in the "
                  "Nsight GUI. Costs an extra capture; needs Nsight mounted on the "
                  "trainer. Occupancy counters stay unavailable (they need "
                  "CAP_SYS_ADMIN).",
    )

    # -- outcome --------------------------------------------------------------
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=QUEUED)
    last_error = models.TextField(blank=True)
    output_dir = models.CharField(max_length=1024, blank=True)
    results = models.JSONField(null=True, blank=True)

    # Denormalized from ``results`` on ingest, purely so the changelist is legible.
    # Taken from the first cell, which is the batch-1 single-stream case operators
    # compare across bundles.
    p50_ms = models.FloatField(null=True, blank=True)
    fps = models.FloatField(null=True, blank=True)
    gpu_mem_mb = models.FloatField(null=True, blank=True)
    contended = models.BooleanField(
        default=False,
        help_text="Another process was on the GPU during the run, so these numbers are "
                  "not comparable with an idle-GPU baseline.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Bundle benchmark"
        verbose_name_plural = "Bundle benchmarks"

    def __str__(self) -> str:
        return f"{self.bundle_name or self.bundle_path} #{self.pk}"

    # -- derived --------------------------------------------------------------
    def resolved_output_dir(self) -> str:
        """Where the harness writes ``benchmark.json`` for this run."""
        root = config_gen._resolve(_runs_root())
        return str(root / BENCHMARK_SUBDIR / str(self.pk))

    def absolute_bundle_dir(self):
        """The bundle directory this row names, validated against the bundle root.

        Reuses :func:`training.services.bundles.resolve`, so a path escaping the root
        or naming a directory with no manifest raises ``BundleError`` here rather than
        reaching the trainer as a confusing 400.
        """
        from training.services import bundles

        return bundles.resolve(self.bundle_path)

    def resolved_images_path(self) -> str:
        """The frame source: the explicit path if given, else the dataset's images.

        Raises ``ValueError`` when neither is set — checked at form time, but also
        here, because a re-run of a row whose dataset was since deleted would
        otherwise reach the trainer with an empty path.
        """
        if self.images_path:
            return self.images_path
        if self.dataset is None:
            raise ValueError("no images: set a dataset or an explicit images path")
        from fleet.services import lsapi
        from fleet.services import paths as fleet_paths

        dataset_dir = fleet_paths.source_root() / self.dataset.name
        return str(lsapi.image_source_dir(dataset_dir))

    def ingest(self, result: dict) -> None:
        """Store the harness result and lift the headline numbers out of it.

        The first cell is the headline: the batch sizes are swept in the order given
        and the first is what an operator compares across bundles.
        """
        self.results = result
        cells = [cell for cell in (result.get("cells") or []) if not cell.get("error")]
        headline = cells[0] if cells else {}
        self.p50_ms = (headline.get("latency_ms") or {}).get("p50")
        self.fps = headline.get("fps")
        self.gpu_mem_mb = (headline.get("mem") or {}).get("load_delta_mb")
        self.contended = bool((result.get("trust") or {}).get("contended"))
        bundle = result.get("bundle") or {}
        self.bundle_name = bundle.get("name") or self.bundle_name
        self.fmt = bundle.get("fmt") or self.fmt

    def nsys_report_path(self):
        """The captured ``.nsys-rep``, or ``None``.

        Rebuilt from ``output_dir`` plus the filename the harness recorded rather than
        from a stored absolute path, so the download view has a directory to confine the
        lookup to and a run cannot be made to serve a file from anywhere else.
        """
        info = (self.results or {}).get("nsys") or {}
        if not info.get("available") or not info.get("report_file") or not self.output_dir:
            return None
        candidate = Path(self.output_dir) / "nsys" / Path(info["report_file"]).name
        return candidate if candidate.is_file() else None


def _runs_root() -> str:
    from training.models import TrainingSettings

    return TrainingSettings.load().runs_root
