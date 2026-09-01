"""The Bundle Benchmarks admin section.

Three surfaces, all over one model:

* the changelist — the runs tab, with the headline numbers denormalized onto it;
* the **add form**, which doubles as the launcher: saving a row enqueues the job, so
  there is no separate intermediate action page to keep in step with the model;
* two custom pages, ``report/`` and ``compare/``, rendered from the stored result JSON.

Rows are records of a measurement, so they cannot be edited after the fact — a row
whose parameters were changed after the run would describe a benchmark that never
happened. Re-measuring is "Re-run selected…", which clones.
"""

import json
from pathlib import PurePosixPath

import django_rq
from django import forms
from django.contrib import admin, messages
from django.http import FileResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.html import format_html, format_html_join

from benchmarks import jobs
from benchmarks.models import BundleBenchmark
from fleet.admin import _status_badge
from training.services import bundles


def _queue():
    return django_rq.get_queue("default")


def _parse_int_list(raw: str, label: str) -> list[int]:
    """Validate a comma-separated positive-integer list, or raise a form error."""
    try:
        values = [int(part) for part in str(raw).replace(" ", "").split(",") if part]
    except ValueError:
        raise forms.ValidationError(f"{label} must be a comma-separated list of integers.")
    if not values:
        raise forms.ValidationError(f"{label} cannot be empty.")
    if any(value < 1 for value in values):
        raise forms.ValidationError(f"{label} must be positive.")
    return values


class BundleBenchmarkForm(forms.ModelForm):
    """The launcher. Bundles have no DB rows, so the choices are a filesystem scan."""

    bundle_path = forms.ChoiceField(
        label="Bundle",
        help_text="Scanned from the bundle root in Training settings. Copying a bundle "
                  "directory in there is all it takes for it to appear here.",
    )

    class Meta:
        model = BundleBenchmark
        fields = [
            "bundle_path", "dataset", "images_path", "max_images",
            "batch_sizes", "concurrency", "warmup", "calls", "min_duration_s",
            "measure_stages", "skip_bare_model", "profile_nsys",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        listed = bundles.list_bundles()
        self.fields["bundle_path"].choices = [
            (entry["relpath"],
             "{name} [{pipeline}] · {kind} · {size} MB{broken}".format(
                 name=entry["name"], pipeline=entry["pipeline"] or "?",
                 kind=entry["kind"], size=entry["size_mb"],
                 broken="" if entry["ok"] else "  — BROKEN, will fail"))
            for entry in listed
        ]
        if not listed:
            self.fields["bundle_path"].help_text = (
                f"No bundles found under {bundles.bundles_root()}. Export one, or copy "
                f"a bundle directory in there."
            )

    def clean_batch_sizes(self):
        _parse_int_list(self.cleaned_data["batch_sizes"], "Batch sizes")
        return self.cleaned_data["batch_sizes"]

    def clean_concurrency(self):
        _parse_int_list(self.cleaned_data["concurrency"], "Concurrency")
        return self.cleaned_data["concurrency"]

    def clean(self):
        cleaned = super().clean()
        if not cleaned.get("dataset") and not (cleaned.get("images_path") or "").strip():
            raise forms.ValidationError(
                "Pick a dataset or give an explicit images path — a bundle benchmark "
                "runs on real frames, so there is nothing to measure without them."
            )
        relpath = cleaned.get("bundle_path")
        if relpath:
            # Structural validation now rather than as a failed job later. The load
            # test is deliberately not run here: it costs a real trainer round trip,
            # and the benchmark itself is a far more thorough load test.
            report = bundles.validate(relpath, load_test=False)
            if not report["ok"]:
                failed = "; ".join(
                    f"{check['label']}: {check['detail']}"
                    for check in report["checks"] if check["status"] == "fail"
                )
                raise forms.ValidationError(f"That bundle will not run — {failed}")
            self.instance.bundle_name = report.get("name") or relpath
            # The format is the model artifact's own suffix. Taken from the manifest
            # rather than from the validate() report, whose `defaults` speak the
            # pipeline-metadata vocabulary and carry no format.
            manifest = bundles.read_manifest(relpath)
            model_rel = (manifest.get("artifacts") or {}).get("model") or ""
            self.instance.fmt = PurePosixPath(model_rel).suffix.lstrip(".")
        return cleaned


@admin.register(BundleBenchmark)
class BundleBenchmarkAdmin(admin.ModelAdmin):
    form = BundleBenchmarkForm
    list_display = [
        "created_at", "bundle_column", "fmt", "status_badge", "sweep",
        "latency_column", "fps", "gpu_mem_mb", "flags", "report_link",
    ]
    list_filter = ["status", "fmt", "contended"]
    search_fields = ["bundle_name", "bundle_path"]
    date_hierarchy = "created_at"
    ordering = ["-created_at"]
    actions = ["rerun_selected", "compare_selected", "stop_selected"]

    readonly_fields = [
        "status", "last_error", "output_dir", "bundle_name", "fmt",
        "p50_ms", "fps", "gpu_mem_mb", "contended",
        "created_at", "started_at", "finished_at", "results_summary",
    ]

    def has_change_permission(self, request, obj=None):
        """A finished run is a record of a measurement, not a form.

        Editing ``batch_sizes`` on a completed row would leave it describing a
        benchmark that never ran. Re-measure with "Re-run selected…" instead.
        """
        return False

    def get_fields(self, request, obj=None):
        if obj is None:
            return [
                "bundle_path", "dataset", "images_path", "max_images",
                "batch_sizes", "concurrency", "warmup", "calls", "min_duration_s",
                "measure_stages", "skip_bare_model", "profile_nsys",
            ]
        return [
            "bundle_name", "bundle_path", "fmt", "dataset", "images_path", "max_images",
            "batch_sizes", "concurrency", "warmup", "calls", "min_duration_s",
            "measure_stages", "skip_bare_model", "profile_nsys",
            "status", "last_error", "output_dir",
            "p50_ms", "fps", "gpu_mem_mb", "contended",
            "created_at", "started_at", "finished_at", "results_summary",
        ]

    # -- launching ------------------------------------------------------------
    def save_model(self, request, obj, form, change):
        """Saving the add form *is* the launch — the row and the job are one action."""
        super().save_model(request, obj, form, change)
        if change:
            return
        obj.output_dir = obj.resolved_output_dir()
        obj.save(update_fields=["output_dir"])
        _queue().enqueue(jobs.run_bundle_benchmark, obj.pk, job_timeout=jobs.JOB_TIMEOUT)
        self.message_user(
            request,
            f"Benchmark queued for {obj.bundle_name or obj.bundle_path} "
            f"(batches {obj.batch_sizes}, streams {obj.concurrency}) — refresh to see "
            f"progress. The trainer runs one GPU job at a time.",
        )

    @admin.action(description="Re-run selected…")
    def rerun_selected(self, request, queryset):
        """Clone each selected row and enqueue it.

        Cloning rather than resetting keeps the old numbers: the whole point of a
        history is being able to see that a bundle got slower.
        """
        queue = _queue()
        cloned = 0
        for benchmark in queryset:
            clone = BundleBenchmark.objects.create(
                bundle_path=benchmark.bundle_path, bundle_name=benchmark.bundle_name,
                fmt=benchmark.fmt, dataset=benchmark.dataset,
                images_path=benchmark.images_path, max_images=benchmark.max_images,
                warmup=benchmark.warmup, calls=benchmark.calls,
                min_duration_s=benchmark.min_duration_s,
                batch_sizes=benchmark.batch_sizes, concurrency=benchmark.concurrency,
                measure_stages=benchmark.measure_stages,
                skip_bare_model=benchmark.skip_bare_model,
                profile_nsys=benchmark.profile_nsys,
            )
            clone.output_dir = clone.resolved_output_dir()
            clone.save(update_fields=["output_dir"])
            queue.enqueue(jobs.run_bundle_benchmark, clone.pk, job_timeout=jobs.JOB_TIMEOUT)
            cloned += 1
        self.message_user(
            request,
            f"{cloned} benchmark(s) queued as new rows. They run one at a time: the "
            f"trainer refuses a second GPU job, and two benchmarks sharing a GPU would "
            f"both measure the contention rather than the bundle.",
        )

    @admin.action(description="Compare selected…")
    def compare_selected(self, request, queryset):
        finished = [b for b in queryset if b.results]
        if len(finished) < 2:
            self.message_user(
                request, "Select at least two finished benchmarks to compare.",
                level=messages.WARNING)
            return None
        ids = ",".join(str(b.pk) for b in sorted(finished, key=lambda b: b.created_at))
        url = reverse(f"admin:{self._url_name('compare')}")
        return redirect(f"{url}?runs={ids}")

    @admin.action(description="Stop selected")
    def stop_selected(self, request, queryset):
        from training.services import runner

        stopped = 0
        for benchmark in queryset.filter(status=BundleBenchmark.RUNNING):
            try:
                runner.stop_benchmark(benchmark)
                stopped += 1
            except Exception as exc:  # noqa: BLE001 - report per row, keep going
                self.message_user(request, f"#{benchmark.pk}: {exc}",
                                  level=messages.WARNING)
        self.message_user(
            request,
            f"Stop requested for {stopped} run(s). Verify with nvidia-smi that the "
            f"process is actually gone before starting another — a stop confirmation "
            f"is not proof on its own.",
        )

    # -- custom pages ---------------------------------------------------------
    def _url_name(self, suffix: str) -> str:
        """A URL name scoped to this ModelAdmin's own model.

        Derived from ``self.model`` rather than hardcoded, the way Django's own admin
        derives its changelist/change routes: if this admin is ever subclassed over a
        proxy model for a second tab, two ``path()`` entries would otherwise share one
        name and ``reverse()`` would pick whichever was registered last.
        """
        opts = self.model._meta
        return f"{opts.app_label}_{opts.model_name}_{suffix}"

    def get_urls(self):
        custom = [
            path("report/", self.admin_site.admin_view(self.report_view),
                 name=self._url_name("report")),
            path("compare/", self.admin_site.admin_view(self.compare_view),
                 name=self._url_name("compare")),
            path("nsys/", self.admin_site.admin_view(self.nsys_download_view),
                 name=self._url_name("nsys")),
        ]
        # Custom paths first: the change view's `<path:object_id>/` catch-all would
        # otherwise swallow them.
        return custom + super().get_urls()

    def report_view(self, request):
        benchmark = BundleBenchmark.objects.filter(pk=request.GET.get("run")).first()
        if benchmark is None:
            self.message_user(request, "No such benchmark.", level=messages.WARNING)
            return redirect("..")
        return TemplateResponse(request, "admin/benchmarks/bundle_report.html", {
            **self.admin_site.each_context(request),
            "title": f"Benchmark #{benchmark.pk} — {benchmark.bundle_name or benchmark.bundle_path}",
            "benchmark": benchmark,
            "status_badge_html": _status_badge(benchmark.status),
            "report_json": benchmark.results or {},
            "nsys_url": (
                reverse(f"admin:{self._url_name('nsys')}") + f"?run={benchmark.pk}"
                if benchmark.nsys_report_path() else ""
            ),
            "back_url": reverse(
                f"admin:{self.model._meta.app_label}_{self.model._meta.model_name}_changelist"),
        })

    def nsys_download_view(self, request):
        """Serve one run's ``.nsys-rep`` so it can be opened in the Nsight GUI.

        The path comes from ``BundleBenchmark.nsys_report_path()``, which rebuilds it
        from the run's own ``output_dir`` — the request supplies only a row id, never a
        path, so this cannot be pointed at an arbitrary file.
        """
        benchmark = BundleBenchmark.objects.filter(pk=request.GET.get("run")).first()
        if benchmark is None:
            self.message_user(request, "No such benchmark.", level=messages.WARNING)
            return redirect("..")
        report = benchmark.nsys_report_path()
        if report is None:
            self.message_user(
                request,
                "That run has no Nsight report on disk. Re-run it with the Nsight "
                "kernel profile enabled.",
                level=messages.WARNING)
            return redirect("..")
        return FileResponse(
            report.open("rb"), as_attachment=True,
            filename=f"benchmark-{benchmark.pk}-{report.name}")

    def compare_view(self, request):
        raw = (request.GET.get("runs") or "").split(",")
        ids = [int(item) for item in raw if item.strip().isdigit()]
        rows = list(BundleBenchmark.objects.filter(pk__in=ids).exclude(results=None))
        rows.sort(key=lambda b: ids.index(b.pk) if b.pk in ids else 0)
        if len(rows) < 2:
            self.message_user(request, "Select at least two finished benchmarks.",
                              level=messages.WARNING)
            return redirect("..")
        return TemplateResponse(request, "admin/benchmarks/compare.html", {
            **self.admin_site.each_context(request),
            "title": f"Comparing {len(rows)} benchmarks",
            "compare_json": {
                "runs": [{
                    "id": row.pk,
                    "label": f"#{row.pk} {row.bundle_name or row.bundle_path}",
                    "fmt": row.fmt,
                    "created": row.created_at.isoformat(),
                    "contended": row.contended,
                    "result": row.results,
                } for row in rows],
            },
            "back_url": reverse(
                f"admin:{self.model._meta.app_label}_{self.model._meta.model_name}_changelist"),
        })

    # -- columns --------------------------------------------------------------
    @admin.display(description="bundle", ordering="bundle_name")
    def bundle_column(self, obj):
        return obj.bundle_name or obj.bundle_path

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="sweep")
    def sweep(self, obj):
        return f"b{obj.batch_sizes} × x{obj.concurrency}"

    @admin.display(description="p50", ordering="p50_ms")
    def latency_column(self, obj):
        return "—" if obj.p50_ms is None else f"{obj.p50_ms:.2f} ms"

    @admin.display(description="flags")
    def flags(self, obj):
        """Anything that makes the numbers less than directly comparable."""
        marks = []
        if obj.contended:
            marks.append(("contended", "another process was on the GPU during this run"))
        warnings = (obj.results or {}).get("warnings") or []
        if warnings:
            marks.append((f"{len(warnings)} caveat(s)",
                          "the harness reported caveats — see the report"))
        if not marks:
            return "—"
        # format_html_join rather than joining pre-built HTML: every piece here goes
        # through escaping, so a caveat count or a future label cannot break the markup.
        return format_html(
            '<span style="color:#e3b341;">{}</span>',
            format_html_join(" · ", '<span title="{}">{}</span>',
                             ((title, label) for label, title in marks)),
        )

    @admin.display(description="report")
    def report_link(self, obj):
        if not obj.results:
            return "—"
        url = reverse(f"admin:{self._url_name('report')}")
        return format_html('<a href="{}?run={}">open</a>', url, obj.pk)

    @admin.display(description="result")
    def results_summary(self, obj):
        """A compact dump on the read-only detail page, so the row is self-contained
        even before anyone opens the rendered report."""
        if not obj.results:
            return "—"
        return format_html("<pre style=\"max-height:24em;overflow:auto;\">{}</pre>",
                           json.dumps(obj.results, indent=2)[:20000])
