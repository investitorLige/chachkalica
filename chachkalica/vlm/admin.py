"""VLM admin — the Models, Videos and Dataset runs tabs, and their screens.

Three registered tabs, matching the shape of the request:

* **Models** — pretrained VLMs, added by picking a family, its weights, and the
  prompt to ask of every frame. Nothing is trained here. Its one action runs a
  model over a whole dataset.
* **Videos** — an mp4 library of its own. Three ways to add one: upload from the
  browser, import a file already in the folder, or paste a link.
* **Dataset runs** — the history of "Run on a dataset…", each row opening a
  report: the answers image by image and, when the dataset is labeled and
  grading was asked for, how well they matched (see :mod:`vlm.services.grading`).

``VlmRun``/``VlmAlert`` are deliberately *not* registered: runs are reached from
the "Run VLM Inference…" action's redirect and from the past-runs panel on a
video's change page, via custom URLs on :class:`VlmVideoAdmin` — the same way
``videos.Video``'s play/stream views hang off ``VideoAdmin`` rather than being
their own tab.

The live screen plays the video on the left and polls for alerts on the right.
Polling (rather than websockets) is what fits this project: it is WSGI-only,
with no Channels anywhere. Each poll doubles as the "someone is watching"
heartbeat that keeps the worker running.
"""

from pathlib import Path

from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.html import format_html, format_html_join

import django_rq

from fleet.models import Dataset
from fleet.services import datasets as datasets_svc, lsapi
from videos.services import streaming
from vlm import jobs
from vlm.forms import VlmModelForm, VlmVideoAddForm
from vlm.models import (
    VlmAlert, VlmDatasetResult, VlmDatasetRun, VlmModel, VlmRun, VlmVideo,
)
from vlm.services import (
    dataset_runner, grading, heartbeat, label_render, videos as video_services,
)
from vlm.weights_catalog import is_cached

_STATUS_COLORS = {
    "ready": "#22c55e",
    "ok": "#22c55e",
    "running": "#3b82f6",
    "downloading": "#f59e0b",
    "queued": "#9ca3af",
    "pending": "#9ca3af",
    "cancel_requested": "#f59e0b",
    "cancelled": "#9ca3af",
    "error": "#ef4444",
}

_DEFAULT_FPS = 1.0

#: Images a dataset run defaults to; 0 in the form means "every image". A
#: capped run is the normal way to sanity-check a prompt before spending hours
#: of GPU on 10k images, so the default is a cap rather than the whole dataset.
_DEFAULT_IMAGE_LIMIT = 200

#: How many result rows the report renders. A 10k-image run's full table would
#: be a several-megabyte page nobody reads to the end; the mismatch filter is
#: what makes a big run navigable.
_REPORT_ROW_LIMIT = 300


def _queue():
    return django_rq.get_queue("default")


def _status_badge(value: str):
    if not value:
        return "—"
    color = _STATUS_COLORS.get(value, "#9ca3af")
    return format_html('<b style="color:{};">●</b> {}', color, value)


def _confusion_table(confusion: dict | None) -> dict | None:
    """Shade a grading confusion matrix for the template.

    Counts come out of :func:`vlm.services.grading.metrics` raw; the cells are
    shaded by their share of the ground-truth row — green where the answer named
    the labeled class, red everywhere else — which is the same convention (and
    the same ``--i`` intensity variable) the eval analytics matrices already
    use, so the two pages read the same way.
    """
    if not confusion:
        return None
    columns = confusion["columns"]
    rows = []
    for row in confusion["rows"]:
        total = row["total"] or 1
        rows.append({
            "actual": row["actual"],
            "total": row["total"],
            "cells": [
                {
                    "value": count,
                    "intensity": round(count / total, 4),
                    "is_diagonal": columns[i] == row["actual"],
                }
                for i, count in enumerate(row["counts"])
            ],
        })
    return {"columns": columns, "rows": rows, "total": confusion["total"]}


@admin.register(VlmModel)
class VlmModelAdmin(admin.ModelAdmin):
    form = VlmModelForm

    list_display = ["name", "family", "backend", "weights", "variant",
                    "weights_available", "prompt_preview", "dataset_runs_link",
                    "created_at"]
    list_filter = ["family", "backend"]
    actions = ["run_on_dataset"]
    search_fields = ["name", "description", "weights", "prompt"]
    readonly_fields = ["created_at", "updated_at"]

    fieldsets = [
        (None, {"fields": ["name", "description"]}),
        ("Model", {"fields": ["backend", "family", "weights_choice", "custom_weights",
                              "variant", "quantization"]}),
        ("Prompt", {"fields": ["prompt", "max_new_tokens"]}),
        ("Details", {"classes": ["collapse"], "fields": ["created_at", "updated_at"]}),
    ]

    @admin.display(description="weights in cache", boolean=True)
    def weights_available(self, obj):
        """Whether this row can actually load on this network.

        huggingface.co is unreachable here, so a repo that was never copied into
        data/hf_cache will fail at the first frame of a run. Surfacing it in the
        list makes that a glance rather than a debugging session.
        """
        return is_cached(obj.weights)

    @admin.display(description="prompt")
    def prompt_preview(self, obj):
        text = (obj.prompt or "").strip().replace("\n", " ")
        return (text[:70] + "…") if len(text) > 70 else (text or "—")

    @admin.display(description="dataset runs")
    def dataset_runs_link(self, obj):
        count = obj.dataset_runs.count()
        if not count:
            return "—"
        url = (reverse("admin:vlm_vlmdatasetrun_changelist")
               + f"?vlm_model__id__exact={obj.pk}")
        return format_html('<a href="{}">{} run(s)</a>', url, count)

    # ---------------------------------------------------------------- actions
    @admin.action(description="Run on a dataset…")
    def run_on_dataset(self, request, queryset):
        """Ask this model its prompt about every image of a dataset.

        One action rather than two, with grading as a checkbox inside it,
        because whether a dataset *can* be graded is a property of the dataset
        on disk and not of the operator's intent — the form can answer it
        itself, and a second action would have to ask the same question again.
        The grading configuration deliberately stays out of this form: the
        aliases worth setting are the ones you pick after seeing what the model
        actually says, so they are edited on the report, which can re-grade
        without re-running (see :func:`vlm.services.dataset_runner.regrade`).

        Intermediate-page pattern, same as ``run_vlm_inference``.
        """
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one VLM to run.",
                              level=messages.WARNING)
            return None
        model = queryset.first()

        datasets = list(Dataset.objects.all())
        if not datasets:
            self.message_user(
                request, "There are no datasets — add one under Datasets first.",
                level=messages.WARNING,
            )
            return None

        # Read the labels folder from disk rather than trusting the stored flag:
        # a dataset labeled after its row was created would otherwise offer no
        # grading, and the flag is only refreshed by the fleet-side actions.
        options = []
        for dataset in datasets:
            try:
                has_labels = datasets_svc.detect_labels(dataset, persist=False)
            except OSError:
                has_labels = False
            options.append({"dataset": dataset, "has_labels": has_labels})

        limit_value = request.POST.get("image_limit")
        if limit_value is None:
            limit_value = _DEFAULT_IMAGE_LIMIT
        try:
            dataset_id = int(request.POST.get("dataset") or datasets[0].pk)
        except (TypeError, ValueError):
            dataset_id = datasets[0].pk
        grade = bool(request.POST.get("grade"))
        # Unchecked checkboxes simply do not post, so "absent" means False on a
        # submit but has to mean the default on the first render.
        negation = (bool(request.POST.get("negation_aware"))
                    if request.POST.get("apply") else True)
        errors = []

        if request.POST.get("apply"):
            dataset = Dataset.objects.filter(pk=dataset_id).first()
            try:
                limit = int(limit_value)
            except (TypeError, ValueError):
                limit = -1

            if limit < 0:
                errors.append("Image limit must be 0 (all images) or more.")
            if not is_cached(model.weights):
                errors.append(
                    f"{model.name}: {model.weights} is not in data/hf_cache, so it "
                    f"cannot load on this network."
                )
            if dataset is None:
                errors.append("Choose a dataset to run over.")
            else:
                directory = dataset_runner.dataset_dir(dataset.name)
                classes = []
                if not directory.is_dir():
                    errors.append(f"{dataset.name}: no directory at {directory}.")
                else:
                    if not dataset_runner.select_images(directory, 0):
                        errors.append(f"{dataset.name}: no images under {directory}.")
                    try:
                        classes = dataset_runner.read_classes(directory)
                    except RuntimeError as exc:
                        classes = []
                        if grade:
                            errors.append(str(exc))
                    if grade:
                        if not classes:
                            errors.append(
                                f"{dataset.name} has no classes.txt, so there are no "
                                f"class names to score answers against — uncheck "
                                f"grading, or add one."
                            )
                        if not datasets_svc.detect_labels(dataset, persist=False):
                            errors.append(
                                f"{dataset.name} has no labels/ folder, so there is "
                                f"nothing to grade against — uncheck grading."
                            )

                if not errors:
                    run = VlmDatasetRun.objects.create(
                        dataset=dataset,
                        dataset_name_snapshot=dataset.name,
                        vlm_model=model,
                        model_config_snapshot=model.config(),
                        prompt_snapshot=model.prompt,
                        model_label_snapshot=model.name,
                        image_limit=limit,
                        grade=grade,
                        negation_aware=negation,
                        classes_snapshot=classes,
                        alias_map=grading.default_alias_map(classes) if grade else {},
                        status=VlmDatasetRun.QUEUED,
                    )
                    _queue().enqueue(jobs.run_vlm_on_dataset, run.id,
                                     job_timeout=jobs.VLM_JOB_TIMEOUT)
                    return redirect(
                        reverse("admin:vlm_vlmdatasetrun_report") + f"?run={run.pk}"
                    )

        for error in errors:
            self.message_user(request, error, level=messages.ERROR)

        context = {
            **self.admin_site.each_context(request),
            "title": f"Run on a dataset — {model.name}",
            "model": model,
            "options": options,
            "selected_dataset_id": dataset_id,
            "image_limit": limit_value,
            "grade": grade,
            "negation_aware": negation,
            "weights_cached": is_cached(model.weights),
            "action_name": "run_on_dataset",
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
            "selected": [str(model.pk)],
        }
        return TemplateResponse(request, "admin/vlm/run_dataset.html", context)


@admin.register(VlmVideo)
class VlmVideoAdmin(admin.ModelAdmin):
    add_form = VlmVideoAddForm
    add_fieldsets = ((None, {
        "fields": ("upload", "import_file", "source_url", "quality", "name"),
    }),)

    list_display = ["name", "status_badge", "geometry", "runs_link",
                    "created_at", "play_link"]
    list_filter = ["status"]
    search_fields = ["name", "filename", "source_url"]
    readonly_fields = ["status", "filename", "player", "geometry", "past_runs",
                       "last_error", "created_at", "updated_at"]
    actions = ["run_vlm_inference", "play_video"]

    def get_actions(self, request):
        """Relabel the stock delete — it also removes the file on disk."""
        actions = super().get_actions(request)
        if "delete_selected" in actions:
            func, name, _desc = actions["delete_selected"]
            actions["delete_selected"] = (
                func, name, "Delete selected video(s) (also deletes the file on disk)",
            )
        return actions

    # ------------------------------------------------------------------ forms
    def get_form(self, request, obj=None, **kwargs):
        defaults = {}
        if obj is None:
            defaults["form"] = self.add_form
        defaults.update(kwargs)
        return super().get_form(request, obj, **defaults)

    def get_fieldsets(self, request, obj=None):
        if obj is None:
            return self.add_fieldsets
        return [
            (None, {"fields": ["name", "status", "player"]}),
            ("File", {"fields": ["filename", "source_url", "quality", "geometry"]}),
            ("Runs", {"fields": ["past_runs"]}),
            ("Details", {"classes": ["collapse"],
                         "fields": ["last_error", "created_at", "updated_at"]}),
        ]

    def save_model(self, request, obj, form, change):
        if change:
            super().save_model(request, obj, form, change)
            return

        upload = form.cleaned_data.get("upload")
        import_file = form.cleaned_data.get("import_file")
        source_url = form.cleaned_data.get("source_url")

        if upload:
            obj.filename = video_services.save_upload(upload)
            obj.source_url = ""
            obj.status = VlmVideo.READY
            if not obj.name:
                obj.name = jobs._unique_name(obj.filename.rsplit(".", 1)[0])
            super().save_model(request, obj, form, change)
            video_services.apply_probe(obj)
            return

        if import_file:
            obj.filename = import_file
            obj.source_url = ""
            obj.status = VlmVideo.READY
            super().save_model(request, obj, form, change)
            video_services.apply_probe(obj)
            return

        # Download flow: create the row now, let the worker fetch the bytes.
        obj.source_url = source_url
        obj.filename = ""
        obj.status = VlmVideo.DOWNLOADING
        if not obj.name:
            obj.name = jobs._unique_name(source_url)
        super().save_model(request, obj, form, change)
        _queue().enqueue(jobs.download_vlm_video, obj.id,
                         job_timeout=jobs.DOWNLOAD_JOB_TIMEOUT)
        self.message_user(
            request, f"Download queued for {source_url} — refresh to see it turn ready.",
        )

    # --------------------------------------------------------------- displays
    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="geometry")
    def geometry(self, obj):
        if not obj or not obj.source_fps:
            return "—"
        parts = [f"{obj.source_fps:.3g} fps"]
        if obj.width and obj.height:
            parts.append(f"{obj.width}×{obj.height}")
        if obj.duration_seconds:
            parts.append(f"{obj.duration_seconds:.0f}s")
        return " · ".join(parts)

    @admin.display(description="runs")
    def runs_link(self, obj):
        latest = obj.runs.first()
        if latest is None:
            return "—"
        url = reverse("admin:vlm_vlmvideo_run_live") + f"?run={latest.pk}"
        return format_html('<a href="{}">{} run(s)</a>', url, obj.runs.count())

    @admin.display(description="")
    def play_link(self, obj):
        if not obj.exists():
            return "—"
        url = reverse("admin:vlm_vlmvideo_play") + f"?video={obj.pk}"
        return format_html('<a class="button" href="{}">▶ play</a>', url)

    @admin.display(description="Player")
    def player(self, obj):
        if obj is None or not obj.pk or not obj.exists():
            return "—"
        url = reverse("admin:vlm_vlmvideo_stream") + f"?video={obj.pk}"
        return format_html(
            '<video controls preload="metadata" style="max-width:640px;width:100%;'
            'border-radius:6px" src="{}"></video>',
            url,
        )

    @admin.display(description="Past runs")
    def past_runs(self, obj):
        """The runs list, here rather than as its own tab — the section has two."""
        if obj is None or not obj.pk:
            return "—"
        runs = list(obj.runs.all()[:20])
        if not runs:
            return "No runs yet — use the “Run VLM Inference…” action."
        rows = format_html_join(
            "",
            '<li><a href="{}">{} @ {} fps</a> — {} · {} alert(s) · {}</li>',
            (
                (reverse("admin:vlm_vlmvideo_run_live") + f"?run={run.pk}",
                 run.model_label_snapshot or "VLM", run.fps, run.status,
                 run.alerts_count, run.created_at.strftime("%Y-%m-%d %H:%M"))
                for run in runs
            ),
        )
        return format_html('<ul style="margin:0;padding-left:18px">{}</ul>', rows)

    # ---------------------------------------------------------------- actions
    @admin.action(description="Play selected video…")
    def play_video(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one video to play.",
                              level=messages.WARNING)
            return None
        video = queryset.first()
        if not video.exists():
            self.message_user(request, f"{video.name}: no file on disk yet.",
                              level=messages.WARNING)
            return None
        return redirect(reverse("admin:vlm_vlmvideo_play") + f"?video={video.pk}")

    @admin.action(description="Run VLM Inference…")
    def run_vlm_inference(self, request, queryset):
        """Pick a VLM and a sampling rate, then open the live screen.

        Returns a ``TemplateResponse`` in place of the changelist for the form
        step — the intermediate-page pattern ``videos.admin.run_inference`` uses
        — and redirects to the live view once the run exists.
        """
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one video to run a VLM over.",
                              level=messages.WARNING)
            return None
        video = queryset.first()
        if not video.exists():
            self.message_user(request, f"{video.name}: no file on disk yet.",
                              level=messages.WARNING)
            return None

        models = list(VlmModel.objects.all())
        if not models:
            self.message_user(
                request, "Add a VLM under Models first — there is nothing to run.",
                level=messages.WARNING,
            )
            return None

        fps_value = request.POST.get("fps") or _DEFAULT_FPS
        try:
            model_id = int(request.POST.get("vlm_model") or models[0].pk)
        except (TypeError, ValueError):
            model_id = models[0].pk
        errors = []

        if request.POST.get("apply"):
            try:
                fps = float(fps_value)
            except (TypeError, ValueError):
                fps = 0.0
            model = VlmModel.objects.filter(pk=model_id).first()

            if fps <= 0:
                errors.append("Frames per second must be greater than zero.")
            if video.source_fps and fps > video.source_fps:
                errors.append(
                    f"{video.name} only has {video.source_fps:.3g} fps — "
                    f"sampling faster than that just repeats frames."
                )
            if model is None:
                errors.append("Choose a VLM to run.")
            elif not is_cached(model.weights):
                errors.append(
                    f"{model.name}: {model.weights} is not in data/hf_cache, so it "
                    f"cannot load on this network."
                )

            if not errors:
                run = VlmRun.objects.create(
                    video=video,
                    vlm_model=model,
                    model_config_snapshot=model.config(),
                    prompt_snapshot=model.prompt,
                    model_label_snapshot=model.name,
                    fps=fps,
                    status=VlmRun.QUEUED,
                )
                _queue().enqueue(jobs.run_vlm_inference, run.id,
                                 job_timeout=jobs.VLM_JOB_TIMEOUT)
                return redirect(
                    reverse("admin:vlm_vlmvideo_run_live") + f"?run={run.pk}"
                )

        for error in errors:
            self.message_user(request, error, level=messages.ERROR)

        context = {
            **self.admin_site.each_context(request),
            "title": f"Run VLM Inference — {video.name}",
            "video": video,
            "models": models,
            "selected_model_id": model_id,
            "fps": fps_value,
            "default_fps": _DEFAULT_FPS,
            "action_name": "run_vlm_inference",
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
            "selected": [str(video.pk)],
        }
        return TemplateResponse(request, "admin/vlm/run_vlm_inference.html", context)

    # ------------------------------------------------------------------- urls
    def get_urls(self):
        custom = [
            path("play/", self.admin_site.admin_view(self.play_view),
                 name="vlm_vlmvideo_play"),
            path("stream/", self.admin_site.admin_view(self.stream_view),
                 name="vlm_vlmvideo_stream"),
            path("run-live/", self.admin_site.admin_view(self.run_live_view),
                 name="vlm_vlmvideo_run_live"),
            path("run-alerts/", self.admin_site.admin_view(self.alerts_view),
                 name="vlm_vlmvideo_run_alerts"),
            path("run-cancel/", self.admin_site.admin_view(self.cancel_view),
                 name="vlm_vlmvideo_run_cancel"),
        ]
        return custom + super().get_urls()

    def play_view(self, request):
        video = VlmVideo.objects.filter(pk=request.GET.get("video")).first()
        if video is None:
            raise Http404("unknown video")
        context = {
            **self.admin_site.each_context(request),
            "title": f"Play — {video.name}",
            "video": video,
            "exists": video.exists(),
            "stream_url": reverse("admin:vlm_vlmvideo_stream") + f"?video={video.pk}",
        }
        return TemplateResponse(request, "admin/vlm/video_player.html", context)

    def stream_view(self, request):
        video = VlmVideo.objects.filter(pk=request.GET.get("video")).first()
        if video is None or not video.exists():
            raise Http404("video file not found")
        return streaming.serve(request, video.path())

    def run_live_view(self, request):
        """The live screen: video on the left, alerts on the right.

        A finished run renders every alert server-side and the page never starts
        polling; a live one renders what exists so far and lets the JS take over
        from the last ``seq``.
        """
        run = VlmRun.objects.select_related("video").filter(
            pk=request.GET.get("run")).first()
        if run is None:
            raise Http404("unknown run")

        alerts = list(run.alerts.all())
        context = {
            **self.admin_site.each_context(request),
            "title": f"VLM — {run.video.name}",
            "run": run,
            "video": run.video,
            "alerts": alerts,
            "last_seq": alerts[-1].seq if alerts else 0,
            "is_terminal": run.is_terminal(),
            "stream_url": reverse("admin:vlm_vlmvideo_stream") + f"?video={run.video.pk}",
            "alerts_url": reverse("admin:vlm_vlmvideo_run_alerts") + f"?run={run.pk}",
            "cancel_url": reverse("admin:vlm_vlmvideo_run_cancel") + f"?run={run.pk}",
            "back_url": reverse("admin:vlm_vlmvideo_changelist"),
        }
        return TemplateResponse(request, "admin/vlm/run_live.html", context)

    def alerts_view(self, request):
        """New alerts since ``after``, and the run's progress.

        This call *is* the viewer heartbeat — the worker pauses when it stops
        arriving, so there is no separate endpoint to keep in sync with it.
        """
        run = VlmRun.objects.filter(pk=request.GET.get("run")).first()
        if run is None:
            raise Http404("unknown run")

        heartbeat.mark_viewed(run.pk)

        try:
            after = int(request.GET.get("after", 0))
        except (TypeError, ValueError):
            after = 0

        alerts = VlmAlert.objects.filter(run=run, seq__gt=after).order_by("seq")
        return JsonResponse({
            "run": {
                "status": run.status,
                "frames_processed": run.frames_processed,
                "frames_total": run.frames_total,
                "alerts_count": run.alerts_count,
                "last_error": run.last_error,
                "terminal": run.is_terminal(),
            },
            "alerts": [
                {
                    "seq": alert.seq,
                    "t": alert.video_timestamp_seconds,
                    "text": alert.text,
                    "latency_ms": alert.latency_ms,
                }
                for alert in alerts
            ],
        })

    def cancel_view(self, request):
        """Ask a running job to stop after its current frame.

        POST-only, so the CSRF middleware protects it; the live page sends the
        admin's own token.
        """
        if request.method != "POST":
            raise Http404("POST only")
        # The live page puts ?run= in the URL; accept a form field too so the
        # endpoint is usable from a plain <form> as well as from fetch().
        run_id = request.GET.get("run") or request.POST.get("run")
        run = VlmRun.objects.filter(pk=run_id).first()
        if run is None:
            raise Http404("unknown run")
        if not run.is_terminal():
            run.status = VlmRun.CANCEL_REQUESTED
            run.save(update_fields=["status"])
        return JsonResponse({"status": run.status})


@admin.register(VlmDatasetRun)
class VlmDatasetRunAdmin(admin.ModelAdmin):
    """The Dataset runs tab: every "Run on a dataset…", each opening a report.

    A tab, unlike ``VlmRun`` — a video run belongs to its video's page, but a
    dataset run has no such home, and the reason to make one is usually to
    compare it against another (this prompt vs that one, this checkpoint vs
    that one) which wants a list.

    Rows are not editable: everything on a run is either a snapshot of what it
    ran with or a measurement of what happened, and the one thing that *is*
    meant to change afterwards — the grading configuration — is changed by
    re-grading from the report, which rewrites the scored rows to match.
    """

    list_display = ["dataset_name_snapshot", "model_label_snapshot", "status_badge",
                    "progress", "accuracy_display", "errors_count",
                    "created_at", "report_link"]
    list_display_links = None
    list_filter = ["status", "grade", "vlm_model"]
    search_fields = ["dataset_name_snapshot", "model_label_snapshot", "prompt_snapshot"]
    date_hierarchy = "created_at"

    def has_add_permission(self, request):
        """Runs come from the Models tab's action, never from an add form."""
        return False

    def has_change_permission(self, request, obj=None):
        return False

    # --------------------------------------------------------------- displays
    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="images")
    def progress(self, obj):
        if obj.images_total:
            return f"{obj.images_processed}/{obj.images_total}"
        return str(obj.images_processed)

    @admin.display(description="accuracy")
    def accuracy_display(self, obj):
        """Exact-match accuracy, or why there isn't one."""
        if not obj.grade:
            return "not graded"
        accuracy = obj.accuracy()
        if accuracy is None:
            return "—"
        graded = (obj.metrics or {}).get("graded_images") or 0
        return format_html("<b>{}%</b> <span style='color:#666'>of {}</span>",
                           round(100 * accuracy, 1), graded)

    @admin.display(description="")
    def report_link(self, obj):
        url = reverse("admin:vlm_vlmdatasetrun_report") + f"?run={obj.pk}"
        return format_html('<a class="button" href="{}">open report</a>', url)

    # ------------------------------------------------------------------- urls
    def get_urls(self):
        custom = [
            path("report/", self.admin_site.admin_view(self.report_view),
                 name="vlm_vlmdatasetrun_report"),
            path("report-progress/", self.admin_site.admin_view(self.progress_view),
                 name="vlm_vlmdatasetrun_progress"),
            path("report-cancel/", self.admin_site.admin_view(self.cancel_view),
                 name="vlm_vlmdatasetrun_cancel"),
            path("report-regrade/", self.admin_site.admin_view(self.regrade_view),
                 name="vlm_vlmdatasetrun_regrade"),
            path("report-image/", self.admin_site.admin_view(self.image_view),
                 name="vlm_vlmdatasetrun_image"),
        ]
        return custom + super().get_urls()

    def _get_run(self, request) -> VlmDatasetRun:
        run = VlmDatasetRun.objects.select_related("dataset", "vlm_model").filter(
            pk=request.GET.get("run") or request.POST.get("run")).first()
        if run is None:
            raise Http404("unknown dataset run")
        return run

    def report_view(self, request):
        """The run's report: configuration, progress, analytics, answers.

        A finished run renders entirely server-side; a live one renders what
        exists so far and lets the JS extend it from the last ``seq``. Unlike
        the video live screen this poll is *not* a heartbeat — a dataset run
        keeps going whether or not anyone is looking at it.
        """
        run = self._get_run(request)
        only_mismatches = request.GET.get("only") == "mismatches"

        results = run.results.all()
        if only_mismatches:
            results = results.filter(matched=False)
        rows = list(results[:_REPORT_ROW_LIMIT])
        shown = len(rows)
        total_rows = results.count()

        metrics = run.metrics or {}
        context = {
            **self.admin_site.each_context(request),
            "title": f"VLM on {run.dataset_name_snapshot}",
            "run": run,
            "rows": rows,
            "shown": shown,
            "total_rows": total_rows,
            "row_limit": _REPORT_ROW_LIMIT,
            "only_mismatches": only_mismatches,
            # The mismatch view is a filtered slice, so the live poll must not
            # append to it — a new answer that matched does not belong there.
            "live_append": not only_mismatches,
            "last_seq": rows[-1].seq if rows and not only_mismatches else 0,
            "is_terminal": run.is_terminal(),
            "metrics": metrics,
            "confusion": _confusion_table(metrics.get("confusion")),
            "alias_rows": [
                {"name": name,
                 "aliases": ", ".join((run.alias_map or {}).get(name) or [])}
                for name in (run.classes_snapshot or [])
            ],
            "progress_url": reverse("admin:vlm_vlmdatasetrun_progress") + f"?run={run.pk}",
            "cancel_url": reverse("admin:vlm_vlmdatasetrun_cancel") + f"?run={run.pk}",
            "regrade_url": reverse("admin:vlm_vlmdatasetrun_regrade") + f"?run={run.pk}",
            "image_url": reverse("admin:vlm_vlmdatasetrun_image"),
            "report_url": reverse("admin:vlm_vlmdatasetrun_report") + f"?run={run.pk}",
            "back_url": reverse("admin:vlm_vlmdatasetrun_changelist"),
        }
        return TemplateResponse(request, "admin/vlm/dataset_run.html", context)

    def progress_view(self, request):
        """Counters, headline scores, and the answers since ``after``."""
        run = self._get_run(request)
        try:
            after = int(request.GET.get("after", 0))
        except (TypeError, ValueError):
            after = 0

        rows = run.results.filter(seq__gt=after).order_by("seq")[:_REPORT_ROW_LIMIT]
        metrics = run.metrics or {}
        return JsonResponse({
            "run": {
                "status": run.status,
                "images_processed": run.images_processed,
                "images_total": run.images_total,
                "errors_count": run.errors_count,
                "last_error": run.last_error,
                "terminal": run.is_terminal(),
                "graded_images": metrics.get("graded_images"),
                "exact_matches": metrics.get("exact_matches"),
                "exact_match_accuracy": metrics.get("exact_match_accuracy"),
            },
            "results": [
                {
                    "pk": row.pk,
                    "seq": row.seq,
                    "image": row.image_filename,
                    "text": row.text,
                    "error": row.error,
                    "latency_ms": row.latency_ms,
                    "has_label": row.has_label,
                    "gt": row.gt_classes,
                    "predicted": row.predicted_classes,
                    "matched": row.matched,
                }
                for row in rows
            ],
        })

    def cancel_view(self, request):
        """Ask a running job to stop after its current image.

        POST-only so the CSRF middleware protects it; the report posts a plain
        form rather than fetch(), since there is nothing to keep on the page.
        """
        if request.method != "POST":
            raise Http404("POST only")
        run = self._get_run(request)
        if not run.is_terminal():
            run.status = VlmDatasetRun.CANCEL_REQUESTED
            run.save(update_fields=["status"])
            self.message_user(request, "Stopping after the current image.")
        return redirect(reverse("admin:vlm_vlmdatasetrun_report") + f"?run={run.pk}")

    def regrade_view(self, request):
        """Re-score a run's stored answers under an edited alias table.

        The GPU work is already done and the answers do not change, so this is
        the cheap half of a run — which is exactly why the aliases are edited
        here, after seeing what the model actually says, rather than guessed at
        before it ran.
        """
        if request.method != "POST":
            raise Http404("POST only")
        run = self._get_run(request)

        if not run.classes_snapshot:
            self.message_user(
                request,
                "This run has no classes.txt snapshot, so there is nothing to "
                "grade against. Re-run it over a dataset that has one.",
                level=messages.ERROR,
            )
            return redirect(reverse("admin:vlm_vlmdatasetrun_report") + f"?run={run.pk}")

        alias_map = {}
        for name in run.classes_snapshot:
            raw = request.POST.get(f"alias__{name}", "")
            alias_map[name] = [term.strip() for term in raw.split(",") if term.strip()]

        metrics = dataset_runner.regrade(
            run, alias_map=alias_map,
            negation=bool(request.POST.get("negation_aware")),
        )
        graded = metrics.get("graded_images") or 0
        accuracy = metrics.get("exact_match_accuracy")
        self.message_user(
            request,
            f"Re-graded {graded} labeled image(s)" + (
                f" — exact-match accuracy {round(100 * accuracy, 1)}%."
                if accuracy is not None else "."
            ),
        )
        return redirect(reverse("admin:vlm_vlmdatasetrun_report") + f"?run={run.pk}")

    def image_view(self, request):
        """Stream one result's source image — with ``?labels=1``, annotated.

        Resolved by result row rather than by an index into the dataset, so a
        page of 300 thumbnails costs 300 stats instead of 300 directory
        listings, and the containment check is what keeps a stored filename
        from addressing anything outside its own dataset.

        The report's thumbnails ask for the plain file and the click-through
        asks for the annotated copy, so the "what was actually labeled here?"
        question a mismatch raises is one click away, without paying to decode
        and redraw 300 images per page load.
        """
        result = VlmDatasetResult.objects.select_related("run").filter(
            pk=request.GET.get("result")).first()
        if result is None:
            raise Http404("unknown result")

        directory = dataset_runner.dataset_dir(result.run.dataset_name_snapshot)
        image_dir = lsapi.image_source_dir(directory)
        candidate = (image_dir / Path(result.image_filename).name).resolve()
        try:
            candidate.relative_to(image_dir.resolve())
        except ValueError:
            raise Http404("image outside its dataset")
        if not candidate.is_file():
            raise Http404("image not found on disk")

        if not request.GET.get("labels"):
            return FileResponse(open(candidate, "rb"))

        # Shapes are re-read from the label file rather than taken off the
        # result row: the row stores which classes were labeled, which is all
        # the scoring needs, but drawing needs the geometry too.
        shapes = datasets_svc.label_shapes(
            directory / datasets_svc.LABELS_SUBDIR,
            candidate.name,
            result.run.classes_snapshot or [],
        )
        if not shapes:
            # Nothing annotated (or nothing to annotate with) — the plain image
            # is the honest answer, not an error.
            return FileResponse(open(candidate, "rb"))
        try:
            rendered = label_render.render(candidate, shapes)
        except RuntimeError as exc:
            raise Http404(str(exc))
        return HttpResponse(rendered, content_type="image/jpeg")
