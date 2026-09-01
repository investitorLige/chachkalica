"""Admin = the training operator console.

The Experiment page composes a friendy_chachkalica config from inline dataset/model
rows; the "Generate config & create run" action validates the roster, writes the
YAML, and records a :class:`TrainingRun`. Executing that run against the
trainer service is a later phase — for now the action ends at a generated,
ready-to-run config.
"""

import json
import re
from pathlib import Path

import django_rq
from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.core import signing
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils import timezone
from django.utils.html import format_html_join
from django.utils.http import urlencode

from fleet.admin import _status_badge
from fleet.models import Annotator, Dataset
from fleet.services import lsapi
from fleet.services.paths import source_root
from training import jobs
from training.models import (
    BuildNode,
    EvalRun,
    Experiment,
    ExperimentDataset,
    ExperimentModel,
    ExportRun,
    FineTuningRun,
    RunResult,
    TrainedModel,
    TrainingRun,
    TrainingSettings,
)
from training import model_specs, pipelines
from training.forms import ExperimentModelForm
from training.services import (
    buildnode, combine, config_gen, exports, extend, finetune, ingest, pipeline_meta, promote,
    runner, teardown,
)


_HARD_IMAGE_TOKEN_SALT = "training.hard-image-path"


def _queue():
    return django_rq.get_queue("default")


def _models_title(models) -> str:
    return " + ".join(m.name for m in models)


def _hard_image_token(image_path):
    return signing.dumps(str(image_path), salt=_HARD_IMAGE_TOKEN_SALT, compress=True)


def _hard_image_path_from_request(request, images):
    """Resolve an immutable signed image token, with index fallback for old pages."""
    token = request.GET.get("image_token")
    if token:
        try:
            return signing.loads(
                token,
                salt=_HARD_IMAGE_TOKEN_SALT,
                max_age=3600,
            )
        except signing.BadSignature as exc:
            raise Http404("invalid or expired image token") from exc

    index = _preview_index(request, len(images))
    raw_path = images[index].get("image_path")
    if not raw_path:
        raise Http404("no image path recorded")
    return raw_path


@admin.register(TrainingSettings)
class TrainingSettingsAdmin(admin.ModelAdmin):
    list_display = ["__str__", "configs_root", "runs_root", "exports_root",
                    "default_device", "service_base_url"]

    def has_add_permission(self, request):
        return not TrainingSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


class ExperimentDatasetInline(admin.TabularInline):
    model = ExperimentDataset
    extra = 1
    autocomplete_fields = ["dataset", "annotator"]
    fields = [
        "dataset", "role", "label_source", "annotator", "explicit_labels_path",
        "aug_hflip", "aug_hflip_fraction", "aug_scale_crop", "aug_scale_crop_fraction",
    ]

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        # Tabular inlines squeeze help_text into a 10px hover-only icon in the
        # column header, which reads as missing. Mirror it onto the widget so
        # hovering the checkbox/input itself shows the tooltip too.
        field = super().formfield_for_dbfield(db_field, request, **kwargs)
        if field is not None and field.help_text:
            field.widget.attrs.setdefault("title", field.help_text)
        return field

    class Media:
        # Shows each augmentation's fraction input only while its checkbox is
        # ticked (train rows). Pure progressive enhancement — with JS off the
        # inputs stay visible and model.clean() still validates them.
        js = ("training/experiment_dataset_aug.js",)


class ExperimentModelInline(admin.StackedInline):
    # Stacked (not tabular) so each model gets a vertical form: the form exposes
    # a field per builder option of every arch (size/variant, thresholds, backbone
    # weights …) and JS hides the ones not matching the selected arch. num_classes
    # stays optional — blank means "auto", resolved per train dataset.
    model = ExperimentModel
    form = ExperimentModelForm
    extra = 1

    def get_fields(self, request, obj=None):
        # arch first, then every arch's builder-option widgets (JS shows only the
        # selected arch's), then the per-arch pretrained-weights dropdowns and the
        # shared custom path/URL field, then the shared knobs. All these fields are
        # declared on the form, so listing them here is safe for the inline
        # formset factory.
        return [
            "arch",
            *model_specs.spec_field_names(),
            *model_specs.weights_field_names(),
            "weights_custom",
            "num_classes",
            "params",
        ]

    class Media:
        js = ("training/experiment_model_form.js",)


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ["name", "dataset_count", "model_count", "epochs", "device", "updated_at"]
    search_fields = ["name", "description"]
    inlines = [ExperimentDatasetInline, ExperimentModelInline]
    actions = ["generate_run"]
    fieldsets = [
        (None, {"fields": ["name", "description"]}),
        (
            "Training",
            {
                "fields": [
                    "epochs", "batch_size", "num_workers", "device", "seed",
                    "amp", "gradient_clip_norm", "early_stopping_patience",
                    "best_metric", "val_interval",
                    "optimizer_name", "lr", "weight_decay", "optimizer_params",
                    "scheduler_name", "scheduler_params",
                ]
            },
        ),
        (
            "Evaluation",
            {"fields": ["eval_batch_size", "eval_num_workers", "eval_score_threshold",
                        "eval_operating_nms_threshold", "iou_thresholds"]},
        ),
        (
            "Training / Eval pipeline",
            {
                "fields": ["pipeline", "detector_checkpoint", "detector_expand_ratio",
                           "detector_min_box_size",
                           "tile_size_px", "tile_width_pct", "tile_height_pct", "overlap",
                           "merge_nms_iou"],
                "description": "Optionally run train, val, and test through a chachak "
                               "pipeline. Only the selected pipeline's fields apply.",
            },
        ),
    ]

    class Media:
        js = ("training/experiment_pipeline_form.js",)

    def get_queryset(self, request):
        # Fine-tune experiments are auto-generated bookkeeping for the "Fine-tune
        # model…" action on Trained models (see training.services.finetune) — an
        # operator never builds or edits one by hand, so this tab stays about the
        # configs they actually author. Its TrainingRun still shows up, just under
        # "Fine-tuning runs" instead of "Training runs" (see FineTuningRunAdmin).
        return super().get_queryset(request).filter(is_finetune=False)

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        # Offer only pipelines supported end-to-end today (train + val + test):
        # blank = full-frame, plus every pipeline in TRAINABLE_PIPELINES. `chain`
        # stays hidden (no single train-time transform).
        field = super().formfield_for_dbfield(db_field, request, **kwargs)
        if db_field.name == "pipeline" and field is not None:
            field.choices = [("", "---------"), *pipelines.EXPERIMENT_PIPELINE_CHOICES]
        return field

    @admin.display(description="datasets")
    def dataset_count(self, obj):
        return obj.datasets.count()

    @admin.display(description="models")
    def model_count(self, obj):
        return obj.models.count()

    @admin.action(description="Generate config & create run…")
    def generate_run(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one experiment.", level=messages.WARNING)
            return None
        experiment = queryset.first()

        # Validate the roster up front so both the preview and the apply path
        # surface the same friendly error.
        ts = TrainingSettings.load()
        try:
            if request.POST.get("apply"):
                run = TrainingRun.objects.create(experiment=experiment)
                yaml_path, _text = config_gen.write_config(experiment, run)
                _queue().enqueue(jobs.run_training, run.pk, job_timeout=jobs.JOB_TIMEOUT)
                run.status = TrainingRun.QUEUED
                run.save(update_fields=["status"])
                self.message_user(
                    request,
                    f"Run #{run.pk} queued — config written to {yaml_path}. "
                    "Watch the Training runs page for progress.",
                )
                return None
            provisional_output = config_gen._resolve(ts.runs_root) / f"{experiment.name}-<run id>"
            yaml_text = config_gen.build_yaml(experiment, provisional_output)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            self.message_user(request, f"Cannot generate config: {exc}", level=messages.ERROR)
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": "Generate training config",
            "experiment": experiment,
            "datasets": list(experiment.datasets.all()),
            "models": list(experiment.models.all()),
            "yaml_text": yaml_text,
            "configs_root": config_gen._resolve(ts.configs_root),
            "action": "generate_run",
            "selected": [str(experiment.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/launch_run.html", context)


class RunResultInline(admin.TabularInline):
    model = RunResult
    extra = 0
    can_delete = False
    fields = ["run_name", "model_arch", "train_dataset_name", "map50", "map50_95",
              "best_epoch", "best_checkpoint", "error"]
    readonly_fields = fields

    def has_add_permission(self, request, obj=None):
        return False

    @admin.display(description="mAP50")
    def map50(self, obj):
        return obj.metric("map50")

    @admin.display(description="mAP50-95")
    def map50_95(self, obj):
        return obj.metric("map50_95")


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ["__str__", "experiment", "status_badge", "config_yaml_path", "created_at"]
    list_filter = ["status", "experiment"]
    inlines = [RunResultInline]
    actions = [
        "launch_selected", "resume_selected", "extend_run", "pause_selected",
        "live_training_report", "ingest_selected", "reconcile_selected", "kill_run_gracefully",
    ]
    readonly_fields = [
        "experiment", "status", "epoch_progress", "config_yaml_path", "output_dir",
        "last_error", "started_at", "finished_at", "results", "created_at",
    ]

    def has_add_permission(self, request):
        # Runs are created by the Experiment action, not by hand.
        return False

    def get_queryset(self, request):
        # Fine-tune runs live under their own "Fine-tuning runs" tab (see
        # FineTuningRunAdmin below) so they don't clutter this one — same split
        # as ExperimentAdmin.get_queryset.
        return super().get_queryset(request).filter(experiment__is_finetune=False)

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="Epoch progress")
    def epoch_progress(self, obj):
        """Per-epoch lines read live from each run's history.yaml on disk.

        Reflects training as it happens (history.yaml is rewritten each epoch);
        reload the page to refresh. One block per internal run, since a run fans
        out into several train-dataset × model pairings.
        """
        from training.services import progress

        histories = progress.run_histories(obj.output_dir)
        if not histories:
            return "No epoch history yet — reload while the run is training."
        return format_html_join(
            "",
            "<div style='margin-bottom:1em'><strong>{}</strong>"
            "<pre style='max-height:24em;overflow:auto;margin:.3em 0;padding:.6em;"
            "background:#1e1e1e;color:#d4d4d4;border-radius:4px;font-size:12px;"
            "line-height:1.6'>{}</pre></div>",
            (
                (h["run_name"], "\n".join(progress.epoch_line(e) for e in h["epochs"]))
                for h in histories
            ),
        )

    def _url_name(self, suffix: str) -> str:
        """A URL name scoped to this ModelAdmin's own model.

        FineTuningRunAdmin subclasses this admin over the FineTuningRun proxy
        (see training.models.FineTuningRun) rather than overriding get_urls, so
        deriving the name from ``self.model`` — the same trick Django's own
        admin uses for its changelist/add/change routes — is what keeps its
        routes (``training_finetuningrun_live_report`` etc.) distinct from
        TrainingRunAdmin's, instead of two ``path()`` entries silently sharing
        one name and ``reverse()`` picking whichever was registered last.
        """
        opts = self.model._meta
        return f"{opts.app_label}_{opts.model_name}_{suffix}"

    def get_urls(self):
        custom = [
            path("report/", self.admin_site.admin_view(self.live_report_view),
                 name=self._url_name("live_report")),
            path("report/metrics/", self.admin_site.admin_view(self.live_report_metrics),
                 name=self._url_name("live_report_metrics")),
            path("hard-images/image/", self.admin_site.admin_view(self.hard_images_image),
                 name=self._url_name("hard_images_image")),
            path("hard-images/data/", self.admin_site.admin_view(self.hard_images_data),
                 name=self._url_name("hard_images_data")),
        ]
        return custom + super().get_urls()

    @admin.action(description="Live training report")
    def live_training_report(self, request, queryset):
        """Open the combined live report: epoch status, metric charts, hard images."""
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one training run.", level=messages.WARNING)
            return None
        run = queryset.first()
        return redirect(reverse(f"admin:{self._url_name('live_report')}") + "?run=" + str(run.pk))

    def _hard_image_artifacts(self, run):
        """Return available ``val_hard_images.json`` files for a TrainingRun."""
        root = Path(run.output_dir) if run.output_dir else None
        if not root or not root.is_dir():
            return []
        artifacts = []
        for path_ in sorted(root.glob("*/val_hard_images.json")):
            try:
                payload = json.loads(path_.read_text())
            except (OSError, ValueError):
                continue
            images = payload.get("images", []) if isinstance(payload, dict) else []
            artifacts.append({
                "run_name": path_.parent.name,
                "path": path_,
                "payload": payload,
                "count": len(images),
            })
        return artifacts

    def _selected_hard_image_artifact(self, run, run_name=None):
        artifacts = self._hard_image_artifacts(run)
        if not artifacts:
            return None
        if run_name:
            for artifact in artifacts:
                if artifact["run_name"] == run_name:
                    return artifact
            return None
        return artifacts[0]

    def _report_data(self, run):
        """Reshape a run's live state for the report page + poll endpoint.

        Unions the internal runs that have per-epoch ``history.yaml`` with those
        that have a ``val_hard_images.json`` artifact (a run may have one before
        the other), so the picker lists every internal run either surfaces.
        """
        from training.services import progress

        histories = {h["run_name"]: h["epochs"] for h in progress.run_histories(run.output_dir)}
        artifacts = {a["run_name"]: a for a in self._hard_image_artifacts(run)}
        runs = []
        for name in sorted(set(histories) | set(artifacts)):
            artifact = artifacts.get(name)
            runs.append({
                "run_name": name,
                "epochs": [progress.flatten_epoch(e) for e in histories.get(name, [])],
                "has_hard_images": artifact is not None,
                "hard_image_count": artifact["count"] if artifact else 0,
                "query": urlencode({"run": run.pk, "run_name": name}),
            })
        return {
            "status": run.status,
            "total_epochs": run.experiment.epochs if run.experiment else None,
            "runs": runs,
        }

    def live_report_view(self, request):
        """Render the combined live training report for one TrainingRun."""
        run = TrainingRun.objects.filter(pk=request.GET.get("run")).first()
        if run is None:
            raise Http404("live report requires ?run=")
        # Ranking metric / thresholds are shared across a run's hard-image
        # artifacts, so pull them from whichever one exists (if any) for the
        # viewer's caption.
        artifacts = self._hard_image_artifacts(run)
        payload = artifacts[0]["payload"] if artifacts else {}
        context = {
            **self.admin_site.each_context(request),
            "title": f"Live training report - run #{run.pk}",
            "run": run,
            "run_id": run.pk,
            "status_badge_html": _status_badge(run.status),
            "report_json": self._report_data(run),
            "metric": payload.get("metric", ""),
            "metric_description": payload.get("metric_description", ""),
            "iou_threshold": payload.get("iou_threshold"),
            "score_threshold": payload.get("score_threshold"),
            "operating_nms_threshold": payload.get("operating_nms_threshold"),
            "max_display_predictions": payload.get("max_display_predictions"),
            "metrics_url": reverse(f"admin:{self._url_name('live_report_metrics')}"),
            "hard_image_url": reverse(f"admin:{self._url_name('hard_images_image')}"),
            "hard_image_data_url": reverse(f"admin:{self._url_name('hard_images_data')}"),
            "back_label": "Back to training runs",
        }
        return TemplateResponse(request, "admin/training/live_report.html", context)

    def live_report_metrics(self, request):
        """JSON poll target feeding the live report's charts, table and status."""
        run = TrainingRun.objects.filter(pk=request.GET.get("run")).first()
        if run is None:
            return JsonResponse({"error": "unknown training run"}, status=400)
        return JsonResponse(self._report_data(run))

    def hard_images_image(self, request):
        """Stream one live hard image for a TrainingRun."""
        run = TrainingRun.objects.filter(pk=request.GET.get("run")).first()
        if run is None:
            raise Http404("unknown training run")
        artifact = self._selected_hard_image_artifact(run, request.GET.get("run_name"))
        images = artifact["payload"].get("images", []) if artifact else []
        raw_path = _hard_image_path_from_request(request, images)
        image_path = Path(raw_path).resolve()
        root = source_root().resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise Http404("image not found under source root")
        return FileResponse(open(image_path, "rb"))

    def hard_images_data(self, request):
        """Return one live hard-image payload entry for a TrainingRun."""
        run = TrainingRun.objects.filter(pk=request.GET.get("run")).first()
        if run is None:
            return JsonResponse({"error": "unknown training run"}, status=400)
        artifact = self._selected_hard_image_artifact(run, request.GET.get("run_name"))
        images = artifact["payload"].get("images", []) if artifact else []
        if not images:
            return JsonResponse({"error": "no hard images for this training run"}, status=400)
        index = _preview_index(request, len(images))
        entry = images[index]
        return JsonResponse({
            "predictions": entry.get("predictions", []),
            "ground_truth": entry.get("ground_truth", []),
            "image": entry.get("image_name", ""),
            "image_token": _hard_image_token(entry.get("image_path", "")),
            "difficulty": entry.get("difficulty"),
            "precision": entry.get("precision"),
            "recall": entry.get("recall"),
            "f1": entry.get("f1"),
            "missed": entry.get("missed"),
            "false_positives": entry.get("false_positives"),
            "wrong_class": entry.get("wrong_class"),
            "loc_error": entry.get("loc_error"),
            "localization_penalty": entry.get("localization_penalty"),
            "total_errors": entry.get("total_errors"),
            "num_predictions": entry.get("num_predictions"),
            "num_ground_truth": entry.get("num_ground_truth"),
            "index": index,
            "count": len(images),
        })

    @admin.action(description="Launch / relaunch on trainer service")
    def launch_selected(self, request, queryset):
        queue = _queue()
        prev_job = None  # chain so a multi-select launch doesn't race the trainer's single slot
        for run in queryset:
            if not run.config_yaml_path:
                self.message_user(request, f"Run #{run.pk} has no config; skipped.",
                                  level=messages.WARNING)
                continue
            prev_job = queue.enqueue(
                jobs.run_training, run.pk,
                depends_on=jobs.depends_on(prev_job), job_timeout=jobs.JOB_TIMEOUT,
            )
            run.status = TrainingRun.QUEUED
            run.save(update_fields=["status"])
        self.message_user(request, "Launch job(s) queued — refresh to see progress.")

    @admin.action(description="Resume from checkpoint (last.pt)")
    def resume_selected(self, request, queryset):
        """Relaunch, passing resume=True so the trainer picks up each sub-run's
        last.pt instead of retraining from scratch.

        Meant for runs stranded at ``running``/``queued`` by a trainer restart,
        or for a ``paused`` run — run "Reconcile status from trainer / disk"
        first so the row reflects reality before resuming a stranded run.
        """
        queue = _queue()
        prev_job = None  # chain so a multi-select resume doesn't race the trainer's single slot
        for run in queryset:
            if not run.config_yaml_path:
                self.message_user(request, f"Run #{run.pk} has no config; skipped.",
                                  level=messages.WARNING)
                continue
            prev_job = queue.enqueue(
                jobs.run_training, run.pk, resume=True,
                depends_on=jobs.depends_on(prev_job), job_timeout=jobs.JOB_TIMEOUT,
            )
            run.status = TrainingRun.QUEUED
            run.save(update_fields=["status"])
        self.message_user(request, "Resume job(s) queued — refresh to see progress.")

    @admin.action(description="Extend run (more epochs)…")
    def extend_run(self, request, queryset):
        """Give a finished run more epochs (and optionally a new lr), then resume it.

        A plain "Resume from checkpoint" on an already-finished run is a no-op:
        its config still says the original epoch count, and its output dir
        already has a ``result.yaml`` the trainer treats as "done, skip". This
        bumps the source Experiment's ``epochs``/``lr``, rewrites the run's
        config, moves the stale ``result.yaml`` aside (see
        ``training.services.extend``), and queues the same resume job
        "Resume from checkpoint" does — one click instead of three.
        """
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one run to extend.",
                              level=messages.WARNING)
            return None
        run = queryset.first()
        if run.status not in (TrainingRun.OK, TrainingRun.ERROR, TrainingRun.PAUSED):
            self.message_user(
                request,
                f"Run #{run.pk} is {run.status}; extend only a finished, errored, or "
                "paused run.",
                level=messages.WARNING,
            )
            return None

        if request.POST.get("apply"):
            additional_epochs = _int_or_none(request.POST.get("additional_epochs"))
            if not additional_epochs or additional_epochs < 1:
                self.message_user(request, "Enter how many more epochs to train.",
                                  level=messages.WARNING)
                return None
            new_lr = _float_or_none(request.POST.get("new_lr"))
            try:
                extend.extend_run(run, additional_epochs, new_lr)
            except ValueError as exc:
                self.message_user(request, f"Cannot extend run #{run.pk}: {exc}",
                                  level=messages.ERROR)
                return None

            queue = _queue()
            queue.enqueue(jobs.run_training, run.pk, resume=True, job_timeout=jobs.JOB_TIMEOUT)
            run.status = TrainingRun.QUEUED
            run.save(update_fields=["status"])
            self.message_user(
                request,
                f"Run #{run.pk} extended by {additional_epochs} epoch(s)"
                + (f" at lr={new_lr}" if new_lr is not None else "")
                + " and queued to resume.",
            )
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": f"Extend run #{run.pk}",
            "run": run,
            "current_epochs": run.experiment.epochs,
            "current_lr": run.experiment.lr,
            "action": "extend_run",
            "selected": [str(run.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/extend_run.html", context)

    @admin.action(description="Pause run (stop, keep row for later resume)")
    def pause_selected(self, request, queryset):
        """Stop the trainer process but leave the run's files/DB row intact.

        Sets the row to ``paused`` *before* asking the trainer to stop, so the
        poller in ``jobs.run_training`` (which checks the DB status each loop
        iteration) sees the pause and exits cleanly instead of racing the
        trainer's own post-kill status and marking the run ``error``. Resume
        later with "Resume from checkpoint (last.pt)".
        """
        for run in queryset:
            if run.status not in (TrainingRun.RUNNING, TrainingRun.QUEUED):
                self.message_user(
                    request, f"Run #{run.pk} is {run.status}, not running/queued; skipped.",
                    level=messages.WARNING,
                )
                continue
            run.status = TrainingRun.PAUSED
            run.save(update_fields=["status"])
            try:
                result = runner.stop(run)
                outcome = result.get("outcome") or result.get("status")
            except Exception as exc:  # noqa: BLE001 - row is already paused either way
                self.message_user(request, f"Run #{run.pk}: paused, but stop request failed: {exc}",
                                  level=messages.WARNING)
                continue
            self.message_user(request, f"Run #{run.pk}: paused ({outcome}).")

    @admin.action(description="Ingest results from output dir (no rerun)")
    def ingest_selected(self, request, queryset):
        from training.services import ingest

        for run in queryset:
            if not run.output_dir or not ingest.is_complete(run.output_dir):
                self.message_user(request, f"Run #{run.pk}: no finished output to ingest.",
                                  level=messages.WARNING)
                continue
            summary = ingest.ingest_run(run)
            run.status = TrainingRun.OK
            run.save(update_fields=["status"])
            self.message_user(request, f"Run #{run.pk}: ingested {summary['run_results']} result(s).")

    @admin.action(description="Reconcile status from trainer / disk")
    def reconcile_selected(self, request, queryset):
        from training.services import reconcile

        for run in queryset:
            outcome = reconcile.reconcile_run(run)
            self.message_user(request, f"Run #{run.pk}: {outcome}")

    @admin.action(description="Kill run gracefully (stop, delete run + files)")
    def kill_run_gracefully(self, request, queryset):
        """Stop the training process, then delete the run's files and DB row.

        Two-step: the first click shows a confirm page (with a warning for any
        promoted models whose checkpoints would be removed); the form re-POSTs
        with ``apply=1`` to actually tear down.
        """
        if request.POST.get("apply"):
            for run in queryset:
                outcome = teardown.kill_run(run)
                self.message_user(
                    request,
                    f"Run #{outcome['run_id']}: stopped ({outcome['stopped']}); "
                    f"removed {len(outcome['removed_paths'])} path(s); row deleted.",
                    level=messages.WARNING if outcome["errors"] else messages.SUCCESS,
                )
                for err in outcome["errors"]:
                    self.message_user(request, f"Run #{outcome['run_id']}: {err}",
                                      level=messages.ERROR)
            return None

        runs = list(queryset)
        affected = [
            {"run": run,
             "models": list(TrainedModel.objects.filter(source_run_result__run=run))}
            for run in runs
        ]
        context = {
            **self.admin_site.each_context(request),
            "title": "Kill training run(s) gracefully",
            "affected": affected,
            "action": "kill_run_gracefully",
            "selected": [str(run.pk) for run in runs],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/kill_run.html", context)


@admin.register(FineTuningRun)
class FineTuningRunAdmin(TrainingRunAdmin):
    """"Fine-tuning runs" tab — the same epoch/status/live-report machinery as
    TrainingRunAdmin, scoped to runs created by TrainedModelAdmin.fine_tune.

    A plain subclass over the FineTuningRun proxy (see training.models): every
    action, inline, and the live report view/urls (see TrainingRunAdmin._url_name)
    come along unchanged, just filtered to this one queryset — the whole point
    is that a fine-tune run gets identical visibility (epochs, metrics, live
    report) without also cluttering the ordinary Training runs tab.
    """

    list_display = ["__str__", "finetune_source", "status_badge", "config_yaml_path", "created_at"]

    def get_queryset(self, request):
        return TrainingRun.objects.filter(experiment__is_finetune=True)

    @admin.display(description="fine-tuned from")
    def finetune_source(self, obj):
        return obj.experiment.finetune_source if obj.experiment_id else None


@admin.register(RunResult)
class RunResultAdmin(admin.ModelAdmin):
    list_display = ["run_name", "run", "model_arch", "train_dataset_name",
                    "map50", "map50_95", "best_epoch", "error"]
    list_filter = ["model_arch", "train_dataset_name"]
    search_fields = ["run_name", "train_dataset_name"]
    actions = ["show_best_epoch_stats", "promote_selected"]

    def has_add_permission(self, request):
        return False

    @admin.display(description="mAP50")
    def map50(self, obj):
        return obj.metric("map50")

    @admin.display(description="mAP50-95")
    def map50_95(self, obj):
        return obj.metric("map50_95")

    @admin.action(description="Show best-epoch statistics")
    def show_best_epoch_stats(self, request, queryset):
        import yaml

        from training.services import progress

        if queryset.count() != 1:
            self.message_user(request, "Select exactly one result.", level=messages.WARNING)
            return None
        rr = queryset.first()
        entry = progress.best_epoch_entry(rr.run_dir, rr.best_epoch)

        def _dump(value):
            return yaml.safe_dump(value, sort_keys=False, default_flow_style=False) if value else ""

        context = {
            **self.admin_site.each_context(request),
            "title": f"Best-epoch statistics — {rr.run_name}",
            "rr": rr,
            "entry": entry,
            "entry_yaml": _dump(entry),
            "val_yaml": _dump(rr.val_metrics),
            "test_yaml": _dump(rr.test_metrics),
        }
        return TemplateResponse(request, "admin/training/best_epoch_stats.html", context)

    @admin.action(description="Promote to model registry")
    def promote_selected(self, request, queryset):
        promoted = 0
        for rr in queryset:
            try:
                tm = promote.promote_run_result(rr)
            except ValueError as exc:
                self.message_user(request, f"{rr.run_name}: {exc}", level=messages.WARNING)
                continue
            promoted += 1
            self.message_user(request, f"Registered model {tm.name!r}.")
        if promoted:
            self.message_user(request, f"Promoted {promoted} model(s) — see Trained models.")


def _float_or_none(raw):
    raw = (raw or "").strip()
    return float(raw) if raw else None


def _int_or_none(raw):
    raw = (raw or "").strip()
    return int(float(raw)) if raw else None


def _preview_index(request, count):
    """Parse a 0-based ``?index=`` and bound it to ``[0, count)`` or 404."""
    if count <= 0:
        raise Http404("dataset has no images")
    try:
        index = int(request.GET.get("index", 0))
    except (TypeError, ValueError):
        raise Http404("bad index")
    if not 0 <= index < count:
        raise Http404("index out of range")
    return index


def _preview_label_dir(request, dataset):
    """Resolve the ground-truth label dir from the preview query, or None.

    Returns None when no label source was chosen or the resolved dir is missing,
    so the viewer can disable the GT toggle instead of erroring.
    """
    label_source = request.GET.get("label_source") or ""
    if not label_source:
        return None
    annotator = Annotator.objects.filter(pk=request.GET.get("annotator")).first()
    try:
        label_dir = config_gen.resolve_label_dir(
            dataset, label_source, annotator,
            (request.GET.get("explicit_labels_path") or "").strip())
    except (ValueError, FileNotFoundError):
        return None
    return label_dir if Path(label_dir).is_dir() else None


def _read_gt_boxes(label_dir, image_path, class_names):
    """Parse a YOLO ``<stem>.txt`` into normalized center-xywh GT box dicts.

    Matches the pairing Friendy uses (``stem.txt`` or ``image.jpg.txt``); lines
    are ``class_id cx cy w h`` already normalized, so they draw on the same path
    as predictions. Extra values (polygons/OBB) beyond the first five are ignored.
    """
    label_dir = Path(label_dir)
    candidates = [label_dir / f"{image_path.stem}.txt", label_dir / f"{image_path.name}.txt"]
    label_file = next((c for c in candidates if c.exists()), None)
    if label_file is None:
        return []
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        boxes.append({"cx": cx, "cy": cy, "w": w, "h": h,
                      "class_id": class_id, "class_name": name})
    return boxes


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ["name", "stage", "arch", "num_classes", "map50", "map50_95", "created_at"]
    list_filter = ["stage", "arch"]
    search_fields = ["name", "description"]
    actions = [
        "evaluate", "fine_tune", "preview_on_dataset", "export_onnx", "export_trt",
        "export_pt_bundle", "view_hard_val_images",
    ]
    readonly_fields = ["source_run_result", "parent_model", "created_at", "updated_at"]

    @admin.display(description="mAP50")
    def map50(self, obj):
        return obj.metrics.get("map50") if isinstance(obj.metrics, dict) else None

    @admin.display(description="mAP50-95")
    def map50_95(self, obj):
        return obj.metrics.get("map50_95") if isinstance(obj.metrics, dict) else None

    # Field visibility per pipeline (the template shows/hides these). A regular
    # eval (blank pipeline) shows none of them; keep this in sync with
    # ``config_gen.build_pipeline_request`` and ``PipelineEvalRun.DETECTOR_PIPELINES``.
    NO_PIPELINE = ("", "No pipeline — regular eval")

    @admin.action(description="Evaluate…")
    def evaluate(self, request, queryset):
        """Evaluate one model — optionally through a chachak pipeline.

        Merges the old "Evaluate on a dataset" and "Evaluate with a pipeline"
        actions: leave *Pipeline* blank for a plain :class:`EvalRun`, or pick one
        to create a :class:`PipelineEvalRun`. The form renders only the knobs the
        chosen pipeline uses (detector / tiling / chain).

        Selecting 2+ models evaluates them *combined*: each model's predictions
        are remapped onto the dataset's class space and merged per image, scored
        as a single result. Only valid for complementary detectors — models
        whose class lists share any class name are refused (they'd double-detect
        that class), see :func:`combine.overlapping_class_names`.
        """
        from eval_pipelines.models import PipelineEvalRun

        models = list(queryset)
        if not models:
            self.message_user(request, "Select at least one model to evaluate.",
                              level=messages.WARNING)
            return None

        overlap = combine.overlapping_class_names(models) if len(models) > 1 else set()
        if overlap:
            self.message_user(
                request,
                "Can't combine these models — they share class(es) "
                f"{', '.join(sorted(overlap))}. Combined evaluation only supports "
                "models with disjoint class spaces.",
                level=messages.WARNING,
            )
            return None

        model = models[0]
        extra_models = models[1:]

        if request.POST.get("apply"):
            dataset = Dataset.objects.filter(pk=request.POST.get("dataset") or None).first()
            if dataset is None:
                self.message_user(request, "Choose a dataset.", level=messages.WARNING)
                return None
            label_source = request.POST.get("label_source") or EvalRun.SOURCE
            annotator = Annotator.objects.filter(pk=request.POST.get("annotator") or None).first()
            explicit = request.POST.get("explicit_labels_path", "")
            pipeline = (request.POST.get("pipeline") or "").strip()

            def _threshold(name, default):
                raw = (request.POST.get(name) or "").strip()
                try:
                    return float(raw) if raw else default
                except ValueError:
                    return default

            def _map_score_threshold():
                return _threshold("map_score_threshold", 0.001)

            def _score_threshold():
                return _threshold("score_threshold", 0.25)

            if not pipeline:
                # No pipeline chosen — a plain EvalRun (the old "evaluate on a dataset").
                eval_run = EvalRun.objects.create(
                    trained_model=model, dataset=dataset, label_source=label_source,
                    annotator=annotator, explicit_labels_path=explicit,
                    map_score_threshold=_map_score_threshold(),
                    score_threshold=_score_threshold(),
                )
                if extra_models:
                    eval_run.combined_models.set(extra_models)
                try:
                    config_gen.write_eval_request(eval_run)
                except (ValueError, FileNotFoundError, RuntimeError) as exc:
                    eval_run.delete()
                    self.message_user(request, f"Cannot build eval request: {exc}",
                                      level=messages.ERROR)
                    return None
                _queue().enqueue(jobs.run_eval, eval_run.pk, job_timeout=jobs.JOB_TIMEOUT)
                eval_run.status = EvalRun.QUEUED
                eval_run.save(update_fields=["status"])
                self.message_user(
                    request,
                    f"Eval #{eval_run.pk} queued for {_models_title(models)} on {dataset.name}.")
                return None

            # A pipeline was chosen — a PipelineEvalRun.
            def _float(name):
                raw = (request.POST.get(name) or "").strip()
                return float(raw) if raw else None

            chain = [c.strip() for c in (request.POST.get("chain") or "").split(",") if c.strip()]
            # Non-null field: fall back to the model default when the form omits it
            # (a legitimate 0.0 must survive, so test for None explicitly).
            expand_ratio = _float("detector_expand_ratio")
            if expand_ratio is None:
                expand_ratio = PipelineEvalRun._meta.get_field("detector_expand_ratio").get_default()
            pe = PipelineEvalRun.objects.create(
                trained_model=model, dataset=dataset, label_source=label_source,
                annotator=annotator, explicit_labels_path=explicit,
                pipeline=pipeline,
                detector_checkpoint=(request.POST.get("detector_checkpoint") or "").strip(),
                detector_expand_ratio=expand_ratio,
                detector_min_box_size=_float("detector_min_box_size"),
                tile_size_px=_int_or_none(request.POST.get("tile_size_px")),
                tile_width_pct=_float("tile_width_pct"),
                tile_height_pct=_float("tile_height_pct"),
                overlap=_float("overlap"),
                merge_nms_iou=_float("merge_nms_iou"),
                chain=chain,
                map_score_threshold=_map_score_threshold(),
                score_threshold=_score_threshold(),
            )
            if extra_models:
                pe.combined_models.set(extra_models)
            try:
                config_gen.write_pipeline_request(pe)
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                pe.delete()
                self.message_user(request, f"Cannot build pipeline request: {exc}",
                                  level=messages.ERROR)
                return None
            _queue().enqueue(jobs.run_pipeline_eval, pe.pk, job_timeout=jobs.JOB_TIMEOUT)
            pe.status = PipelineEvalRun.QUEUED
            pe.save(update_fields=["status"])
            self.message_user(
                request,
                f"Pipeline eval #{pe.pk} ({pipeline}) queued for "
                f"{_models_title(models)} on {dataset.name}.")
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": f"Evaluate {_models_title(models)}",
            "model": model,
            "models": models,
            "is_combined": len(models) > 1,
            "datasets": Dataset.objects.all(),
            "annotators": Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"),
            "label_source_choices": PipelineEvalRun._meta.get_field("label_source").choices,
            "pipeline_choices": [self.NO_PIPELINE, *PipelineEvalRun.PIPELINE_CHOICES],
            "detector_pipelines": " ".join(sorted(PipelineEvalRun.DETECTOR_PIPELINES)),
            "tiling_pipelines": " ".join([
                PipelineEvalRun.BATCH_DETECT, PipelineEvalRun.BATCH_PEOPLE, PipelineEvalRun.CHAIN]),
            "chain_pipelines": PipelineEvalRun.CHAIN,
            # Every pipeline merges several predictions per image back together;
            # only the no-pipeline / raw option has nothing to merge.
            "merging_pipelines": " ".join(
                value for value, _label in PipelineEvalRun.PIPELINE_CHOICES),
            # Pre-fill the pipeline fields from the experiment the model came from,
            # so a later manual eval defaults to the params the model was trained
            # with. Empty dict when the model has no originating experiment or that
            # experiment had no pipeline. Only applied for a single selected model.
            "pipeline_defaults": self._experiment_pipeline_defaults(model) if len(models) == 1 else {},
            "action": "evaluate",
            "selected": [str(m.pk) for m in models],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/evaluate_model.html", context)

    @admin.action(description="Fine-tune model…")
    def fine_tune(self, request, queryset):
        """Fine-tune one trained model on a new dataset.

        A smaller sibling of ``ExperimentAdmin.generate_run``: pick a dataset
        (plus an optional val dataset), a handful of training hyperparameters,
        and an optional freeze-backbone toggle, then hand off to
        ``training.services.finetune.create_run`` to build the same
        Experiment/dataset/model/TrainingRun rows the full flow would — pre-
        seeded with this model's architecture, checkpoint (as a Friendy warm
        start) and frozen pipeline geometry, so none of that is asked for
        twice. Unlike an ordinary experiment, the result is auto-promoted
        straight to a new TrainedModel the moment the run succeeds (see
        ``jobs.finalize_success``) — watch it under "Fine-tuning runs", not
        "Training runs".
        """
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to fine-tune.",
                              level=messages.WARNING)
            return None
        source = queryset.first()
        if not source.checkpoint_path:
            self.message_user(request, f"{source.name} has no checkpoint to fine-tune from.",
                              level=messages.ERROR)
            return None

        if request.POST.get("apply"):
            train_dataset = Dataset.objects.filter(
                pk=request.POST.get("train_dataset") or None).first()
            if train_dataset is None:
                self.message_user(request, "Choose a fine-tune dataset.", level=messages.WARNING)
                return None
            label_source = request.POST.get("label_source") or ExperimentDataset.SOURCE
            annotator = Annotator.objects.filter(pk=request.POST.get("annotator") or None).first()
            explicit_labels_path = request.POST.get("explicit_labels_path", "")
            if label_source == ExperimentDataset.ANNOTATOR and annotator is None:
                self.message_user(request, "Pick an annotator for 'annotator output'.",
                                  level=messages.WARNING)
                return None
            if label_source == ExperimentDataset.EXPLICIT and not explicit_labels_path.strip():
                self.message_user(request, "Enter an explicit labels path.", level=messages.WARNING)
                return None

            val_dataset = Dataset.objects.filter(pk=request.POST.get("val_dataset") or None).first()
            val_label_source = request.POST.get("val_label_source") or ExperimentDataset.SOURCE
            val_annotator = Annotator.objects.filter(
                pk=request.POST.get("val_annotator") or None).first()
            val_explicit_labels_path = request.POST.get("val_explicit_labels_path", "")
            if val_dataset is not None:
                if val_label_source == ExperimentDataset.ANNOTATOR and val_annotator is None:
                    self.message_user(
                        request, "Pick an annotator for the val dataset's 'annotator output'.",
                        level=messages.WARNING)
                    return None
                if val_label_source == ExperimentDataset.EXPLICIT and not val_explicit_labels_path.strip():
                    self.message_user(
                        request, "Enter an explicit labels path for the val dataset.",
                        level=messages.WARNING)
                    return None

            early_stopping_patience = _int_or_none(request.POST.get("early_stopping_patience"))
            if early_stopping_patience is not None and val_dataset is None:
                self.message_user(
                    request,
                    "Early stopping needs a validation dataset — pick one or clear the "
                    "patience field.",
                    level=messages.WARNING,
                )
                return None

            req = finetune.FineTuneRequest(
                train_dataset=train_dataset,
                label_source=label_source,
                annotator=annotator,
                explicit_labels_path=explicit_labels_path,
                val_dataset=val_dataset,
                val_label_source=val_label_source,
                val_annotator=val_annotator,
                val_explicit_labels_path=val_explicit_labels_path,
                name=(request.POST.get("name") or "").strip(),
                epochs=_int_or_none(request.POST.get("epochs")) or 20,
                batch_size=_int_or_none(request.POST.get("batch_size")) or 4,
                num_workers=_int_or_none(request.POST.get("num_workers")),
                lr=_float_or_none(request.POST.get("lr")) or 2e-5,
                freeze_backbone=bool(request.POST.get("freeze_backbone")),
                early_stopping_patience=early_stopping_patience,
            )
            try:
                run = finetune.create_run(source, req)
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                self.message_user(request, f"Cannot start fine-tuning: {exc}", level=messages.ERROR)
                return None
            self.message_user(
                request,
                f"Fine-tune run #{run.pk} queued from {source.name!r} on {train_dataset.name}. "
                "Watch the Fine-tuning runs page for progress — it promotes a new model "
                "automatically when the run finishes.",
            )
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": f"Fine-tune {source.name}",
            "source": source,
            "datasets": Dataset.objects.all(),
            "annotators": Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"),
            "label_source_choices": ExperimentDataset.LABEL_SOURCE_CHOICES,
            "default_name": f"{source.name}-ft",
            "action": "fine_tune",
            "selected": [str(source.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/fine_tune.html", context)

    @staticmethod
    def _experiment_pipeline_defaults(model) -> dict:
        """The model's frozen pipeline metadata, shaped for a form template.

        Shared by the eval and preview forms, which offer the same pipeline
        vocabulary. Blank for a full-frame model, since the eval form's "no
        pipeline" option *is* the plain-eval path — there is nothing to prefill.
        ``chain`` becomes the comma-separated string the form field takes.
        """
        meta = pipeline_meta.for_trained_model(model)
        if meta["pipeline"] == pipeline_meta.RAW:
            return {}
        return {**meta, "chain": ", ".join(meta["chain"])}

    # ------------------------------------------------------------------ preview
    RAW_PIPELINE = ("raw", "Raw model (no pipeline)")

    @admin.action(description="Preview model on selected dataset…")
    def preview_on_dataset(self, request, queryset):
        """Open the interactive box-preview viewer for one model on a dataset.

        Unlike the eval actions this persists nothing — on submit it just
        redirects to the viewer with the chosen settings as query params.
        """
        from eval_pipelines.models import PipelineEvalRun

        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to preview.",
                              level=messages.WARNING)
            return None
        model = queryset.first()

        if request.POST.get("apply"):
            dataset = Dataset.objects.filter(pk=request.POST.get("dataset") or None).first()
            if dataset is None:
                self.message_user(request, "Choose a dataset.", level=messages.WARNING)
                return None
            params = {
                "model": model.pk,
                "dataset": dataset.pk,
                "pipeline": request.POST.get("pipeline") or self.RAW_PIPELINE[0],
                "detector_checkpoint": (request.POST.get("detector_checkpoint") or "").strip(),
                "detector_expand_ratio": (request.POST.get("detector_expand_ratio") or "").strip(),
                "detector_min_box_size": (request.POST.get("detector_min_box_size") or "").strip(),
                "tile_size_px": (request.POST.get("tile_size_px") or "").strip(),
                "tile_width_pct": (request.POST.get("tile_width_pct") or "").strip(),
                "tile_height_pct": (request.POST.get("tile_height_pct") or "").strip(),
                "overlap": (request.POST.get("overlap") or "").strip(),
                "merge_nms_iou": (request.POST.get("merge_nms_iou") or "").strip(),
                "chain": (request.POST.get("chain") or "").strip(),
                "score": (request.POST.get("score") or "").strip(),
                "label_source": request.POST.get("label_source") or "",
                "annotator": request.POST.get("annotator") or "",
                "explicit_labels_path": (request.POST.get("explicit_labels_path") or "").strip(),
            }
            query = urlencode({k: v for k, v in params.items() if v not in ("", None)})
            return redirect(reverse("admin:training_trainedmodel_preview") + "?" + query)

        context = {
            **self.admin_site.each_context(request),
            "title": f"Preview {model.name} on a dataset",
            "model": model,
            "datasets": Dataset.objects.all(),
            "annotators": Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"),
            "label_source_choices": PipelineEvalRun._meta.get_field("label_source").choices,
            "pipeline_choices": [self.RAW_PIPELINE, *PipelineEvalRun.PIPELINE_CHOICES],
            "detector_pipelines": " ".join(sorted(PipelineEvalRun.DETECTOR_PIPELINES)),
            "tiling_pipelines": " ".join([
                PipelineEvalRun.BATCH_DETECT, PipelineEvalRun.BATCH_PEOPLE, PipelineEvalRun.CHAIN]),
            "chain_pipelines": PipelineEvalRun.CHAIN,
            # Every pipeline merges several predictions per image back together;
            # only the no-pipeline / raw option has nothing to merge.
            "merging_pipelines": " ".join(
                value for value, _label in PipelineEvalRun.PIPELINE_CHOICES),
            # Same "serve it the way it was trained" prefill as the video/camera
            # inference forms, off the same frozen record — a preview run through
            # different geometry than the model was trained with is misleading, so
            # the trained pipeline is the default and overriding it is deliberate.
            "pipeline_defaults": self._experiment_pipeline_defaults(model),
            "action": "preview_on_dataset",
            "selected": [str(model.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/preview_model.html", context)

    # ------------------------------------------------------------------- ONNX export
    def _export_checkpoints(self, model):
        """(label, checkpoint_path) pairs for the model's best and last ``.pt``.

        ``best`` comes from the source run result (falling back to the model's own
        promoted ``checkpoint_path``); ``last`` only exists when a source run
        result is present. Missing/blank paths are dropped.
        """
        run_result = model.source_run_result
        best = ((getattr(run_result, "best_checkpoint", "") or "").strip()
                if run_result else "")
        best = best or (model.checkpoint_path or "").strip()
        last = ((getattr(run_result, "last_checkpoint", "") or "").strip()
                if run_result else "")

        pairs = []
        if best:
            pairs.append(("best", best))
        if last and last != best:
            pairs.append(("last", last))
        return pairs

    @staticmethod
    def _onnx_stem(model_name: str) -> str:
        """Filesystem-safe basename for a model's exported artifacts."""
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
        return slug or "model"

    @staticmethod
    def _trained_size_hint(checkpoints) -> dict:
        """Best-effort ``runner.inspect_checkpoint`` on the first checkpoint, to
        prefill/confirm the export forms' static input size.

        Never raises — a trainer-service hiccup here must not block opening the
        form; the operator just sees no hint and can still type a size by hand
        (or, for ONNX, the export itself still resolves the size correctly).
        """
        if not checkpoints:
            return {}
        _, checkpoint_path = checkpoints[0]
        try:
            return runner.inspect_checkpoint(checkpoint_path)
        except Exception:  # noqa: BLE001 - best-effort hint only
            return {}

    @classmethod
    def _fp16_trusted(cls, checkpoints) -> bool:
        """Whether this arch's fp16 engine is known to reproduce its fp32 output.

        Defaults to True when the hint is unavailable, so a trainer-service hiccup
        never silently downgrades an arch that is fine in fp16.
        """
        return bool(cls._trained_size_hint(checkpoints).get("fp16_trusted", True))

    @classmethod
    def _auto_precision(cls, checkpoints) -> str:
        """What ``precision="auto"`` would build for this arch on the trainer.

        Not derivable from ``fp16_trusted`` alone: an arch whose blanket-cast fp16
        is untrusted can still have a clean fp16 engine via ModelOpt AutoCast (see
        trt_export/arch/__init__.py's ARCH_CAST_BACKEND), and whether that route is
        available depends on nvidia-modelopt being installed on the trainer. Falls
        back to the old fp16_trusted rule when the hint predates this field.
        """
        hint = cls._trained_size_hint(checkpoints)
        value = hint.get("auto_precision")
        if value in ("fp16", "fp32", "bf16"):
            return value
        return "fp16" if hint.get("fp16_trusted", True) else "fp32"

    @classmethod
    def _batch_aware(cls, checkpoints) -> bool:
        """Whether this arch's engine decodes a batch profile above 1 correctly.

        Defaults to False when the hint is unavailable (older trainer service, or
        the hint failed) — the safe direction to guess, since assuming batch-safety
        it never claimed would let a bad request through instead of just hiding a
        form field.
        """
        return bool(cls._trained_size_hint(checkpoints).get("batch_aware", False))

    @admin.action(description="Export best + last to ONNX…")
    def export_onnx(self, request, queryset):
        """Queue a model's best and last ``.pt`` for ONNX export under a chosen directory.

        Creates one ``ExportRun`` per checkpoint and enqueues ``jobs.run_export_onnx``
        on the ``django_rq`` worker — the export itself (plus its pipeline sidecar and
        best-effort infer bundle) runs there against the trainer service's
        ``/export_onnx`` (torch env), instead of blocking this request. Writes
        ``<name>-best.onnx`` / ``<name>-last.onnx`` (each with a sibling ``.meta.json``)
        into the operator's directory. Relative paths resolve against the project root,
        like the other training paths.
        """
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to export.",
                              level=messages.WARNING)
            return None
        model = queryset.first()
        checkpoints = self._export_checkpoints(model)
        if not checkpoints:
            self.message_user(
                request, f"{model.name}: no checkpoint on record to export.",
                level=messages.WARNING)
            return None

        if request.POST.get("apply"):
            output_dir = (request.POST.get("output_dir") or "").strip()
            if not output_dir:
                self.message_user(request, "Enter an output directory.",
                                  level=messages.WARNING)
                return None
            out_dir = config_gen._resolve(output_dir)
            stem = self._onnx_stem(model.name)
            queue = _queue()
            for label, checkpoint in checkpoints:
                onnx_path = out_dir / f"{stem}-{label}.onnx"
                export_run = ExportRun.objects.create(
                    model=model, kind=ExportRun.ONNX, checkpoint_label=label,
                    checkpoint_path=checkpoint, output_path=str(onnx_path),
                )
                queue.enqueue(
                    jobs.run_export_onnx, export_run.pk, job_timeout=jobs.EXPORT_ONNX_JOB_TIMEOUT)
            self.message_user(
                request,
                f"Queued {len(checkpoints)} ONNX export job(s) for {model.name} — see "
                f"Export runs for progress.")
            return None

        default_dir = exports.exports_root()
        trained_size = self._trained_size_hint(checkpoints).get("trained_size")
        context = {
            **self.admin_site.each_context(request),
            "title": f"Export {model.name} to ONNX",
            "model": model,
            "checkpoints": [{"label": label, "path": path} for label, path in checkpoints],
            "default_output_dir": str(default_dir),
            "trained_size_hint": f"{trained_size[0]}×{trained_size[1]}" if trained_size else "",
            "action": "export_onnx",
            "selected": [str(model.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/export_onnx.html", context)

    # ------------------------------------------------------------------- TensorRT export
    @admin.action(description="Export best + last to TensorRT…")
    def export_trt(self, request, queryset):
        """Queue TensorRT engine builds for a model's best and last ``.pt``.

        Creates one ``ExportRun`` per checkpoint and enqueues ``jobs.run_export_trt``
        on the ``django_rq`` worker — the build itself (plus its pipeline sidecar and
        best-effort infer bundle) runs there against the trainer service's
        ``/export_trt`` (GPU env), instead of blocking this request. Writes
        ``<name>-best.engine`` / ``<name>-last.engine`` (each with sibling
        ``.meta.json`` + ``.engine.json``) into the operator's directory. Relative
        paths resolve against the project root.

        NOTE: unlike ONNX export, this uses the GPU and competes with active training;
        engines are non-portable (tied to the trainer's GPU + TensorRT version). Check
        "Export runs" afterward for whether the requested precision actually built (an
        fp16 request can silently fall back to fp32).
        """
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to export.",
                              level=messages.WARNING)
            return None
        model = queryset.first()
        checkpoints = self._export_checkpoints(model)
        if not checkpoints:
            self.message_user(
                request, f"{model.name}: no checkpoint on record to export.",
                level=messages.WARNING)
            return None

        if request.POST.get("apply"):
            output_dir = (request.POST.get("output_dir") or "").strip()
            if not output_dir:
                self.message_user(request, "Enter an output directory.",
                                  level=messages.WARNING)
                return None
            # The form's own default comes from fp16_trusted below; an explicit
            # posted value still wins, so an operator can deliberately build fp16
            # for an untrusted arch. Only the fallback for a missing/garbage value
            # follows the flag — never silently upgrade to fp16 for an arch whose
            # fp16 engine is known not to match its fp32 output.
            posted = (request.POST.get("precision") or "").strip()
            if posted in ("fp16", "fp32", "bf16", "auto"):
                precision = posted
            else:
                precision = self._auto_precision(checkpoints)
            # An unchecked box posts nothing, so absence *is* the False -- no whitelist
            # needed here, unlike precision above.
            gpu_infer = bool(request.POST.get("gpu_infer"))
            # Optional static input size ("640" or "640x640"): pins a fixed engine
            # profile (min==opt==max), which is how Faster R-CNN gets FP16.
            input_size = (request.POST.get("input_size") or "").strip()
            input_hw = None
            if input_size:
                parts = input_size.lower().replace("×", "x").split("x")
                try:
                    dims = [int(p) for p in parts]
                    if len(dims) == 1:
                        input_hw = (dims[0], dims[0])
                    elif len(dims) == 2:
                        input_hw = (dims[0], dims[1])
                    else:
                        raise ValueError
                    if any(d <= 0 for d in input_hw):
                        raise ValueError
                except ValueError:
                    self.message_user(
                        request,
                        f"Invalid input size {input_size!r}; use a number (e.g. 640) or HxW "
                        f"(e.g. 640x640).", level=messages.WARNING)
                    return None
            # Batch profile. Only offered at all when the arch's engine actually
            # decodes B > 1 correctly (see _batch_aware) — the template hides the
            # checkbox/range fields otherwise, so an unaware arch always falls
            # through to the fixed branch at batch_size's default of "1".
            batch_aware = self._batch_aware(checkpoints)
            variable_batch = bool(request.POST.get("variable_batch"))
            if not batch_aware:
                min_batch = opt_batch = max_batch = 1
            elif variable_batch:
                min_raw = (request.POST.get("min_batch") or "").strip()
                max_raw = (request.POST.get("max_batch") or "").strip()
                try:
                    min_batch, max_batch = int(min_raw), int(max_raw)
                    if min_batch < 1 or max_batch < min_batch:
                        raise ValueError
                except ValueError:
                    self.message_user(
                        request,
                        f"Invalid batch range (min={min_raw!r}, max={max_raw!r}); "
                        f"need 1 <= min <= max.", level=messages.WARNING)
                    return None
                opt_batch = max_batch
            else:
                batch_raw = (request.POST.get("batch_size") or "1").strip()
                try:
                    min_batch = opt_batch = max_batch = int(batch_raw)
                    if min_batch < 1:
                        raise ValueError
                except ValueError:
                    self.message_user(
                        request,
                        f"Invalid batch size {batch_raw!r}; enter a positive integer.",
                        level=messages.WARNING)
                    return None

            # "Build on": blank = the local trainer (the only option before build
            # nodes existed). A node instead compiles on ITS GPU and returns a whole
            # bundle, which is a different job — see jobs.run_export_remote.
            node = None
            node_pk = (request.POST.get("node") or "").strip()
            if node_pk:
                node = BuildNode.objects.filter(
                    pk=node_pk, status=BuildNode.ACTIVE).first()
                if node is None:
                    self.message_user(
                        request, "That build node is gone or retired; nothing queued.",
                        level=messages.WARNING)
                    return None

            if node is not None and max_batch > 1:
                self.message_user(
                    request,
                    f"Batch profile (batch {min_batch}-{max_batch}) is ignored for a "
                    f"remote build node — {node.name} will build batch-1, like every "
                    f"other bundle it assembles.", level=messages.WARNING)

            out_dir = config_gen._resolve(output_dir)
            stem = self._onnx_stem(model.name)
            queue = _queue()
            for label, checkpoint in checkpoints:
                engine_path = out_dir / f"{stem}-{label}.engine"
                export_run = ExportRun.objects.create(
                    model=model, kind=ExportRun.TRT, checkpoint_label=label,
                    checkpoint_path=checkpoint, output_path=str(engine_path),
                    precision=precision, input_hw=list(input_hw) if input_hw else None,
                    gpu_infer=gpu_infer,
                    min_batch=min_batch, opt_batch=opt_batch, max_batch=max_batch,
                    node=node,
                )
                if node is None:
                    queue.enqueue(
                        jobs.run_export_trt, export_run.pk,
                        job_timeout=jobs.EXPORT_TRT_JOB_TIMEOUT)
                else:
                    queue.enqueue(
                        jobs.run_export_remote, export_run.pk,
                        job_timeout=jobs.EXPORT_REMOTE_JOB_TIMEOUT)
            where = f"on {node.name}" if node else "on the local trainer"
            batch_note = f", batch {min_batch}-{max_batch}" if max_batch > 1 else ""
            self.message_user(
                request,
                f"Queued {len(checkpoints)} TensorRT export job(s) for {model.name} "
                f"({precision}{batch_note}) {where} — see Export runs for progress.")
            return None

        default_dir = exports.exports_root()
        hint = self._trained_size_hint(checkpoints)
        trained_size = hint.get("trained_size")
        # Absent (older trainer service, or the hint failed) -> treat as trusted, so
        # the form keeps its historical fp16 default for every other arch.
        fp16_trusted = hint.get("fp16_trusted", True)
        auto_precision = self._auto_precision(checkpoints)
        # True when the arch's plain fp16 is on the safety floor but AutoCast
        # rescues it — the form should then point at fp16, not fp32, and say why.
        autocast_rescued = not fp16_trusted and auto_precision == "fp16"
        # Absent -> treat as NOT batch-aware, the opposite direction from fp16_trusted
        # above: hiding the batch field on a missing hint is safe, defaulting it open
        # is not (see _batch_aware).
        batch_aware = hint.get("batch_aware", False)
        context = {
            **self.admin_site.each_context(request),
            "title": f"Export {model.name} to TensorRT",
            "model": model,
            "checkpoints": [{"label": label, "path": path} for label, path in checkpoints],
            "default_output_dir": str(default_dir),
            "default_input_size": f"{trained_size[0]}x{trained_size[1]}" if trained_size else "",
            "fp16_trusted": fp16_trusted,
            "auto_precision": auto_precision,
            "autocast_rescued": autocast_rescued,
            "batch_aware": batch_aware,
            "build_nodes": BuildNode.objects.filter(status=BuildNode.ACTIVE),
            "action": "export_trt",
            "selected": [str(model.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/export_trt.html", context)

    # ------------------------------------------------------------------- .pt bundle export
    @admin.action(description="Export .pt bundle…")
    def export_pt_bundle(self, request, queryset):
        """Package the model's best checkpoint into a self-contained ``.tar.gz``
        another machine can catalog as a Trained model.

        Unlike ``export_onnx``/``export_trt`` this doesn't touch the trainer
        service at all — it's a plain-file operation (copy + hash + tar) against
        the checkpoint already on the shared filesystem, so there's no precision/
        input-size/build-node choice to make, and only the promoted ``best``
        checkpoint is bundled (not ``last``). ``exports.build_pt_bundle`` folds in
        the model's catalog record, its frozen pipeline metadata (plus the
        person-detector checkpoint it depends on, if any), and every training
        hyperparameter/dataset/metric on record for the run it came from — see
        ``docs/pt-bundles.md``.
        """
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to export.",
                              level=messages.WARNING)
            return None
        model = queryset.first()
        best = next(
            (path for label, path in self._export_checkpoints(model) if label == "best"), None)
        if not best:
            self.message_user(
                request, f"{model.name}: no checkpoint on record to export.",
                level=messages.WARNING)
            return None

        if request.POST.get("apply"):
            output_dir = (request.POST.get("output_dir") or "").strip()
            if not output_dir:
                self.message_user(request, "Enter an output directory.",
                                  level=messages.WARNING)
                return None
            out_dir = config_gen._resolve(output_dir)
            archive_path = out_dir / f"{self._onnx_stem(model.name)}-ptbundle.tar.gz"
            export_run = ExportRun.objects.create(
                model=model, kind=ExportRun.PT, checkpoint_label="best",
                checkpoint_path=best, output_path=str(archive_path),
            )
            _queue().enqueue(
                jobs.run_export_pt_bundle, export_run.pk,
                job_timeout=jobs.EXPORT_PT_BUNDLE_JOB_TIMEOUT)
            self.message_user(
                request,
                f"Queued .pt bundle export for {model.name} — see Export runs for progress.")
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": f"Export {model.name} as a .pt bundle",
            "model": model,
            "checkpoint_path": best,
            "default_output_dir": str(exports.exports_root()),
            "action": "export_pt_bundle",
            "selected": [str(model.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/export_pt_bundle.html", context)

    def get_urls(self):
        custom = [
            path("preview/", self.admin_site.admin_view(self.preview_view),
                 name="training_trainedmodel_preview"),
            path("preview/image/", self.admin_site.admin_view(self.preview_image),
                 name="training_trainedmodel_preview_image"),
            path("preview/data/", self.admin_site.admin_view(self.preview_data),
                 name="training_trainedmodel_preview_data"),
            path("hard-images/", self.admin_site.admin_view(self.hard_images_view),
                 name="training_trainedmodel_hard_images"),
            path("hard-images/image/", self.admin_site.admin_view(self.hard_images_image),
                 name="training_trainedmodel_hard_images_image"),
            path("hard-images/data/", self.admin_site.admin_view(self.hard_images_data),
                 name="training_trainedmodel_hard_images_data"),
        ]
        return custom + super().get_urls()

    def preview_view(self, request):
        """Render the viewer shell; the browser pulls images + boxes per index."""
        model = TrainedModel.objects.filter(pk=request.GET.get("model")).first()
        dataset = Dataset.objects.filter(pk=request.GET.get("dataset")).first()
        if model is None or dataset is None:
            raise Http404("preview requires ?model= and ?dataset=")

        images = lsapi.list_dataset_images(source_root() / dataset.name)
        has_labels = _preview_label_dir(request, dataset) is not None

        context = {
            **self.admin_site.each_context(request),
            "title": f"Preview {model.name} on {dataset.name}",
            "model": model,
            "dataset": dataset,
            "pipeline": request.GET.get("pipeline") or self.RAW_PIPELINE[0],
            "image_count": len(images),
            "has_labels": has_labels,
            "query": request.GET.urlencode(),
        }
        return TemplateResponse(request, "admin/training/preview_viewer.html", context)

    def preview_image(self, request):
        """Stream the raw bytes of the dataset image at ``?index=``."""
        dataset = Dataset.objects.filter(pk=request.GET.get("dataset")).first()
        if dataset is None:
            raise Http404("unknown dataset")
        images = lsapi.list_dataset_images(source_root() / dataset.name)
        index = _preview_index(request, len(images))
        return FileResponse(open(images[index], "rb"))

    def preview_data(self, request):
        """Run inference for one image and return predictions (+ optional GT)."""
        model = TrainedModel.objects.filter(pk=request.GET.get("model")).first()
        dataset = Dataset.objects.filter(pk=request.GET.get("dataset")).first()
        if model is None or dataset is None:
            return JsonResponse({"error": "unknown model or dataset"}, status=400)

        images = lsapi.list_dataset_images(source_root() / dataset.name)
        if not images:
            return JsonResponse({"error": "dataset has no images"}, status=400)
        index = _preview_index(request, len(images))
        image_path = images[index]

        score = _float_or_none(request.GET.get("score"))
        chain = [
            c.strip() for c in (request.GET.get("chain") or "").split(",")
            if c.strip()
        ]
        try:
            payload = config_gen.build_preview_request(
                model,
                request.GET.get("pipeline") or self.RAW_PIPELINE[0],
                str(image_path),
                detector_checkpoint=(request.GET.get("detector_checkpoint") or "").strip(),
                detector_expand_ratio=_float_or_none(request.GET.get("detector_expand_ratio")),
                detector_min_box_size=_float_or_none(request.GET.get("detector_min_box_size")),
                tile_size_px=_int_or_none(request.GET.get("tile_size_px")),
                tile_width_pct=_float_or_none(request.GET.get("tile_width_pct")),
                tile_height_pct=_float_or_none(request.GET.get("tile_height_pct")),
                overlap=_float_or_none(request.GET.get("overlap")),
                merge_nms_iou=_float_or_none(request.GET.get("merge_nms_iou")),
                chain=chain,
                score_threshold=0.05 if score is None else score,
            )
            result = runner.predict_image(payload)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001 — surface trainer/network errors to the viewer
            return JsonResponse({"error": f"inference failed: {exc}"}, status=502)

        label_dir = _preview_label_dir(request, dataset)
        ground_truth = (
            _read_gt_boxes(label_dir, image_path, config_gen.dataset_classes(dataset))
            if label_dir is not None else []
        )
        return JsonResponse({
            "predictions": result.get("boxes", []),
            "ground_truth": ground_truth,
            "classes": result.get("classes", {}),
            "image": image_path.name,
            "index": index,
            "count": len(images),
        })

    # -------------------------------------------------------- hardest val images
    @admin.action(description="View 50 hardest val images…")
    def view_hard_val_images(self, request, queryset):
        """Open the viewer for the val images this model's training run struggled with.

        The artifact (``<run_dir>/val_hard_images.json``) is produced by the trainer at
        end-of-run, so this action just checks it exists and redirects to the viewer.
        """
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model.", level=messages.WARNING)
            return None
        model = queryset.first()
        artifact = self._hard_images_path(model)
        if artifact is None or not artifact.exists():
            self.message_user(
                request,
                f"No hardest-val-images artifact for {model.name} — its training run "
                "predates this feature or had no val set with labels.",
                level=messages.WARNING,
            )
            return None
        return redirect(
            reverse("admin:training_trainedmodel_hard_images") + "?model=" + str(model.pk))

    def _hard_images_path(self, model):
        """Locate ``val_hard_images.json`` in the source run's output dir, or None."""
        run_result = getattr(model, "source_run_result", None)
        run_dir = (getattr(run_result, "run_dir", "") or "").strip() if run_result else ""
        if not run_dir:
            return None
        return Path(run_dir) / "val_hard_images.json"

    def _load_hard_images(self, model):
        """Parse the hard-images artifact for ``model``, or None if missing/unreadable."""
        artifact = self._hard_images_path(model)
        if artifact is None or not artifact.exists():
            return None
        try:
            return json.loads(artifact.read_text())
        except (ValueError, OSError):
            return None

    def hard_images_view(self, request):
        """Render the viewer shell; the browser pulls images + precomputed boxes per index."""
        model = TrainedModel.objects.filter(pk=request.GET.get("model")).first()
        if model is None:
            raise Http404("hard-images requires ?model=")
        payload = self._load_hard_images(model)
        if payload is None:
            raise Http404("no hard-images artifact for this model")
        images = payload.get("images", [])
        context = {
            **self.admin_site.each_context(request),
            "title": f"Hardest val images — {model.name}",
            "model": model,
            "subject_name": model.name,
            "image_count": len(images),
            "metric": payload.get("metric", ""),
            "metric_description": payload.get("metric_description", ""),
            "iou_threshold": payload.get("iou_threshold"),
            "score_threshold": payload.get("score_threshold"),
            "operating_nms_threshold": payload.get("operating_nms_threshold"),
            "max_display_predictions": payload.get("max_display_predictions"),
            "query": urlencode({"model": model.pk}),
            "run_choices": [],
            "run_choices_json": "[]",
            "back_label": "Back to models",
        }
        return TemplateResponse(request, "admin/training/hard_images_viewer.html", context)

    def hard_images_image(self, request):
        """Stream the raw bytes of the hard image at ``?index=`` (guarded to source_root)."""
        model = TrainedModel.objects.filter(pk=request.GET.get("model")).first()
        if model is None:
            raise Http404("unknown model")
        payload = self._load_hard_images(model)
        images = payload.get("images", []) if payload else []
        raw_path = _hard_image_path_from_request(request, images)
        image_path = Path(raw_path).resolve()
        root = source_root().resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise Http404("image not found under source root")
        return FileResponse(open(image_path, "rb"))

    def hard_images_data(self, request):
        """Return the precomputed predictions + ground truth + difficulty for ``?index=``."""
        model = TrainedModel.objects.filter(pk=request.GET.get("model")).first()
        if model is None:
            return JsonResponse({"error": "unknown model"}, status=400)
        payload = self._load_hard_images(model)
        images = payload.get("images", []) if payload else []
        if not images:
            return JsonResponse({"error": "no hard images for this model"}, status=400)
        index = _preview_index(request, len(images))
        entry = images[index]
        return JsonResponse({
            "predictions": entry.get("predictions", []),
            "ground_truth": entry.get("ground_truth", []),
            "image": entry.get("image_name", ""),
            "image_token": _hard_image_token(entry.get("image_path", "")),
            "difficulty": entry.get("difficulty"),
            "precision": entry.get("precision"),
            "recall": entry.get("recall"),
            "f1": entry.get("f1"),
            "missed": entry.get("missed"),
            "false_positives": entry.get("false_positives"),
            "wrong_class": entry.get("wrong_class"),
            "loc_error": entry.get("loc_error"),
            "localization_penalty": entry.get("localization_penalty"),
            "total_errors": entry.get("total_errors"),
            "num_predictions": entry.get("num_predictions"),
            "num_ground_truth": entry.get("num_ground_truth"),
            "index": index,
            "count": len(images),
        })


@admin.register(ExportRun)
class ExportRunAdmin(admin.ModelAdmin):
    """Queued ONNX/TensorRT export jobs — one row per checkpoint per action.

    Created by ``TrainedModelAdmin.export_onnx``/``export_trt`` and driven by
    ``training.jobs.run_export_onnx``/``run_export_trt``/``run_export_remote`` on
    the ``django_rq`` worker; nothing here is editable by hand.
    """

    list_display = [
        "__str__", "model", "kind", "checkpoint_label", "built_on", "status_badge",
        "built_precision", "output_path", "bundle_dir", "created_at",
    ]
    list_filter = ["kind", "status", "node", "model"]
    readonly_fields = [
        "model", "kind", "checkpoint_label", "checkpoint_path", "output_path",
        "precision", "input_hw", "gpu_infer", "node", "remote_build_id", "status", "result",
        "bundle_dir", "bundle_error", "last_error", "started_at", "finished_at",
        "created_at",
    ]

    def has_add_permission(self, request):
        # Rows are created by the export actions, not by hand.
        return False

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="built on", ordering="node")
    def built_on(self, obj):
        """Which machine compiled this. Matters because an engine only runs there."""
        if obj.node is None:
            return "local trainer"
        gpu = (obj.result or {}).get("gpu_name") or obj.node.gpu_name
        return f"{obj.node.name} ({gpu})" if gpu else obj.node.name

    @admin.display(description="precision")
    def built_precision(self, obj):
        """The engine's actual precision vs. what was requested — an fp16
        request can silently fall back to fp32 (e.g. Faster R-CNN with a
        dynamic profile; see ``export_trt.html``'s help text)."""
        if obj.kind != ExportRun.TRT or not obj.precision:
            return ""
        built = (obj.result or {}).get("precision")
        if not built:
            return obj.precision
        if built != obj.precision:
            return f"{obj.precision} → {built} (fallback)"
        return built


@admin.register(BuildNode)
class BuildNodeAdmin(admin.ModelAdmin):
    """Machines that can compile TensorRT engines on their own GPU.

    A node is registered by hand — name, URL, token — and everything else on the
    row is filled in by pinging it. See ``docs/build-nodes.md`` for standing one
    up; the short version is that the token here must match the ``BUILDNODE_TOKEN``
    the node was started with.
    """

    list_display = [
        "name", "base_url", "status", "health_badge", "gpu_name",
        "tensorrt_version", "person_detector", "load", "last_seen_at",
    ]
    list_filter = ["status", "last_status"]
    search_fields = ["name", "base_url", "gpu_name"]
    actions = ["ping_selected", "generate_token"]
    readonly_fields = [
        "last_status", "last_error", "last_seen_at", "last_checked_at",
        "gpu_name", "compute_capability", "tensorrt_version", "driver_version",
        "health_detail", "created_at",
    ]
    fieldsets = [
        (None, {"fields": ["name", "base_url", "token", "status", "notes"]}),
        ("Last health check", {
            "fields": [
                "last_status", "last_error", "last_seen_at", "last_checked_at",
                "gpu_name", "compute_capability", "tensorrt_version",
                "driver_version", "health_detail",
            ],
        }),
    ]

    @admin.display(description="health", ordering="last_status")
    def health_badge(self, obj):
        return _status_badge(obj.last_status)

    @admin.display(description="person detector")
    def person_detector(self, obj):
        """Whether the node can supply a detector for a person-crop bundle.

        Worth a column of its own: a node without one builds plain-frame bundles
        fine and fails only on the pipelines that need cropping.
        """
        detector = (obj.last_health or {}).get("person_detector") or {}
        if not obj.last_health:
            return "—"
        if not detector.get("present"):
            return "missing"
        kinds = [k for k in ("onnx", "trt_graph") if detector.get(k)]
        return ", ".join(kinds) or "present"

    @admin.display(description="load")
    def load(self, obj):
        health = obj.last_health or {}
        if not health:
            return "—"
        return "busy" if health.get("busy") else f"idle ({health.get('queue_len', 0)} queued)"

    @admin.display(description="full /health response")
    def health_detail(self, obj):
        if not obj.last_health:
            return "Never pinged."
        return json.dumps(obj.last_health, indent=2)

    @admin.action(description="Ping selected build nodes")
    def ping_selected(self, request, queryset):
        """Refresh each node's health snapshot, synchronously.

        Deliberately not an RQ job: it is one short GET per node, and going through
        the worker would redisplay the page with the PREVIOUS snapshot while
        implying it was current. An operator pressing "ping" wants the answer now.
        """
        ok, bad = [], []
        for node in queryset:
            body = buildnode.record_health(node)
            (ok if body.get("status") == "ok" else bad).append(node)
        if ok:
            self.message_user(
                request,
                "Reachable: " + ", ".join(
                    f"{n.name} ({n.gpu_name or 'unknown GPU'}, TRT {n.tensorrt_version or '?'})"
                    for n in ok))
        for node in bad:
            self.message_user(
                request, f"{node.name}: {node.last_error or 'unhealthy'}",
                level=messages.WARNING)

    @admin.action(description="Generate a token for selected nodes")
    def generate_token(self, request, queryset):
        """Mint a token to copy into the node's ``BUILDNODE_TOKEN``.

        Convenience only — the node is authoritative. Setting this does not change
        anything on the node, so it is refused once a node is answering, where it
        would silently break a working link.
        """
        import secrets

        changed, skipped = [], []
        for node in queryset:
            if node.last_status == BuildNode.OK:
                skipped.append(node.name)
                continue
            node.token = secrets.token_hex(20)
            node.save(update_fields=["token"])
            changed.append(node)
        for node in changed:
            self.message_user(
                request,
                f"{node.name}: set BUILDNODE_TOKEN={node.token} on the node, then ping it.")
        if skipped:
            self.message_user(
                request,
                f"Left alone (currently healthy, a new token would break them): "
                f"{', '.join(skipped)}. Retire or clear the health first if you really "
                f"mean to rotate.",
                level=messages.WARNING)


# EvalRun's admin lives in ``eval_pipelines.admin`` (as the "Base Eval" proxy)
# so the base and pipeline evals sit together under the "Eval Pipelines" section.
