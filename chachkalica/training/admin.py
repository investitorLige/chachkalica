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
    EvalRun,
    Experiment,
    ExperimentDataset,
    ExperimentModel,
    RunResult,
    TrainedModel,
    TrainingRun,
    TrainingSettings,
)
from training import model_specs, pipelines
from training.forms import ExperimentModelForm
from training.services import combine, config_gen, ingest, promote, runner, teardown


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
    list_display = ["__str__", "configs_root", "runs_root", "default_device", "service_base_url"]

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
        "launch_selected", "resume_selected", "pause_selected", "live_training_report",
        "ingest_selected", "reconcile_selected", "kill_run_gracefully",
    ]
    readonly_fields = [
        "experiment", "status", "epoch_progress", "config_yaml_path", "output_dir",
        "last_error", "started_at", "finished_at", "results", "created_at",
    ]

    def has_add_permission(self, request):
        # Runs are created by the Experiment action, not by hand.
        return False

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

    def get_urls(self):
        custom = [
            path("report/", self.admin_site.admin_view(self.live_report_view),
                 name="training_trainingrun_live_report"),
            path("report/metrics/", self.admin_site.admin_view(self.live_report_metrics),
                 name="training_trainingrun_live_report_metrics"),
            path("hard-images/image/", self.admin_site.admin_view(self.hard_images_image),
                 name="training_trainingrun_hard_images_image"),
            path("hard-images/data/", self.admin_site.admin_view(self.hard_images_data),
                 name="training_trainingrun_hard_images_data"),
        ]
        return custom + super().get_urls()

    @admin.action(description="Live training report")
    def live_training_report(self, request, queryset):
        """Open the combined live report: epoch status, metric charts, hard images."""
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one training run.", level=messages.WARNING)
            return None
        run = queryset.first()
        return redirect(reverse("admin:training_trainingrun_live_report") + "?run=" + str(run.pk))

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
            "metrics_url": reverse("admin:training_trainingrun_live_report_metrics"),
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
        for run in queryset:
            if not run.config_yaml_path:
                self.message_user(request, f"Run #{run.pk} has no config; skipped.",
                                  level=messages.WARNING)
                continue
            queue.enqueue(jobs.run_training, run.pk, job_timeout=jobs.JOB_TIMEOUT)
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
        for run in queryset:
            if not run.config_yaml_path:
                self.message_user(request, f"Run #{run.pk} has no config; skipped.",
                                  level=messages.WARNING)
                continue
            queue.enqueue(jobs.run_training, run.pk, resume=True, job_timeout=jobs.JOB_TIMEOUT)
            run.status = TrainingRun.QUEUED
            run.save(update_fields=["status"])
        self.message_user(request, "Resume job(s) queued — refresh to see progress.")

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
    actions = ["evaluate", "preview_on_dataset", "export_onnx", "export_trt", "view_hard_val_images"]
    readonly_fields = ["source_run_result", "created_at", "updated_at"]

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
                tile_width_pct=_float("tile_width_pct"),
                tile_height_pct=_float("tile_height_pct"),
                overlap=_float("overlap"),
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

    @staticmethod
    def _experiment_pipeline_defaults(model) -> dict:
        """Pipeline config saved on the experiment this model was trained in."""
        rr = getattr(model, "source_run_result", None)
        experiment = getattr(getattr(rr, "run", None), "experiment", None)
        if experiment is None or not experiment.pipeline:
            return {}
        return {
            "pipeline": experiment.pipeline,
            "detector_checkpoint": experiment.detector_checkpoint,
            "detector_expand_ratio": experiment.detector_expand_ratio,
            "tile_width_pct": experiment.tile_width_pct,
            "tile_height_pct": experiment.tile_height_pct,
            "overlap": experiment.overlap,
            "chain": ", ".join(experiment.chain or []),
        }

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
                "tile_width_pct": (request.POST.get("tile_width_pct") or "").strip(),
                "tile_height_pct": (request.POST.get("tile_height_pct") or "").strip(),
                "overlap": (request.POST.get("overlap") or "").strip(),
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

    @admin.action(description="Export best + last to ONNX…")
    def export_onnx(self, request, queryset):
        """Export a model's best and last ``.pt`` to ONNX under a chosen directory.

        Drives the trainer service's synchronous ``/export_onnx`` (the export must
        run in the trainer's torch env), writing ``<name>-best.onnx`` /
        ``<name>-last.onnx`` (each with a sibling ``.meta.json``) into the operator's
        directory. Relative paths resolve against the project root, like the other
        training paths.
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
            exported = 0
            for label, checkpoint in checkpoints:
                onnx_path = out_dir / f"{stem}-{label}.onnx"
                try:
                    result = runner.export_onnx(checkpoint, onnx_path)
                except Exception as exc:  # noqa: BLE001 - surface service/network errors
                    self.message_user(
                        request, f"{label} ({checkpoint}): {exc}", level=messages.ERROR)
                    continue
                exported += 1
                self.message_user(
                    request,
                    f"Exported {label} → {result.get('onnx_path')} "
                    f"(+ {Path(result.get('meta_path', '')).name}).")
            if exported:
                self.message_user(request, f"Exported {exported} checkpoint(s) for {model.name}.")
            return None

        ts = TrainingSettings.load()
        default_dir = config_gen._resolve(ts.runs_root) / "onnx_exports"
        context = {
            **self.admin_site.each_context(request),
            "title": f"Export {model.name} to ONNX",
            "model": model,
            "checkpoints": [{"label": label, "path": path} for label, path in checkpoints],
            "default_output_dir": str(default_dir),
            "action": "export_onnx",
            "selected": [str(model.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/export_onnx.html", context)

    # ------------------------------------------------------------------- TensorRT export
    @admin.action(description="Export best + last to TensorRT…")
    def export_trt(self, request, queryset):
        """Build TensorRT engines for a model's best and last ``.pt`` under a chosen dir.

        Drives the trainer service's synchronous ``/export_trt`` (the build must run
        in the trainer's GPU env), writing ``<name>-best.engine`` / ``<name>-last.engine``
        (each with sibling ``.meta.json`` + ``.engine.json``) into the operator's
        directory. Relative paths resolve against the project root.

        NOTE: unlike ONNX export, this uses the GPU and competes with active training;
        engines are non-portable (tied to the trainer's GPU + TensorRT version).
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
            precision = (request.POST.get("precision") or "fp16").strip()
            if precision not in ("fp16", "fp32"):
                precision = "fp16"
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
            out_dir = config_gen._resolve(output_dir)
            stem = self._onnx_stem(model.name)
            exported = 0
            for label, checkpoint in checkpoints:
                engine_path = out_dir / f"{stem}-{label}.engine"
                try:
                    result = runner.export_trt(
                        checkpoint, engine_path, precision=precision, input_hw=input_hw)
                except Exception as exc:  # noqa: BLE001 - surface service/network errors
                    self.message_user(
                        request, f"{label} ({checkpoint}): {exc}", level=messages.ERROR)
                    continue
                exported += 1
                built = result.get("precision", precision)
                self.message_user(
                    request,
                    f"Built {label} → {result.get('engine_path')} "
                    f"(+ {Path(result.get('meta_path', '')).name}) [{built}].")
                # Loud warning when an fp16 request silently downgraded — for
                # Faster R-CNN this means the input size wasn't pinned.
                if precision == "fp16" and built != "fp16":
                    self.message_user(
                        request,
                        f"{label}: requested FP16 but built {built.upper()} (the FP16 build "
                        f"produced no engine). For Faster R-CNN, set a static Input size to "
                        f"get FP16.", level=messages.WARNING)
            if exported:
                self.message_user(
                    request, f"Built {exported} engine(s) for {model.name} ({precision}).")
            return None

        ts = TrainingSettings.load()
        default_dir = config_gen._resolve(ts.runs_root) / "trt_exports"
        context = {
            **self.admin_site.each_context(request),
            "title": f"Export {model.name} to TensorRT",
            "model": model,
            "checkpoints": [{"label": label, "path": path} for label, path in checkpoints],
            "default_output_dir": str(default_dir),
            "action": "export_trt",
            "selected": [str(model.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/training/export_trt.html", context)

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
                tile_width_pct=_float_or_none(request.GET.get("tile_width_pct")),
                tile_height_pct=_float_or_none(request.GET.get("tile_height_pct")),
                overlap=_float_or_none(request.GET.get("overlap")),
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


# EvalRun's admin lives in ``eval_pipelines.admin`` (as the "Base Eval" proxy)
# so the base and pipeline evals sit together under the "Eval Pipelines" section.
