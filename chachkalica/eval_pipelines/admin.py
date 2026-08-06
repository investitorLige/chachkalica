"""The "Eval Pipelines" admin section: Base Eval + Pipeline Eval + combined list.

Base Eval is the existing :class:`training.EvalRun` re-homed here via the
:class:`~eval_pipelines.models.BaseEval` proxy. Pipeline Eval is
:class:`~eval_pipelines.models.PipelineEvalRun`, split into one list per pipeline
type. The single "All pipeline evals" list is backed by
:class:`~eval_pipelines.models.CombinedEval` — a read-only union of base and
pipeline evals so both sit in one changelist and can be Analyzed together.

Every list shares the read-only, job-driven shape, the "Analyze" comparison
action, and a "Promote predictions to source labels" action that turns a finished
run's predictions into the dataset's source-of-truth labels.
"""

import django_rq
from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.template.response import TemplateResponse

from fleet.admin import _status_badge
from fleet.services import datasets as datasets_svc
from training import jobs
from training.models import EvalRun
from training.services import config_gen, runner

from eval_pipelines.models import (
    BaseEval,
    BatchDetectEval,
    BatchPeopleEval,
    ChainEval,
    CombinedEval,
    PeopleDetectFirstEval,
    PipelineEvalRun,
)


def _queue():
    return django_rq.get_queue("default")


def _analyze(model_admin, request, queryset, title):
    """Shared "Analyze / compare" action body for base and pipeline evals."""
    from training.services import eval_analytics

    runs = [e for e in queryset if isinstance(e.metrics, dict) and e.metrics]
    skipped = [e for e in queryset if not (isinstance(e.metrics, dict) and e.metrics)]
    if skipped:
        model_admin.message_user(
            request,
            "Skipped eval(s) with no ingested metrics yet: "
            + ", ".join(f"#{e.pk}" for e in skipped),
            level=messages.WARNING,
        )
    if not runs:
        model_admin.message_user(request, "No evaluated metrics to analyze.",
                                 level=messages.WARNING)
        return None

    context = {
        **model_admin.admin_site.each_context(request),
        "title": title,
        **eval_analytics.compare(runs),
    }
    return TemplateResponse(request, "admin/training/eval_analytics.html", context)


class EvalDisplayMixin:
    """Shared metric columns for every eval list (base, pipeline, combined)."""

    @admin.display(description="models")
    def models_display(self, obj):
        """The primary model, plus any combined ones ("A + B").

        ``CombinedEval`` rows (the read-only union view) have no
        ``combined_models`` — the view only carries the primary FK — so those
        just show the one model name.
        """
        combined = getattr(obj, "combined_models", None)
        extra = combined.all() if combined is not None else []
        if not extra:
            return obj.trained_model.name
        return " + ".join([obj.trained_model.name, *(m.name for m in extra)])

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="mAP50")
    def map50(self, obj):
        return obj.metric("map50")

    @admin.display(description="mAP50-95")
    def map50_95(self, obj):
        return obj.metric("map50_95")

    @admin.display(description="eval time")
    def eval_time(self, obj):
        seconds = obj.metric("eval_seconds")
        return f"{seconds:.1f}s" if isinstance(seconds, (int, float)) else "—"


class PromoteLabelsMixin:
    """Shared "promote predictions to source labels" action.

    Reads a finished run's predictions trainer-side (they are torch pickles the
    app can't open), remaps the predicted classes onto the dataset's own class
    space by name, keeps boxes above an operator-chosen confidence, backs up any
    existing source labels, and writes the predictions as the dataset's source
    ``labels/``. Refreshes the dataset's ``has_labels`` flag afterwards.

    Subclasses set :attr:`promote_kind` (``"base"`` / ``"pipeline"``); the
    combined list overrides :meth:`_promote_target` to resolve each row back to
    its real eval before promoting.
    """

    promote_kind = None

    def _promote_target(self, obj):
        """Return ``(real_eval, kind)`` for one selected row."""
        return obj, self.promote_kind

    @admin.action(description="Promote predictions to source labels…")
    def promote_labels(self, request, queryset):
        targets = [self._promote_target(obj) for obj in queryset]

        if request.POST.get("apply"):
            try:
                threshold = float(request.POST.get("score_threshold"))
            except (TypeError, ValueError):
                self.message_user(request, "Enter a numeric confidence threshold.",
                                  level=messages.ERROR)
                return None
            for eval_obj, kind in targets:
                label = f"{kind} eval #{eval_obj.pk}"
                try:
                    payload = config_gen.build_promote_payload(eval_obj, kind, threshold)
                    summary = runner.promote_labels(payload)
                except (ValueError, RuntimeError, OSError) as exc:
                    self.message_user(request, f"{label}: {exc}", level=messages.ERROR)
                    continue
                # Files changed on disk — refresh the source-labels flag so the
                # admin/training/project-creation paths see the new labels.
                datasets_svc.detect_labels(eval_obj.dataset)
                msg = (f"{label}: wrote {summary['labels_written']} label file(s) "
                       f"({summary['boxes_written']} box(es) ≥ {threshold:g}) to "
                       f"{summary['labels_dir']}")
                if summary["backed_up"]:
                    msg += (f"; backed up {summary['backed_up']} prior label(s) to "
                            f"{summary['backup_dir']}")
                if summary["dropped_unmapped"]:
                    msg += (f"; dropped {summary['dropped_unmapped']} prediction(s) whose "
                            "class is not in the dataset")
                self.message_user(request, msg)
            return None

        default_threshold = next(
            (o.score_threshold for o, _ in targets if o.score_threshold is not None), None
        ) or 0.25
        context = {
            **self.admin_site.each_context(request),
            "title": "Promote predictions to source labels",
            "targets": [{"label": str(o), "dataset": o.dataset.name} for o, _ in targets],
            "default_threshold": default_threshold,
            "action": "promote_labels",
            "selected": [str(o.pk) for o in queryset],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/eval_pipelines/promote_labels.html", context)


@admin.register(BaseEval)
class BaseEvalAdmin(EvalDisplayMixin, PromoteLabelsMixin, admin.ModelAdmin):
    promote_kind = "base"
    list_display = ["__str__", "models_display", "dataset", "status_badge",
                    "map50", "map50_95", "eval_time", "created_at"]
    list_filter = ["status", "trained_model"]
    actions = ["analyze_selected", "launch_selected", "reconcile_selected", "promote_labels"]
    readonly_fields = [
        "trained_model", "dataset", "label_source", "annotator", "explicit_labels_path",
        "score_threshold",
        "status", "request_yaml_path", "output_dir", "metrics", "last_error",
        "started_at", "finished_at", "created_at",
    ]

    def has_add_permission(self, request):
        return False

    @admin.action(description="Analyze / compare metrics of selected eval(s)…")
    def analyze_selected(self, request, queryset):
        return _analyze(self, request, queryset, "Eval metrics comparison")

    @admin.action(description="Launch / relaunch on trainer service")
    def launch_selected(self, request, queryset):
        queue = _queue()
        prev_job = None  # chain so a multi-select launch doesn't race the trainer's single slot
        for eval_run in queryset:
            if not eval_run.request_yaml_path:
                self.message_user(request, f"Eval #{eval_run.pk} has no request; skipped.",
                                  level=messages.WARNING)
                continue
            prev_job = queue.enqueue(
                jobs.run_eval, eval_run.pk,
                depends_on=jobs.depends_on(prev_job), job_timeout=jobs.JOB_TIMEOUT,
            )
            eval_run.status = BaseEval.QUEUED
            eval_run.save(update_fields=["status"])
        self.message_user(request, "Eval job(s) queued — refresh to see progress.")

    @admin.action(description="Reconcile status from trainer / disk")
    def reconcile_selected(self, request, queryset):
        from training.services import reconcile

        for eval_run in queryset:
            outcome = reconcile.reconcile_eval(eval_run)
            self.message_user(request, f"Eval #{eval_run.pk}: {outcome}")


class PipelineEvalRunAdmin(EvalDisplayMixin, PromoteLabelsMixin, admin.ModelAdmin):
    """Shared admin for the per-pipeline proxies (Batch detect, Chain, …).

    Each pipeline type gets its own list via a proxy subclass registered below, so
    a run lands under the list matching the pipeline chosen in the "Evaluate with
    a pipeline…" action. ``pipeline`` is dropped from the columns/filters since
    every row in a given list shares it.
    """

    promote_kind = "pipeline"
    list_display = ["__str__", "models_display", "dataset", "status_badge",
                    "map50", "map50_95", "eval_time", "created_at"]
    list_filter = ["status", "trained_model"]
    actions = ["analyze_selected", "launch_selected", "reconcile_selected", "promote_labels"]
    readonly_fields = [
        "trained_model", "dataset", "label_source", "annotator", "explicit_labels_path",
        "pipeline", "detector_checkpoint", "detector_expand_ratio",
        "tile_width_pct", "tile_height_pct", "overlap", "chain",
        "score_threshold",
        "status", "request_yaml_path", "output_dir", "metrics", "last_error",
        "started_at", "finished_at", "created_at",
    ]

    def has_add_permission(self, request):
        return False

    @admin.action(description="Analyze / compare metrics of selected pipeline eval(s)…")
    def analyze_selected(self, request, queryset):
        return _analyze(self, request, queryset, "Pipeline eval metrics comparison")

    @admin.action(description="Launch / relaunch on trainer service")
    def launch_selected(self, request, queryset):
        queue = _queue()
        prev_job = None  # chain so a multi-select launch doesn't race the trainer's single slot
        for pe in queryset:
            if not pe.request_yaml_path:
                self.message_user(request, f"Pipeline eval #{pe.pk} has no request; skipped.",
                                  level=messages.WARNING)
                continue
            prev_job = queue.enqueue(
                jobs.run_pipeline_eval, pe.pk,
                depends_on=jobs.depends_on(prev_job), job_timeout=jobs.JOB_TIMEOUT,
            )
            pe.status = PipelineEvalRun.QUEUED
            pe.save(update_fields=["status"])
        self.message_user(request, "Pipeline eval job(s) queued — refresh to see progress.")

    @admin.action(description="Reconcile status from trainer / disk")
    def reconcile_selected(self, request, queryset):
        from training.services import reconcile

        for pe in queryset:
            outcome = reconcile.reconcile_pipeline(pe)
            self.message_user(request, f"Pipeline eval #{pe.pk}: {outcome}")


@admin.register(CombinedEval)
class CombinedEvalAdmin(EvalDisplayMixin, PromoteLabelsMixin, admin.ModelAdmin):
    """The combined "All pipeline evals" list — base + every pipeline in one place.

    Read-only union view (see :class:`CombinedEval`): rows can't be edited here,
    but "Analyze / compare" spans base and pipeline runs at once, and the launch /
    reconcile / promote actions route each selected row back to its real
    :class:`EvalRun` / :class:`PipelineEvalRun` by ``kind``.
    """

    list_display = ["__str__", "models_display", "pipeline", "dataset", "status_badge",
                    "map50", "map50_95", "eval_time", "created_at"]
    list_display_links = None
    list_filter = ["status", "pipeline", "trained_model"]
    actions = ["analyze_selected", "launch_selected", "reconcile_selected", "promote_labels",
               "delete_selected_evals"]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False  # backed by a DB view — read-only

    def has_delete_permission(self, request, obj=None):
        return False

    def _promote_target(self, obj):
        if obj.kind == CombinedEval.BASE:
            return EvalRun.objects.get(pk=obj.orig_id), "base"
        return PipelineEvalRun.objects.get(pk=obj.orig_id), "pipeline"

    @admin.action(description="Analyze / compare metrics of selected eval(s)…")
    def analyze_selected(self, request, queryset):
        return _analyze(self, request, queryset, "Eval metrics comparison")

    @admin.action(description="Launch / relaunch on trainer service")
    def launch_selected(self, request, queryset):
        queue = _queue()
        prev_job = None  # chain so a multi-select launch doesn't race the trainer's single slot
        for row in queryset:
            eval_obj, kind = self._promote_target(row)
            if not eval_obj.request_yaml_path:
                self.message_user(request, f"{kind} eval #{eval_obj.pk} has no request; skipped.",
                                  level=messages.WARNING)
                continue
            job = jobs.run_eval if kind == "base" else jobs.run_pipeline_eval
            prev_job = queue.enqueue(
                job, eval_obj.pk,
                depends_on=jobs.depends_on(prev_job), job_timeout=jobs.JOB_TIMEOUT,
            )
            eval_obj.status = eval_obj.QUEUED
            eval_obj.save(update_fields=["status"])
        self.message_user(request, "Eval job(s) queued — refresh to see progress.")

    @admin.action(description="Reconcile status from trainer / disk")
    def reconcile_selected(self, request, queryset):
        from training.services import reconcile

        for row in queryset:
            eval_obj, kind = self._promote_target(row)
            outcome = (reconcile.reconcile_eval(eval_obj) if kind == "base"
                       else reconcile.reconcile_pipeline(eval_obj))
            self.message_user(request, f"{kind} eval #{eval_obj.pk}: {outcome}")

    @admin.action(description="Delete selected eval(s) and their generated files…")
    def delete_selected_evals(self, request, queryset):
        """Delete the real base/pipeline eval(s) behind selected combined rows.

        This view is otherwise read-only (see :meth:`has_delete_permission`), so
        it's the one place that needs its own delete action. Deleting the real
        row fires the post_delete cleanup signal (``training.signals`` for base
        evals, ``eval_pipelines.signals`` for pipeline evals), which removes the
        run's output dir and generated request YAML from disk.
        """
        targets = [self._promote_target(obj) for obj in queryset]

        if request.POST.get("apply"):
            for eval_obj, kind in targets:
                label = f"{kind} eval #{eval_obj.pk}"
                eval_obj.delete()
                self.message_user(request, f"Deleted {label} and its generated files.")
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": "Delete selected eval(s)",
            "targets": [{"label": str(o), "dataset": o.dataset.name} for o, _ in targets],
            "action": "delete_selected_evals",
            "selected": [str(o.pk) for o in queryset],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/eval_pipelines/delete_evals.html", context)


# One admin list per pipeline type, all sharing the behaviour above. Each proxy's
# manager scopes the queryset to its pipeline, so these lists partition the runs.
admin.site.register(BatchDetectEval, PipelineEvalRunAdmin)
admin.site.register(PeopleDetectFirstEval, PipelineEvalRunAdmin)
admin.site.register(BatchPeopleEval, PipelineEvalRunAdmin)
admin.site.register(ChainEval, PipelineEvalRunAdmin)
