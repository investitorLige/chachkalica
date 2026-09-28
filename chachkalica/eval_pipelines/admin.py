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

import json
from pathlib import Path

import django_rq
from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.http import FileResponse, Http404, HttpResponseRedirect, JsonResponse
from django.template.loader import render_to_string
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_POST
from django.utils.http import urlencode
from django.utils.safestring import mark_safe

from fleet import admin as fleet_admin
from fleet import jobs as fleet_jobs
from fleet.admin import _status_badge
from fleet.models import DatasetMeasureRun
from fleet.services import datasets as datasets_svc
from fleet.services import measures as measures_svc
from fleet.services.paths import source_root
from training import jobs
from training.admin import _hard_image_path_from_request, _hard_image_token, _preview_index
from training.models import EvalRun
from training.services import config_gen, eval_analytics, runner, tag_analytics, test_sets

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


def _hard_images_artifact(output_dir):
    """The one ``*_hard_images.json`` file in an eval's ``output_dir``, or None.

    Named ``eval_hard_images.json`` for a base eval and
    ``predictions_hard_images.json`` for a chachak pipeline eval (single-model
    or combined-checkpoint) — matched by glob so this doesn't have to know
    which. Missing for evals that predate the feature or had no ground truth.
    """
    if not output_dir:
        return None
    matches = sorted(Path(output_dir).glob("*_hard_images.json"))
    return matches[0] if matches else None


#: Said on the compare page when an eval has no match table, in place of the
#: Tag analytics button. Naming the action beats omitting the button silently,
#: which is indistinguishable from the feature not existing.
NO_MATCH_TABLE_NOTE = (
    "no match table — run “Build tag analytics data” on this eval to slice it"
)


def _attach_eval_links(columns):
    """Add the per-eval viewer links to each compare-page column, in place.

    ``hard_images_url`` is None when that eval wrote no worst-images artifact.
    ``tag_analytics_url`` now hangs on the match table alone, which is genuinely
    the page's one requirement: tag analytics renders from the match outcomes and
    reports whatever else is missing rather than refusing. Previously this gate
    was right by accident and wrong in effect — a missing tag sidecar still let
    the button through, and the page it led to then declined to render.
    """
    for column in columns:
        query = urlencode({"kind": column["kind"], "eval": column["orig_id"]})
        has_hard_images = _hard_images_artifact(column.get("output_dir")) is not None
        column["hard_images_url"] = (
            reverse("admin:eval_pipelines_combinedeval_hard_images") + "?" + query
        ) if has_hard_images else None
        has_matches = tag_analytics.match_table_artifact(column.get("output_dir")) is not None
        column["tag_analytics_url"] = (
            reverse("admin:eval_pipelines_combinedeval_tag_analytics") + "?" + query
        ) if has_matches else None
        column["tag_analytics_note"] = "" if has_matches else NO_MATCH_TABLE_NOTE
    return columns


def _resolve_eval_for_hard_images(request):
    """Look up the real ``EvalRun``/``PipelineEvalRun`` a hard-images URL points at.

    The worst-images viewer is shared by every eval list (see
    :meth:`CombinedEvalAdmin.hard_images_view`), so it's addressed by
    ``?kind=base|pipeline&eval=<pk>`` rather than a list-specific pk space.
    """
    kind = request.GET.get("kind")
    pk = request.GET.get("eval")
    model = PipelineEvalRun if kind == CombinedEval.PIPELINE else EvalRun
    obj = model.objects.filter(pk=pk).first()
    if obj is None:
        raise Http404("unknown eval")
    return obj


def _load_hard_images_for(eval_obj):
    """Parse the eval's hard-images artifact, or None if missing/unreadable."""
    artifact = _hard_images_artifact(getattr(eval_obj, "output_dir", ""))
    if artifact is None:
        return None
    try:
        return json.loads(artifact.read_text())
    except (ValueError, OSError):
        return None


def _analyze(model_admin, request, queryset, title):
    """Shared "Analyze / compare" action body for base and pipeline evals."""
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

    compared = eval_analytics.compare(runs)
    _attach_eval_links(compared["columns"])
    context = {
        **model_admin.admin_site.each_context(request),
        "title": title,
        # How the per-tag section addresses these same evals: the kind/orig_id
        # pair every column already carries, so no new identity is invented.
        "eval_tokens": ",".join(
            f"{column['kind']}:{column['orig_id']}" for column in compared["columns"]),
        "tag_compare_url": reverse("admin:eval_pipelines_combinedeval_tag_compare"),
        **compared,
    }
    return TemplateResponse(request, "admin/training/eval_analytics.html", context)


class EvalDisplayMixin:
    """Shared metric columns for every eval list (base, pipeline, combined)."""

    class Media:
        css = {"all": ("eval_pipelines/metrics_summary.css",)}

    @admin.display(description="metrics")
    def metrics_pretty(self, obj):
        """Headline tiles + per-class table + confusion matrix, not a JSON dump.

        Swapped in for the raw ``metrics`` field in ``readonly_fields`` — left
        alone, Django's default readonly rendering for a JSONField is a single
        unbroken ``json.dumps(...)`` line. ``eval_analytics.summary`` does the
        reshaping (shared with the "Analyze / compare" action); this just
        renders it.
        """
        html = render_to_string("admin/eval_pipelines/metrics_summary.html",
                                 eval_analytics.summary(obj))
        return mark_safe(html)

    @admin.display(description="class map")
    def class_map_display(self, obj):
        """How the dataset's classes were translated before scoring, if at all.

        Worth a row of its own rather than a raw JSONField dump: it is the one
        thing that changes what the metrics beside it *mean*. An eval that
        collapsed four weapon classes onto a single-class detector, or dropped
        the two it could not predict, does not report the same number as one
        that scored the dataset as labelled — and months later the request YAML
        is not where anyone looks first.
        """
        from training.services import class_sync

        return class_sync.describe(obj.class_map) or "— scored the dataset's own classes"

    @admin.display(description="models")
    def models_display(self, obj):
        """The primary model, plus any combined ones ("A + B").

        ``model_label()`` rather than the FK's name: an eval of an exported
        artifact or a bundle has no catalogue entry, and names itself by the
        path it stamped on itself when it was queued.

        ``CombinedEval`` rows (the read-only union view) have no
        ``combined_models`` — the view only carries the primary FK — so those
        just show the one model name.
        """
        combined = getattr(obj, "combined_models", None)
        extra = combined.all() if combined is not None else []
        if not extra:
            return obj.model_label()
        return " + ".join([obj.model_label(), *(m.name for m in extra)])

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
    combined list overrides :meth:`_eval_target` to resolve each row back to
    its real eval before promoting.
    """

    promote_kind = None

    def _eval_target(self, obj):
        """Return ``(real_eval, kind)`` for one selected row.

        Shared with :class:`TagAnalyticsMixin`, which needs the same
        resolution: the combined list's rows come from a database view and have
        to be traced back to a real eval before anything reads their files.
        """
        return obj, self.promote_kind

    @admin.action(description="Promote predictions to source labels…")
    def promote_labels(self, request, queryset):
        targets = [self._eval_target(obj) for obj in queryset]

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


def _tag_analytics_url(eval_obj, kind: str) -> str:
    """Link to the shared tag-analytics report for one eval."""
    return reverse("admin:eval_pipelines_combinedeval_tag_analytics") + "?" + urlencode(
        {"kind": kind, "eval": eval_obj.pk}
    )


#: Ceiling on a compare request. Each column parses its own match table, so a
#: hand-edited URL asking for 200 is a way to make the box read them all.
MAX_COMPARE_COLUMNS = 12


def _eval_by_kind(kind: str, pk):
    """One eval addressed as ``base:12`` / ``pipeline:7`` is."""
    model = PipelineEvalRun if kind == CombinedEval.PIPELINE else EvalRun
    return model.objects.filter(pk=pk).first()


def _resolve_eval_for_tags(request):
    """The real ``EvalRun``/``PipelineEvalRun`` a tag-analytics URL points at.

    Addressed by ``?kind=base|pipeline&eval=<pk>`` for the same reason the
    worst-images viewer is: one report serves every eval list, so it cannot be
    hung off any one list's pk space.
    """
    kind = request.GET.get("kind")
    model = PipelineEvalRun if kind == CombinedEval.PIPELINE else EvalRun
    obj = model.objects.filter(pk=request.GET.get("eval")).first()
    if obj is None:
        raise Http404("unknown eval")
    return obj, (CombinedEval.PIPELINE if kind == CombinedEval.PIPELINE else CombinedEval.BASE)


def _load_tag_analysis(eval_obj):
    """``((table, index, sources), None)`` for an eval, or ``(None, fatal)``.

    The match table is the one hard requirement: without it there are no images,
    no ground-truth rows and nothing to score, so its absence is the only thing
    that blanks the page. Everything else — the annotator's tag answers, and in
    time the measured sidecar — costs some tags and not the page, so it comes
    back as a *source* row naming what is missing and what produces it. That is
    the difference between "tag analytics needs annotators" and "tag analytics
    works on every eval, and here is what would make it say more".
    """
    artifact = tag_analytics.match_table_artifact(getattr(eval_obj, "output_dir", ""))
    if artifact is None:
        return None, (
            "This eval has no match table. Evals run before tag analytics existed "
            "did not write one — select the eval and run “Build tag analytics data…”, "
            "which rebuilds it from the predictions it already saved (no GPU, no "
            "re-inference)."
        )
    try:
        table = tag_analytics.load_table(artifact)
    except (OSError, ValueError) as exc:
        return None, f"Could not read {artifact.name}: {exc}"

    detail = f"{artifact.name}, version {table.version}"
    if table.backfilled:
        detail += ", rebuilt from the saved predictions"
    # Two separate gaps, two separate sentences: version 2 added ground-truth
    # extents (the box-size tag) and version 3 prediction boxes (the whole
    # per-person section). Naming only the older one would leave somebody on a
    # version 2 table wondering why People says nothing.
    missing = []
    if not table.has_geometry:
        missing.append("no per-box geometry, so the box-size tag is unavailable")
    if table.version < 3:
        missing.append("no prediction geometry, so people cannot be scored")
    sources = [{
        "key": "match_table", "label": "Match outcomes",
        "state": "ok" if not missing else "partial",
        "detail": detail + (" — " + "; ".join(missing) if missing else ""),
        "path": str(artifact),
    }]

    labels_dir = None
    try:
        labels_dir = config_gen.resolve_label_dir(
            eval_obj.dataset, eval_obj.label_source,
            getattr(eval_obj, "annotator", None),
            getattr(eval_obj, "explicit_labels_path", "") or "",
        )
    except ValueError as exc:
        sources.append({"key": "annotator_tags", "label": "Annotator answers",
                        "state": "missing", "path": "",
                        "detail": f"Cannot locate this eval's labels: {exc}"})

    document = tag_analytics.load_tag_document(labels_dir) if labels_dir else None
    specs = datasets_svc.dataset_tag_specs(eval_obj.dataset)
    reasons = {}
    if labels_dir is not None:
        if document is None:
            missing = (
                f"No annotation_tags.json next to this eval's labels ({labels_dir}). "
                "Tag answers land there when the dataset's annotator projects are "
                "synced — define tags on the dataset, sync, and re-run or rebuild."
            )
            sources.append({"key": "annotator_tags", "label": "Annotator answers",
                            "state": "missing", "detail": missing,
                            "path": str(labels_dir)})
            # Every declared tag is missing for the same one reason; saying it
            # per tag beats making the reader infer it from an absence.
            reasons = {str(spec["name"]): missing for spec in specs}
        else:
            sources.append({
                "key": "annotator_tags", "label": "Annotator answers", "state": "ok",
                "detail": f"{len(document.get('tags') or [])} tag(s) answered over "
                          f"{len(document.get('images') or {})} image(s)",
                "path": str(labels_dir),
            })

    # Image measurements are a property of the dataset, not of any one eval or
    # label set, so they are looked up by dataset rather than beside the labels.
    measures = measures_svc.load(eval_obj.dataset.name)
    state, detail = measures_svc.freshness(eval_obj.dataset.name, measures)
    sources.append({
        "key": "measures", "label": "Image measurements",
        "state": "ok" if state == "ok" else ("partial" if state == "stale" else "missing"),
        "detail": detail,
        "path": str(measures_svc.measures_path(eval_obj.dataset.name)),
    })
    if state == "stale":
        # Still describes the frames it was taken from, so it is offered with a
        # warning rather than thrown away.
        for _name, _key, label, _unit in tag_analytics.IMAGE_MEASURE_TAGS:
            reasons.setdefault(label, detail)

    box_measures = tag_analytics.load_box_measures(labels_dir) if labels_dir else None
    if box_measures is None:
        box_detail = (
            "Person size and pose have not been measured for this label set. "
            "This one needs the trainer's GPU — a person detector and the posture "
            "engine over each frame once."
        )
        box_state = "missing"
    elif not box_measures.get("complete", True):
        done = (box_measures.get("totals") or {}).get("images_done", "?")
        total = (box_measures.get("totals") or {}).get("images", "?")
        box_state = "partial"
        box_detail = (
            f"Measured {done} of {total} frame(s) before the pass was stopped. The "
            "buckets below are cut over what it did measure, which is a smaller "
            "population than the eval."
        )
    else:
        totals = box_measures.get("totals") or {}
        box_state = "ok"
        box_detail = (
            f"{totals.get('boxes', 0)} box(es) measured against "
            f"{box_measures.get('person_source', 'unknown')} person boxes; "
            f"{totals.get('without_person', 0)} had no person around them; "
            f"{totals.get('people_found', 0)} person(s) found"
        )
    sources.append({"key": "box_measures", "label": "Box measures (person / pose)",
                    "state": box_state, "detail": box_detail,
                    "path": str(labels_dir or "")})

    # People are the third population this page can score, and they need both
    # halves: the sidecar's person boxes and the match table's prediction
    # geometry. build_persons returns None when either is missing, and the page
    # prints which rather than rendering an empty section.
    persons = tag_analytics.build_persons(table, box_measures)
    if persons is not None and persons.count:
        person_detail = (
            f"{persons.count} person(s) over {persons.measured_images} measured frame(s)"
            + (f"; {persons.drifted_images} frame(s) skipped for row drift"
               if persons.drifted_images else ""))
        person_cause = ""
    elif table.version < 3:
        person_detail, person_cause = tag_analytics.NO_PREDICTION_GEOMETRY_DETAIL, "match_table"
    else:
        person_detail, person_cause = tag_analytics.NO_PERSONS_DETAIL, "measures"
    sources.append({
        "key": "persons", "label": "People (per-person scoring)",
        "state": "ok" if not person_cause else "missing",
        # Which pass to run differs by cause, and the page puts the button for
        # it on this row, so the cause travels rather than being re-guessed
        # from the wording of the sentence above.
        "cause": person_cause,
        "detail": person_detail,
        "path": str(labels_dir or ""),
    })

    index = tag_analytics.build_tag_index(
        table, document, declared_specs=specs, measures=measures,
        box_measures=box_measures, reasons=reasons, persons=persons)
    return (table, index, sources), None


class TagAnalyticsMixin:
    """"Tag analytics" + "Build tag analytics data" actions for an eval list.

    The report itself lives on the combined list (one page, every eval kind —
    see :meth:`CombinedEvalAdmin.tag_analytics_view`); these actions are how
    each list reaches it and how an older eval gets the data it needs first.
    """

    @admin.action(description="Tag analytics (per-tag / cross-tag metrics)…")
    def open_tag_analytics(self, request, queryset):
        rows = list(queryset)
        if len(rows) != 1:
            self.message_user(
                request,
                "Select exactly one eval — tag analytics slices a single eval's own "
                "predictions. Use “Analyze / compare” to put several side by side.",
                level=messages.WARNING,
            )
            return None
        eval_obj, kind = self._eval_target(rows[0])
        return HttpResponseRedirect(_tag_analytics_url(eval_obj, kind))

    @admin.action(description="Build tag analytics data (rebuild match table)")
    def build_tag_data(self, request, queryset):
        """Rebuild each selected eval's match table from its saved predictions.

        Synchronous, like "Promote predictions to source labels": the trainer
        reads the predictions file and the label files and writes one JSON. It
        loads no model and never touches the GPU, so it does not queue behind a
        training run — but it is not instant on a large eval, which is why the
        result is reported per row rather than fired and forgotten.
        """
        for row in queryset:
            eval_obj, kind = self._eval_target(row)
            label = f"{kind} eval #{eval_obj.pk}"
            try:
                payload = config_gen.build_match_table_payload(eval_obj, kind)
                summary = runner.build_match_table(payload)
            except (ValueError, RuntimeError, OSError) as exc:
                self.message_user(request, f"{label}: {exc}", level=messages.ERROR)
                continue
            note = ""
            if summary["images_without_labels"]:
                note = (f"; {summary['images_without_labels']} image(s) had no readable "
                        "label file and contribute no ground truth")
            self.message_user(
                request,
                f"{label}: built {Path(summary['match_table_path']).name} — "
                f"{summary['images']} image(s), {summary['ground_truth_boxes']} "
                f"ground-truth box(es), {summary['predictions']} prediction(s){note}",
            )


@admin.register(BaseEval)
class BaseEvalAdmin(EvalDisplayMixin, PromoteLabelsMixin, TagAnalyticsMixin, admin.ModelAdmin):
    # Rendered by class_map_display instead — the raw JSONField would show the
    # same thing again, as an editable widget on an otherwise read-only page.
    exclude = ["class_map"]
    promote_kind = "base"
    list_display = ["__str__", "models_display", "dataset", "status_badge",
                    "map50", "map50_95", "eval_time", "created_at"]
    list_filter = ["status", "model_source", "trained_model"]
    actions = ["analyze_selected", "open_tag_analytics", "build_tag_data",
               "launch_selected", "reconcile_selected", "promote_labels"]
    readonly_fields = [
        "trained_model", "model_source", "artifact_path", "bundle_path",
        "model_label_snapshot",
        "dataset", "label_source", "annotator", "explicit_labels_path",
        "class_map_display",
        "score_threshold",
        "status", "request_yaml_path", "output_dir", "metrics_pretty", "last_error",
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


class PipelineEvalRunAdmin(EvalDisplayMixin, PromoteLabelsMixin, TagAnalyticsMixin, admin.ModelAdmin):
    """Shared admin for the per-pipeline proxies (Batch detect, Chain, …).

    Each pipeline type gets its own list via a proxy subclass registered below, so
    a run lands under the list matching the pipeline chosen in the "Evaluate with
    a pipeline…" action. ``pipeline`` is dropped from the columns/filters since
    every row in a given list shares it.
    """

    # Rendered by class_map_display instead — the raw JSONField would show the
    # same thing again, as an editable widget on an otherwise read-only page.
    exclude = ["class_map"]
    promote_kind = "pipeline"
    list_display = ["__str__", "models_display", "dataset", "status_badge",
                    "map50", "map50_95", "eval_time", "created_at"]
    list_filter = ["status", "model_source", "trained_model"]
    actions = ["analyze_selected", "open_tag_analytics", "build_tag_data",
               "launch_selected", "reconcile_selected", "promote_labels"]
    readonly_fields = [
        "trained_model", "model_source", "artifact_path", "bundle_path",
        "model_label_snapshot",
        "dataset", "label_source", "annotator", "explicit_labels_path",
        "class_map_display",
        "pipeline", "detector_checkpoint", "detector_expand_ratio",
        "tile_width_pct", "tile_height_pct", "overlap", "chain",
        "score_threshold",
        "status", "request_yaml_path", "output_dir", "metrics_pretty", "last_error",
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
class CombinedEvalAdmin(EvalDisplayMixin, PromoteLabelsMixin, TagAnalyticsMixin, admin.ModelAdmin):
    """The combined "All pipeline evals" list — base + every pipeline in one place.

    Read-only union view (see :class:`CombinedEval`): rows can't be edited here,
    but "Analyze / compare" spans base and pipeline runs at once, and the launch /
    reconcile / promote actions route each selected row back to its real
    :class:`EvalRun` / :class:`PipelineEvalRun` by ``kind``.
    """

    list_display = ["__str__", "models_display", "pipeline", "dataset", "status_badge",
                    "map50", "map50_95", "eval_time", "created_at"]
    list_display_links = None
    list_filter = ["status", "pipeline", "model_source", "trained_model"]
    actions = ["analyze_selected", "open_tag_analytics", "build_tag_data",
               "launch_selected", "reconcile_selected", "promote_labels",
               "delete_selected_evals"]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False  # backed by a DB view — read-only

    def has_delete_permission(self, request, obj=None):
        return False

    def _eval_target(self, obj):
        if obj.kind == CombinedEval.BASE:
            return EvalRun.objects.get(pk=obj.orig_id), "base"
        return PipelineEvalRun.objects.get(pk=obj.orig_id), "pipeline"

    def get_urls(self):
        custom = [
            path("tag-analytics/", self.admin_site.admin_view(self.tag_analytics_view),
                 name="eval_pipelines_combinedeval_tag_analytics"),
            path("tag-analytics/data/", self.admin_site.admin_view(self.tag_analytics_data),
                 name="eval_pipelines_combinedeval_tag_analytics_data"),
            path("tag-analytics/compare/", self.admin_site.admin_view(self.tag_compare_data),
                 name="eval_pipelines_combinedeval_tag_compare"),
            path("tag-analytics/measure/", self.admin_site.admin_view(self.tag_measure),
                 name="eval_pipelines_combinedeval_tag_measure"),
            path("hard-images/", self.admin_site.admin_view(self.hard_images_view),
                 name="eval_pipelines_combinedeval_hard_images"),
            path("hard-images/image/", self.admin_site.admin_view(self.hard_images_image),
                 name="eval_pipelines_combinedeval_hard_images_image"),
            path("hard-images/data/", self.admin_site.admin_view(self.hard_images_data),
                 name="eval_pipelines_combinedeval_hard_images_data"),
        ]
        return custom + super().get_urls()

    # ------------------------------------------------------------- worst images
    # Linked from the "Analyze / compare" page (see ``_attach_hard_images_links``),
    # not from a changelist action — one viewer shared by every eval kind/list,
    # addressed by ``?kind=&eval=`` rather than a list-specific pk space. Mirrors
    # ``TrainedModelAdmin``'s "hardest val images" viewer (``training.admin``),
    # but the worst 10% rather than a fixed top-50 — see
    # ``friendy_chachkalica.metrics.EVAL_HARD_IMAGES_FRACTION``.
    # ------------------------------------------------------- tag analytics
    # One report for every eval list, addressed by ``?kind=&eval=`` like the
    # worst-images viewer. The page renders the per-tag tables server-side and
    # then talks to ``tag_analytics_data`` for anything the operator builds by
    # hand — a cross-tab, an arbitrary intersection of tag values. Those are
    # numpy passes over a table already in memory, so a round trip is cheaper
    # (and far less code) than shipping the whole table to the browser and
    # re-implementing AP in JavaScript.
    def tag_analytics_view(self, request):
        eval_obj, kind = _resolve_eval_for_tags(request)
        loaded, problem = _load_tag_analysis(eval_obj)
        context = {
            **self.admin_site.each_context(request),
            "title": f"Tag analytics — {eval_obj}",
            "subject_name": str(eval_obj),
            "dataset": eval_obj.dataset.name,
            "fatal": problem,
            "query": urlencode({"kind": kind, "eval": eval_obj.pk}),
            "data_url": reverse("admin:eval_pipelines_combinedeval_tag_analytics_data"),
            "measure_url": reverse("admin:eval_pipelines_combinedeval_tag_measure"),
            # Independent of this eval's own table: a set still running (or one
            # that failed) should not hide how its siblings did.
            "test_sets": test_sets.section(eval_obj, kind),
        }
        if loaded is not None:
            table, index, sources = loaded
            context["sources"] = sources
            context["report"] = tag_analytics.report(table, index, eval_obj.metrics)
        return TemplateResponse(request, "admin/eval_pipelines/tag_analytics.html", context)

    @method_decorator(require_POST)
    def tag_measure(self, request):
        """Queue an image-measurement pass for this eval's dataset, then come back.

        The person looking at a "not measured yet" row is exactly the person who
        wants it measured; making them find the Datasets list and remember which
        dataset this eval used is a detour with no purpose. The bulk path stays
        on the Dataset action.

        POST because it starts work — a URL someone pasted should not.
        """
        eval_obj, kind = _resolve_eval_for_tags(request)
        wanted = request.POST.get("measure_kind") or DatasetMeasureRun.IMAGE
        if wanted == DatasetMeasureRun.BOX:
            run = DatasetMeasureRun.objects.create(
                dataset=eval_obj.dataset, kind=DatasetMeasureRun.BOX,
                label_source=eval_obj.label_source,
                annotator=getattr(eval_obj, "annotator", None),
                explicit_labels_path=getattr(eval_obj, "explicit_labels_path", "") or "",
            )
            try:
                config_gen.write_measures_request(run)
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                run.delete()
                messages.error(request, f"Cannot measure this label set: {exc}")
                return HttpResponseRedirect(_tag_analytics_url(eval_obj, kind))
            _queue().enqueue(jobs.build_box_measures, run.pk, job_timeout=jobs.JOB_TIMEOUT)
            messages.info(
                request,
                f"Measuring {run.labels_dir_snapshot} against "
                f"{run.person_source} person boxes — it takes the trainer's one job "
                "slot, so queued evals will wait. Track it under Fleet > Image "
                "measures runs, then reload this page.",
            )
            return HttpResponseRedirect(_tag_analytics_url(eval_obj, kind))

        run = DatasetMeasureRun.objects.create(dataset=eval_obj.dataset)
        django_rq.get_queue("default").enqueue(
            fleet_jobs.measure_dataset_images, run.id,
            job_timeout=fleet_admin.MEASURE_JOB_TIMEOUT)
        messages.info(
            request,
            f"Measuring {eval_obj.dataset.name}'s images — track it under "
            "Fleet > Image measures runs, then reload this page.",
        )
        return HttpResponseRedirect(_tag_analytics_url(eval_obj, kind))

    def tag_compare_data(self, request):
        """One tag's slices scored across several evals, as JSON.

        ``?evals=base:12,pipeline:7`` — the ``kind``/``orig_id`` pair the compare
        page's columns already carry. ``?mode=tags`` lists what can be compared;
        otherwise ``&tag=&metric=`` fills the grid.

        Lazy on purpose. Each column parses its own match table (tens of MB on a
        large eval), so doing this on page load would make the compare page pay
        for a section most visits never open.
        """
        columns, notes = [], []
        for token in (request.GET.get("evals") or "").split(","):
            kind, _, pk = token.strip().partition(":")
            if not pk:
                continue
            if len(columns) >= MAX_COMPARE_COLUMNS:
                notes.append(f"Only the first {MAX_COMPARE_COLUMNS} evals were compared.")
                break
            eval_obj = _eval_by_kind(kind, pk)
            if eval_obj is None:
                continue
            loaded, problem = _load_tag_analysis(eval_obj)
            if loaded is None:
                notes.append(f"#{pk}: {problem}")
                continue
            table, index, _sources = loaded
            columns.append({
                "label": eval_obj.model_label(), "dataset": eval_obj.dataset.name,
                "table": table, "index": index,
            })

        if not columns:
            return JsonResponse(
                {"error": "; ".join(notes) or "No comparable evals were named."}, status=400)

        if request.GET.get("mode") == "tags":
            return JsonResponse({
                "tags": tag_analytics.comparable_tags(columns),
                "columns": [column["label"] for column in columns],
                # The metric list follows the chosen tag's scope, and the page
                # needs both lists before the first grid is built.
                "frame_metrics": tag_analytics.FRAME_METRICS,
                "box_metrics": tag_analytics.BOX_METRICS,
                "person_metrics": tag_analytics.PERSON_METRICS,
                "notes": notes,
            })

        try:
            payload = tag_analytics.compare_tag(
                columns, request.GET.get("tag", ""), request.GET.get("metric", ""))
        except tag_analytics.UnknownClause as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        payload["notes"] = notes
        return JsonResponse(payload)

    def tag_analytics_data(self, request):
        """Score one operator-built slice, or one cross-tab, as JSON.

        ``?mode=slice&clauses=[["weather","rain"],["shift","night"]]`` intersects
        the clauses; ``?mode=cross&tag_a=&tag_b=&metric=`` fills a grid with one
        metric over every combination of two tags' values;
        ``?mode=scorecard`` scores every value of every usable tag at once.

        All three accept ``&threshold=`` to score at a confidence other than the
        eval's own. Nothing is re-run for it: the match table holds every
        prediction down to its score floor, so a different cut is a filter over
        rows already on disk (see ``tag_analytics.resolve_score_cut``).
        """
        eval_obj, _kind = _resolve_eval_for_tags(request)
        loaded, problem = _load_tag_analysis(eval_obj)
        if loaded is None:
            return JsonResponse({"error": problem}, status=400)
        table, index, _sources = loaded

        try:
            # Absent means "the eval's own operating point"; a blank or
            # unparseable one is a client bug worth reporting, not worth
            # silently scoring at a threshold nobody asked for.
            raw_threshold = request.GET.get("threshold")
            threshold = None if raw_threshold in (None, "") else float(raw_threshold)

            if request.GET.get("mode") == "scorecard":
                return JsonResponse(tag_analytics.scorecard(table, index, threshold))
            if request.GET.get("mode") == "cross":
                return JsonResponse(tag_analytics.cross_tab(
                    table, index,
                    request.GET.get("tag_a", ""), request.GET.get("tag_b", ""),
                    request.GET.get("metric", ""), threshold,
                ))
            clauses = json.loads(request.GET.get("clauses") or "[]")
            if not isinstance(clauses, list):
                raise ValueError("clauses must be a list of [tag, value] pairs")
            pairs = [(str(pair[0]), str(pair[1])) for pair in clauses]
            return JsonResponse(
                tag_analytics.slice_metrics(table, index, pairs, threshold))
        except tag_analytics.UnknownClause as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        except (ValueError, TypeError, IndexError) as exc:
            return JsonResponse({"error": f"bad filter: {exc}"}, status=400)

    def hard_images_view(self, request):
        """Render the viewer shell; the browser pulls images + precomputed boxes per index."""
        eval_obj = _resolve_eval_for_hard_images(request)
        payload = _load_hard_images_for(eval_obj)
        if payload is None:
            raise Http404("no worst-images artifact for this eval")
        images = payload.get("images", [])
        context = {
            **self.admin_site.each_context(request),
            "title": f"Worst images — {eval_obj}",
            "subject_name": str(eval_obj),
            "image_count": len(images),
            "metric": payload.get("metric", ""),
            "metric_description": payload.get("metric_description", ""),
            "iou_threshold": payload.get("iou_threshold"),
            "score_threshold": payload.get("score_threshold"),
            "operating_nms_threshold": payload.get("operating_nms_threshold"),
            "max_display_predictions": payload.get("max_display_predictions"),
            "query": urlencode({
                "kind": request.GET.get("kind", CombinedEval.BASE), "eval": eval_obj.pk,
            }),
            "run_choices": [],
            "run_choices_json": "[]",
            "back_label": "Back to eval",
        }
        return TemplateResponse(request, "admin/training/hard_images_viewer.html", context)

    def hard_images_image(self, request):
        """Stream the raw bytes of the worst image at ``?index=`` (guarded to source_root)."""
        eval_obj = _resolve_eval_for_hard_images(request)
        payload = _load_hard_images_for(eval_obj)
        images = payload.get("images", []) if payload else []
        raw_path = _hard_image_path_from_request(request, images)
        image_path = Path(raw_path).resolve()
        root = source_root().resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise Http404("image not found under source root")
        return FileResponse(open(image_path, "rb"))

    def hard_images_data(self, request):
        """Return the precomputed predictions + ground truth + difficulty for ``?index=``."""
        eval_obj = _resolve_eval_for_hard_images(request)
        payload = _load_hard_images_for(eval_obj)
        images = payload.get("images", []) if payload else []
        if not images:
            return JsonResponse({"error": "no worst images for this eval"}, status=400)
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

    @admin.action(description="Analyze / compare metrics of selected eval(s)…")
    def analyze_selected(self, request, queryset):
        return _analyze(self, request, queryset, "Eval metrics comparison")

    @admin.action(description="Launch / relaunch on trainer service")
    def launch_selected(self, request, queryset):
        queue = _queue()
        prev_job = None  # chain so a multi-select launch doesn't race the trainer's single slot
        for row in queryset:
            eval_obj, kind = self._eval_target(row)
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
            eval_obj, kind = self._eval_target(row)
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
        targets = [self._eval_target(obj) for obj in queryset]

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
