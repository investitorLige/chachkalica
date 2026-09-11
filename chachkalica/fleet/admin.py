"""Admin = the fleet operator console.

Long operations (provision / setup / sync / remove) are enqueued as rq jobs —
each action enqueues one job per selected row, flips the row to ``queued``, and
returns immediately. The row's status column then reflects
``queued -> running -> ok/error`` as the worker picks it up (refresh to see it).
"""

import json
from collections import Counter
from pathlib import Path

import cv2
import django_rq
from django import forms
from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.core.exceptions import ValidationError as DjangoValidationError
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils import timezone
from django.utils.html import format_html
from django.utils.http import urlencode

from admin_sections.scoping import scope_queryset
from fleet import jobs
from fleet.models import (
    AnnotationTag, Annotator, Dataset, DatasetInferenceResult, DatasetInferenceRun,
    FleetSettings, GroundingSamRun, Project,
)
from fleet.services import analytics as analytics_svc
from fleet.services import annotation_tags as annotation_tags_svc
from fleet.services import data_quality_solve
from fleet.services import dataset_inference
from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services import merge as merge_svc
from fleet.services import overlap as overlap_svc
from fleet.services import split as split_svc
from fleet.services.paths import source_root
from training.services import inference_form
from videos.services import inference

_STATUS_COLORS = {
    "ok": "#22c55e",
    "running": "#f59e0b",
    "queued": "#9ca3af",
    "paused": "#3b82f6",
    "warning": "#f97316",
    "error": "#ef4444",
}


#: Cap on resolve cards per page. The cards carry the operator's decision back
#: in one form field, so an unbounded report would eventually outgrow
#: DATA_UPLOAD_MAX_MEMORY_SIZE — and a five-figure card count is not a page
#: anyone can work through anyway. Past this, prune what's shown and re-run.
_OVERLAP_MAX_CARDS = 1500

#: Width the resolve grid asks for its thumbnails at. Wide enough to tell two
#: similar frames apart at a glance, small enough that a few hundred cards
#: aren't a multi-gigabyte page load.
_OVERLAP_THUMB_WIDTH = 220

#: How many images a dataset inference run measures by default. Enough for a
#: stable median (and a p90 that means something) without tying up the GPU for
#: an afternoon on a 79k-image dataset; 0 in the form means all of them.
_DEFAULT_IMAGE_LIMIT = 200

#: Untimed calls before the timed loop. Three is enough to get past a model load
#: and a TensorRT plan deserialization, both of which are one-off and neither of
#: which is a frame time.
_DEFAULT_WARMUP = 3

#: Width the dataset inference report asks for its thumbnails at, and how many
#: rows one page of it renders. A run may hold thousands of images; the report is
#: for reading the timing distribution and spot-checking detections, not for
#: flipping through the whole dataset.
_RUN_THUMB_WIDTH = 260
_RUN_ROW_LIMIT = 300


def _queue():
    return django_rq.get_queue("default")


def _thumbnail_response(path: Path, width: int) -> HttpResponse:
    """Re-encode one image down to ``width`` px as JPEG (never upscales)."""
    image = cv2.imread(str(path))
    if image is None:
        raise Http404("unreadable image")
    height, original_width = image.shape[:2]
    if original_width > width:
        image = cv2.resize(image, (width, max(1, round(height * width / original_width))),
                           interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise Http404("could not encode thumbnail")
    return HttpResponse(buffer.tobytes(), content_type="image/jpeg")


def _status_badge(value: str):
    if not value:
        return "—"
    color = _STATUS_COLORS.get(value, "#9ca3af")
    return format_html(
        '<b style="color:{};">●</b> {}', color, value
    )


def _stage_rows(timings: dict) -> list[dict]:
    """The per-stage table for a run's report, biggest share first.

    Read as *shares* of the frame, not as absolutes: attributing stages needs a
    sync at every boundary, which inflates the total it measures — the same
    caveat the bundle benchmark's stage pass carries, and the reason the report
    prints the share beside the millisecond mean rather than only the latter.
    """
    shares = (timings or {}).get("stage_share") or {}
    means = (timings or {}).get("stage_ms_mean") or {}
    rows = [
        {
            # "preprocess_ms" -> "preprocess": the trainer names its stages with
            # the unit attached (they arrive as timing keys), which reads as
            # noise once the column header says milliseconds.
            "name": stage[:-3] if stage.endswith("_ms") else stage,
            "share": share,
            "percent": round(100 * share, 1),
            "mean_ms": means.get(stage),
        }
        for stage, share in shares.items()
    ]
    return sorted(rows, key=lambda r: r["share"], reverse=True)


def _run_geometry_rows(run) -> list[tuple[str, object]]:
    """``(label, value)`` for the pipeline knobs this run actually used.

    Only the knobs the run's pipeline uses are listed — a raw run has no tile
    size and printing a blank one invites the reader to wonder what it was. Blank
    means "chachak's default", which is what the value column says.
    """
    from training import pipelines

    uses_detector = pipelines.needs_detector(run.pipeline, run.chain) or \
        run.pipeline == pipelines.CHAIN
    uses_tiling = run.pipeline in (
        pipelines.BATCH_DETECT, pipelines.BATCH_PEOPLE, pipelines.CHAIN)

    rows: list[tuple[str, object]] = [("pipeline", run.pipeline)]
    if run.pipeline == pipelines.CHAIN:
        rows.append(("chain", ", ".join(run.chain or [])))
    if uses_detector:
        rows += [
            ("detector checkpoint", run.detector_checkpoint),
            ("person-box expand ratio", run.detector_expand_ratio),
            ("person-crop minimum size (px)", run.detector_min_box_size),
        ]
    if uses_tiling:
        rows += [
            ("tile size (px)", run.tile_size_px),
            ("tile width %", run.tile_width_pct),
            ("tile height %", run.tile_height_pct),
            ("tile overlap", run.overlap),
        ]
    if run.pipeline != run.RAW:
        rows.append(("merge NMS IoU", run.merge_nms_iou))
    rows += [
        ("score threshold", run.score_threshold),
        ("images", f"{run.image_limit or 'all'}"),
        ("warmup calls", run.warmup),
    ]
    return [(label, "chachak default" if value is None or value == "" else value)
            for label, value in rows]


def _dataset_run_form_values(defaults: dict, posted) -> dict:
    """The values to render the dataset-run form with: the shared pipeline half
    (prefilled from the model's own metadata) plus this form's own two knobs.

    ``image_limit``/``warmup`` are properties of the *measurement*, not of the
    model, so they never come from a model's pipeline metadata — same split as
    the video form's ``frame_stride``.
    """
    return inference_form.form_values(defaults, posted, extra={
        "image_limit": _DEFAULT_IMAGE_LIMIT,
        "warmup": _DEFAULT_WARMUP,
    })


@admin.register(FleetSettings)
class FleetSettingsAdmin(admin.ModelAdmin):
    list_display = ["__str__", "base_port", "image_name", "webhook_url"]

    def has_add_permission(self, request):
        # Singleton: only ever one row.
        return not FleetSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(Annotator)
class AnnotatorAdmin(admin.ModelAdmin):
    list_display = ["username", "status", "port", "container_name", "ls_url", "last_action", "status_badge"]
    list_filter = ["status", "last_status"]
    search_fields = ["username", "email", "container_name"]
    readonly_fields = ["ls_url", "last_action", "last_status", "last_error", "last_run_at", "created_at", "updated_at"]
    actions = ["provision_selected", "remove_selected", "purge_selected"]
    # email/password are editable; leave any of port/container/volume/password
    # blank and save() fills them in. `token` is intentionally not on the form —
    # it is generated and rotated automatically during provisioning.
    fieldsets = [
        (None, {"fields": ["username", "email", "password", "status"]}),
        (
            "Container (leave blank to auto-fill)",
            {"fields": ["port", "container_name", "volume_name", "ls_url"]},
        ),
        (
            "Last job",
            {
                "classes": ["collapse"],
                "fields": ["last_action", "last_status", "last_error", "last_run_at", "created_at", "updated_at"],
            },
        ),
    ]

    @admin.display(description="last status", ordering="last_status")
    def status_badge(self, obj):
        return _status_badge(obj.last_status)

    def _enqueue_each(self, request, queryset, func, action_label, **kwargs):
        queue = _queue()
        for annotator in queryset:
            queue.enqueue(func, annotator.id, **kwargs)
            annotator.last_action = action_label
            annotator.last_status = "queued"
            annotator.last_error = ""
            annotator.last_run_at = timezone.now()
            annotator.save(update_fields=["last_action", "last_status", "last_error", "last_run_at"])
        self.message_user(request, f"{queryset.count()} {action_label} job(s) queued — refresh to see progress.")

    @admin.action(description="Provision / restore selected annotators")
    def provision_selected(self, request, queryset):
        self._enqueue_each(request, queryset, jobs.provision_annotator, "provision")

    @admin.action(description="Remove containers (keep volume + row)")
    def remove_selected(self, request, queryset):
        self._enqueue_each(request, queryset, jobs.remove_annotator, "remove", purge=False)

    @admin.action(description="Purge (delete container, volume, AND row)")
    def purge_selected(self, request, queryset):
        queue = _queue()
        for annotator in queryset:
            queue.enqueue(jobs.remove_annotator, annotator.id, purge=True)
        self.message_user(request, f"{queryset.count()} purge job(s) queued — rows are removed once complete.")


class DatasetAdminForm(forms.ModelForm):
    """Turn ``name`` into a dropdown of folders found under the source root.

    Listing the on-disk source directories (rather than free text) keeps the
    name in lockstep with what's actually present to set up. When adding, only
    folders not yet registered *on this same admin front* are offered — main
    admin sees (and so excludes) every dataset everywhere, but a section only
    excludes names already used by a row that's actually visible there, so the
    same on-disk dataset can get its own independent row per project admin
    without two rows of it ever colliding on one front. ``request`` is set by
    ``DatasetAdmin.get_form`` below; its absence (e.g. a form built directly,
    outside the admin) falls back to the original global check. When editing,
    the current name stays selectable even if its folder has since gone missing.
    """

    class Meta:
        model = Dataset
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name_field = self._meta.model._meta.get_field("name")
        current = self.instance.name if self.instance and self.instance.pk else None

        taken_qs = scope_queryset(Dataset.objects.all(), Dataset, getattr(self, "request", None))
        taken = set(taken_qs.values_list("name", flat=True))
        taken.discard(current)
        choices = [(d, d) for d in self._source_dirs() if d not in taken]
        if current and current not in {value for value, _ in choices}:
            choices.insert(0, (current, f"{current} (missing on disk)"))

        self.fields["name"] = forms.ChoiceField(
            choices=[("", "— select a source folder —")] + choices,
            label=name_field.verbose_name.capitalize(),
            help_text=name_field.help_text,
        )

    @staticmethod
    def _source_dirs() -> list[str]:
        try:
            root = source_root()
            return sorted(
                p.name for p in root.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
        except (FileNotFoundError, NotADirectoryError, OSError):
            return []


def _tag_row(tag) -> dict:
    """One saved tag in the shape the editor template renders a row from —
    the same shape ``annotation_tags.posted_rows`` produces, so a form that
    failed validation redraws exactly what was typed instead of the stale
    database state."""
    return {
        "scope": tag.scope,
        "name": tag.name,
        "widget": tag.widget,
        "choices": tag.cleaned_choices(),
        "max_rating": tag.max_rating,
        "required": tag.required,
    }


def _tag_editor_context(dataset, rows=None) -> dict:
    """Everything ``_annotation_tag_sections.html`` needs to draw the editor.

    ``rows`` is the posted set when a save failed validation (so the page
    redraws what was typed rather than the stale database state) and None
    otherwise. Row indices are re-densified here: they only have to be unique
    within one page, and the two rendered tables post as one flat list.
    """
    indexed = rows if rows is not None else [_tag_row(tag) for tag in dataset.tags.all()]
    for index, row in enumerate(indexed):
        row["index"] = index
    return {
        "frame_rows": [r for r in indexed if r["scope"] != AnnotationTag.REGION],
        "region_rows": [r for r in indexed if r["scope"] == AnnotationTag.REGION],
        "next_index": len(indexed),
        "scope_frame": AnnotationTag.FRAME,
        "scope_region": AnnotationTag.REGION,
        "widget_choices": AnnotationTag.WIDGET_CHOICES,
        "choice_widgets": json.dumps(sorted(lsapi.CHOICE_WIDGETS)),
        "rating_widget": AnnotationTag.RATING,
        "project_count": dataset.projects.count(),
    }


def _preview_bounded_index(request, count):
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

    Returns None when the resolved dir is missing so the viewer can say
    "no labels" instead of erroring. ``config_gen`` is imported lazily to avoid
    a fleet -> training import cycle at module load.
    """
    from training.services import config_gen

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


def _read_label_shapes(label_dir, image_path, class_names):
    """Parse an image's label ``.txt`` into normalized draw shapes.

    Shared with the VLM dataset report, which burns the same shapes onto the
    image server-side, so the parsing lives beside ``labels_source_dir``.
    """
    return datasets_svc.label_shapes(label_dir, image_path.name, class_names)


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    form = DatasetAdminForm
    list_display = ["name", "storage_type", "storage_root", "has_labels"]
    readonly_fields = ["has_labels"]
    search_fields = ["name"]
    actions = [
        "run_model_inference",
        "edit_annotation_tags",
        "setup_for_all_active",
        "sync_all_projects",
        "setup_sync_one_annotator",
        "promote_annotator_labels",
        "generate_grounding_sam_labels",
        "merge_selected",
        "split_selected",
        "analyze_selected",
        "check_duplicate_images",
        "check_overlapping_images",
        "preview_labels",
    ]

    def get_form(self, request, obj=None, **kwargs):
        # DatasetAdminForm needs to know which admin front it's rendering on
        # (main vs. a section), to scope its "already taken" dropdown check
        # accordingly — this is the standard way to hand a request through to
        # a ModelForm: get_form() builds a fresh Form subclass per request, so
        # stamping the class attribute here can't leak across requests.
        Form = super().get_form(request, obj, **kwargs)
        Form.request = request
        return Form

    def save_model(self, request, obj, form, change):
        # Refresh has_labels from disk whenever a dataset is added/edited here,
        # so the flag reflects whether a source labels/ folder is present.
        super().save_model(request, obj, form, change)
        datasets_svc.detect_labels(obj)

    def _save_posted_tags(self, request, dataset):
        """Save the tag editor's rows onto a dataset.

        Returns ``(ok, rows)``: on success ``rows`` is None so the caller
        redraws from the database, on failure it is the posted set so the
        operator's typing survives the error. Every message is already on the
        request by the time this returns.
        """
        rows = annotation_tags_svc.posted_rows(request.POST)
        try:
            annotation_tags_svc.save_tags(dataset, rows)
        except DjangoValidationError as exc:
            for message in exc.messages:
                self.message_user(request, message, level=messages.ERROR)
            return False, rows

        count = len(rows)
        self.message_user(
            request,
            f"Saved {count} annotation tag{'' if count == 1 else 's'} on "
            f"{dataset.name!r}. New projects get them automatically.",
        )
        return True, None

    def _push_tags_to_projects(self, request, dataset):
        """Queue the rewrite of every existing project's labeling interface.

        Setting a dataset up again cannot do this itself: project creation is
        idempotent on the title, so an existing project is skipped rather than
        rebuilt, and an edited tag would only reach annotators set up from here
        on.
        """
        projects = dataset.projects.count()
        if not projects:
            self.message_user(
                request,
                f"{dataset.name!r} has no Label Studio projects yet — nothing to push to.",
                level=messages.WARNING,
            )
            return
        _queue().enqueue(jobs.apply_dataset_tags, dataset.id)
        for project in dataset.projects.all():
            project.last_status = "queued"
            project.last_run_at = timezone.now()
            project.save(update_fields=["last_status", "last_run_at"])
        self.message_user(
            request,
            f"Pushing the new interface onto {projects} existing project(s) — "
            "watch the Projects page for the outcome.",
        )

    @admin.action(description="Edit annotation tags (frame-wide / box-wide)…")
    def edit_annotation_tags(self, request, queryset):
        """Edit a dataset's tags on their own, without setting anything up.

        The same editor is embedded in both setup actions, which is where tags
        are normally written; this is the way back to them afterwards — to add
        one mid-project and push it onto the projects annotators are already
        working in.
        """
        datasets = list(queryset)
        if len(datasets) != 1:
            self.message_user(
                request, "Select exactly one dataset to edit its tags.", level=messages.WARNING
            )
            return None
        dataset = datasets[0]

        rows = None
        if request.POST.get("apply"):
            ok, rows = self._save_posted_tags(request, dataset)
            if ok and request.POST.get("push_existing"):
                self._push_tags_to_projects(request, dataset)

        # Preview the interface a project would get. A dataset whose classes.txt
        # is missing can still have its tags edited — only the preview is lost.
        try:
            preview = datasets_svc.dataset_label_config(dataset)
            preview_error = ""
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            preview, preview_error = "", str(exc)

        context = {
            **self.admin_site.each_context(request),
            **_tag_editor_context(dataset, rows),
            "title": f"Annotation tags — {dataset.name}",
            "dataset": dataset,
            "preview": preview,
            "preview_error": preview_error,
            "action": "edit_annotation_tags",
            "selected": [str(dataset.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/fleet/annotation_tags.html", context)

    @admin.action(description="Set up selected dataset(s) for all active annotators")
    def setup_for_all_active(self, request, queryset):
        """Create every active annotator's project for the selected dataset(s).

        Confirms first rather than firing straight off, because this is the
        moment the labeling interface is decided: a project bakes in the
        dataset's annotation tags when it is created, and creation is
        idempotent on the project title, so a tag added later needs an explicit
        push to reach it. The confirmation page is therefore the tag editor —
        the questions annotators will answer, settled before the projects that
        ask them exist.

        The editor needs one dataset to edit; select several and the page lists
        what each already carries and sets them up unchanged.
        """
        datasets = list(queryset)
        active = list(Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"))
        if not active:
            self.message_user(request, "No active annotators to set up.", level=messages.WARNING)
            return None

        # Tags belong to one dataset; with several selected there is no single
        # set to edit, so the page falls back to showing each one's.
        single = datasets[0] if len(datasets) == 1 else None

        rows = None
        if request.POST.get("apply"):
            saved = True
            if single is not None:
                saved, rows = self._save_posted_tags(request, single)
                if saved and request.POST.get("push_existing"):
                    self._push_tags_to_projects(request, single)
            if saved:
                queue = _queue()
                count = 0
                for dataset in datasets:
                    for annotator in active:
                        queue.enqueue(jobs.setup_project, dataset.id, annotator.id)
                        count += 1
                self.message_user(
                    request, f"{count} setup job(s) queued (datasets × active annotators)."
                )
                return None
            # Invalid tags: nothing was queued and nothing was saved. Fall
            # through and redraw so the operator can fix the row — setting the
            # projects up now would bake in the interface they were editing.

        context = {
            **self.admin_site.each_context(request),
            "title": "Set up for all active annotators",
            "datasets": datasets,
            "dataset": single,
            "annotators": active,
            "tag_summaries": None if single else [
                {"dataset": d, "tags": list(d.tags.all())} for d in datasets
            ],
            "action": "setup_for_all_active",
            "selected": [str(d.pk) for d in datasets],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
            "submit_label": "Set up projects",
        }
        if single is not None:
            context.update(_tag_editor_context(single, rows))
        return TemplateResponse(request, "admin/fleet/setup_datasets.html", context)

    @admin.action(description="Sync all projects of selected dataset(s)")
    def sync_all_projects(self, request, queryset):
        queue = _queue()
        projects = Project.objects.filter(dataset__in=queryset)
        for project in projects:
            queue.enqueue(jobs.sync_project, project.id)
            project.last_status = "queued"
            project.last_run_at = timezone.now()
            project.save(update_fields=["last_status", "last_run_at"])
        self.message_user(request, f"{projects.count()} sync job(s) queued — refresh to see progress.")

    @admin.action(description="Set up + sync selected dataset(s) for one annotator…")
    def setup_sync_one_annotator(self, request, queryset):
        """Set up, then sync, for a single chosen annotator.

        Carries the same tag editor as the all-annotators action, for the same
        reason: this is a project-creation path, and a project's labeling
        interface is fixed the moment it is created.
        """
        datasets = list(queryset)
        single = datasets[0] if len(datasets) == 1 else None

        rows = None
        if request.POST.get("apply"):
            annotator = Annotator.objects.filter(pk=request.POST.get("annotator")).first()
            if annotator is None:
                self.message_user(request, "Choose an annotator.", level=messages.WARNING)
                return None
            saved = True
            if single is not None:
                saved, rows = self._save_posted_tags(request, single)
                if saved and request.POST.get("push_existing"):
                    self._push_tags_to_projects(request, single)
            if saved:
                queue = _queue()
                for dataset in datasets:
                    queue.enqueue(jobs.setup_and_sync_project, dataset.id, annotator.id)
                self.message_user(
                    request,
                    f"{len(datasets)} setup+sync job(s) queued for {annotator.username} — "
                    "refresh the Projects page to see progress.",
                )
                return None
            # Invalid tags — redraw rather than set up against a half-edited set.

        active = list(Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"))
        if not active:
            self.message_user(request, "No active annotators to set up.", level=messages.WARNING)
            return None
        context = {
            **self.admin_site.each_context(request),
            "title": "Set up + sync for one annotator",
            "datasets": datasets,
            "dataset": single,
            "annotators": active,
            "chosen_annotator": request.POST.get("annotator") or "",
            "tag_summaries": None if single else [
                {"dataset": d, "tags": list(d.tags.all())} for d in datasets
            ],
            "action": "setup_sync_one_annotator",
            "selected": [str(d.pk) for d in datasets],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        if single is not None:
            context.update(_tag_editor_context(single, rows))
        return TemplateResponse(request, "admin/fleet/pick_annotator.html", context)

    @admin.action(description="Promote an annotator's annotations to source labels…")
    def promote_annotator_labels(self, request, queryset):
        """Move an annotator's synced labels into each dataset's source labels/ folder.

        Runs synchronously (a local file move — no Label Studio calls), so the
        result is reported straight back rather than queued. The operator picks
        one annotator, then that annotator's ``.txt`` output is moved from
        ``target/<dataset>/<username>/`` to ``source/<dataset>/labels/`` for every
        selected dataset.
        """
        datasets = sorted(queryset, key=lambda d: d.name)

        if request.POST.get("apply"):
            annotator = Annotator.objects.filter(pk=request.POST.get("annotator")).first()
            if annotator is None:
                self.message_user(request, "Choose an annotator.", level=messages.WARNING)
                return None
            for dataset in datasets:
                try:
                    result = datasets_svc.promote_annotator_labels(dataset, annotator)
                except (FileNotFoundError, OSError) as exc:
                    self.message_user(request, f"{dataset.name}: {exc}", level=messages.ERROR)
                    continue
                tags_note = (" Tag answers (annotation_tags.json) came across too."
                             if result["tags_promoted"] else "")
                self.message_user(
                    request,
                    f"{dataset.name}: promoted {result['moved']} label file(s) from "
                    f"{annotator.username} to {result['dest']}.{tags_note}",
                )
            return None

        active = list(Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"))
        if not active:
            self.message_user(request, "No active annotators to promote from.", level=messages.WARNING)
            return None
        context = {
            **self.admin_site.each_context(request),
            "title": "Promote annotator annotations to source",
            "datasets": datasets,
            "annotators": active,
            "action": "promote_annotator_labels",
            "selected": [str(d.pk) for d in datasets],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/fleet/promote_annotator.html", context)

    @admin.action(description="Generate labels with Grounding SAM…")
    def generate_grounding_sam_labels(self, request, queryset):
        """Auto-label a dataset's unlabeled images from classes.txt via Grounding SAM.

        Runs off-request (the backend call is slow model inference) — one
        GroundingSamRun row + RQ job per dataset, queued straight away. Progress
        is tracked on the run (Fleet > Grounding SAM runs), not here. Images that
        already have a source label file are left untouched; this only fills in
        what's missing.
        """
        datasets = sorted(queryset, key=lambda d: d.name)
        valid = []
        invalid = []
        for dataset in datasets:
            try:
                names, _tools = lsapi.parse_classes_file(source_root() / dataset.name / "classes.txt")
            except (FileNotFoundError, RuntimeError) as exc:
                invalid.append((dataset, str(exc)))
                continue
            valid.append((dataset, names))

        if request.POST.get("apply"):
            try:
                confidence = float(request.POST.get("confidence", ""))
            except ValueError:
                confidence = -1
            if not 0 < confidence <= 1:
                self.message_user(
                    request, "Confidence must be a number between 0 and 1.", level=messages.WARNING
                )
                return None
            for dataset, _names in valid:
                run = GroundingSamRun.objects.create(dataset=dataset, confidence=confidence)
                _queue().enqueue(jobs.generate_grounding_sam_labels, run.id)
            if valid:
                self.message_user(
                    request,
                    f"{len(valid)} dataset(s) queued for Grounding SAM labeling — "
                    "track progress under Fleet > Grounding SAM runs.",
                )
            for dataset, reason in invalid:
                self.message_user(request, f"{dataset.name}: {reason}", level=messages.ERROR)
            return None

        if not valid:
            self.message_user(
                request, "None of the selected datasets have a readable classes.txt.",
                level=messages.WARNING,
            )
            return None

        for dataset in datasets:
            dataset.has_labels_now = datasets_svc.detect_labels(dataset, persist=False)

        context = {
            **self.admin_site.each_context(request),
            "title": "Generate labels with Grounding SAM",
            "valid": valid,
            "invalid": invalid,
            "action": "generate_grounding_sam_labels",
            "selected": [str(d.pk) for d in datasets],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/fleet/generate_grounding_sam.html", context)

    @admin.action(description="Run model inference…")
    def run_model_inference(self, request, queryset):
        """Run one model over this dataset's images and measure how fast it is.

        The same two-step wizard as the video action ("Run model inference…" on
        Videos), over a folder instead of a clip: step 1 picks where the model
        comes from — trained catalogue, exported ``.onnx``/``.engine``, or an
        infer bundle — and step 2 configures that model plus the pipeline the
        images go through, arriving prefilled from the model's own recorded
        pipeline (see ``training.services.pipeline_meta``).

        It exists because the trained-models tab's preview viewer only runs a
        catalogued ``.pt``, while what gets deployed is an exported artifact or a
        bundle — and "how many frames a second does *that* do, on real frames,
        through the pipeline it will be served with" is the question this
        answers. The measured numbers live on the run (Fleet > Dataset inference
        runs), which is where this redirects.

        Both halves of the form — the model list and the pipeline knobs — are
        built and parsed by ``training.services.inference_form``, the same code
        the video action uses, so the two forms cannot drift apart.
        """
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one dataset to run a model on.",
                              level=messages.WARNING)
            return None
        dataset = queryset.first()
        directory = dataset_inference.dataset_dir(dataset.name)

        base_context = {
            **self.admin_site.each_context(request),
            "dataset": dataset,
            "action": "run_model_inference",
            "selected": [str(dataset.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }

        # ------------------------------------------------- step 1: model source
        model_source = request.POST.get("model_source") or ""
        if model_source not in dict(DatasetInferenceRun.MODEL_SOURCE_CHOICES):
            if request.POST.get("configure") or request.POST.get("apply"):
                self.message_user(request, "Choose which kind of model to run.",
                                  level=messages.WARNING)
            return TemplateResponse(
                request, "admin/fleet/dataset_inference_source.html", {
                    **base_context,
                    "title": f"Run model inference — {dataset.name}",
                    **inference_form.source_step_context(),
                })

        # ------------------------------------------------------- step 2: submit
        if request.POST.get("apply"):
            run, error = self._build_dataset_inference_run(request, dataset, model_source)
            if error:
                self.message_user(request, error, level=messages.WARNING)
            else:
                run.save()
                _queue().enqueue(jobs.run_dataset_inference, run.id,
                                 job_timeout=jobs.DATASET_INFERENCE_JOB_TIMEOUT)
                return redirect(
                    reverse("admin:fleet_datasetinferencerun_report") + f"?run={run.pk}"
                )

        # ------------------------------------------ step 2: render the full form
        image_count = len(dataset_inference.select_images(directory, 0)) \
            if directory.is_dir() else 0
        model_ctx = inference_form.model_context(model_source, request.POST)
        context = {
            **base_context,
            "title": f"Run model inference — {dataset.name}",
            **model_ctx,
            **inference_form.pipeline_context(),
            "values": _dataset_run_form_values(model_ctx["defaults"], request.POST),
            "image_count": image_count,
            "images_dir": str(lsapi.image_source_dir(directory))
                          if directory.is_dir() else str(directory),
            "visible_to_trainer": dataset_inference.visible_to_trainer(directory),
            "shared_data_root": str(dataset_inference.shared_data_root()),
        }
        return TemplateResponse(request, "admin/fleet/dataset_inference.html", context)

    def _build_dataset_inference_run(self, request, dataset, model_source):
        """Validate the step-2 POST into an unsaved :class:`DatasetInferenceRun`.

        Returns ``(run, None)`` or ``(None, message)`` on the first problem found,
        so the caller can re-render the form with a warning. The checks a dataset
        run adds over a video's are about the images being *reachable*: a run is
        pointless if the directory is empty, and impossible if the trainer cannot
        see it (images are handed over as paths, never copied).
        """
        model_fields, error = inference_form.parse_model_source(request.POST, model_source)
        if error:
            return None, error
        knobs, error = inference_form.parse_pipeline_fields(request.POST)
        if error:
            return None, error

        # 0 means "every image", so this one is not the positive-int parser.
        image_limit = inference_form.parse_optional_int(
            request.POST.get("image_limit"), 0, 1000000)
        if image_limit is inference_form._INVALID:
            return None, "Image limit must be 0 (all images) or more."
        warmup = inference_form.parse_optional_int(request.POST.get("warmup"), 0, 1000)
        if warmup is inference_form._INVALID:
            return None, "Warmup calls must be 0 or more."

        directory = dataset_inference.dataset_dir(dataset.name)
        if not directory.is_dir():
            return None, f"{dataset.name}: no dataset directory at {directory}."
        if not dataset_inference.visible_to_trainer(directory):
            return None, (
                f"{directory} is outside {dataset_inference.shared_data_root()}, which "
                f"is the only tree the trainer can open — images are handed to it as "
                f"paths, not copied. Move the dataset under the data root (or point "
                f"source_dir there) to run a model on it."
            )
        if not dataset_inference.select_images(directory, 0):
            return None, f"{dataset.name}: no images found under {directory}."

        run = DatasetInferenceRun(
            dataset=dataset,
            dataset_name_snapshot=dataset.name,
            model_source=model_source,
            image_limit=0 if image_limit is None else image_limit,
            warmup=3 if warmup is None else warmup,
            **model_fields,
            **knobs,
        )

        # A bundle's geometry is the bundle's, not the form's — the server-side
        # half of the locked fields on the form, re-read from the manifest so a
        # stale page still runs what the bundle says today.
        try:
            run.sync_bundle()
        except ValueError as exc:
            return None, str(exc)

        # Pre-flight the exact payload the run will send, so a missing artifact,
        # an out-of-root path or a detector-less person pipeline is reported here
        # rather than as a failed job.
        try:
            inference.build_predict_payload(run)
        except RuntimeError as exc:
            return None, str(exc)

        # Snapshotted after sync_bundle, so a bundle run records the bundle it
        # actually resolved.
        run.model_label_snapshot = run.model_label()
        return run, None

    @admin.action(description="Merge selected datasets into a new dataset…")
    def merge_selected(self, request, queryset):
        datasets = sorted(queryset, key=lambda d: d.name)
        if len(datasets) < 2:
            self.message_user(request, "Select at least two datasets to merge.", level=messages.WARNING)
            return None

        if request.POST.get("apply"):
            new_name = (request.POST.get("new_name") or "").strip()
            try:
                kept, _dropped, tools = merge_svc.compute_intersection(datasets)
            except Exception as exc:
                self.message_user(request, f"Cannot merge: {exc}", level=messages.ERROR)
                return None
            if not new_name:
                self.message_user(request, "Enter a name for the merged dataset.", level=messages.WARNING)
                return None
            if Dataset.objects.filter(name=new_name).exists():
                self.message_user(request, f"A dataset named {new_name!r} already exists.", level=messages.ERROR)
                return None
            if not kept:
                self.message_user(request, "The selected datasets share no common class names.", level=messages.ERROR)
                return None
            if not tools:
                self.message_user(request, "The selected datasets share no common labeling tools.", level=messages.ERROR)
                return None
            _queue().enqueue(jobs.merge_datasets, [d.id for d in datasets], new_name)
            self.message_user(
                request,
                f"Merge queued — dataset {new_name!r} will appear here when the worker finishes.",
            )
            return None

        try:
            kept, dropped, tools = merge_svc.compute_intersection(datasets)
        except Exception as exc:
            self.message_user(request, f"Cannot merge: {exc}", level=messages.ERROR)
            return None
        cloud = [d.name for d in datasets if d.storage_type != Dataset.LOCAL]
        # Annotate each row with a fresh labels check so the preview can say whose
        # annotations will be carried over and remapped (vs. images only).
        for d in datasets:
            d.will_carry_labels = datasets_svc.detect_labels(d, persist=False)
        any_labeled = any(d.will_carry_labels for d in datasets)
        context = {
            **self.admin_site.each_context(request),
            "title": "Merge datasets",
            "datasets": datasets,
            "kept": kept,
            "dropped": dropped,
            "tools": tools,
            "cloud": cloud,
            "any_labeled": any_labeled,
            "action": "merge_selected",
            "selected": [str(d.pk) for d in datasets],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/fleet/merge_datasets.html", context)

    @admin.action(description="Split into two datasets by percentage…")
    def split_selected(self, request, queryset):
        datasets = list(queryset)
        if len(datasets) != 1:
            self.message_user(request, "Select exactly one dataset to split.", level=messages.WARNING)
            return None
        dataset = datasets[0]

        if request.POST.get("apply"):
            left_name = (request.POST.get("left_name") or "").strip()
            right_name = (request.POST.get("right_name") or "").strip()
            try:
                left_percent = int(request.POST.get("left_percent", ""))
            except ValueError:
                left_percent = -1
            shuffle = bool(request.POST.get("shuffle"))
            if not left_name or not right_name:
                self.message_user(request, "Enter a name for both split datasets.", level=messages.WARNING)
                return None
            if not 1 <= left_percent <= 99:
                self.message_user(request, "Split percentage must be between 1 and 99.", level=messages.WARNING)
                return None
            for name in (left_name, right_name):
                if Dataset.objects.filter(name=name).exists():
                    self.message_user(request, f"A dataset named {name!r} already exists.", level=messages.ERROR)
                    return None
            if left_name == right_name:
                self.message_user(request, "The two split dataset names must differ.", level=messages.ERROR)
                return None
            _queue().enqueue(
                jobs.split_dataset, dataset.id, left_name, right_name, left_percent, shuffle=shuffle
            )
            self.message_user(
                request,
                f"Split queued — {left_name!r} and {right_name!r} will appear here when the worker "
                f"finishes, and {dataset.name!r} will be removed.",
            )
            return None

        if dataset.storage_type != Dataset.LOCAL:
            self.message_user(request, "Split supports local-storage datasets only.", level=messages.ERROR)
            return None
        try:
            image_count = split_svc.dataset_image_count(dataset)
        except (FileNotFoundError, RuntimeError) as exc:
            self.message_user(request, f"Cannot split: {exc}", level=messages.ERROR)
            return None
        if not image_count:
            self.message_user(request, f"{dataset.name!r} has no images to split.", level=messages.ERROR)
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": "Split dataset",
            "dataset": dataset,
            "image_count": image_count,
            "has_labels": datasets_svc.detect_labels(dataset, persist=False),
            "action": "split_selected",
            "selected": [str(dataset.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/fleet/split_dataset.html", context)

    @admin.action(description="Analyze labeled dataset(s) — class distribution…")
    def analyze_selected(self, request, queryset):
        reports = []
        skipped = []
        for dataset in sorted(queryset, key=lambda d: d.name):
            # Refresh the flag from disk so labels added after the row was created
            # are still recognised; analytics needs a source labels/ folder.
            if not datasets_svc.detect_labels(dataset, persist=False):
                skipped.append(dataset.name)
                continue
            try:
                reports.append(analytics_svc.analyze_dataset(dataset))
            except (FileNotFoundError, RuntimeError) as exc:
                self.message_user(request, f"{dataset.name}: {exc}", level=messages.ERROR)

        if skipped:
            self.message_user(
                request,
                "Skipped unlabeled dataset(s) (no source labels/ folder): "
                + ", ".join(skipped),
                level=messages.WARNING,
            )
        if not reports:
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": "Dataset analytics",
            "reports": reports,
            "selected": [str(report["dataset"].pk) for report in reports],
            "action": "analyze_selected",
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/fleet/dataset_analytics.html", context)

    @admin.action(description="Check for duplicate images in selected dataset(s)…")
    def check_duplicate_images(self, request, queryset):
        """Find duplicate/near-duplicate images *within* each selected dataset.

        This used to be one of the checks in ``analyze_selected``, but it is the
        only one that hashes image bytes rather than reading label files — on a
        large dataset it dominated the runtime of a report operators otherwise
        wanted to run often, so it lives on its own action.

        Matching is the same as ``check_overlapping_images``: exact file MD5
        first, then an 8x8 difference hash for the same photo re-exported at a
        different size/compression. Runs synchronously (disk I/O, not model
        inference), so a big dataset can take a minute or two. Pruning keeps the
        alphabetically first image of each group and is enqueued as an rq job,
        which re-hashes on the worker rather than trusting this report.
        """
        datasets = sorted(queryset, key=lambda d: d.name)
        if not datasets:
            return None

        cloud = [d.name for d in datasets if d.storage_type != Dataset.LOCAL]
        if cloud:
            self.message_user(
                request,
                "Duplicate checking only supports local-storage datasets: " + ", ".join(cloud),
                level=messages.ERROR,
            )
            return None

        if request.POST.get("apply"):
            prune_ids = {str(pk) for pk in request.POST.getlist("prune")}
            targets = [d for d in datasets if str(d.pk) in prune_ids]
            if not targets:
                self.message_user(request, "Select at least one dataset to prune from.",
                                  level=messages.WARNING)
                return None
            for dataset in targets:
                _queue().enqueue(
                    jobs.prune_intra_duplicates, dataset.id,
                    job_timeout=jobs.PRUNE_INTRA_DUPLICATES_JOB_TIMEOUT,
                )
            self.message_user(
                request,
                f"Prune queued for {', '.join(d.name for d in targets)} — re-hashes each dataset and "
                "deletes the extra copies on the worker; check /django-rq/ for progress and the "
                "backup dir once it finishes.",
            )
            return None

        reports = []
        for dataset in datasets:
            try:
                reports.append(overlap_svc.find_intra_duplicates(dataset))
            except (FileNotFoundError, RuntimeError) as exc:
                self.message_user(request, f"{dataset.name}: {exc}", level=messages.ERROR)
        if not reports:
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": "Duplicate images",
            "reports": [
                {
                    **report,
                    "groups": [
                        {"kind": kind, "keep": group[0], "extras": group[1:]}
                        for kind, groups in (("exact", report["exact_groups"]),
                                             ("near", report["near_groups"]))
                        for group in groups
                    ],
                    "duplicate_extra": (report["exact_duplicate_extra"]
                                        + report["near_duplicate_extra"]),
                }
                for report in reports
            ],
            "action": "check_duplicate_images",
            "selected": [str(report["dataset"].pk) for report in reports],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/fleet/dataset_duplicates.html", context)

    @admin.action(description="Check for overlapping images…")
    def check_overlapping_images(self, request, queryset):
        """Find images duplicated across the selected datasets, and resolve them.

        Every image is matched by file MD5 (byte-identical copies) and, failing
        that, by an 8x8 difference hash (the same photo re-exported at a
        different size/compression — the common case for datasets pulled
        through Roboflow/Kaggle more than once). Runs synchronously like
        ``analyze_selected`` — it's disk I/O, not model inference — so large
        datasets can take a minute or two.

        The findings are shown twice over. First a table per dataset pair: which
        image matched which, unchanged from what this action has always
        reported. Then a *resolve* section that regroups those matches into one
        card per picture listing every copy of it, with the survivor picked per
        copy and bulk buttons for the usual "keep everything from this one"
        answer. The regrouping is the point — a near-duplicate matches several
        images on the other side, so pair rows let an operator make
        contradictory choices about the same file, while a group has exactly one
        survivor by construction (see ``overlap.group_overlaps``).

        Pruning is enqueued as an rq job: backing up and deleting thousands of
        full-resolution frames outruns the request timeout on its own. Unlike
        the whole-dataset prunes it does *not* re-hash on the worker — the
        operator's per-copy choice is the input, and re-deriving it would throw
        that choice away.
        """
        datasets = sorted(queryset, key=lambda d: d.name)
        if len(datasets) < 2:
            self.message_user(request, "Select at least two datasets to check for overlaps.", level=messages.WARNING)
            return None

        cloud = [d.name for d in datasets if d.storage_type != Dataset.LOCAL]
        if cloud:
            self.message_user(
                request,
                "Overlap checking only supports local-storage datasets: " + ", ".join(cloud),
                level=messages.ERROR,
            )
            return None

        if request.POST.get("apply"):
            return self._queue_overlap_prune(request)

        # Dataset.name isn't unique at the DB level (two sections may each hold a
        # row for the same folder), and every path here is source_root()/name —
        # so two such rows are the *same files* and would show up as overlapping
        # copies of each other. Pruning "one of them" would delete the other's
        # images out from under it, so say so rather than let it happen quietly.
        counts = Counter(dataset.name for dataset in datasets)
        shared = sorted(name for name, rows in counts.items() if rows > 1)
        if shared:
            self.message_user(
                request,
                "Selected more than one dataset row pointing at the same folder ("
                + ", ".join(shared) + ") — they share their images on disk, so every "
                "image will look overlapping and pruning either row deletes the same files. "
                "Deselect the duplicates.",
                level=messages.ERROR,
            )
            return None

        try:
            prints = {d.pk: overlap_svc.fingerprint_dataset(d) for d in datasets}
        except (FileNotFoundError, NotADirectoryError) as exc:
            self.message_user(request, f"Could not read a dataset's images: {exc}",
                              level=messages.ERROR)
            return None

        reports = overlap_svc.find_overlaps(datasets, prints)
        grouped = overlap_svc.group_overlaps(datasets, prints)
        listed = grouped["groups"][:_OVERLAP_MAX_CARDS]
        context = {
            **self.admin_site.each_context(request),
            "title": "Overlapping images",
            "datasets": datasets,
            "reports": reports,
            "grouped": grouped,
            # Only the cards the page will actually render, numbered for the
            # "group N had no copy kept" error the prune handler can raise.
            "resolve_groups": [
                {**group, "number": number}
                for number, group in enumerate(listed, start=1)
            ],
            "hidden_group_count": len(grouped["groups"]) - len(listed),
            "max_cards": _OVERLAP_MAX_CARDS,
            "thumb_url": reverse("admin:fleet_dataset_overlap_image"),
            "preview_url": reverse("admin:fleet_dataset_overlap_preview"),
            "thumb_width": _OVERLAP_THUMB_WIDTH,
            "action": "check_overlapping_images",
            "selected": [str(d.pk) for d in datasets],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/fleet/dataset_overlaps.html", context)

    # ---------------------------------------------------------- overlap resolve
    @staticmethod
    def _posted_overlap_groups(request) -> list[dict]:
        """The resolve form's groups and choices, as ``{kind, tokens, delete}``.

        The whole decision travels in ONE field, assembled by the page's script,
        for two reasons. It has to travel at all — recomputing the groups here
        would mean fingerprinting every image again (the most expensive thing in
        this module) to answer a *different* question: what overlaps right now,
        not what the operator was looking at when they chose. And it has to be
        one field rather than a hidden input plus a radio pair per copy, because
        that shape hits Django's 1000-field ``DATA_UPLOAD_MAX_NUMBER_FIELDS``
        ceiling at a couple of hundred groups — well inside the range this
        section exists to handle.

        Anything malformed is dropped rather than raising: the field is rebuilt
        from scratch on every submit, so a bad one means a bug on the page, and
        the caller's "nothing to prune" path is the safe answer. Tokens are
        validated against the datasets' own image listings at deletion time.
        """
        try:
            payload = json.loads(request.POST.get("selection") or "{}")
            raw_groups = payload["groups"]
        except (TypeError, ValueError, KeyError):
            return []

        groups = []
        for entry in raw_groups if isinstance(raw_groups, list) else ():
            try:
                tokens = [str(token) for token in entry["tokens"]]
                deleting = {str(token) for token in entry.get("delete", ())}
            except (TypeError, ValueError, KeyError):
                continue
            if tokens:
                groups.append({
                    "kind": str(entry.get("kind", "near")),
                    "tokens": tokens,
                    "delete": [token for token in tokens if token in deleting],
                })
        return groups

    def _queue_overlap_prune(self, request):
        """Validate the resolve form's per-copy choices and enqueue the deletion."""
        groups = self._posted_overlap_groups(request)
        if not groups:
            self.message_user(request, "Nothing to prune — the form carried no duplicate groups.",
                              level=messages.WARNING)
            return None

        doomed: dict[int, list[str]] = {}
        wiped = []
        for number, group in enumerate(groups, start=1):
            if len(group["delete"]) == len(group["tokens"]):
                wiped.append(number)
                continue
            for token in group["delete"]:
                pk, _, name = token.partition(":")
                if pk.isdigit() and name:
                    doomed.setdefault(int(pk), []).append(name)

        # Every group must keep a survivor: the point of the action is to
        # de-duplicate, never to lose the picture from every dataset at once.
        if wiped:
            numbers = ", ".join(str(n) for n in wiped[:10])
            self.message_user(
                request,
                f"Nothing was pruned: group{'' if len(wiped) == 1 else 's'} {numbers}"
                + (" …" if len(wiped) > 10 else "")
                + " had every copy marked for deletion. Keep one copy per group and resubmit.",
                level=messages.ERROR,
            )
            return None
        if not doomed:
            self.message_user(request, "Nothing selected for deletion.", level=messages.WARNING)
            return None

        total = sum(len(names) for names in doomed.values())
        _queue().enqueue(
            jobs.prune_overlap_selection, doomed,
            job_timeout=jobs.PRUNE_OVERLAP_SELECTION_JOB_TIMEOUT,
        )
        self.message_user(
            request,
            f"Prune queued for {total} image{'' if total == 1 else 's'} across "
            f"{len(doomed)} dataset{'' if len(doomed) == 1 else 's'} — each image and its label "
            "file is copied into the backup dir before deletion; check /django-rq/ for progress.",
        )
        return None

    def overlap_preview_view(self, request):
        """Full-screen pager over the duplicate groups, one group per step.

        Reached by submitting the resolve form to this URL, so it inherits the
        exact groups on screen for free — no query string long enough to hold
        them, and no second fingerprinting pass. That also means it is POST-only
        and not bookmarkable, which is fine for a look-before-you-delete view.
        """
        if request.method != "POST":
            raise Http404("the overlap preview opens from the overlap report")

        groups = self._posted_overlap_groups(request)
        names = {}
        for token in (token for group in groups for token in group["tokens"]):
            pk, _, _name = token.partition(":")
            if pk.isdigit():
                names[int(pk)] = None
        for dataset in Dataset.objects.filter(pk__in=list(names)):
            names[dataset.pk] = dataset.name

        payload = []
        for group in groups:
            copies = []
            for token in group["tokens"]:
                pk, _, name = token.partition(":")
                if not pk.isdigit() or not name:
                    continue
                copies.append({"dataset": int(pk), "dataset_name": names.get(int(pk)) or "?",
                               "name": name})
            if copies:
                payload.append({"kind": group["kind"], "copies": copies})

        context = {
            **self.admin_site.each_context(request),
            "title": "Preview overlapping images",
            "groups_payload": payload,
            "group_count": len(payload),
            "image_url": reverse("admin:fleet_dataset_overlap_image"),
            "thumb_width": _OVERLAP_THUMB_WIDTH,
        }
        return TemplateResponse(request, "admin/fleet/dataset_overlap_viewer.html", context)

    def overlap_image(self, request):
        """Stream one dataset image by filename, downscaled when ``?w=`` is given.

        Addressed by name, not by the label viewer's ``?index=``: a prune shifts
        every index after the deleted file, and the resolve grid outlives the
        prune it queues. ``?w=`` matters for the grid specifically — a few
        hundred cards pulling 4K frames at full size is tens of gigabytes, so
        the thumbnails ask for a downscaled re-encode instead.
        """
        dataset = Dataset.objects.filter(pk=request.GET.get("dataset")).first()
        if dataset is None:
            raise Http404("unknown dataset")

        image = overlap_svc.resolve_dataset_image(
            lsapi.image_source_dir(source_root() / dataset.name), request.GET.get("name") or "")
        if image is None:
            raise Http404("unknown image")

        try:
            width = int(request.GET.get("w") or 0)
        except (TypeError, ValueError):
            raise Http404("bad width")
        if width <= 0:
            return FileResponse(open(image, "rb"))
        return _thumbnail_response(image, min(max(width, 32), 1024))

    # -------------------------------------------------------------- label preview
    @admin.action(description="Preview dataset labels…")
    def preview_labels(self, request, queryset):
        """Open the interactive label viewer for one dataset.

        Draws the on-disk annotations (source labels, an annotator's output, or an
        explicit path) on each image and lets the operator scroll through — no model,
        no inference. Like the model preview this persists nothing: on submit it just
        redirects to the viewer with the chosen label source as query params.
        """
        from training.models import ExperimentDataset

        if queryset.count() != 1:
            self.message_user(request, "Select exactly one dataset to preview.",
                              level=messages.WARNING)
            return None
        dataset = queryset.first()

        if request.POST.get("apply"):
            params = {
                "dataset": dataset.pk,
                "label_source": request.POST.get("label_source") or ExperimentDataset.SOURCE,
                "annotator": request.POST.get("annotator") or "",
                "explicit_labels_path": (request.POST.get("explicit_labels_path") or "").strip(),
            }
            query = urlencode({k: v for k, v in params.items() if v not in ("", None)})
            return redirect(reverse("admin:fleet_dataset_preview") + "?" + query)

        context = {
            **self.admin_site.each_context(request),
            "title": f"Preview labels for {dataset.name}",
            "dataset": dataset,
            "annotators": Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"),
            "label_source_choices": ExperimentDataset.LABEL_SOURCE_CHOICES,
            "default_label_source": ExperimentDataset.SOURCE,
            "action": "preview_labels",
            "selected": [str(dataset.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/fleet/dataset_preview.html", context)

    def get_urls(self):
        custom = [
            path("dataset-preview/", self.admin_site.admin_view(self.preview_view),
                 name="fleet_dataset_preview"),
            path("dataset-preview/image/", self.admin_site.admin_view(self.preview_image),
                 name="fleet_dataset_preview_image"),
            path("dataset-preview/data/", self.admin_site.admin_view(self.preview_data),
                 name="fleet_dataset_preview_data"),
            path("dataset-quality/fix/", self.admin_site.admin_view(self.quality_fix),
                 name="fleet_dataset_quality_fix"),
            path("dataset-overlaps/preview/", self.admin_site.admin_view(self.overlap_preview_view),
                 name="fleet_dataset_overlap_preview"),
            path("dataset-overlaps/image/", self.admin_site.admin_view(self.overlap_image),
                 name="fleet_dataset_overlap_image"),
        ]
        return custom + super().get_urls()

    def quality_fix(self, request):
        """Apply one source-label quality repair from the analytics page."""
        if request.method != "POST":
            return JsonResponse({"error": "POST required"}, status=405)

        dataset = Dataset.objects.filter(pk=request.POST.get("dataset_id")).first()
        if dataset is None:
            return JsonResponse({"error": "unknown dataset"}, status=400)

        issue = request.POST.get("issue") or ""
        action = request.POST.get("action") or None

        try:
            result = data_quality_solve.solve_dataset_quality(dataset, issue, action)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        return JsonResponse(result)

    def preview_view(self, request):
        """Render the viewer shell; the browser pulls images + labels per index."""
        dataset = Dataset.objects.filter(pk=request.GET.get("dataset")).first()
        if dataset is None:
            raise Http404("preview requires ?dataset=")

        images = lsapi.list_dataset_images(source_root() / dataset.name)
        has_labels = _preview_label_dir(request, dataset) is not None

        context = {
            **self.admin_site.each_context(request),
            "title": f"Labels — {dataset.name}",
            "dataset": dataset,
            "image_count": len(images),
            "has_labels": has_labels,
            "query": request.GET.urlencode(),
        }
        return TemplateResponse(request, "admin/fleet/dataset_preview_viewer.html", context)

    def preview_image(self, request):
        """Stream the raw bytes of the dataset image at ``?index=``."""
        dataset = Dataset.objects.filter(pk=request.GET.get("dataset")).first()
        if dataset is None:
            raise Http404("unknown dataset")
        images = lsapi.list_dataset_images(source_root() / dataset.name)
        index = _preview_bounded_index(request, len(images))
        return FileResponse(open(images[index], "rb"))

    def preview_data(self, request):
        """Return the label shapes for one image (no inference)."""
        from training.services import config_gen

        dataset = Dataset.objects.filter(pk=request.GET.get("dataset")).first()
        if dataset is None:
            return JsonResponse({"error": "unknown dataset"}, status=400)

        images = lsapi.list_dataset_images(source_root() / dataset.name)
        if not images:
            return JsonResponse({"error": "dataset has no images"}, status=400)
        index = _preview_bounded_index(request, len(images))
        image_path = images[index]
        dimensions = analytics_svc._image_dimensions(image_path)
        width, height = dimensions if dimensions is not None else (None, None)


        label_dir = _preview_label_dir(request, dataset)
        shapes = (
            _read_label_shapes(label_dir, image_path, config_gen.dataset_classes(dataset))
            if label_dir is not None else []
        )
        return JsonResponse({
            "shapes": shapes,
            "image": image_path.name,
            "width": width,
            "height": height,
            "index": index,
            "count": len(images),
        })


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ["title", "annotator", "dataset", "ls_project_id", "webhook_id", "status_badge"]
    list_filter = ["dataset", "last_status"]
    search_fields = ["title", "annotator__username", "dataset__name"]
    readonly_fields = ["last_status", "last_error", "last_run_at", "created_at"]
    actions = ["sync_selected"]

    @admin.display(description="last status", ordering="last_status")
    def status_badge(self, obj):
        return _status_badge(obj.last_status)

    @admin.action(description="Sync selected projects from Label Studio")
    def sync_selected(self, request, queryset):
        queue = _queue()
        for project in queryset:
            queue.enqueue(jobs.sync_project, project.id)
            project.last_status = "queued"
            project.last_run_at = timezone.now()
            project.save(update_fields=["last_status", "last_run_at"])
        self.message_user(request, f"{queryset.count()} sync job(s) queued — refresh to see progress.")


@admin.register(GroundingSamRun)
class GroundingSamRunAdmin(admin.ModelAdmin):
    """Progress tracker for Grounding SAM auto-labeling jobs.

    Purely a monitoring view — runs are created from the Dataset admin's
    "Generate labels with Grounding SAM…" action, never here. Refresh the list
    to watch ``images_processed`` move along as the worker gets through a
    dataset's batches.
    """

    list_display = [
        "dataset", "status_badge", "progress", "images_labeled",
        "detections_written", "confidence", "created_at", "finished_at",
    ]
    list_filter = ["status", "dataset"]
    ordering = ["-created_at"]
    fields = [
        "dataset", "confidence", "status_badge", "progress", "images_labeled",
        "detections_written", "error", "created_at", "started_at", "finished_at",
    ]
    readonly_fields = fields

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="progress")
    def progress(self, obj):
        return f"{obj.images_processed}/{obj.images_total}"


@admin.register(DatasetInferenceRun)
class DatasetInferenceRunAdmin(admin.ModelAdmin):
    """The Dataset inference runs tab: every "Run model inference…", each opening
    a timing report.

    A tab rather than something hanging off the dataset's page, for the reason
    the VLM dataset runs are one: the point of making a run is usually to compare
    it with another (this engine against that ONNX, fp16 against fp32, this tile
    size against that one), and comparing wants a list. The changelist therefore
    carries the two headline numbers — fps and median latency — so a comparison
    often needs no report opened at all.

    Rows are not editable: every field is either a snapshot of what ran or a
    measurement of what happened.
    """

    list_display = ["dataset_name_snapshot", "model_label_column", "pipeline",
                    "status_badge", "progress", "fps_column", "p50_column",
                    "detections_total", "errors_count", "created_at", "report_link"]
    list_display_links = None
    list_filter = ["status", "model_source", "pipeline", "trained_model"]
    search_fields = ["dataset_name_snapshot", "model_label_snapshot",
                     "artifact_path", "bundle_path"]
    date_hierarchy = "created_at"

    def has_add_permission(self, request):
        """Runs come from the Datasets tab's action, never from an add form."""
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("dataset", "trained_model")

    # --------------------------------------------------------------- displays
    @admin.display(description="model")
    def model_label_column(self, obj):
        return obj.model_label()

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="images")
    def progress(self, obj):
        if obj.images_total:
            return f"{obj.images_processed}/{obj.images_total}"
        return str(obj.images_processed)

    @admin.display(description="fps", ordering="fps")
    def fps_column(self, obj):
        """Model fps, marked when it is really the round trip.

        A run measured against a trainer that reports no per-call timing has only
        the HTTP round trip to divide into a second, and that is a slower number
        for a reason that has nothing to do with the model — so it is labelled
        rather than shown as if it were the same measurement.
        """
        if obj.fps is None:
            return "—"
        if (obj.timings or {}).get("timing_source") == "request":
            return format_html("<b>{}</b> <span style='color:#666'>(round trip)</span>",
                               f"{obj.fps:.1f}")
        return format_html("<b>{}</b>", f"{obj.fps:.1f}")

    @admin.display(description="p50 ms", ordering="p50_ms")
    def p50_column(self, obj):
        return f"{obj.p50_ms:.1f}" if obj.p50_ms is not None else "—"

    @admin.display(description="")
    def report_link(self, obj):
        url = reverse("admin:fleet_datasetinferencerun_report") + f"?run={obj.pk}"
        return format_html('<a class="button" href="{}">open report</a>', url)

    # ------------------------------------------------------------------- urls
    def get_urls(self):
        custom = [
            path("report/", self.admin_site.admin_view(self.report_view),
                 name="fleet_datasetinferencerun_report"),
            path("report-progress/", self.admin_site.admin_view(self.progress_view),
                 name="fleet_datasetinferencerun_progress"),
            path("report-cancel/", self.admin_site.admin_view(self.cancel_view),
                 name="fleet_datasetinferencerun_cancel"),
            path("report-image/", self.admin_site.admin_view(self.image_view),
                 name="fleet_datasetinferencerun_image"),
        ]
        return custom + super().get_urls()

    def _get_run(self, request) -> DatasetInferenceRun:
        run = DatasetInferenceRun.objects.select_related("dataset", "trained_model").filter(
            pk=request.GET.get("run") or request.POST.get("run")).first()
        if run is None:
            raise Http404("unknown dataset inference run")
        return run

    def report_view(self, request):
        """The run's report: what ran, what it measured, and the images.

        A finished run renders entirely server-side; a live one renders what
        exists so far and lets the JS extend it from the last ``seq``. The poll
        is not a heartbeat — the run keeps going whether or not anyone is
        looking, and Stop is the only thing that ends it early.
        """
        run = self._get_run(request)
        only_detections = request.GET.get("only") == "detections"

        results = run.results.all()
        if only_detections:
            results = results.filter(detections__gt=0)
        rows = list(results[:_RUN_ROW_LIMIT])

        timings = run.timings or {}
        context = {
            **self.admin_site.each_context(request),
            "title": f"{run.model_label()} on {run.dataset_name_snapshot}",
            "run": run,
            "rows": rows,
            "shown": len(rows),
            "total_rows": results.count(),
            "row_limit": _RUN_ROW_LIMIT,
            "thumb_width": _RUN_THUMB_WIDTH,
            "only_detections": only_detections,
            # A filtered slice must not be appended to by the live poll: an
            # image that detected nothing does not belong in it.
            "live_append": not only_detections,
            "last_seq": rows[-1].seq if rows and not only_detections else 0,
            "is_terminal": run.is_terminal(),
            "timings": timings,
            "stage_rows": _stage_rows(timings),
            "geometry": _run_geometry_rows(run),
            "progress_url": reverse("admin:fleet_datasetinferencerun_progress") + f"?run={run.pk}",
            "cancel_url": reverse("admin:fleet_datasetinferencerun_cancel") + f"?run={run.pk}",
            "image_url": reverse("admin:fleet_datasetinferencerun_image"),
            "report_url": reverse("admin:fleet_datasetinferencerun_report") + f"?run={run.pk}",
            "back_url": reverse("admin:fleet_datasetinferencerun_changelist"),
        }
        return TemplateResponse(request, "admin/fleet/dataset_inference_report.html",
                                context)

    def progress_view(self, request):
        """Counters, the headline timings, and the results since ``after``."""
        run = self._get_run(request)
        try:
            after = int(request.GET.get("after", 0))
        except (TypeError, ValueError):
            after = 0

        rows = run.results.filter(seq__gt=after).order_by("seq")[:_RUN_ROW_LIMIT]
        timings = run.timings or {}
        return JsonResponse({
            "run": {
                "status": run.status,
                "images_processed": run.images_processed,
                "images_total": run.images_total,
                "errors_count": run.errors_count,
                "detections_total": run.detections_total,
                "last_error": run.last_error,
                "terminal": run.is_terminal(),
                "fps": run.fps,
                "p50_ms": run.p50_ms,
                "timing_source": timings.get("timing_source"),
            },
            "results": [
                {
                    "pk": row.pk,
                    "seq": row.seq,
                    "image": row.image_filename,
                    "detections": row.detections,
                    "model_ms": row.model_ms,
                    "request_ms": row.request_ms,
                    "error": row.error,
                }
                for row in rows
            ],
        })

    def cancel_view(self, request):
        """Ask a running job to stop after its current image.

        POST-only so the CSRF middleware protects it. A cancelled run keeps the
        images it already measured — they are paid for, and their numbers are
        still true of the model.
        """
        if request.method != "POST":
            raise Http404("POST only")
        run = self._get_run(request)
        if not run.is_terminal():
            run.status = DatasetInferenceRun.CANCEL_REQUESTED
            run.save(update_fields=["status"])
            self.message_user(request, "Stopping after the current image.")
        return redirect(reverse("admin:fleet_datasetinferencerun_report") + f"?run={run.pk}")

    def image_view(self, request):
        """One result's image: a thumbnail by default, its boxes drawn with ``?boxes=1``.

        Resolved by result row rather than by an index into the dataset, so a page
        of thumbnails costs one stat each instead of a directory listing each, and
        the containment check in
        :func:`fleet.services.dataset_inference.result_image_path` keeps a stored
        filename from addressing anything outside its own dataset.

        The thumbnails ask for the plain file (re-encoded small) and the
        click-through asks for the annotated full-size copy, so a page of them
        does not pay to decode and redraw every image — the same split the VLM
        dataset report uses.
        """
        result = DatasetInferenceResult.objects.select_related("run").filter(
            pk=request.GET.get("result")).first()
        if result is None:
            raise Http404("unknown result")

        try:
            path = dataset_inference.result_image_path(result)
        except ValueError:
            raise Http404("image outside its dataset")
        except FileNotFoundError:
            raise Http404("image not found on disk")

        if not request.GET.get("boxes"):
            return _thumbnail_response(path, _RUN_THUMB_WIDTH)
        try:
            return HttpResponse(dataset_inference.render_result(result),
                                content_type="image/jpeg")
        except RuntimeError as exc:
            raise Http404(str(exc))
