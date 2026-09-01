"""Admin = the fleet operator console.

Long operations (provision / setup / sync / remove) are enqueued as rq jobs —
each action enqueues one job per selected row, flips the row to ``queued``, and
returns immediately. The row's status column then reflects
``queued -> running -> ok/error`` as the worker picks it up (refresh to see it).
"""

from pathlib import Path

import django_rq
from django import forms
from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils import timezone
from django.utils.html import format_html
from django.utils.http import urlencode

from fleet import jobs
from fleet.models import Annotator, Dataset, FleetSettings, GroundingSamRun, Project
from fleet.services import analytics as analytics_svc
from fleet.services import data_quality_solve
from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services import merge as merge_svc
from fleet.services import overlap as overlap_svc
from fleet.services import split as split_svc
from fleet.services.paths import source_root

_STATUS_COLORS = {
    "ok": "#22c55e",
    "running": "#f59e0b",
    "queued": "#9ca3af",
    "paused": "#3b82f6",
    "warning": "#f97316",
    "error": "#ef4444",
}


def _queue():
    return django_rq.get_queue("default")


def _status_badge(value: str):
    if not value:
        return "—"
    color = _STATUS_COLORS.get(value, "#9ca3af")
    return format_html(
        '<b style="color:{};">●</b> {}', color, value
    )


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
    folders not yet registered are offered; when editing, the current name stays
    selectable even if its folder has since gone missing.
    """

    class Meta:
        model = Dataset
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name_field = self._meta.model._meta.get_field("name")
        current = self.instance.name if self.instance and self.instance.pk else None

        taken = set(Dataset.objects.values_list("name", flat=True))
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

    Uses the same pairing the rest of the app uses (``<stem>.txt`` then
    ``<image>.txt``) and :func:`txt_format.parse_label_text`, which auto-detects
    the app's ``W H`` header vs header-less YOLO and splits polygons from boxes.
    Each shape is ``{class_id, class_name, kind, bbox, polygon}`` with normalized
    center-xywh ``bbox`` and, for polygons, a flat normalized ``polygon`` list.
    """
    from fleet.reconcile import txt_format

    label_dir = Path(label_dir)
    candidates = [label_dir / f"{image_path.stem}.txt", label_dir / f"{image_path.name}.txt"]
    label_file = next((c for c in candidates if c.exists()), None)
    if label_file is None:
        return []
    _w, _h, objects = txt_format.parse_label_text(label_file.read_text())
    shapes = []
    for obj in objects:
        class_id = obj["class_id"]
        name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        cx, cy, w, h = obj["bbox"]
        polygon = obj.get("polygon")
        shapes.append({
            "class_id": class_id,
            "class_name": name,
            "kind": "polygon" if polygon else "box",
            "bbox": {"cx": cx, "cy": cy, "w": w, "h": h},
            "polygon": polygon or [],
        })
    return shapes


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    form = DatasetAdminForm
    list_display = ["name", "storage_type", "storage_root", "has_labels"]
    readonly_fields = ["has_labels"]
    search_fields = ["name"]
    actions = [
        "setup_for_all_active",
        "sync_all_projects",
        "setup_sync_one_annotator",
        "promote_annotator_labels",
        "generate_grounding_sam_labels",
        "merge_selected",
        "split_selected",
        "analyze_selected",
        "check_overlapping_images",
        "preview_labels",
    ]

    def save_model(self, request, obj, form, change):
        # Refresh has_labels from disk whenever a dataset is added/edited here,
        # so the flag reflects whether a source labels/ folder is present.
        super().save_model(request, obj, form, change)
        datasets_svc.detect_labels(obj)

    @admin.action(description="Set up selected dataset(s) for all active annotators")
    def setup_for_all_active(self, request, queryset):
        queue = _queue()
        active = list(Annotator.objects.filter(status=Annotator.ACTIVE))
        if not active:
            self.message_user(request, "No active annotators to set up.", level="warning")
            return
        count = 0
        for dataset in queryset:
            for annotator in active:
                queue.enqueue(jobs.setup_project, dataset.id, annotator.id)
                count += 1
        self.message_user(request, f"{count} setup job(s) queued (datasets × active annotators).")

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
        datasets = list(queryset)
        if request.POST.get("apply"):
            annotator = Annotator.objects.filter(pk=request.POST.get("annotator")).first()
            if annotator is None:
                self.message_user(request, "Choose an annotator.", level=messages.WARNING)
                return None
            queue = _queue()
            for dataset in datasets:
                queue.enqueue(jobs.setup_and_sync_project, dataset.id, annotator.id)
            self.message_user(
                request,
                f"{len(datasets)} setup+sync job(s) queued for {annotator.username} — "
                "refresh the Projects page to see progress.",
            )
            return None

        active = list(Annotator.objects.filter(status=Annotator.ACTIVE).order_by("username"))
        if not active:
            self.message_user(request, "No active annotators to set up.", level=messages.WARNING)
            return None
        context = {
            **self.admin_site.each_context(request),
            "title": "Set up + sync for one annotator",
            "datasets": datasets,
            "annotators": active,
            "action": "setup_sync_one_annotator",
            "selected": [str(d.pk) for d in datasets],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
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
                self.message_user(
                    request,
                    f"{dataset.name}: promoted {result['moved']} label file(s) from "
                    f"{annotator.username} to {result['dest']}.",
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

    @admin.action(description="Check for overlapping images…")
    def check_overlapping_images(self, request, queryset):
        """Find images duplicated across two or more datasets.

        Every image is matched by file MD5 (byte-identical copies) and, failing
        that, by an 8x8 difference hash (the same photo re-exported at a
        different size/compression — the common case for datasets pulled
        through Roboflow/Kaggle more than once). Runs synchronously like
        ``analyze_selected`` — it's disk I/O, not model inference — so large
        datasets can take a minute or two.

        Pruning the duplicates found is only offered when exactly two datasets
        are selected: with three or more it's ambiguous which side(s) an
        operator means to keep. Pruning itself is enqueued as an rq job — it
        re-hashes and then deletes on the worker — since a two-dataset prune
        can run long enough to hit the request timeout.
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
            if len(datasets) != 2:
                self.message_user(request, "Pruning is only available for exactly two datasets.", level=messages.ERROR)
                return None
            prune_left = bool(request.POST.get("prune_left"))
            prune_right = bool(request.POST.get("prune_right"))
            if not prune_left and not prune_right:
                self.message_user(request, "Select at least one dataset to prune from.", level=messages.WARNING)
                return None
            _queue().enqueue(
                jobs.prune_overlaps,
                datasets[0].id,
                datasets[1].id,
                prune_left=prune_left,
                prune_right=prune_right,
                job_timeout=jobs.PRUNE_OVERLAPS_JOB_TIMEOUT,
            )
            self.message_user(
                request,
                "Prune queued — re-hashes both datasets and deletes duplicates on the worker; "
                "check /django-rq/ for progress and the backup dir once it finishes.",
            )
            return None

        reports = overlap_svc.find_overlaps(datasets)
        context = {
            **self.admin_site.each_context(request),
            "title": "Overlapping images",
            "datasets": datasets,
            "reports": reports,
            "can_prune": len(datasets) == 2,
            "action": "check_overlapping_images",
            "selected": [str(d.pk) for d in datasets],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/fleet/dataset_overlaps.html", context)

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

        if issue == "duplicate_images":
            # Unlike the label-file repairs below, pruning duplicates re-hashes
            # every image in the dataset — that can run long enough to hit the
            # request timeout, so it's queued the same way as check_overlapping_images.
            if action not in (None, "", "prune"):
                return JsonResponse({"error": "duplicate images only support prune"}, status=400)
            _queue().enqueue(
                jobs.prune_intra_duplicates, dataset.id,
                job_timeout=jobs.PRUNE_INTRA_DUPLICATES_JOB_TIMEOUT,
            )
            return JsonResponse({"queued": True, "dataset": dataset.name, "issue": issue})

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
