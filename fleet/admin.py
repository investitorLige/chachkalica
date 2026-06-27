"""Admin = the fleet operator console.

Long operations (provision / setup / sync / remove) are enqueued as rq jobs —
each action enqueues one job per selected row, flips the row to ``queued``, and
returns immediately. The row's status column then reflects
``queued -> running -> ok/error`` as the worker picks it up (refresh to see it).
"""

import django_rq
from django import forms
from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.template.response import TemplateResponse
from django.utils import timezone
from django.utils.html import format_html

from fleet import jobs
from fleet.models import Annotator, Dataset, FleetSettings, Project
from fleet.services import analytics as analytics_svc
from fleet.services import datasets as datasets_svc
from fleet.services import merge as merge_svc
from fleet.services.paths import source_root

_STATUS_COLORS = {
    "ok": "#22c55e",
    "running": "#f59e0b",
    "queued": "#9ca3af",
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
        "merge_selected",
        "analyze_selected",
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
        }
        return TemplateResponse(request, "admin/fleet/dataset_analytics.html", context)


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
