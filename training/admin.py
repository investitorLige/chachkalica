"""Admin = the training operator console.

The Experiment page composes a friendy_mercury config from inline dataset/model
rows; the "Generate config & create run" action validates the roster, writes the
YAML, and records a :class:`TrainingRun`. Executing that run against the
trainer service is a later phase — for now the action ends at a generated,
ready-to-run config.
"""

from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.template.response import TemplateResponse

from fleet.admin import _status_badge
from training.models import (
    Experiment,
    ExperimentDataset,
    ExperimentModel,
    TrainingRun,
    TrainingSettings,
)
from training.services import config_gen


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
    fields = ["order", "dataset", "role", "label_source", "annotator", "explicit_labels_path"]


class ExperimentModelInline(admin.TabularInline):
    model = ExperimentModel
    extra = 1
    fields = ["order", "arch", "num_classes", "params"]


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
                    "amp", "gradient_clip_norm",
                    "optimizer_name", "lr", "weight_decay", "optimizer_params",
                    "scheduler_name", "scheduler_params",
                ]
            },
        ),
        (
            "Evaluation",
            {"fields": ["eval_batch_size", "eval_num_workers", "eval_score_threshold", "iou_thresholds"]},
        ),
    ]

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
                self.message_user(
                    request,
                    f"Run #{run.pk} created — config written to {yaml_path}. "
                    "Execution against the trainer service lands in the next phase.",
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


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ["__str__", "experiment", "status_badge", "config_yaml_path", "created_at"]
    list_filter = ["status", "experiment"]
    readonly_fields = [
        "experiment", "status", "config_yaml_path", "output_dir",
        "last_error", "started_at", "finished_at", "results", "created_at",
    ]

    def has_add_permission(self, request):
        # Runs are created by the Experiment action, not by hand.
        return False

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)
