"""Signal handlers wiring model deletion to on-disk cleanup.

Deleting a row on any training-app admin tab — runs, results, models, exports,
evals — should also remove its artifacts under ``data/`` (checkpoints, logs,
generated config), the one exception being ``fleet.Dataset`` (never touched
here or anywhere else). ``post_delete`` fires for single deletes, bulk "delete
selected", and cascades alike, so this is the one place that covers every
deletion path. Cleanup never raises — a filesystem hiccup must not roll back
the row deletion.
"""

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from training.models import EvalRun, ExportRun, FineTuningRun, RunResult, TrainedModel, TrainingRun
from training.services import cleanup

logger = logging.getLogger(__name__)


# FineTuningRun is a proxy of TrainingRun (same table, see its docstring) so
# it needs its own sender registration: Django's delete Collector groups
# instances by the exact class the queryset/instance was fetched as, so a
# delete made through the proxy manager (the "Fine-tuning runs" tab) fires
# post_delete with sender=FineTuningRun, never sender=TrainingRun — a
# `sender=TrainingRun` receiver alone silently never sees it.
@receiver(post_delete, sender=TrainingRun, dispatch_uid="training_run_cleanup")
@receiver(post_delete, sender=FineTuningRun, dispatch_uid="finetuning_run_cleanup")
def _training_run_deleted(sender, instance, **kwargs):
    try:
        cleanup.remove_run_artifacts(instance)
    except Exception:  # noqa: BLE001 - never let cleanup break the delete
        logger.exception("cleanup failed for %s #%s", sender.__name__, instance.pk)


@receiver(post_delete, sender=EvalRun, dispatch_uid="eval_run_cleanup")
def _eval_run_deleted(sender, instance, **kwargs):
    try:
        cleanup.remove_eval_artifacts(instance)
    except Exception:  # noqa: BLE001
        logger.exception("cleanup failed for EvalRun #%s", instance.pk)


@receiver(post_delete, sender=RunResult, dispatch_uid="run_result_cleanup")
def _run_result_deleted(sender, instance, **kwargs):
    try:
        cleanup.remove_run_result_artifacts(instance)
    except Exception:  # noqa: BLE001
        logger.exception("cleanup failed for RunResult #%s", instance.pk)


@receiver(post_delete, sender=TrainedModel, dispatch_uid="trained_model_cleanup")
def _trained_model_deleted(sender, instance, **kwargs):
    try:
        cleanup.remove_trained_model_artifacts(instance)
    except Exception:  # noqa: BLE001
        logger.exception("cleanup failed for TrainedModel #%s", instance.pk)


@receiver(post_delete, sender=ExportRun, dispatch_uid="export_run_cleanup")
def _export_run_deleted(sender, instance, **kwargs):
    try:
        cleanup.remove_export_artifacts(instance)
    except Exception:  # noqa: BLE001
        logger.exception("cleanup failed for ExportRun #%s", instance.pk)
