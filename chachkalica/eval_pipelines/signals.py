"""Signal handlers wiring PipelineEvalRun deletion to on-disk cleanup.

Mirrors ``training.signals`` for the base ``EvalRun``: deleting a pipeline eval
(from any per-pipeline list, or via the combined list's delete action) should
also remove its output dir and generated request YAML. ``post_delete`` fires
for single deletes, bulk "delete selected", and cascades alike, so this is the
one place that covers every deletion path. Cleanup never raises — a filesystem
hiccup must not roll back the row deletion.
"""

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from eval_pipelines.models import PipelineEvalRun
from training.services import cleanup

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=PipelineEvalRun, dispatch_uid="pipeline_eval_run_cleanup")
def _pipeline_eval_run_deleted(sender, instance, **kwargs):
    try:
        cleanup.remove_eval_artifacts(instance)
    except Exception:  # noqa: BLE001 - never let cleanup break the delete
        logger.exception("cleanup failed for PipelineEvalRun #%s", instance.pk)
