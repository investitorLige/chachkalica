"""Signal handlers wiring eval-pipeline model deletion to on-disk cleanup.

Mirrors ``training.signals`` for the base ``EvalRun``: deleting an eval row —
from the Base Eval tab, any per-pipeline list, or via the combined list's
delete action — should also remove its output dir and generated request YAML.

Every list an operator actually deletes from here is backed by a *proxy*
model (``BaseEval`` proxies ``training.EvalRun``; ``BatchDetectEval`` /
``PeopleDetectFirstEval`` / ``BatchPeopleEval`` / ``ChainEval`` all proxy
``PipelineEvalRun``) — Django's delete Collector groups instances by the exact
class the queryset/instance was fetched as, so a delete made through a proxy's
own manager fires ``post_delete`` with that proxy as ``sender``, never the
concrete model underneath. A receiver registered only on the concrete model
silently never sees those deletes, so every proxy gets its own registration
here even though they all run the same cleanup. ``post_delete`` fires for
single deletes, bulk "delete selected", and cascades alike, so this is the one
place that covers every deletion path. Cleanup never raises — a filesystem
hiccup must not roll back the row deletion.
"""

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from eval_pipelines.models import (
    BaseEval,
    BatchDetectEval,
    BatchPeopleEval,
    ChainEval,
    PeopleDetectFirstEval,
    PipelineEvalRun,
)
from training.services import cleanup

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=PipelineEvalRun, dispatch_uid="pipeline_eval_run_cleanup")
@receiver(post_delete, sender=BatchDetectEval, dispatch_uid="batch_detect_eval_cleanup")
@receiver(post_delete, sender=PeopleDetectFirstEval, dispatch_uid="people_detect_first_eval_cleanup")
@receiver(post_delete, sender=BatchPeopleEval, dispatch_uid="batch_people_eval_cleanup")
@receiver(post_delete, sender=ChainEval, dispatch_uid="chain_eval_cleanup")
def _pipeline_eval_run_deleted(sender, instance, **kwargs):
    try:
        cleanup.remove_eval_artifacts(instance)
    except Exception:  # noqa: BLE001 - never let cleanup break the delete
        logger.exception("cleanup failed for %s #%s", sender.__name__, instance.pk)


@receiver(post_delete, sender=BaseEval, dispatch_uid="base_eval_cleanup")
def _base_eval_deleted(sender, instance, **kwargs):
    try:
        cleanup.remove_eval_artifacts(instance)
    except Exception:  # noqa: BLE001
        logger.exception("cleanup failed for BaseEval #%s", instance.pk)
