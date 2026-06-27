"""Signal receivers for the fleet app.

Wired up from ``FleetConfig.ready()``.
"""

import logging

from django.db.models.signals import pre_delete
from django.dispatch import receiver

from fleet.models import Dataset
from fleet.services import datasets as datasets_svc

logger = logging.getLogger(__name__)


@receiver(pre_delete, sender=Dataset)
def delete_dataset_projects(sender, instance: Dataset, **kwargs):
    """Tear down each annotator's Label Studio project before the dataset goes.

    Runs while the ``Project`` rows still exist (the cascade fires after this),
    so we still know every ``ls_project_id``. Best-effort — failures are logged,
    never raised, so a down container can't block the deletion.
    """
    try:
        results = datasets_svc.teardown_dataset_projects(instance)
    except Exception:  # noqa: BLE001 - never let cleanup block a delete
        logger.exception("Failed tearing down LS projects for dataset %s", instance.name)
        return
    for entry in results:
        logger.info(
            "dataset %s teardown: %s project=%s -> %s",
            instance.name, entry["username"], entry["project_id"], entry["status"],
        )
