"""Signal handlers wiring Video/InferenceJob deletion to on-disk cleanup.

Mirrors ``training.signals``: deleting a :class:`~videos.models.Video` should
remove its file under ``videos_root``, and deleting an
:class:`~videos.models.InferenceJob` should remove its annotated output video.
``post_delete`` is used rather than a ``ModelAdmin`` override so cleanup also
fires when a row disappears by cascade — e.g. deleting a
:class:`training.TrainedModel` CASCADEs to its ``InferenceJob`` rows, which
never goes through ``InferenceJobAdmin``'s own delete flow. ``post_delete``
fires for single deletes, bulk "delete selected", and cascades alike, so this
is the one place that covers every deletion path. Cleanup never raises — a
filesystem hiccup must not roll back the row deletion.
"""

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from videos.models import InferenceJob, Video

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=Video, dispatch_uid="video_cleanup")
def _video_deleted(sender, instance, **kwargs):
    try:
        if instance.filename:
            instance.path().unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - never let cleanup break the delete
        logger.exception("cleanup failed for Video #%s", instance.pk)


@receiver(post_delete, sender=InferenceJob, dispatch_uid="inference_job_cleanup")
def _inference_job_deleted(sender, instance, **kwargs):
    try:
        if instance.output_filename:
            instance.output_path().unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        logger.exception("cleanup failed for InferenceJob #%s", instance.pk)
