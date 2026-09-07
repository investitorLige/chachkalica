"""Delete a VLM video's file when its row goes away.

Mirrors ``videos.signals``: ``post_delete`` rather than a ``ModelAdmin``
override, so cleanup also fires for bulk deletes and cascades. Cleanup never
raises — a filesystem hiccup must not roll back the row deletion.
"""

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from vlm.models import VlmVideo

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=VlmVideo, dispatch_uid="vlm_video_cleanup")
def _vlm_video_deleted(sender, instance, **kwargs):
    try:
        if instance.filename:
            instance.path().unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - never let cleanup break the delete
        logger.exception("cleanup failed for VlmVideo #%s", instance.pk)
