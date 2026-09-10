"""Signal handlers wiring Video/Render deletion to on-disk cleanup.

Mirrors ``videos.signals``: deleting a :class:`~marketing_studio.models.Video`
removes its file under the studio's video root, and deleting a
:class:`~marketing_studio.models.Render` removes its output video.
``post_delete`` rather than a ``ModelAdmin`` override so cleanup also fires on
cascade (deleting a Video CASCADEs to its renders) and on bulk "delete
selected", which never go through the admin's own delete flow. Cleanup never
raises — a filesystem hiccup must not roll back the row deletion.
"""

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from marketing_studio.models import Render, Video

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=Video, dispatch_uid="marketing_video_cleanup")
def _video_deleted(sender, instance, **kwargs):
    try:
        if instance.filename:
            instance.path().unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - never let cleanup break the delete
        logger.exception("cleanup failed for marketing Video #%s", instance.pk)


@receiver(post_delete, sender=Render, dispatch_uid="marketing_render_cleanup")
def _render_deleted(sender, instance, **kwargs):
    try:
        if instance.output_filename:
            instance.output_path().unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        logger.exception("cleanup failed for Render #%s", instance.pk)
