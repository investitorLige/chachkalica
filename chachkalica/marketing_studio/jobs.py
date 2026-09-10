"""RQ tasks for the studio: downloading a clip, and running a render.

Mirrors :mod:`videos.jobs` — flip the row to running, do the work, record
``ok``/``error``. Failures are recorded AND re-raised so they also land in
django-rq's failed-job registry. Both tasks reuse the ``videos.services``
implementations, differing only in the root they are pointed at.
"""

from django.db import IntegrityError
from django.utils import timezone

from marketing_studio.models import Render, Video
from marketing_studio.services.paths import marketing_videos_root, output_root
from videos.services import downloader, inference

# The queue's DEFAULT_TIMEOUT is only 900s, and a render runs a model over every
# frame and re-encodes; give it the same hour videos.jobs gives its own.
RENDER_JOB_TIMEOUT = 3600
JOB_TIMEOUT = 1800


def _unique_name(base: str, exclude_pk: int | None = None) -> str:
    """A ``Video.name`` derived from ``base`` that no other row uses."""
    base = base.strip() or "video"
    candidate = base
    n = 2
    while Video.objects.filter(name=candidate).exclude(pk=exclude_pk).exists():
        candidate = f"{base} ({n})"
        n += 1
    return candidate


def download_video(video_id: int) -> dict:
    video = Video.objects.get(pk=video_id)
    video.status = Video.DOWNLOADING
    video.last_error = ""
    video.updated_at = timezone.now()
    video.save(update_fields=["status", "last_error", "updated_at"])

    quality = int(video.quality) if video.quality.isdigit() else None
    try:
        root = marketing_videos_root()
        root.mkdir(parents=True, exist_ok=True)
        path = downloader.download(video.source_url, root, quality)
    except Exception as exc:
        video.refresh_from_db()
        video.status = Video.ERROR
        video.last_error = str(exc)
        video.save(update_fields=["status", "last_error", "updated_at"])
        raise

    video.refresh_from_db()
    video.filename = path.name
    video.name = _unique_name(path.stem, exclude_pk=video.pk)
    video.status = Video.READY
    video.last_error = ""
    try:
        video.save(update_fields=["filename", "name", "status", "last_error", "updated_at"])
    except IntegrityError:
        # Extremely unlikely race on the unique name — fall back to a pk suffix.
        video.name = f"{path.stem} #{video.pk}"
        video.save(update_fields=["filename", "name", "status", "last_error", "updated_at"])
    return {"filename": video.filename, "name": video.name}


def run_render(render_id: int) -> dict:
    render = Render.objects.select_related("video").get(pk=render_id)
    render.status = Render.RUNNING
    render.started_at = timezone.now()
    render.last_error = ""
    render.save(update_fields=["status", "started_at", "last_error"])

    try:
        # root= is what keeps this render out of the Videos tab's inferred/
        # folder; see marketing_studio.services.paths.output_root.
        result = inference.run_inference_on_video(render, root=output_root())
    except Exception as exc:
        render.refresh_from_db()
        render.status = Render.ERROR
        render.last_error = str(exc)
        render.finished_at = timezone.now()
        render.save(update_fields=["status", "last_error", "finished_at"])
        raise

    render.refresh_from_db()
    render.status = Render.OK
    render.frames_total = result["frames_total"]
    render.frames_processed = result["frames_processed"]
    render.finished_at = timezone.now()
    render.save(update_fields=["status", "frames_total", "frames_processed", "finished_at"])
    return result
