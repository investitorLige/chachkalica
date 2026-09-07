"""RQ tasks for the VLM section: downloading a video, and running a VLM over
a video or a dataset.

Mirrors :mod:`videos.jobs` — flip the row's status, do the work, record the
outcome. Failures are recorded AND re-raised so they also land in django-rq's
failed-job registry.
"""

from django.db import IntegrityError
from django.utils import timezone

from fleet.services.paths import vlm_videos_root
from videos.services import downloader
from vlm.models import VlmDatasetRun, VlmRun, VlmVideo
from vlm.services import dataset_runner, runner, videos as video_services

# The "default" queue's DEFAULT_TIMEOUT is 900s. Downloads can exceed it, and a
# VLM run over a long video certainly will, so both are enqueued with an
# explicit job_timeout — the same per-call override videos.jobs already uses
# rather than standing up a second queue.
DOWNLOAD_JOB_TIMEOUT = 1800

# A VLM answer takes on the order of a second per frame, so an hour of footage
# sampled at 1 fps is already hours of work. Budget generously; the run stops
# early on its own when nobody is watching.
VLM_JOB_TIMEOUT = 6 * 3600


def _unique_name(base: str, exclude_pk: int | None = None) -> str:
    """A ``VlmVideo.name`` derived from ``base`` that no other row uses."""
    base = (base or "").strip() or "video"
    candidate = base
    n = 2
    while VlmVideo.objects.filter(name=candidate).exclude(pk=exclude_pk).exists():
        candidate = f"{base} ({n})"
        n += 1
    return candidate


def download_vlm_video(video_id: int) -> dict:
    video = VlmVideo.objects.get(pk=video_id)
    video.status = VlmVideo.DOWNLOADING
    video.last_error = ""
    video.save(update_fields=["status", "last_error", "updated_at"])

    quality = int(video.quality) if video.quality.isdigit() else None
    try:
        path = downloader.download(video.source_url, vlm_videos_root(), quality)
    except Exception as exc:
        video.refresh_from_db()
        video.status = VlmVideo.ERROR
        video.last_error = str(exc)
        video.save(update_fields=["status", "last_error", "updated_at"])
        raise

    video.refresh_from_db()
    video.filename = path.name
    video.name = _unique_name(path.stem, exclude_pk=video.pk)
    video.status = VlmVideo.READY
    video.last_error = ""
    try:
        video.save(update_fields=["filename", "name", "status", "last_error", "updated_at"])
    except IntegrityError:
        # Extremely unlikely race on the unique name — fall back to a pk suffix.
        video.name = f"{path.stem} #{video.pk}"
        video.save(update_fields=["filename", "name", "status", "last_error", "updated_at"])

    video_services.apply_probe(video)
    return {"filename": video.filename, "name": video.name}


def run_vlm_inference(run_id: int) -> dict:
    run = VlmRun.objects.select_related("video", "vlm_model").get(pk=run_id)
    run.status = VlmRun.RUNNING
    run.started_at = timezone.now()
    run.last_error = ""
    run.save(update_fields=["status", "started_at", "last_error"])

    try:
        result = runner.run_vlm_on_video(run)
    except Exception as exc:
        # Refresh before writing so a cancel requested mid-flight is not
        # clobbered by this error write — the ordering videos.jobs relies on too.
        run.refresh_from_db()
        run.status = VlmRun.ERROR
        run.last_error = str(exc)
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "last_error", "finished_at"])
        raise

    run.refresh_from_db()
    run.status = VlmRun.CANCELLED if result["cancelled"] else VlmRun.OK
    run.finished_at = timezone.now()
    run.save(update_fields=["status", "finished_at"])
    return result


def run_vlm_on_dataset(run_id: int) -> dict:
    """Run one :class:`~vlm.models.VlmDatasetRun` to completion.

    The same shape as :func:`run_vlm_inference` — flip to running, do the work,
    record the outcome — and it shares that function's ``VLM_JOB_TIMEOUT``: a
    dataset of a few thousand images at a second an image is the same order of
    work as an hour of video at 1 fps. The difference is that nothing pauses
    this one, so the budget has to cover the whole dataset rather than however
    long someone keeps a tab open.
    """
    run = VlmDatasetRun.objects.select_related("dataset", "vlm_model").get(pk=run_id)
    run.status = VlmDatasetRun.RUNNING
    run.started_at = timezone.now()
    run.last_error = ""
    run.save(update_fields=["status", "started_at", "last_error"])

    try:
        result = dataset_runner.run_vlm_on_dataset(run)
    except Exception as exc:
        # Refresh before writing so a cancel requested mid-flight is not
        # clobbered by this error write — the ordering videos.jobs relies on too.
        run.refresh_from_db()
        run.status = VlmDatasetRun.ERROR
        run.last_error = str(exc)
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "last_error", "finished_at"])
        raise

    run.refresh_from_db()
    run.status = VlmDatasetRun.CANCELLED if result["cancelled"] else VlmDatasetRun.OK
    run.finished_at = timezone.now()
    run.save(update_fields=["status", "finished_at"])
    return result
