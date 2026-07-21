"""RQ task for downloading a video off the request path.

Mirrors :mod:`fleet.jobs`: flip the row to ``downloading``, run the download,
then record ``ready``/``error`` on the row. Failures are recorded AND re-raised
so they also land in django-rq's failed-job registry.
"""

from django.db import IntegrityError
from django.utils import timezone

from fleet.services.paths import videos_root
from videos.models import FrameExtractionJob, Video
from videos.services import downloader, frame_extraction

# The queue's DEFAULT_TIMEOUT is only 900s; a long/densely-sampled video can run
# past that, so extract_frames is always enqueued with job_timeout=JOB_TIMEOUT.
JOB_TIMEOUT = 1800


def _unique_name(base: str, exclude_pk: int | None = None) -> str:
    """A ``Video.name`` derived from ``base`` that no other row uses."""
    base = base.strip() or "video"
    candidate = base
    n = 2
    while (
        Video.objects.filter(name=candidate).exclude(pk=exclude_pk).exists()
    ):
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
        path = downloader.download(video.source_url, videos_root(), quality)
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


def extract_frames(job_id: int) -> dict:
    job = FrameExtractionJob.objects.select_related("video").get(pk=job_id)
    job.status = FrameExtractionJob.RUNNING
    job.started_at = timezone.now()
    job.last_error = ""
    job.save(update_fields=["status", "started_at", "last_error"])

    classes = [line.strip() for line in job.classes_text.splitlines() if line.strip()]
    try:
        result = frame_extraction.extract_to_dataset(
            job.video, job.dataset_name, job.every_n_frames, job.max_frames_per_video,
            job.image_ext, classes,
            only_with_people=job.only_with_people,
            similarity_threshold=job.similarity_threshold,
        )
    except Exception as exc:
        job.refresh_from_db()
        job.status = FrameExtractionJob.ERROR
        job.last_error = str(exc)
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "last_error", "finished_at"])
        raise

    job.refresh_from_db()
    job.status = FrameExtractionJob.OK
    job.frames_extracted = result["frames"]
    job.frames_dropped_similar = result.get("dropped_similar")
    job.frames_dropped_no_person = result.get("dropped_no_person")
    job.dataset_id = result["dataset_id"]
    job.finished_at = timezone.now()
    job.save(update_fields=[
        "status", "frames_extracted", "frames_dropped_similar", "frames_dropped_no_person",
        "dataset", "finished_at",
    ])
    return result
