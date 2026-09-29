"""RQ tasks: run a studio prompt's previews, and run a dataset build.

Same shape as :mod:`vlm.jobs` — flip the row's status, do the work, record the
outcome; failures are recorded AND re-raised so they also land in django-rq's
failed-job registry. Both go on the "default" queue with an explicit
``job_timeout`` rather than a queue of their own.
"""

import logging

from django.utils import timezone

from genaug.models import AugBuild, AugSessionPrompt, Status
from genaug.services import builder, previews

logger = logging.getLogger(__name__)

#: A handful of images, but the first call may load FireRed from disk.
PREVIEW_JOB_TIMEOUT = 2 * 3600

#: Minutes per image on a 16 GB card, times thousands of images, is days.
BUILD_JOB_TIMEOUT = 7 * 24 * 3600


def run_prompt_previews(prompt_id: int) -> dict:
    prompt = AugSessionPrompt.objects.select_related("session", "session__dataset",
                                                     "session__annotator").get(pk=prompt_id)
    prompt.status, prompt.last_error = Status.RUNNING, ""
    prompt.save(update_fields=["status", "last_error"])
    try:
        result = previews.run_prompt(prompt)
    except Exception as exc:
        prompt.status, prompt.last_error = Status.ERROR, str(exc)
        prompt.save(update_fields=["status", "last_error"])
        raise
    prompt.status = Status.OK if result["done"] else Status.ERROR
    if not result["done"]:
        first = prompt.previews.exclude(error="").values_list("error", flat=True).first()
        prompt.last_error = first or "every preview failed"
    prompt.save(update_fields=["status", "last_error"])
    return result


def run_build(build_id: int) -> dict:
    build = AugBuild.objects.select_related("source_dataset").get(pk=build_id)
    build.status, build.last_error = Status.RUNNING, ""
    build.started_at = timezone.now()
    build.save(update_fields=["status", "last_error", "started_at"])
    try:
        result = builder.run_build(build)
    except Exception as exc:
        build.refresh_from_db()
        build.status, build.last_error = Status.ERROR, str(exc)
        build.finished_at = timezone.now()
        build.save(update_fields=["status", "last_error", "finished_at"])
        raise
    build.refresh_from_db()
    build.status = Status.CANCELLED if result["cancelled"] else Status.OK
    build.waiting_reason = ""
    build.finished_at = timezone.now()
    build.save(update_fields=["status", "waiting_reason", "finished_at"])
    logger.info("genaug build %s done: %s", build.pk, summary(build))
    return result


def summary(build: AugBuild) -> str:
    judged = build.variants_accepted + build.variants_rejected
    rate = f"{100 * build.variants_accepted / judged:.2f}%" if judged else "n/a"
    return (
        f"Source images considered: {build.images_total}\n"
        f"Images augmented:         {build.images_selected}\n"
        f"Variants attempted:       {build.variants_attempted}\n"
        f"Accepted:                 {build.variants_accepted}\n"
        f"Rejected:                 {build.variants_rejected}\n"
        f"Errors:                   {build.variants_errored}\n"
        f"Acceptance rate:          {rate}"
    )
