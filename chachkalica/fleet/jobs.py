"""RQ task functions — the long operations run here, off the request path.

Each is a thin wrapper: flip the row to ``running``, call the service function,
then record the outcome (``ok``/``error`` + ``last_error``/``last_run_at``) on
the row. Failures are recorded AND re-raised, so they also land in django-rq's
failed-job registry (visible under /django-rq/ in the admin).

Admin actions and management commands enqueue these via ``django_rq``.
"""

from django.utils import timezone

from fleet.models import Annotator, Dataset, GroundingSamRun, Project
from fleet.services import datasets as datasets_svc
from fleet.services import grounding_sam as grounding_sam_svc
from fleet.services import merge as merge_svc
from fleet.services import overlap as overlap_svc
from fleet.services import provisioning, sync as sync_svc
from fleet.services import split as split_svc


def _mark_running(obj, action: str | None):
    if action is not None:
        obj.last_action = action
    obj.last_status = "running"
    obj.last_error = ""
    obj.last_run_at = timezone.now()
    obj.save()


def _mark_done(obj, status: str, error: str = ""):
    obj.last_status = status
    obj.last_error = error
    obj.last_run_at = timezone.now()
    obj.save()


def provision_annotator(annotator_id: int) -> dict:
    annotator = Annotator.objects.get(pk=annotator_id)
    _mark_running(annotator, "provision")
    try:
        result = provisioning.add_annotator(annotator)
    except Exception as exc:
        annotator.refresh_from_db()
        _mark_done(annotator, "error", str(exc))
        raise
    annotator.refresh_from_db()  # service updated token/status
    _mark_done(annotator, "ok", "")
    return result


def remove_annotator(annotator_id: int, purge: bool = False) -> dict:
    annotator = Annotator.objects.get(pk=annotator_id)
    if purge:
        # The row is deleted by the service, so there is nothing to record on it.
        return provisioning.remove_annotator(annotator, purge=True)

    _mark_running(annotator, "remove")
    try:
        result = provisioning.remove_annotator(annotator, purge=False)
    except Exception as exc:
        annotator.refresh_from_db()
        _mark_done(annotator, "error", str(exc))
        raise
    annotator.refresh_from_db()  # service set status -> retired
    _mark_done(annotator, "ok", "")
    return result


def setup_project(dataset_id: int, annotator_id: int) -> dict:
    """Create (or refresh) one annotator's project for a dataset + its webhook."""
    dataset = Dataset.objects.get(pk=dataset_id)
    annotator = Annotator.objects.get(pk=annotator_id)
    results = datasets_svc.setup_dataset(dataset, [annotator])
    result = results[0] if results else {"status": "no result"}

    # setup_dataset created/updated the Project row; record outcome on it.
    project = Project.objects.filter(annotator=annotator, dataset=dataset).first()
    if project:
        status = "error" if str(result.get("status", "")).startswith("skipped") else "ok"
        _mark_done(project, status, "" if status == "ok" else result.get("status", ""))
    return result


def setup_and_sync_project(dataset_id: int, annotator_id: int) -> dict:
    """Set up one annotator's project for a dataset, then sync it — in order.

    Running both in a single job guarantees the sync never races ahead of the
    Project row that setup creates. Both steps mark Project.last_status.
    """
    setup_result = setup_project(dataset_id, annotator_id)
    project = Project.objects.filter(annotator_id=annotator_id, dataset_id=dataset_id).first()
    if project and project.ls_project_id is not None:
        return sync_project(project.id)
    return setup_result


def generate_grounding_sam_labels(run_id: int) -> dict:
    """Auto-label a dataset's unlabeled images via the Grounding SAM backend.

    The run row is updated (queued -> running -> ok/error) around the call;
    progress counters (images_processed/images_labeled/detections_written)
    are updated by the service itself as it works through each batch.
    """
    run = GroundingSamRun.objects.get(pk=run_id)
    run.status = GroundingSamRun.RUNNING
    run.started_at = timezone.now()
    run.save(update_fields=["status", "started_at"])
    try:
        result = grounding_sam_svc.generate_labels_for_dataset(
            run.dataset, confidence=run.confidence, run=run
        )
    except Exception as exc:
        run.status = GroundingSamRun.ERROR
        run.error = str(exc)
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error", "finished_at"])
        raise
    run.status = GroundingSamRun.OK
    run.finished_at = timezone.now()
    run.save(update_fields=["status", "finished_at"])
    return result


def merge_datasets(dataset_ids: list[int], new_name: str) -> dict:
    """Merge several datasets into a new dataset (images + merged classes.txt)."""
    datasets = list(Dataset.objects.filter(pk__in=dataset_ids))
    # Preserve a deterministic order (by name) regardless of pk ordering.
    datasets.sort(key=lambda d: d.name)
    return merge_svc.merge_datasets(datasets, new_name)


def split_dataset(
    dataset_id: int, left_name: str, right_name: str, left_percent: int, *, shuffle: bool
) -> dict:
    """Split a dataset into two new datasets by percentage, then delete the source."""
    dataset = Dataset.objects.get(pk=dataset_id)
    return split_svc.split_by_percentage(dataset, left_name, right_name, left_percent, shuffle=shuffle)


def prune_overlaps(left_id: int, right_id: int, *, prune_left: bool, prune_right: bool) -> dict:
    """Re-fingerprint a dataset pair and delete the duplicate copies found.

    Re-hashes rather than reusing the report shown on screen, since that report
    may be stale by the time the worker picks this up (the picture on disk is
    the only thing safe to prune from).
    """
    left = Dataset.objects.get(pk=left_id)
    right = Dataset.objects.get(pk=right_id)
    report = overlap_svc.compare_pair(left, right)
    return overlap_svc.prune_overlaps(report, prune_left=prune_left, prune_right=prune_right)


def prune_intra_duplicates(dataset_id: int) -> dict:
    """Re-fingerprint one dataset and delete its duplicate/near-duplicate images.

    Queued rather than run inline from the analytics page for the same reason
    as ``prune_overlaps``: re-hashing every image can run long enough to hit
    the request timeout.
    """
    dataset = Dataset.objects.get(pk=dataset_id)
    return overlap_svc.prune_intra_duplicates(dataset)


def sync_project(project_id: int) -> dict:
    project = Project.objects.get(pk=project_id)
    _mark_running(project, None)
    try:
        result = sync_svc.sync_project(project)
    except Exception as exc:
        project.refresh_from_db()
        _mark_done(project, "error", str(exc))
        raise
    errors = result.get("errors") or []
    _mark_done(project, "ok" if not errors else "warning", "; ".join(errors)[:1000])
    return result
