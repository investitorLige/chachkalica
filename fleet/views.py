"""Webhook receiver — one Label Studio annotation event -> one file op.

Replaces the standalone coco_sync Flask app. Label Studio is configured (by the
setup-dataset step) to POST here on annotation create / update / delete, with
identity in the query string:

    POST /hook?annotator=alice&dataset=dataset1&project_id=3

Tokens/ports now come from the database (the Annotator row) instead of a
mounted YAML. Each event writes, overwrites, or deletes exactly one
``target/<dataset>/<annotator>/<image>.txt``; COCO is assembled in bulk by the
sync service. The pure reconcile modules are reused unchanged.
"""

import json
import logging
from pathlib import Path

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from fleet.models import Annotator
from fleet.reconcile import labels, txt_format, validate, writer
from fleet.services import lsapi
from fleet.services.paths import annotator_base_url, source_root, target_root

log = logging.getLogger("fleet.webhook")

_CREATE_UPDATE = {"ANNOTATION_CREATED", "ANNOTATION_UPDATED"}
_DELETE = {"ANNOTATIONS_DELETED", "ANNOTATION_DELETED"}


def _active_annotator(username: str) -> Annotator:
    return Annotator.objects.get(username=username, status=Annotator.ACTIVE)


def _base_url(annotator: Annotator) -> str:
    # Fleet state records http://localhost:<port>, which is wrong from here:
    # the receiver reaches the instances via the docker host gateway.
    return annotator_base_url(annotator)


def _name_to_index(source: Path, dataset: str) -> tuple[dict[str, int], int]:
    class_names = labels.load_class_names(source / dataset / "classes.txt")
    return labels.name_to_index(class_names), len(class_names)


def _write_from_result(target: Path, dataset, annotator_name, filename, result, name_to_index, num_classes) -> str:
    """Write (or delete, if empty) the txt for one image from an LS result."""
    width, height, objects = txt_format.result_to_image(result, name_to_index)
    objects, warnings = validate.clean_objects(objects, num_classes)
    for warning in warnings:
        log.warning("%s/%s %s: %s", dataset, annotator_name, filename, warning)

    path = writer.label_path(target, dataset, annotator_name, filename)
    if not objects:
        deleted = writer.delete(path)
        return "deleted (no regions)" if deleted else "noop (no regions)"
    writer.write_atomic(path, txt_format.objects_to_text(width, height, objects))
    return f"wrote {len(objects)} object(s)"


def _task_image_value(body: dict) -> str | None:
    task = body.get("task")
    if isinstance(task, dict):
        return (task.get("data") or {}).get("image")
    return None


def _handle_create_update(body, annotator: Annotator, dataset, source: Path, target: Path) -> str:
    annotation = body.get("annotation") or {}
    result = annotation.get("result") or []

    # Geometry is in the payload; we only need the task to learn the image
    # filename. Use the inlined task if present, else fetch it by id.
    image_value = _task_image_value(body)
    if not image_value:
        task_id = annotation.get("task")
        if not isinstance(task_id, int):
            raise ValueError("payload annotation has no integer task id")
        task = lsapi.get_task(ls_url=_base_url(annotator), api_token=annotator.token, task_id=task_id)
        image_value = (task.get("data") or {}).get("image")
    filename = txt_format.image_filename_from_value(image_value)
    if not filename:
        raise ValueError("could not resolve image filename for annotation")

    name_to_index, num_classes = _name_to_index(source, dataset)
    return _write_from_result(target, dataset, annotator.username, filename, result, name_to_index, num_classes)


def _handle_delete(annotator: Annotator, dataset, project_id, source: Path, target: Path) -> str:
    # ANNOTATIONS_DELETED names only the removed annotation ids (no task), and
    # they are already gone, so reconcile the whole (small) project: rewrite each
    # task's txt from its surviving annotation, or delete it.
    if project_id is None:
        raise ValueError("delete event needs project_id in the webhook URL to reconcile")

    name_to_index, num_classes = _name_to_index(source, dataset)
    tasks = lsapi.list_project_tasks(
        ls_url=_base_url(annotator), api_token=annotator.token, project_id=project_id
    )
    written = removed = 0
    for task in tasks:
        filename = txt_format.image_filename_from_value((task.get("data") or {}).get("image"))
        if not filename:
            continue
        survivor = txt_format.latest_annotation(task)
        result = survivor.get("result") or [] if survivor else []
        outcome = _write_from_result(target, dataset, annotator.username, filename, result, name_to_index, num_classes)
        if outcome.startswith("wrote"):
            written += 1
        elif outcome.startswith("deleted"):
            removed += 1
    return f"reconciled {len(tasks)} task(s): {written} written, {removed} removed"


@require_GET
def health(request):
    return JsonResponse({"ok": True})


@csrf_exempt
@require_POST
def hook(request):
    annotator_name = request.GET.get("annotator")
    dataset = request.GET.get("dataset")
    if not annotator_name or not dataset:
        return JsonResponse({"error": "annotator and dataset query params are required"}, status=400)
    project_id = request.GET.get("project_id")
    project_id = int(project_id) if project_id else None

    try:
        body = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        body = {}
    action = body.get("action", "")

    try:
        annotator = _active_annotator(annotator_name)
    except Annotator.DoesNotExist:
        return JsonResponse({"error": f"unknown active annotator {annotator_name!r}"}, status=404)

    source = source_root()
    target = target_root()

    try:
        if action in _CREATE_UPDATE:
            outcome = _handle_create_update(body, annotator, dataset, source, target)
        elif action in _DELETE:
            outcome = _handle_delete(annotator, dataset, project_id, source, target)
        else:
            return JsonResponse({"ignored": action}, status=200)
    except Exception as exc:  # noqa: BLE001 - log and 500 so LS retries
        log.exception("hook failed for %s/%s action=%s", dataset, annotator_name, action)
        return JsonResponse({"error": str(exc)}, status=500)

    log.info("%s %s/%s -> %s", action, dataset, annotator_name, outcome)
    return JsonResponse({"ok": True, "outcome": outcome}, status=200)
