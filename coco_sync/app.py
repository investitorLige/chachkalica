"""Webhook receiver: turn each Label Studio annotation event into one file op.

Label Studio is configured (by `fleet.py setup-dataset`) to POST here on
annotation create / update / delete, with the annotator, dataset and project
encoded in the query string:

    POST /hook?annotator=alice&dataset=dataset1&project_id=3

Each event writes, overwrites, or deletes exactly one
`target/<dataset>/<annotator>/<image>.txt`. No COCO is touched here — that is
assembled in bulk by `fleet.py sync`.

Create/update: write the submitted annotation's regions (this naturally
captures box edits and removals, since the payload carries the full surviving
set). Delete: re-read the task; if it still has another annotation, rewrite
from that, otherwise remove the file.
"""

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request

from . import labels, ls_client, txt_format, validate, writer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("coco_sync")

SOURCE_ROOT = Path(os.getenv("SOURCE_ROOT", "/data/source"))
TARGET_ROOT = Path(os.getenv("TARGET_ROOT", "/data/target"))

_CREATE_UPDATE = {"ANNOTATION_CREATED", "ANNOTATION_UPDATED"}
_DELETE = {"ANNOTATIONS_DELETED", "ANNOTATION_DELETED"}

app = Flask(__name__)


def _name_to_index(dataset: str) -> tuple[dict[str, int], int]:
    class_names = labels.load_class_names(SOURCE_ROOT / dataset / "classes.txt")
    return labels.name_to_index(class_names), len(class_names)


def _write_from_result(dataset, annotator, filename, result, name_to_index, num_classes) -> str:
    """Write (or delete, if empty) the txt for one image from an LS result."""
    width, height, objects = txt_format.result_to_image(result, name_to_index)
    objects, warnings = validate.clean_objects(objects, num_classes)
    for warning in warnings:
        log.warning("%s/%s %s: %s", dataset, annotator, filename, warning)

    path = writer.label_path(TARGET_ROOT, dataset, annotator, filename)
    if not objects:
        deleted = writer.delete(path)
        return "deleted (no regions)" if deleted else "noop (no regions)"
    writer.write_atomic(path, txt_format.objects_to_text(width, height, objects))
    return f"wrote {len(objects)} object(s)"


def _task_image_value(body: dict) -> str | None:
    """Image value if Label Studio inlined the task in the payload."""
    task = body.get("task")
    if isinstance(task, dict):
        return (task.get("data") or {}).get("image")
    return None


def _handle_create_update(body, annotator, dataset) -> str:
    annotation = body.get("annotation") or {}
    result = annotation.get("result") or []

    # Geometry is in the payload; we only need the task to learn the image
    # filename. Use the inlined task if present, else fetch it by id.
    image_value = _task_image_value(body)
    if not image_value:
        task_id = annotation.get("task")
        if not isinstance(task_id, int):
            raise ValueError("payload annotation has no integer task id")
        image_value = (ls_client.get_task(annotator, task_id).get("data") or {}).get("image")
    filename = txt_format.image_filename_from_value(image_value)
    if not filename:
        raise ValueError("could not resolve image filename for annotation")

    name_to_index, num_classes = _name_to_index(dataset)
    return _write_from_result(dataset, annotator, filename, result, name_to_index, num_classes)


def _handle_delete(annotator, dataset, project_id) -> str:
    # Label Studio's ANNOTATIONS_DELETED payload names the removed annotations
    # only by id (no task), and the annotations are already gone, so we cannot
    # map them back to an image directly. Instead reconcile the whole project
    # (it is one annotator's single dataset project, so this stays small):
    # rewrite each task's txt from its surviving annotation, or delete it.
    if project_id is None:
        raise ValueError("delete event needs project_id in the webhook URL to reconcile")

    name_to_index, num_classes = _name_to_index(dataset)
    tasks = ls_client.list_project_tasks(annotator, project_id)
    written = removed = 0

    for task in tasks:
        filename = txt_format.image_filename_from_value((task.get("data") or {}).get("image"))
        if not filename:
            continue
        survivor = txt_format.latest_annotation(task)
        result = survivor.get("result") or [] if survivor else []
        outcome = _write_from_result(dataset, annotator, filename, result, name_to_index, num_classes)
        if outcome.startswith("wrote"):
            written += 1
        elif outcome.startswith("deleted"):
            removed += 1

    return f"reconciled {len(tasks)} task(s): {written} written, {removed} removed"


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/hook")
def hook():
    annotator = request.args.get("annotator")
    dataset = request.args.get("dataset")
    if not annotator or not dataset:
        return jsonify({"error": "annotator and dataset query params are required"}), 400
    project_id = request.args.get("project_id", type=int)

    body = request.get_json(force=True, silent=True) or {}
    action = body.get("action", "")

    try:
        if action in _CREATE_UPDATE:
            outcome = _handle_create_update(body, annotator, dataset)
        elif action in _DELETE:
            outcome = _handle_delete(annotator, dataset, project_id)
        else:
            return jsonify({"ignored": action}), 200
    except Exception as exc:  # noqa: BLE001 - log and 500 so LS retries
        log.exception("hook failed for %s/%s action=%s", dataset, annotator, action)
        return jsonify({"error": str(exc)}), 500

    log.info("%s %s/%s -> %s", action, dataset, annotator, outcome)
    return jsonify({"ok": True, "outcome": outcome}), 200
