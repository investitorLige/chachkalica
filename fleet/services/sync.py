"""Reconcile the target/ tree from authoritative Label Studio state.

Mirrors the old ``fleet.py sync``: for each project, pull the export snapshot,
rewrite each image's per-image ``.txt`` from its latest annotation (pruning
stale ones), then assemble the ``<username>.coco.json``. Uses the persisted
``ls_project_id`` directly instead of re-looking-up by title.
"""

import json

from fleet.models import FleetSettings, Project
from fleet.reconcile import coco, labels, txt_format, validate, writer
from fleet.services import lsapi
from fleet.services.paths import annotator_base_url, source_root, target_root


def sync_project(project: Project) -> dict:
    """Rebuild one project's per-image txts and COCO file from LS state."""
    fs = FleetSettings.load()
    src = source_root(fs)
    tgt = target_root(fs)

    annotator = project.annotator
    dataset = project.dataset.name
    username = annotator.username

    if project.ls_project_id is None:
        raise RuntimeError(f"{project} has no Label Studio project id — run setup-dataset first")

    classes_file = src / dataset / "classes.txt"
    if not classes_file.exists():
        raise RuntimeError(f"missing classes file: {classes_file}")
    class_names = labels.load_class_names(classes_file)
    name_to_index = labels.name_to_index(class_names)

    tasks = lsapi.fetch_project_annotations(
        ls_url=annotator_base_url(annotator), api_token=annotator.token, project_id=project.ls_project_id
    )

    label_dir = writer.labels_dir(tgt, dataset, username)
    present: set[str] = set()
    for task in tasks:
        image_value = (task.get("data") or {}).get("image")
        filename = txt_format.image_filename_from_value(image_value)
        if not filename:
            continue
        annotation = txt_format.latest_annotation(task)
        result = annotation.get("result") or [] if annotation else []
        width, height, objects = txt_format.result_to_image(result, name_to_index)
        objects, _ = validate.clean_objects(objects, len(class_names))

        path = writer.label_path(tgt, dataset, username, filename)
        if objects:
            writer.write_atomic(path, txt_format.objects_to_text(width, height, objects))
            present.add(path.name)
        else:
            writer.delete(path)

    # Prune txts whose task/annotation no longer exists in Label Studio.
    pruned = 0
    if label_dir.exists():
        for stale in label_dir.glob("*.txt"):
            if stale.name not in present:
                stale.unlink()
                pruned += 1

    doc = coco.build_coco(label_dir, class_names)
    errors = validate.validate_coco(doc)
    coco_file = writer.coco_path(tgt, dataset, username)
    writer.write_atomic(coco_file, json.dumps(doc, indent=2))

    return {
        "username": username,
        "dataset": dataset,
        "images": len(doc["images"]),
        "annotations": len(doc["annotations"]),
        "pruned": pruned,
        "errors": errors,
        "coco_path": str(coco_file),
    }


def sync_projects(projects: list[Project]) -> list[dict]:
    """Sync several projects, skipping any whose container is not running."""
    results: list[dict] = []
    for project in projects:
        if not lsapi.container_running(project.annotator.container_name):
            results.append({
                "username": project.annotator.username,
                "dataset": project.dataset.name,
                "status": "skipped (container not running)",
            })
            continue
        results.append(sync_project(project))
    return results
