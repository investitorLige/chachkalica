"""Reconcile the target/ tree from authoritative Label Studio state.

Mirrors the old ``fleet.py sync``: for each project, pull the export snapshot,
rewrite each image's per-image ``.txt`` from its latest annotation (pruning
stale ones), then assemble the ``<username>.coco.json``. Uses the persisted
``ls_project_id`` directly instead of re-looking-up by title.

Annotation tag answers come out in the same pass, as an ``annotation_tags.json``
sidecar in the label directory (see :mod:`fleet.reconcile.tag_values`). Sync,
not the webhook, owns that file for the same reason it owns the COCO document:
it is one file describing the whole project, and a per-event read-modify-write
of it would race itself across webhook threads and processes. Tag answers
therefore land at sync time, and the file is a snapshot of Label Studio's
state exactly like the ``.txt``s beside it.
"""

import json

from fleet.models import FleetSettings, Project
from fleet.reconcile import coco, labels, tag_values, txt_format, validate, writer
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

    tag_specs = [tag.to_spec() for tag in project.dataset.tags.all()]
    tagged_images: dict[str, dict] = {}

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

        if tag_specs:
            frame, regions = tag_values.collect_answers(result, tag_specs)
            # Built from the cleaned objects, so a box's position here is the
            # line it is about to get in the .txt — the join key the sidecar
            # records. A dropped object shifts the rows of every box after it,
            # which is exactly why this cannot be computed from the raw result.
            entry = tag_values.image_entry(objects, frame, regions)
            if entry is not None:
                tagged_images[filename] = entry

        path = writer.label_path(tgt, dataset, username, filename)
        if objects:
            writer.write_atomic(path, txt_format.objects_to_text(width, height, objects))
            legacy_path = writer.legacy_label_path(tgt, dataset, username, filename)
            if legacy_path != path:
                writer.delete(legacy_path)
            present.add(path.name)
        else:
            writer.delete(path)
            writer.delete(writer.legacy_label_path(tgt, dataset, username, filename))

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

    tags_file = tag_values.tags_path(label_dir)
    if tag_specs:
        document = tag_values.build_document(
            dataset=dataset, annotator=username, specs=tag_specs, images=tagged_images,
        )
        writer.write_atomic(tags_file, json.dumps(document, indent=2))
    else:
        # The dataset's tags were deleted (or never existed): drop a stale
        # sidecar rather than leave one describing controls that are gone.
        writer.delete(tags_file)

    return {
        "username": username,
        "dataset": dataset,
        "images": len(doc["images"]),
        "annotations": len(doc["annotations"]),
        "pruned": pruned,
        "errors": errors,
        "coco_path": str(coco_file),
        "tagged_images": len(tagged_images),
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
