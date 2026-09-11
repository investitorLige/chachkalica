"""Create Label Studio projects for a dataset and wire up their webhooks.

Mirrors the old ``fleet.py setup-dataset``: for each selected annotator create
a ``"<dataset> — <username>"`` project (idempotent), register the annotation
webhook pointing back at the /hook receiver, and persist a ``Project`` row that
remembers the real ``ls_project_id`` + ``webhook_id``.
"""

import shutil
from pathlib import Path

import requests

from fleet.models import Annotator, Dataset, FleetSettings, Project, project_title
from fleet.reconcile import tag_values, writer
from fleet.services import lsapi
from fleet.services.paths import annotator_base_url, source_root, target_root

WEBHOOK_ACTIONS = ["ANNOTATION_CREATED", "ANNOTATION_UPDATED", "ANNOTATIONS_DELETED"]

# A dataset's optional pre-existing labels live under source/<name>/labels/.
LABELS_SUBDIR = "labels"


def labels_source_dir(dataset: Dataset, fs: FleetSettings | None = None) -> Path:
    """Path to a dataset's optional source labels folder (may not exist)."""
    fs = fs or FleetSettings.load()
    return source_root(fs) / dataset.name / LABELS_SUBDIR


def detect_labels(dataset: Dataset, *, persist: bool = True) -> bool:
    """Return whether the dataset has a non-empty source labels/ folder.

    When ``persist`` is set and the result differs from the stored flag, update
    ``Dataset.has_labels`` so the admin and project-creation path can rely on it.
    """
    labels_dir = labels_source_dir(dataset)
    has = labels_dir.is_dir() and any(
        path.suffix.lower() == ".txt" for path in labels_dir.iterdir()
    )
    if persist and dataset.has_labels != has:
        dataset.has_labels = has
        dataset.save(update_fields=["has_labels"])
    return has


def promote_annotator_labels(dataset: Dataset, annotator: Annotator, fs: FleetSettings | None = None) -> dict:
    """Move an annotator's synced label ``.txt`` files into the source labels/ folder.

    Annotator output lives at ``target/<dataset>/<username>/*.txt`` (written by
    sync); the dataset's source-of-truth labels live at
    ``source/<dataset>/labels/*.txt`` alongside the images and classes.txt in the
    dataset root. This makes the annotator's annotations the source labels by
    moving every ``.txt`` across, overwriting any existing source label of the
    same name. Images and classes.txt in the dataset root are untouched.

    The ``annotation_tags.json`` sidecar travels with the labels it describes:
    its per-box rows are line numbers in these very ``.txt`` files, so leaving
    it behind in target/ would leave the promoted labels with tag answers that
    nothing can find, and copying it late (after the next sync rewrote the
    txts) would leave rows pointing at different boxes.
    """
    fs = fs or FleetSettings.load()
    src_dir = target_root(fs) / dataset.name / annotator.username
    if not src_dir.is_dir():
        raise FileNotFoundError(
            f"No annotator output for {annotator.username} at {src_dir} — "
            "sync the annotator's project first."
        )

    txts = sorted(src_dir.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(
            f"{annotator.username} has no label .txt files at {src_dir} — nothing to promote."
        )

    dest_dir = labels_source_dir(dataset, fs)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for txt in txts:
        # Normalize legacy ``image.jpg.txt`` exports to ``image.txt`` so they
        # overwrite the source label for that image instead of creating a duplicate.
        dest = dest_dir / writer.label_filename(txt.stem)
        shutil.move(str(txt), str(dest))
        legacy_dest = dest_dir / txt.name
        if legacy_dest != dest:
            legacy_dest.unlink(missing_ok=True)

    # Copied, not moved: the annotator's own sidecar stays valid for their
    # target/ labels (which the next sync will rewrite anyway), and a promote
    # must not be the thing that deletes it.
    tags_src = tag_values.tags_path(src_dir)
    tags_dest = tag_values.tags_path(dest_dir)
    tags_promoted = tags_src.exists()
    if tags_promoted:
        shutil.copy2(str(tags_src), str(tags_dest))
    else:
        # A promote from an annotator with no tag answers must not leave the
        # previous annotator's sidecar sitting next to the new labels, where
        # its rows now point at somebody else's boxes.
        tags_dest.unlink(missing_ok=True)

    # Source labels now exist — refresh the flag so the admin/training path sees them.
    detect_labels(dataset)
    return {
        "dataset": dataset.name,
        "annotator": annotator.username,
        "moved": len(txts),
        "dest": str(dest_dir),
        "tags_promoted": tags_promoted,
    }


def dataset_tag_specs(dataset: Dataset) -> list[dict]:
    """The dataset's annotation tags in the ORM-free form the renderer takes."""
    return [tag.to_spec() for tag in dataset.tags.all()]


def dataset_label_config(dataset: Dataset, fs: FleetSettings | None = None) -> str:
    """Build the labeling interface a project for this dataset would get.

    Classes and drawing tools come from the on-disk ``classes.txt``; the extra
    controls come from the dataset's tags. Shared by project creation, the
    "push tags to existing projects" path, and the tag editor's preview, so
    what the editor shows is literally what gets sent.
    """
    fs = fs or FleetSettings.load()
    classes_file = source_root(fs) / dataset.name / "classes.txt"
    lsapi.require_path(classes_file, kind="Classes file")
    return lsapi.build_label_config(classes_file, dataset_tag_specs(dataset))


def apply_tags_to_projects(dataset: Dataset) -> list[dict]:
    """Push the dataset's current label config onto every project it already has.

    Creating a project bakes the tags in, so a tag added afterwards would only
    reach annotators set up from then on. This rewrites the interface of the
    existing projects instead, which keeps every annotation already made —
    the alternative, recreating the project, would throw that work away.

    Best-effort per annotator: a stopped container or an LS rejection (removing
    a control that regions already use) is recorded against that one project and
    never blocks the others.
    """
    label_config = dataset_label_config(dataset)

    results: list[dict] = []
    for project in dataset.projects.select_related("annotator"):
        annotator = project.annotator
        entry = {"username": annotator.username, "project_id": project.ls_project_id}
        if project.ls_project_id is None:
            entry["status"] = "skipped (no LS project)"
        elif not lsapi.container_running(annotator.container_name):
            entry["status"] = "skipped (container not running)"
        else:
            try:
                lsapi.update_project_label_config(
                    ls_url=annotator_base_url(annotator),
                    api_token=annotator.token,
                    project_id=project.ls_project_id,
                    label_config=label_config,
                )
                entry["status"] = "updated"
            except requests.RequestException as exc:
                entry["status"] = f"failed ({exc})"
        results.append(entry)
    return results


def _webhook_target(webhook_base: str, *, dataset: str, username: str, project_id: int) -> str:
    return (
        f"{webhook_base.rstrip('/')}/hook"
        f"?annotator={username}&dataset={dataset}&project_id={project_id}"
    )


def register_webhook(*, annotator: Annotator, project_id: int, dataset: str, webhook_base: str):
    """Point a project's annotation events at the /hook receiver.

    Identity is baked into the URL so the receiver needs no project lookup.
    Idempotent: reuses an existing webhook with the same URL. Returns
    (status, webhook_id) where status is registered/exists/failed.
    """
    headers = {"Authorization": f"Token {annotator.token}"}
    api_base = annotator_base_url(annotator)
    target = _webhook_target(
        webhook_base, dataset=dataset, username=annotator.username, project_id=project_id
    )
    try:
        existing = requests.get(
            f"{api_base}/api/webhooks/",
            headers=headers,
            params={"project": project_id},
            timeout=10,
        )
        existing.raise_for_status()
        hooks = existing.json()
        if isinstance(hooks, dict):
            hooks = hooks.get("results", [])
        for hook in hooks:
            if hook.get("url") == target:
                return "exists", hook.get("id")

        response = requests.post(
            f"{api_base}/api/webhooks/",
            headers=headers,
            json={
                "project": project_id,
                "url": target,
                "send_payload": True,
                "send_for_all_actions": False,
                "actions": WEBHOOK_ACTIONS,
                "is_active": True,
            },
            timeout=10,
        )
        response.raise_for_status()
        return "registered", response.json().get("id")
    except requests.RequestException as exc:
        return f"failed ({exc})", None


def setup_dataset(dataset: Dataset, annotators: list[Annotator]) -> list[dict]:
    """Create the project + webhook for each annotator; persist Project rows."""
    fs = FleetSettings.load()
    src = source_root(fs)
    data_root = src.parent
    dataset_dir = src / dataset.name
    classes_file = dataset_dir / "classes.txt"

    lsapi.require_path(dataset_dir, kind="Dataset directory")
    lsapi.require_path(classes_file, kind="Classes file")

    # Refresh the flag from disk so labels added after the row are still picked up.
    labels_dir = labels_source_dir(dataset, fs) if detect_labels(dataset) else None
    tags = dataset_tag_specs(dataset)

    results: list[dict] = []
    for annotator in annotators:
        title = project_title(dataset.name, annotator.username)
        if not lsapi.container_running(annotator.container_name):
            results.append({"username": annotator.username, "status": "skipped (container not running)"})
            continue

        create = lsapi.create_dataset_project(
            dataset_dir=dataset_dir,
            data_root=data_root,
            classes_file=classes_file,
            ls_url=annotator_base_url(annotator),
            api_token=annotator.token,
            project_title=title,
            storage_type=dataset.storage_type,
            storage_root=dataset.storage_root or None,
            labels_dir=labels_dir,
            tags=tags,
        )
        project_id = create["project_id"]

        hook_status, webhook_id = register_webhook(
            annotator=annotator,
            project_id=project_id,
            dataset=dataset.name,
            webhook_base=fs.webhook_url,
        )

        Project.objects.update_or_create(
            annotator=annotator,
            dataset=dataset,
            defaults={
                "ls_project_id": project_id,
                "webhook_id": webhook_id,
                "title": title,
            },
        )
        results.append({
            "username": annotator.username,
            "project_id": project_id,
            "tasks": create["num_tasks"],
            "predictions": create["num_predictions"],
            "skipped_create": create["skipped"],
            "webhook": hook_status,
            "status": "exists" if create["skipped"] else "created",
        })
    return results


def teardown_dataset_projects(dataset: Dataset) -> list[dict]:
    """Delete every annotator's Label Studio project for this dataset.

    Deleting a project in Label Studio also removes its tasks and webhooks, so
    no separate webhook cleanup is needed. Best-effort: a stopped container or
    an LS error for one annotator never blocks the others (or the dataset
    deletion that triggered this). The caller is responsible for removing the
    ``Project`` rows (the dataset cascade handles that).
    """
    results: list[dict] = []
    for project in dataset.projects.select_related("annotator"):
        annotator = project.annotator
        entry = {"username": annotator.username, "project_id": project.ls_project_id}
        if project.ls_project_id is None:
            entry["status"] = "skipped (no LS project)"
        elif not lsapi.container_running(annotator.container_name):
            entry["status"] = "skipped (container not running)"
        else:
            try:
                lsapi.delete_project(
                    ls_url=annotator_base_url(annotator),
                    api_token=annotator.token,
                    project_id=project.ls_project_id,
                )
                entry["status"] = "deleted"
            except requests.RequestException as exc:
                entry["status"] = f"failed ({exc})"
        results.append(entry)
    return results


def label_shapes(labels_dir: Path, image_filename: str, class_names) -> list[dict]:
    """Parse one image's label file into normalized, named draw shapes.

    Each shape is ``{class_id, class_name, kind, bbox, polygon}`` with a
    normalized centre-xywh ``bbox`` and, for polygons, a flat normalized
    ``polygon`` list — the format both label overlays consume: the dataset
    preview viewer's canvas, and the VLM dataset report's burned-in boxes.
    An unknown class id keeps its number as its name rather than being dropped,
    since the point here is to *show* the annotation, warts and all.

    Returns an empty list when the image has no label file.
    """
    from fleet.reconcile import txt_format

    label_file = find_label_file(Path(labels_dir), image_filename)
    if label_file is None:
        return []
    _w, _h, objects = txt_format.parse_label_text(label_file.read_text(encoding="utf-8"))

    shapes = []
    for obj in objects:
        class_id = obj["class_id"]
        name = (class_names[class_id] if 0 <= class_id < len(class_names)
                else str(class_id))
        cx, cy, w, h = obj["bbox"]
        polygon = obj.get("polygon")
        shapes.append({
            "class_id": class_id,
            "class_name": name,
            "kind": "polygon" if polygon else "box",
            "bbox": {"cx": cx, "cy": cy, "w": w, "h": h},
            "polygon": polygon or [],
        })
    return shapes


def find_label_file(labels_dir: Path, image_filename: str) -> Path | None:
    """Locate one image's YOLO label file inside ``labels_dir``.

    Two names are tried, because both are in use: ``<image_filename>.txt``
    (``img.jpg.txt`` — what this app's own sync writes) and ``<stem>.txt``
    (``img.txt`` — standard YOLO, and what :func:`promote_annotator_labels`
    normalizes to). Returns None when neither exists, which means *unlabeled*
    and is not the same as an existing but empty file.
    """
    for candidate in (
        labels_dir / f"{image_filename}.txt",
        labels_dir / f"{Path(image_filename).stem}.txt",
    ):
        if candidate.exists():
            return candidate
    return None
