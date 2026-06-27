"""Create Label Studio projects for a dataset and wire up their webhooks.

Mirrors the old ``fleet.py setup-dataset``: for each selected annotator create
a ``"<dataset> — <username>"`` project (idempotent), register the annotation
webhook pointing back at the /hook receiver, and persist a ``Project`` row that
remembers the real ``ls_project_id`` + ``webhook_id``.
"""

from pathlib import Path

import requests

from fleet.models import Annotator, Dataset, FleetSettings, Project, project_title
from fleet.services import lsapi
from fleet.services.paths import annotator_base_url, source_root

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
