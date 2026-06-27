"""Label Studio HTTP API + Docker helpers, as pure functions.

Ported from the old top-level ``label-studio.py``. Everything here takes
primitives (urls, tokens, ids, paths) and returns data — no ORM, no config
files, no CLI. The Django service layer, admin actions, management commands,
and the webhook view all call into this module.

The hyphenated ``label-studio.py`` could only be imported via an importlib
shim; as ``fleet.services.lsapi`` it is a normal import.
"""

import json
import shutil
import subprocess
import zipfile
from html import escape
from pathlib import Path
from urllib.parse import quote

import requests

from fleet.reconcile.labels import load_class_names
from fleet.reconcile.txt_format import image_filename_from_value, results_for_label_text

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CONTAINER_DATA_ROOT = "/label-studio/data/local"


# ---------------------------
# Process / path utilities
# ---------------------------
def run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        command = " ".join(cmd)
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Command failed: {command}\n{message}") from exc


def require_path(path: Path, *, kind: str):
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def get_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} must be inside data root {root}") from exc


# ---------------------------
# Docker management
# ---------------------------
def docker_available() -> bool:
    return shutil.which("docker") is not None


def container_exists(container_name: str) -> bool:
    if not docker_available():
        return False
    result = run(["docker", "ps", "-a", "--format", "{{.Names}}"])
    return container_name in result.stdout.splitlines()


def container_running(container_name: str) -> bool:
    if not docker_available():
        return False
    result = run(["docker", "ps", "--format", "{{.Names}}"])
    return container_name in result.stdout.splitlines()


def get_container_info(container_name: str) -> dict:
    result = run(["docker", "inspect", container_name])
    data = json.loads(result.stdout)[0]

    ports = data.get("NetworkSettings", {}).get("Ports", {})
    host_port = None
    for bindings in ports.values():
        if bindings:
            host_port = int(bindings[0]["HostPort"])
            break

    volume_name = None
    data_root = None
    for mount in data.get("Mounts", []):
        if mount.get("Destination") == "/label-studio/data":
            volume_name = mount.get("Name") or mount.get("Source")
        if mount.get("Destination") == "/label-studio/data/local":
            data_root = mount.get("Source")

    return {
        "container_name": data.get("Name", f"/{container_name}").lstrip("/"),
        "image_name": data.get("Config", {}).get("Image"),
        "port": host_port,
        "volume_name": volume_name,
        "data_root": data_root,
    }


# ---------------------------
# Label config from classes.txt
# ---------------------------
# Maps a tool keyword (as written in classes.txt) to the Label Studio control
# tag, the control's `name`, and any extra attributes. The `name` doubles as a
# dedupe key, so aliases (bbox/rectangle, keypoint/sam) collapse to one control.
LABEL_TOOLS = {
    "bbox": ("RectangleLabels", "bbox", ""),
    "rectangle": ("RectangleLabels", "bbox", ""),
    "polygon": ("PolygonLabels", "segmentation", ""),
    "keypoint": ("KeyPointLabels", "sam_point", ' smart="true"'),
    "sam": ("KeyPointLabels", "sam_point", ' smart="true"'),
}

# Used when classes.txt has no `# tools:` directive. The export pipeline only
# handles bbox + polygon (there is no mask format), so brush is not offered.
DEFAULT_TOOLS = ["bbox", "polygon", "keypoint"]


def parse_classes_file(classes_file: Path) -> tuple[list[str], list[str]]:
    """Parse a classes.txt into (class names, tool keywords).

    Blank lines are ignored. A comment of the form `# tools: bbox, polygon`
    selects which labeling controls to emit; any other `#` line is a comment.
    Every remaining non-empty line is a class name. When no `tools:` directive
    is present, DEFAULT_TOOLS is used."""
    classes: list[str] = []
    tools: list[str] | None = None
    for raw in classes_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, sep, value = line.lstrip("#").strip().partition(":")
            if sep and key.strip().lower() == "tools":
                tools = [t.strip().lower() for t in value.split(",") if t.strip()]
            continue
        classes.append(line)

    if not classes:
        raise RuntimeError(f"{classes_file} has no class names")
    if tools is None:
        tools = list(DEFAULT_TOOLS)
    if not tools:
        raise RuntimeError(f"{classes_file} declares an empty 'tools:' list")
    return classes, tools


def build_label_config(classes_file: Path) -> str:
    classes, tools = parse_classes_file(classes_file)

    labels_xml = "\n".join(
        f'    <Label value="{escape(label)}"/>' for label in classes
    )

    seen: set[str] = set()
    blocks: list[str] = []
    for tool in tools:
        spec = LABEL_TOOLS.get(tool)
        if spec is None:
            raise RuntimeError(
                f"Unknown label tool {tool!r} in {classes_file}. "
                f"Valid tools: {', '.join(sorted(LABEL_TOOLS))}."
            )
        tag, name, extra = spec
        if name in seen:  # alias or duplicate already emitted
            continue
        seen.add(name)
        blocks.append(
            f'  <{tag} name="{name}" toName="image"{extra}>\n'
            f"{labels_xml}\n"
            f"  </{tag}>"
        )
    controls_xml = "\n".join(blocks)

    return f"""
<View>
  <Image name="image" value="$image"/>
{controls_xml}
</View>
"""


# ---------------------------
# Projects
# ---------------------------
def _auth(api_token: str) -> dict:
    return {"Authorization": f"Token {api_token}"}


def list_projects(*, ls_url: str, api_token: str) -> list[dict]:
    """Return every project in the instance, following pagination."""
    projects = []
    url = f"{ls_url}/api/projects"
    while url:
        response = requests.get(url, headers=_auth(api_token))
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            projects.extend(payload)
            break
        projects.extend(payload.get("results", []))
        url = payload.get("next")
    return projects


def find_projects_by_title(*, ls_url: str, api_token: str, title: str) -> list[dict]:
    projects = list_projects(ls_url=ls_url, api_token=api_token)
    return [project for project in projects if project.get("title") == title]


def create_project(*, ls_url: str, api_token: str, title: str, label_config: str) -> int:
    response = requests.post(
        f"{ls_url}/api/projects",
        headers=_auth(api_token),
        json={"title": title, "label_config": label_config},
    )
    response.raise_for_status()
    return response.json()["id"]


def delete_project(*, ls_url: str, api_token: str, project_id: int):
    response = requests.delete(
        f"{ls_url}/api/projects/{project_id}", headers=_auth(api_token)
    )
    response.raise_for_status()


def get_project(*, ls_url: str, api_token: str, project_id: int) -> dict:
    response = requests.get(
        f"{ls_url}/api/projects/{project_id}", headers=_auth(api_token)
    )
    response.raise_for_status()
    return response.json()


def fetch_project_annotations(*, ls_url: str, api_token: str, project_id: int) -> list[dict]:
    """Return every task in a project with its annotations inlined.

    Uses the export snapshot endpoint, which returns tasks each carrying an
    `annotations` list (with `result`) in a single request — the authoritative
    state the sync service reconciles the target against.
    """
    response = requests.get(
        f"{ls_url}/api/projects/{project_id}/export",
        headers=_auth(api_token),
        params={"exportType": "JSON", "download_all_tasks": "true"},
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else payload.get("tasks", [])


# ---------------------------
# Tasks
# ---------------------------
def get_task(*, ls_url: str, api_token: str, task_id: int) -> dict:
    """Fetch one task — the webhook uses this to learn an image's filename."""
    response = requests.get(
        f"{ls_url}/api/tasks/{task_id}", headers=_auth(api_token), timeout=10
    )
    response.raise_for_status()
    return response.json()


def list_project_tasks(*, ls_url: str, api_token: str, project_id: int) -> list[dict]:
    """Return every task in a project, each with its inline `annotations`.

    Used by the webhook's delete handler: the ANNOTATIONS_DELETED payload only
    names the deleted annotation by id (no task), so the only way to learn which
    image to rewrite is to re-read the project's current tasks and reconcile.
    `fields=all` makes each task carry its surviving annotations' `result`.
    """
    headers = _auth(api_token)
    page_size = 100
    tasks: list[dict] = []
    page = 1
    while True:
        response = requests.get(
            f"{ls_url}/api/tasks/",
            headers=headers,
            params={"project": project_id, "fields": "all", "page": page, "page_size": page_size},
            timeout=10,
        )
        if response.status_code == 404:
            break  # Label Studio 404s past the last page of results.
        response.raise_for_status()
        body = response.json()
        batch = body.get("tasks", []) if isinstance(body, dict) else body
        tasks.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return tasks


def import_tasks(*, ls_url: str, api_token: str, project_id: int, tasks: list[dict]):
    response = requests.post(
        f"{ls_url}/api/projects/{project_id}/import",
        headers=_auth(api_token),
        json=tasks,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------
# Storage + task building
# ---------------------------
def image_source_dir(dataset_dir: Path) -> Path:
    """Return the directory that actually holds a dataset's image files.

    Prefer an ``images/`` subdirectory (the standard YOLO layout, alongside the
    ``labels/`` subdir we already read) when it exists and contains at least one
    image; otherwise fall back to images sitting flat in the dataset directory.
    """
    images_subdir = dataset_dir / "images"
    if images_subdir.is_dir() and any(
        p.suffix.lower() in IMAGE_EXTENSIONS for p in images_subdir.iterdir()
    ):
        return images_subdir
    return dataset_dir


def make_tasks_from_data_root(dataset_dir: Path, data_root: Path) -> list[dict]:
    tasks = []
    image_dir = image_source_dir(dataset_dir)
    dataset_relative_path = get_relative_path(image_dir, data_root)
    for img in sorted(image_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_path = quote(f"{dataset_relative_path}/{img.name}")
        tasks.append({"data": {"image": f"/data/local-files/?d={image_path}"}})
    return tasks


def make_tasks_from_cloud_root(*, dataset_dir: Path, data_root: Path, cloud_root: str) -> list[dict]:
    tasks = []
    image_dir = image_source_dir(dataset_dir)
    dataset_relative_path = get_relative_path(image_dir, data_root)
    cloud_root = cloud_root.rstrip("/")
    for img in sorted(image_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_path = quote(f"{dataset_relative_path}/{img.name}")
        tasks.append({
            "data": {"image": f"{cloud_root}/{image_path}"},
            "meta": {"local_path": f"{dataset_relative_path}/{img.name}"},
        })
    return tasks


def load_predictions_for_image(
    labels_dir: Path, image_filename: str, names: list[str]
) -> list[dict] | None:
    """Build a task's `predictions` list from its on-disk label file, or None.

    Looks for ``<image_filename>.txt`` (the app's own convention, e.g.
    ``img01.jpg.txt``) then ``<stem>.txt`` (standard YOLO, e.g. ``img01.txt``).
    Returns None when no label file exists or it yields no usable regions.
    """
    for candidate in (
        labels_dir / f"{image_filename}.txt",
        labels_dir / f"{Path(image_filename).stem}.txt",
    ):
        if candidate.exists():
            results = results_for_label_text(candidate.read_text(encoding="utf-8"), names)
            return [{"model_version": "imported", "result": results}] if results else None
    return None


def attach_predictions(tasks: list[dict], *, labels_dir: Path, classes_file: Path) -> int:
    """Attach imported predictions to each task that has a matching label file.

    Mutates `tasks` in place; returns the number of tasks that gained predictions.
    """
    names = load_class_names(classes_file)
    count = 0
    for task in tasks:
        filename = image_filename_from_value(task["data"]["image"])
        if not filename:
            continue
        predictions = load_predictions_for_image(labels_dir, filename, names)
        if predictions:
            task["predictions"] = predictions
            count += 1
    return count


def create_local_files_storage(
    *, ls_url: str, api_token: str, project_id: int, project_title: str,
    dataset_dir: Path, data_root: Path,
) -> dict:
    dataset_relative_path = get_relative_path(dataset_dir, data_root)
    container_dataset_path = f"{CONTAINER_DATA_ROOT}/{dataset_relative_path}"
    response = requests.post(
        f"{ls_url}/api/storages/localfiles/",
        headers=_auth(api_token),
        json={
            "project": project_id,
            "title": f"{project_title} Local Files",
            "path": container_dataset_path,
            "use_blob_urls": True,
        },
    )
    response.raise_for_status()
    return response.json()


def create_dataset_project(
    *, dataset_dir: Path, data_root: Path, classes_file: Path, ls_url: str,
    api_token: str, project_title: str, storage_type: str = "local",
    storage_root: str | None = None, labels_dir: Path | None = None,
) -> dict:
    """Create a Label Studio project and import image tasks from a dataset path.

    When ``labels_dir`` is given and exists, each task whose image has a matching
    label file gets those regions attached as predictions (pre-annotations).

    Idempotent: if a project with this title already exists, leave it untouched
    and return it (skipped=True) rather than re-importing tasks.
    """
    get_relative_path(dataset_dir, data_root)
    require_path(dataset_dir, kind="Dataset directory")
    require_path(classes_file, kind="Classes file")

    existing = find_projects_by_title(ls_url=ls_url, api_token=api_token, title=project_title)
    if existing:
        return {
            "ls_url": ls_url,
            "project_id": existing[0]["id"],
            "num_tasks": 0,
            "num_predictions": 0,
            "storage_type": storage_type,
            "storage_root": storage_root,
            "import_result": None,
            "skipped": True,
        }

    label_config = build_label_config(classes_file)
    project_id = create_project(
        ls_url=ls_url, api_token=api_token, title=project_title, label_config=label_config
    )

    if storage_type == "local":
        create_local_files_storage(
            ls_url=ls_url, api_token=api_token, project_id=project_id,
            project_title=project_title, dataset_dir=dataset_dir, data_root=data_root,
        )
        tasks = make_tasks_from_data_root(dataset_dir, data_root)
    elif storage_type == "cloud":
        if not storage_root:
            raise RuntimeError("storage root is required when storage type is cloud")
        tasks = make_tasks_from_cloud_root(
            dataset_dir=dataset_dir, data_root=data_root, cloud_root=storage_root
        )
    else:
        raise RuntimeError(f"Unsupported storage type: {storage_type}")

    if not tasks:
        raise RuntimeError("No images found in dataset directory")

    num_predictions = 0
    if labels_dir and labels_dir.is_dir():
        num_predictions = attach_predictions(tasks, labels_dir=labels_dir, classes_file=classes_file)

    result = import_tasks(ls_url=ls_url, api_token=api_token, project_id=project_id, tasks=tasks)

    return {
        "ls_url": ls_url,
        "project_id": project_id,
        "num_tasks": len(tasks),
        "num_predictions": num_predictions,
        "storage_type": storage_type,
        "storage_root": storage_root,
        "import_result": result,
        "skipped": False,
    }


def export_project_archive(
    *, ls_url: str, api_token: str, project_id: int, target_dir: Path, export_type: str = "JSON"
) -> dict:
    """Export a project's annotations from Label Studio into target_dir.

    JSON comes back inline; other formats (YOLO/COCO/VOC) arrive as a zip we
    unpack. Kept for parity with the old CLI's export-project command.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    export_type = export_type.upper()
    response = requests.get(
        f"{ls_url}/api/projects/{project_id}/export",
        headers=_auth(api_token),
        params={"exportType": export_type, "download_all_tasks": "true"},
    )
    response.raise_for_status()
    stem = f"project_{project_id}_{export_type.lower()}"

    if export_type == "JSON":
        output_path = target_dir / f"{stem}.json"
        output_path.write_bytes(response.content)
        return {"output_path": str(output_path.resolve()), "extracted": False}

    archive_path = target_dir / f"{stem}.zip"
    archive_path.write_bytes(response.content)
    extract_dir = target_dir / stem
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)
    return {"output_path": str(extract_dir.resolve()), "extracted": True}
