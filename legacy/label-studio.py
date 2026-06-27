import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from html import escape
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

if __name__ in sys.modules:
    sys.modules.setdefault("label_studio", sys.modules[__name__])


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_INSTANCE_CONFIG = PACKAGE_ROOT / "configs" / "instance" / "local.yaml"
DEFAULT_PROJECT_CONFIG = PACKAGE_ROOT / "configs" / "project" / "default.yaml"
CONTAINER_DATA_ROOT = "/label-studio/data/local"


# ---------------------------
# Utility
# ---------------------------
def run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        command = " ".join(cmd)
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Command failed: {command}\n{message}") from exc


def load_config(path: Path) -> dict:
    path = resolve_package_path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_package_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def require_path(path: Path, *, kind: str):
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def resolve_relative_to(path: str | Path, root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def get_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} must be inside data root {root}") from exc


# ---------------------------
# Docker management
# ---------------------------
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


def docker_available() -> bool:
    return shutil.which("docker") is not None


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


def ensure_container(
    *,
    data_root: Path,
    container_name: str,
    image_name: str,
    port: int,
    volume_name: str,
):
    if not docker_available():
        raise RuntimeError("Docker is not installed or is not available on PATH")

    if container_exists(container_name):
        if not container_running(container_name):
            run(["docker", "start", container_name])
        return get_container_info(container_name)

    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", f"{port}:8080",
        "-v", f"{volume_name}:/label-studio/data",
        "-v", f"{data_root.resolve()}:/label-studio/data/local",
        "-e", "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true",
        "-e", "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data/local",
    ]
    cmd.append(image_name)

    run(cmd)
    return get_container_info(container_name)


def ensure_label_studio_instance(
    *,
    data_root: Path,
    mode: str = "local",
    container_name: str = "label-studio",
    image_name: str = "heartexlabs/label-studio:latest",
    ls_url: str = "http://localhost:8080",
    port: int = 8080,
    volume_name: str = "label-studio-data",
) -> dict:
    """Start the Docker container if needed and wait for Label Studio."""
    data_root.mkdir(parents=True, exist_ok=True)

    if mode == "remote":
        wait_until_ready(ls_url)
        return {
            "mode": mode,
            "container_name": None,
            "image_name": None,
            "port": None,
            "volume_name": None,
            "ls_url": ls_url,
            "container_exists": False,
            "container_running": False,
            "api_ready": True,
            "data_root": str(data_root.resolve()),
        }

    container_info = ensure_container(
        data_root=data_root,
        container_name=container_name,
        image_name=image_name,
        port=port,
        volume_name=volume_name,
    )
    wait_until_ready(ls_url)

    return {
        **container_info,
        "mode": mode,
        "ls_url": ls_url,
        "container_exists": True,
        "container_running": container_running(container_info["container_name"]),
        "api_ready": True,
        "data_root": container_info["data_root"] or str(data_root.resolve()),
    }


def check_label_studio_instance(
    *,
    mode: str = "local",
    container_name: str = "label-studio",
    ls_url: str = "http://localhost:8080",
) -> dict:
    """Return Docker/API status without creating projects or importing tasks."""
    docker_exists = False
    docker_running = False
    if mode == "local":
        docker_exists = container_exists(container_name)
        docker_running = container_running(container_name) if docker_exists else False

    api_ready = False
    try:
        response = requests.get(ls_url, timeout=2)
        api_ready = response.ok
    except requests.RequestException:
        api_ready = False

    return {
        "mode": mode,
        "container_name": container_name,
        "docker_available": docker_available(),
        "container_exists": docker_exists,
        "container_running": docker_running,
        "ls_url": ls_url,
        "api_ready": api_ready,
    }


# ---------------------------
# Label Studio helpers
# ---------------------------
def wait_until_ready(ls_url: str):
    for _ in range(30):
        try:
            requests.get(ls_url, timeout=2)
            return
        except requests.RequestException:
            time.sleep(1)

    raise RuntimeError(f"Label Studio did not start at {ls_url}")


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
# The smart keypoint stays: it is SAM's interactive trigger, and SAM is run
# with SAM_OUTPUT=polygon so its result lands on the polygon control.
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

    # Plain <Label> tags only. `smart` and `showInline` are NOT Label
    # attributes: `showInline` belongs on the control tag (defaults true), and
    # `smart` on a Label makes it a smart-only variant that breaks Polygon/Brush
    # rendering (so only Brush, the default tool, showed). Smart stays on the
    # KeyPoint/SAM *control* tag via LABEL_TOOLS extras.
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


def create_project(
    *,
    ls_url: str,
    api_token: str,
    title: str,
    label_config: str,
    on_conflict: str = "prompt",
) -> int:
    existing_projects = find_projects_by_title(
        ls_url=ls_url,
        api_token=api_token,
        title=title,
    )
    if existing_projects:
        if on_conflict == "overwrite":
            for project in existing_projects:
                delete_project(
                    ls_url=ls_url,
                    api_token=api_token,
                    project_id=project["id"],
                )
        else:
            print(f"Project title already exists: {title}")
            print(
                "Matching project IDs: "
                + ", ".join(str(project["id"]) for project in existing_projects)
            )
            print("Type the exact project title and press Enter to overwrite it.")
            confirmation = input("Project title: ")
            if confirmation != title:
                raise RuntimeError("Failed to create a project")

            for project in existing_projects:
                delete_project(
                    ls_url=ls_url,
                    api_token=api_token,
                    project_id=project["id"],
                )

    response = requests.post(
        f"{ls_url}/api/projects",
        headers={"Authorization": f"Token {api_token}"},
        json={
            "title": title,
            "label_config": label_config,
        },
    )
    response.raise_for_status()
    return response.json()["id"]


def list_projects(*, ls_url: str, api_token: str) -> list[dict]:
    """Return every project in the instance, following pagination."""
    projects = []
    url = f"{ls_url}/api/projects"

    while url:
        response = requests.get(
            url,
            headers={"Authorization": f"Token {api_token}"},
        )
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, list):
            projects.extend(payload)
            break

        projects.extend(payload.get("results", []))
        url = payload.get("next")

    return projects


def find_projects_by_title(
    *,
    ls_url: str,
    api_token: str,
    title: str,
) -> list[dict]:
    projects = list_projects(ls_url=ls_url, api_token=api_token)
    return [project for project in projects if project.get("title") == title]


def fetch_project_annotations(
    *,
    ls_url: str,
    api_token: str,
    project_id: int,
) -> list[dict]:
    """Return every task in a project with its annotations inlined.

    Uses the export snapshot endpoint, which returns tasks each carrying an
    `annotations` list (with `result`) in a single request — the authoritative
    state `fleet.py sync` reconciles the target against.
    """
    response = requests.get(
        f"{ls_url}/api/projects/{project_id}/export",
        headers={"Authorization": f"Token {api_token}"},
        params={"exportType": "JSON", "download_all_tasks": "true"},
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else payload.get("tasks", [])


def delete_project(
    *,
    ls_url: str,
    api_token: str,
    project_id: int,
):
    response = requests.delete(
        f"{ls_url}/api/projects/{project_id}",
        headers={"Authorization": f"Token {api_token}"},
    )
    response.raise_for_status()


def make_tasks_from_data_root(dataset_dir: Path, data_root: Path) -> list[dict]:
    tasks = []
    dataset_relative_path = get_relative_path(dataset_dir, data_root)

    for img in sorted(dataset_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_path = quote(f"{dataset_relative_path}/{img.name}")
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d={image_path}"
            }
        })

    return tasks


def make_tasks_from_cloud_root(
    *,
    dataset_dir: Path,
    data_root: Path,
    cloud_root: str,
) -> list[dict]:
    tasks = []
    dataset_relative_path = get_relative_path(dataset_dir, data_root)
    cloud_root = cloud_root.rstrip("/")

    for img in sorted(dataset_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_path = quote(f"{dataset_relative_path}/{img.name}")
        tasks.append({
            "data": {
                "image": f"{cloud_root}/{image_path}"
            },
            "meta": {
                "local_path": f"{dataset_relative_path}/{img.name}",
            }
        })

    return tasks


def import_tasks(
    *,
    ls_url: str,
    api_token: str,
    project_id: int,
    tasks: list[dict],
):
    response = requests.post(
        f"{ls_url}/api/projects/{project_id}/import",
        headers={"Authorization": f"Token {api_token}"},
        json=tasks,
    )
    response.raise_for_status()
    return response.json()


def list_project_tasks(
    *,
    ls_url: str,
    api_token: str,
    project_id: int,
) -> list[dict]:
    tasks = []
    url = f"{ls_url}/api/tasks?project={project_id}&page_size=100"

    while url:
        response = requests.get(
            url,
            headers={"Authorization": f"Token {api_token}"},
        )
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, list):
            tasks.extend(payload)
            break

        tasks.extend(payload.get("tasks") or payload.get("results", []))
        url = payload.get("next")

    return tasks


def get_project(
    *,
    ls_url: str,
    api_token: str,
    project_id: int,
) -> dict:
    response = requests.get(
        f"{ls_url}/api/projects/{project_id}",
        headers={"Authorization": f"Token {api_token}"},
    )
    response.raise_for_status()
    return response.json()


def export_project(
    *,
    ls_url: str,
    api_token: str,
    project_id: int,
    target_dir: Path,
    export_type: str = "JSON",
) -> dict:
    """Export a project's annotations from Label Studio into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    export_type = export_type.upper()

    response = requests.get(
        f"{ls_url}/api/projects/{project_id}/export",
        headers={"Authorization": f"Token {api_token}"},
        params={"exportType": export_type, "download_all_tasks": "true"},
    )
    response.raise_for_status()

    stem = f"project_{project_id}_{export_type.lower()}"

    # JSON exports are returned inline; everything else (YOLO, COCO, VOC, ...)
    # comes back as a zip archive that we unpack into target_dir.
    if export_type == "JSON":
        output_path = target_dir / f"{stem}.json"
        output_path.write_bytes(response.content)
        return {
            "ls_url": ls_url,
            "project_id": project_id,
            "export_type": export_type,
            "target_dir": str(target_dir.resolve()),
            "output_path": str(output_path.resolve()),
            "extracted": False,
        }

    archive_path = target_dir / f"{stem}.zip"
    archive_path.write_bytes(response.content)

    extract_dir = target_dir / stem
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    return {
        "ls_url": ls_url,
        "project_id": project_id,
        "export_type": export_type,
        "target_dir": str(target_dir.resolve()),
        "output_path": str(extract_dir.resolve()),
        "extracted": True,
    }


def create_local_files_storage(
    *,
    ls_url: str,
    api_token: str,
    project_id: int,
    project_title: str,
    dataset_dir: Path,
    data_root: Path,
) -> dict:
    dataset_relative_path = get_relative_path(dataset_dir, data_root)
    container_dataset_path = f"{CONTAINER_DATA_ROOT}/{dataset_relative_path}"

    response = requests.post(
        f"{ls_url}/api/storages/localfiles/",
        headers={"Authorization": f"Token {api_token}"},
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
    *,
    dataset_dir: Path,
    data_root: Path,
    classes_file: Path,
    ls_url: str,
    api_token: str,
    project_title: str,
    storage_type: str = "local",
    storage_root: str | None = None,
    on_conflict: str = "prompt",
) -> dict:
    """Create a Label Studio project and import image tasks from a dataset path."""
    get_relative_path(dataset_dir, data_root)
    require_path(dataset_dir, kind="Dataset directory")
    require_path(classes_file, kind="Classes file")

    # Idempotent path for batch/fleet creation: if a project with this title
    # already exists, leave it untouched rather than re-importing tasks.
    if on_conflict == "skip":
        existing = find_projects_by_title(
            ls_url=ls_url,
            api_token=api_token,
            title=project_title,
        )
        if existing:
            return {
                "ls_url": ls_url,
                "project_id": existing[0]["id"],
                "num_tasks": 0,
                "data_root": str(data_root.resolve()),
                "dataset_dir": str(dataset_dir.resolve()),
                "storage_type": storage_type,
                "storage_root": storage_root,
                "import_result": None,
                "skipped": True,
            }

    label_config = build_label_config(classes_file)
    project_id = create_project(
        ls_url=ls_url,
        api_token=api_token,
        title=project_title,
        label_config=label_config,
        on_conflict=on_conflict,
    )

    if storage_type == "local":
        create_local_files_storage(
            ls_url=ls_url,
            api_token=api_token,
            project_id=project_id,
            project_title=project_title,
            dataset_dir=dataset_dir,
            data_root=data_root,
        )
        tasks = make_tasks_from_data_root(dataset_dir, data_root)
    elif storage_type == "cloud":
        if not storage_root:
            raise RuntimeError("storage.root is required when storage.type is cloud")
        tasks = make_tasks_from_cloud_root(
            dataset_dir=dataset_dir,
            data_root=data_root,
            cloud_root=storage_root,
        )
    else:
        raise RuntimeError(f"Unsupported storage.type: {storage_type}")

    if not tasks:
        raise RuntimeError("No images found in dataset directory")

    result = import_tasks(
        ls_url=ls_url,
        api_token=api_token,
        project_id=project_id,
        tasks=tasks,
    )

    return {
        "ls_url": ls_url,
        "project_id": project_id,
        "num_tasks": len(tasks),
        "data_root": str(data_root.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "storage_type": storage_type,
        "storage_root": storage_root,
        "import_result": result,
        "skipped": False,
    }


# ---------------------------
# Main reusable function
# ---------------------------
def setup_label_studio(config_path: str):
    """Ensure the Label Studio Docker container exists and return its settings."""
    cfg = load_instance_config(config_path)

    return ensure_label_studio_instance(
        mode=cfg["mode"],
        container_name=cfg["container_name"],
        image_name=cfg["image_name"],
        ls_url=cfg["ls_url"],
        port=cfg["port"],
        volume_name=cfg["volume_name"],
        data_root=cfg["data_root"],
    )


def setup_label_studio_project(
    project_config_path: str | Path = DEFAULT_PROJECT_CONFIG,
    instance_config_path: str | None = None,
):
    project_cfg = load_project_config(project_config_path)
    instance_cfg = load_instance_config(
        instance_config_path or project_cfg["instance_config"]
    )
    project_cfg["dataset_dir"] = resolve_relative_to(
        project_cfg["dataset_dir"],
        instance_cfg["data_root"],
    )
    project_cfg["classes_file"] = resolve_relative_to(
        project_cfg["classes_file"],
        instance_cfg["data_root"],
    )

    ensure_label_studio_instance(
        mode=instance_cfg["mode"],
        container_name=instance_cfg["container_name"],
        image_name=instance_cfg["image_name"],
        ls_url=instance_cfg["ls_url"],
        port=instance_cfg["port"],
        volume_name=instance_cfg["volume_name"],
        data_root=instance_cfg["data_root"],
    )

    return create_dataset_project(
        dataset_dir=project_cfg["dataset_dir"],
        data_root=instance_cfg["data_root"],
        classes_file=project_cfg["classes_file"],
        ls_url=instance_cfg["ls_url"],
        api_token=project_cfg["api_token"],
        project_title=project_cfg["project_title"],
        storage_type=project_cfg["storage_type"],
        storage_root=project_cfg["storage_root"],
    )


def load_instance_config(config_path: str | Path = DEFAULT_INSTANCE_CONFIG) -> dict:
    config = load_config(Path(config_path))
    ls_cfg = config["label_studio"]
    mode = ls_cfg.get("mode", "local")
    if mode not in {"local", "remote"}:
        raise RuntimeError("label_studio.mode must be local or remote")

    return {
        "mode": mode,
        "container_name": ls_cfg.get("container_name", "label-studio"),
        "image_name": ls_cfg.get("image_name", "heartexlabs/label-studio:latest"),
        "ls_url": ls_cfg.get("url", "http://localhost:8080"),
        "port": int(ls_cfg.get("port", 8080)),
        "volume_name": ls_cfg.get("volume_name", "label-studio-data"),
        "data_root": resolve_package_path(ls_cfg.get("data_root", "label_data")),
    }


def load_project_config(config_path: str | Path = DEFAULT_PROJECT_CONFIG) -> dict:
    config_path_obj = resolve_package_path(config_path)
    config = load_config(config_path_obj)
    project_cfg = config["project"]
    paths_cfg = config["paths"]
    auth_cfg = config["auth"]
    storage_cfg = config.get("storage", {})
    default_instance_config = DEFAULT_INSTANCE_CONFIG
    storage_type = storage_cfg.get("type", "local")
    if storage_type not in {"local", "cloud"}:
        raise RuntimeError("storage.type must be local or cloud")

    return {
        "dataset_dir": Path(paths_cfg["dataset_dir"]),
        "classes_file": Path(paths_cfg["classes_file"]),
        "target_dir": Path(paths_cfg.get("target_dir", "target")),
        "api_token": auth_cfg["token"],
        "project_title": project_cfg.get("title", "Label Studio Project"),
        "storage_type": storage_type,
        "storage_root": storage_cfg.get("root"),
        "instance_config": str(resolve_package_path(config.get("instance_config", default_instance_config))),
    }


def resolve_project_id(
    *,
    ls_url: str,
    api_token: str,
    project_id: int | None,
    project_title: str | None,
) -> int:
    if project_id is not None:
        return project_id
    if not project_title:
        raise RuntimeError("Provide --project-id or --project-title")

    projects = find_projects_by_title(
        ls_url=ls_url,
        api_token=api_token,
        title=project_title,
    )
    if not projects:
        raise RuntimeError(f"Project title not found: {project_title}")
    if len(projects) > 1:
        ids = ", ".join(str(project["id"]) for project in projects)
        raise RuntimeError(f"Project title is ambiguous: {project_title} ({ids})")
    return projects[0]["id"]


# ---------------------------
# Optional CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Backward-compatible config path. Used as instance config except for create-project.",
    )
    parser.add_argument(
        "--instance-config",
        help="Path to instance config YAML.",
    )
    parser.add_argument(
        "--project-config",
        default=DEFAULT_PROJECT_CONFIG,
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        help="Existing Label Studio project ID for export-project.",
    )
    parser.add_argument(
        "--project-title",
        help="Existing Label Studio project title for export-project.",
    )
    parser.add_argument(
        "--export-type",
        default="JSON",
        help="Label Studio export format for export-project (e.g. JSON, YOLO, COCO).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("setup", "start", "check", "create-project", "export-project"),
        default="setup",
        help=(
            "setup/start ensure the Docker instance exists; check only reports "
            "status; create-project uses an existing Label Studio instance"
        ),
    )
    args = parser.parse_args()
    instance_config_path = args.config or args.instance_config or DEFAULT_INSTANCE_CONFIG

    if args.command == "setup":
        result = setup_label_studio(instance_config_path)

        print("Label Studio instance:")
        print(f"Mode: {result['mode']}")
        print(f"URL: {result['ls_url']}")
        if result["mode"] == "local":
            print(f"Container: {result['container_name']}")
            print(f"Image: {result['image_name']}")
            print(f"Port: {result['port']}")
            print(f"Volume: {result['volume_name']}")
        print(f"Data root: {result['data_root']}")
        print(f"API ready: {result['api_ready']}")
        print("Put datasets under data root and use dataset paths relative to it.")
        return

    if args.command == "start":
        cfg = load_instance_config(instance_config_path)
        result = ensure_label_studio_instance(
            mode=cfg["mode"],
            container_name=cfg["container_name"],
            image_name=cfg["image_name"],
            ls_url=cfg["ls_url"],
            port=cfg["port"],
            volume_name=cfg["volume_name"],
            data_root=cfg["data_root"],
        )

        print("Label Studio instance:")
        print(f"Mode: {result['mode']}")
        print(f"URL: {result['ls_url']}")
        if result["mode"] == "local":
            print(f"Container: {result['container_name']}")
            print(f"Image: {result['image_name']}")
            print(f"Port: {result['port']}")
            print(f"Volume: {result['volume_name']}")
        print(f"Data root: {result['data_root']}")
        print(f"API ready: {result['api_ready']}")
        print("Put datasets under data root and use dataset paths relative to it.")
        return

    if args.command == "check":
        cfg = load_instance_config(instance_config_path)
        result = check_label_studio_instance(
            mode=cfg["mode"],
            container_name=cfg["container_name"],
            ls_url=cfg["ls_url"],
        )

        print("Label Studio instance:")
        print(f"Mode: {result['mode']}")
        print(f"URL: {result['ls_url']}")
        if result["mode"] == "local":
            print(f"Container exists: {result['container_exists']}")
            print(f"Container running: {result['container_running']}")
        print(f"API ready: {result['api_ready']}")
        return

    project_config_path = args.config or args.project_config

    if args.command == "export-project":
        project_cfg = load_project_config(project_config_path)
        instance_cfg = load_instance_config(args.instance_config or project_cfg["instance_config"])
        target_dir = resolve_relative_to(
            project_cfg["target_dir"],
            instance_cfg["data_root"],
        )
        project_id = resolve_project_id(
            ls_url=instance_cfg["ls_url"],
            api_token=project_cfg["api_token"],
            project_id=args.project_id,
            project_title=args.project_title or project_cfg["project_title"],
        )
        result = export_project(
            ls_url=instance_cfg["ls_url"],
            api_token=project_cfg["api_token"],
            project_id=project_id,
            target_dir=target_dir,
            export_type=args.export_type,
        )

        print("Project exported:")
        print(f"URL: {result['ls_url']}")
        print(f"Project ID: {result['project_id']}")
        print(f"Export type: {result['export_type']}")
        print(f"Target dir: {result['target_dir']}")
        print(f"Output: {result['output_path']}")
        return

    result = setup_label_studio_project(
        project_config_path=project_config_path,
        instance_config_path=args.instance_config,
    )

    print("Project created:")
    print(f"URL: {result['ls_url']}")
    print(f"Project ID: {result['project_id']}")
    print(f"Tasks imported: {result['num_tasks']}")
    print(f"Data root: {result['data_root']}")
    print(f"Dataset: {result['dataset_dir']}")
    print(f"Storage type: {result['storage_type']}")
    if result["storage_root"]:
        print(f"Storage root: {result['storage_root']}")
    print("All labelable data should live under data root.")
    print("Project dataset paths should be relative to that directory.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
