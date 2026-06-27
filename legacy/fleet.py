"""Manage a fleet of per-annotator Label Studio containers.

Each annotator gets their own isolated Label Studio instance:
  - unique container name  (label-studio-<username>)
  - unique host port       (auto-incremented from base_port)
  - unique data volume      (label-studio-<username>-data)
  - shared source mount     data/source -> /label-studio/data/local/source
  - shared target mount     data/target -> /label-studio/data/local/target

On creation each container is bootstrapped with an admin user and an API
token. We pass a token we generate via LABEL_STUDIO_USER_TOKEN (Path A); if the
image ignores it we fall back to logging in and fetching the token (Path B).

Fleet state (ports, passwords, tokens) is written to configs/fleet.local.yaml,
which is gitignored because it holds secrets.

Two stores, kept in lockstep by `add`:
  - configs/fleet.local.yaml  the *live* fleet (annotators with a container now)
  - configs/register.yaml     the *durable* roster, keyed by username; remembers
                              every annotator ever added so `remove` can drop the
                              live entry while `add` later restores the same
                              identity and reattaches the kept volume.
Both hold secrets and are gitignored.
"""

import argparse
import importlib.util
import json
import secrets
import sys
import time
from pathlib import Path

import requests
import yaml

from coco_sync import coco, labels, txt_format, validate, writer

PACKAGE_ROOT = Path(__file__).resolve().parent
FLEET_CONFIG = PACKAGE_ROOT / "configs" / "fleet.local.yaml"
# Durable roster keyed by username. Unlike fleet.local.yaml (the *live* fleet,
# which loses an annotator on `remove`), the register remembers every annotator
# ever added — their email/password/token/port/volume — so a later `add`
# restores the same identity and reattaches the kept volume. Holds secrets, so
# it is gitignored alongside fleet.local.yaml.
REGISTER_CONFIG = PACKAGE_ROOT / "configs" / "register.yaml"
CONTAINER_DOC_ROOT = "/label-studio/data/local"

DEFAULTS = {
    "base_port": 8081,
    "image_name": "heartexlabs/label-studio:latest",
    "source_dir": "data/source",
    "target_dir": "data/target",
    # Where each container reaches the coco_sync webhook receiver. The receiver
    # publishes a host port, so containers hit it via the docker host gateway.
    "webhook_url": "http://host.docker.internal:9000",
}


# Reuse the docker/readiness helpers from label-studio.py (hyphenated filename,
# so load it as a module rather than importing by name).
def _load_label_studio():
    spec = importlib.util.spec_from_file_location(
        "label_studio", PACKAGE_ROOT / "label-studio.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("label_studio", module)
    spec.loader.exec_module(module)
    return module


ls = _load_label_studio()


# ---------------------------
# Fleet state
# ---------------------------
def load_fleet() -> dict:
    if FLEET_CONFIG.exists():
        with FLEET_CONFIG.open("r", encoding="utf-8") as f:
            fleet = yaml.safe_load(f) or {}
    else:
        fleet = {}

    for key, value in DEFAULTS.items():
        fleet.setdefault(key, value)
    fleet.setdefault("annotators", {})
    return fleet


def save_fleet(fleet: dict):
    FLEET_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with FLEET_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(fleet, f, sort_keys=False)


def load_register() -> dict:
    """The durable {username: record} roster (see REGISTER_CONFIG)."""
    if REGISTER_CONFIG.exists():
        with REGISTER_CONFIG.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_register(register: dict):
    REGISTER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with REGISTER_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(register, f, sort_keys=False)


def next_port(fleet: dict, register: dict | None = None) -> int:
    # Reserve ports held by retired-but-remembered annotators too, so a new
    # annotator never steals a port that a later `add <retired>` expects back.
    used = {a["port"] for a in fleet["annotators"].values() if a.get("port")}
    if register:
        used |= {r["port"] for r in register.values() if r.get("port")}
    port = fleet["base_port"]
    while port in used:
        port += 1
    return port


def validate_fleet(fleet: dict):
    """Catch conflicting records before we start any container.

    Records created by `add` are unique by construction; this guards against
    duplicates introduced by hand-editing fleet.local.yaml — two annotators
    sharing a port (bind clash), a container name (Docker clash), or a volume
    name (their projects/annotations would bleed together)."""
    seen: dict[str, dict] = {"port": {}, "container_name": {}, "volume_name": {}}
    for username, record in fleet["annotators"].items():
        for field, owners in seen.items():
            value = record.get(field)
            if value is None:
                continue
            if value in owners:
                raise RuntimeError(
                    f"Fleet config conflict: {username!r} and {owners[value]!r} "
                    f"both use {field}={value!r}. Give each annotator a unique "
                    f"{field} in {FLEET_CONFIG}."
                )
            owners[value] = username


# ---------------------------
# Token bootstrapping
# ---------------------------
def token_works(ls_url: str, token: str) -> bool:
    try:
        response = requests.get(
            f"{ls_url}/api/projects?page_size=1",
            headers={"Authorization": f"Token {token}"},
            timeout=5,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def wait_until_http(ls_url: str, *, timeout: int = 180):
    """Wait until the instance answers HTTP. First boot runs DB migrations and
    can take well over a minute, so this is more patient than ls.wait_until_ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            requests.get(ls_url, timeout=5)
            return
        except requests.RequestException:
            time.sleep(3)
    raise RuntimeError(f"Label Studio did not start at {ls_url}")


def login_session(ls_url: str, email: str, password: str) -> requests.Session:
    session = requests.Session()
    login_url = f"{ls_url}/user/login/"
    session.get(login_url, timeout=10)
    csrf = session.cookies.get("csrftoken", "")
    session.post(
        login_url,
        data={
            "email": email,
            "password": password,
            "csrfmiddlewaretoken": csrf,
        },
        headers={"Referer": login_url},
        timeout=10,
    )
    return session


def enable_legacy_tokens(ls_url: str, email: str, password: str):
    """LS >= 1.23 disables `Token <token>` auth by default. This whole codebase
    uses it, so turn it back on for the org. No-op on versions without the
    endpoint (where legacy tokens are already enabled)."""
    try:
        session = login_session(ls_url, email, password)
        csrf = session.cookies.get("csrftoken", "")
        session.post(
            f"{ls_url}/api/jwt/settings",
            json={"api_tokens_enabled": True, "legacy_api_tokens_enabled": True},
            headers={"X-CSRFToken": csrf, "Referer": ls_url},
            timeout=10,
        )
    except requests.RequestException:
        pass


def fetch_token_via_login(ls_url: str, email: str, password: str) -> str:
    """Path B: establish a session, then read the user's legacy token."""
    session = login_session(ls_url, email, password)
    response = session.get(f"{ls_url}/api/current-user/token", timeout=10)
    response.raise_for_status()
    return response.json()["token"]


def resolve_token(ls_url: str, *, candidate_token: str, email: str, password: str) -> str:
    """Return a working token, trying the pre-set one first (Path A → Path B)."""
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if token_works(ls_url, candidate_token):
            return candidate_token
        try:
            fetched = fetch_token_via_login(ls_url, email, password)
            if fetched and token_works(ls_url, fetched):
                return fetched
        except requests.RequestException:
            pass
        time.sleep(3)

    raise RuntimeError(
        f"Could not obtain a working API token from {ls_url}. "
        "The container may still be migrating, or legacy tokens may be disabled."
    )


# ---------------------------
# Container lifecycle
# ---------------------------
def run_container(
    *,
    container_name: str,
    image_name: str,
    port: int,
    volume_name: str,
    source_path: Path,
    target_path: Path,
    email: str,
    password: str,
    token: str,
):
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", f"{port}:8080",
        # Reach the host-published coco_sync webhook receiver from inside the
        # container (host.docker.internal does not resolve on Linux otherwise).
        "--add-host", "host.docker.internal:host-gateway",
        "-v", f"{volume_name}:/label-studio/data",
        "-v", f"{source_path.resolve()}:{CONTAINER_DOC_ROOT}/source",
        "-v", f"{target_path.resolve()}:{CONTAINER_DOC_ROOT}/target",
        "-e", "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true",
        "-e", f"LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={CONTAINER_DOC_ROOT}",
        "-e", f"LABEL_STUDIO_USERNAME={email}",
        "-e", f"LABEL_STUDIO_PASSWORD={password}",
        "-e", f"LABEL_STUDIO_USER_TOKEN={token}",
        image_name,
    ]
    ls.run(cmd)


def add_annotator(username: str, *, email: str | None = None) -> dict:
    if not ls.docker_available():
        raise RuntimeError("Docker is not installed or is not available on PATH")

    fleet = load_fleet()
    # Surface any conflicting hand-edits before we touch Docker.
    validate_fleet(fleet)
    annotators = fleet["annotators"]
    register = load_register()

    source_path = ls.resolve_package_path(fleet["source_dir"])
    target_path = ls.resolve_package_path(fleet["target_dir"])
    source_path.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)

    if username in annotators and annotators[username].get("token"):
        existing = annotators[username]
        if ls.container_exists(existing["container_name"]):
            print(f"Annotator already provisioned: {username}")
            # Backfill the register for annotators provisioned before it existed.
            if username not in register:
                register[username] = existing
                save_register(register)
            return existing

    # Restore identity from the live record if present, else from the durable
    # register (a previously-removed annotator). Only generate fresh secrets for
    # a genuinely new username. Restoring reuses the same volume, so projects and
    # annotations come back — provided `remove` kept the volume (the default).
    prev = annotators.get(username) or register.get(username) or {}
    if username not in annotators and username in register:
        print(
            f"Restoring {username} from register "
            f"(port {prev.get('port')}, volume {prev.get('volume_name')})."
        )
    port = prev.get("port") or next_port(fleet, register)
    record = {
        "email": email or prev.get("email") or f"{username}@labelers.local",
        "password": prev.get("password") or secrets.token_urlsafe(16),
        "port": port,
        "container_name": prev.get("container_name") or f"label-studio-{username}",
        "volume_name": prev.get("volume_name") or f"label-studio-{username}-data",
        "ls_url": f"http://localhost:{port}",
        "token": prev.get("token") or secrets.token_hex(20),
    }
    # Persist the generated secrets BEFORE we start the container, so a failure
    # mid-bootstrap doesn't orphan a container whose credentials we've forgotten.
    # Write both stores: the live fleet and the durable register stay in lockstep,
    # so the token (rotated below) never drifts between them.
    annotators[username] = record
    register[username] = record
    save_fleet(fleet)
    save_register(register)

    if not ls.container_exists(record["container_name"]):
        run_container(
            container_name=record["container_name"],
            image_name=fleet["image_name"],
            port=record["port"],
            volume_name=record["volume_name"],
            source_path=source_path,
            target_path=target_path,
            email=record["email"],
            password=record["password"],
            token=record["token"],
        )
    elif not ls.container_running(record["container_name"]):
        ls.run(["docker", "start", record["container_name"]])

    wait_until_http(record["ls_url"])
    enable_legacy_tokens(record["ls_url"], record["email"], record["password"])
    record["token"] = resolve_token(
        record["ls_url"],
        candidate_token=record["token"],
        email=record["email"],
        password=record["password"],
    )

    annotators[username] = record
    register[username] = record
    save_fleet(fleet)
    save_register(register)
    return record


def provision_all() -> list[dict]:
    """Provision every annotator declared in the fleet config, honoring any
    hand-edited values and filling in the rest. Brings up containers that are
    not running yet; a no-op for ones already provisioned."""
    fleet = load_fleet()
    validate_fleet(fleet)
    usernames = list(fleet["annotators"].keys())
    if not usernames:
        print(
            "No annotators declared in the fleet config. Add one with "
            "`fleet.py add <username>`, or hand-write an entry under "
            f"`annotators:` in {FLEET_CONFIG} and re-run."
        )
        return []
    return [add_annotator(username) for username in usernames]


def select_annotators(
    fleet: dict, *, username: str | None, all_: bool
) -> list[tuple[str, dict]]:
    """Resolve the --annotator / --all selection into (username, record) pairs."""
    annotators = fleet["annotators"]
    if all_:
        if not annotators:
            raise RuntimeError(
                "No annotators provisioned yet. Run `fleet.py add <username>` first."
            )
        return list(annotators.items())

    assert username is not None, "username is required unless all_ is set"
    record = annotators.get(username)
    if not record:
        raise RuntimeError(
            f"Unknown annotator: {username}. Run `fleet.py add {username}` first."
        )
    return [(username, record)]


# ---------------------------
# Webhook registration
# ---------------------------
WEBHOOK_ACTIONS = ["ANNOTATION_CREATED", "ANNOTATION_UPDATED", "ANNOTATIONS_DELETED"]


def register_webhook(
    *,
    record: dict,
    project_id: int,
    dataset: str,
    username: str,
    webhook_base: str,
):
    """Point a project's annotation events at the coco_sync receiver.

    Identity is baked into the URL so the receiver needs no project lookup.
    Idempotent: skips if a webhook with the same URL already exists.
    """
    ls_url = record["ls_url"]
    headers = {"Authorization": f"Token {record['token']}"}
    target = (
        f"{webhook_base.rstrip('/')}/hook"
        f"?annotator={username}&dataset={dataset}&project_id={project_id}"
    )

    try:
        existing = requests.get(
            f"{ls_url}/api/webhooks/",
            headers=headers,
            params={"project": project_id},
            timeout=10,
        )
        existing.raise_for_status()
        hooks = existing.json()
        if isinstance(hooks, dict):
            hooks = hooks.get("results", [])
        if any(hook.get("url") == target for hook in hooks):
            return "exists"

        response = requests.post(
            f"{ls_url}/api/webhooks/",
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
        return "registered"
    except requests.RequestException as exc:
        return f"failed ({exc})"


# ---------------------------
# Sync: reconcile target/ from authoritative Label Studio state
# ---------------------------
def _project_dataset_owner(title: str) -> tuple[str, str] | None:
    """Split a `"<dataset> — <username>"` project title into its parts."""
    if " — " not in title:
        return None
    dataset, _, owner = title.partition(" — ")
    return dataset.strip(), owner.strip()


def sync_annotator_project(
    *,
    record: dict,
    project_id: int,
    dataset: str,
    username: str,
    source_path: Path,
    target_path: Path,
) -> dict:
    """Rebuild one project's per-image txts and COCO file from LS state."""
    classes_file = source_path / dataset / "classes.txt"
    if not classes_file.exists():
        raise RuntimeError(f"missing classes file: {classes_file}")
    class_names = labels.load_class_names(classes_file)
    name_to_index = labels.name_to_index(class_names)

    tasks = ls.fetch_project_annotations(
        ls_url=record["ls_url"], api_token=record["token"], project_id=project_id
    )

    label_dir = writer.labels_dir(target_path, dataset, username)
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

        path = writer.label_path(target_path, dataset, username, filename)
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
    coco_file = writer.coco_path(target_path, dataset, username)
    writer.write_atomic(coco_file, json.dumps(doc, indent=2))

    return {
        "images": len(doc["images"]),
        "annotations": len(doc["annotations"]),
        "pruned": pruned,
        "errors": errors,
        "coco_path": coco_file,
    }


def sync(*, dataset: str | None = None, username: str | None = None, all_: bool = False):
    """Reconcile target/ for the selected annotator(s) from Label Studio."""
    fleet = load_fleet()
    selected = select_annotators(fleet, username=username, all_=all_)
    source_path = ls.resolve_package_path(fleet["source_dir"])
    target_path = ls.resolve_package_path(fleet["target_dir"])

    for name, record in selected:
        if not ls.container_running(record["container_name"]):
            print(f"{name:20} SKIPPED  (container not running)")
            continue

        projects = ls.list_projects(ls_url=record["ls_url"], api_token=record["token"])
        matched = False
        for project in projects:
            parsed = _project_dataset_owner(project.get("title", ""))
            if not parsed:
                continue
            ds, owner = parsed
            if owner != name:
                continue
            if dataset and ds != dataset:
                continue
            matched = True
            try:
                result = sync_annotator_project(
                    record=record,
                    project_id=project["id"],
                    dataset=ds,
                    username=name,
                    source_path=source_path,
                    target_path=target_path,
                )
            except RuntimeError as exc:
                print(f"{name:20} {ds:14} ERROR    {exc}")
                continue
            warn = f"  ⚠ {len(result['errors'])} coco errors" if result["errors"] else ""
            print(
                f"{name:20} {ds:14} images={result['images']:<4} "
                f"anns={result['annotations']:<4} pruned={result['pruned']:<3}"
                f"-> {result['coco_path'].name}{warn}"
            )
        if not matched:
            scope = f" for dataset {dataset}" if dataset else ""
            print(f"{name:20} no matching projects{scope}")


def setup_dataset(dataset: str, *, username: str | None = None, all_: bool = False):
    """Create a `<dataset> — <username>` project in the selected annotator(s)."""
    fleet = load_fleet()
    selected = select_annotators(fleet, username=username, all_=all_)

    source_path = ls.resolve_package_path(fleet["source_dir"])
    data_root = source_path.parent
    dataset_dir = source_path / dataset
    classes_file = dataset_dir / "classes.txt"

    # Validate the dataset once, up front, before touching any container.
    ls.require_path(dataset_dir, kind="Dataset directory")
    ls.require_path(classes_file, kind="Classes file")

    results = []
    for name, record in selected:
        title = f"{dataset} — {name}"
        if not ls.container_running(record["container_name"]):
            print(f"{name:20} SKIPPED  (container not running: {record['container_name']})")
            results.append((name, None))
            continue

        result = ls.create_dataset_project(
            dataset_dir=dataset_dir,
            data_root=data_root,
            classes_file=classes_file,
            ls_url=record["ls_url"],
            api_token=record["token"],
            project_title=title,
            storage_type="local",
            on_conflict="skip",
        )
        results.append((name, result))

        hook_status = register_webhook(
            record=record,
            project_id=result["project_id"],
            dataset=dataset,
            username=name,
            webhook_base=fleet["webhook_url"],
        )
        if result.get("skipped"):
            print(f"{name:20} exists   project={result['project_id']:<4} webhook={hook_status} ({title})")
        else:
            print(
                f"{name:20} created  project={result['project_id']:<4} "
                f"tasks={result['num_tasks']} webhook={hook_status} ({title})"
            )
    return results


def remove_annotator(username: str, *, purge: bool = False):
    """Stop and remove an annotator's container.

    Default: keep the data volume AND the register entry, so a later
    `add <username>` restores the same identity and reattaches the volume
    (projects + annotations intact). Only the live fleet entry is dropped.

    `purge=True`: also delete the data volume and forget the annotator from the
    register — a full, irreversible teardown.
    """
    fleet = load_fleet()
    register = load_register()
    record = fleet["annotators"].get(username) or register.get(username)
    if not record:
        raise RuntimeError(f"Unknown annotator: {username}")

    container_name = record["container_name"]
    if ls.container_exists(container_name):
        if ls.container_running(container_name):
            ls.run(["docker", "stop", container_name])
        ls.run(["docker", "rm", container_name])

    if purge:
        try:
            ls.run(["docker", "volume", "rm", record["volume_name"]])
        except RuntimeError:
            pass
        register.pop(username, None)

    # The live fleet always loses the entry; the register keeps it unless purged.
    fleet["annotators"].pop(username, None)
    save_fleet(fleet)
    save_register(register)


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Manage per-annotator Label Studio containers")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser(
        "add",
        help="Create/start an annotator container and bootstrap its token. "
        "With no username, provisions every annotator declared in the config.",
    )
    add.add_argument(
        "username",
        nargs="?",
        help="Annotator to provision. Omit to provision all declared in the config.",
    )
    add.add_argument("--email", help="Login email (default <username>@labelers.local)")

    sub.add_parser("list", help="List provisioned annotators")

    setup = sub.add_parser(
        "setup-dataset",
        help="Create a dataset project in one or all annotator containers",
    )
    setup.add_argument("dataset", help="Dataset name under data/source/<dataset>")
    target = setup.add_mutually_exclusive_group(required=True)
    target.add_argument("--annotator", help="Target a single annotator by username")
    target.add_argument("--all", action="store_true", help="Target every provisioned annotator")

    sync_cmd = sub.add_parser(
        "sync",
        help="Reconcile target/ from Label Studio: rebuild per-image txts and "
        "the <username>.coco.json for matching projects. Defaults to all annotators.",
    )
    sync_cmd.add_argument("dataset", nargs="?", help="Only sync this dataset (default: all)")
    sync_target = sync_cmd.add_mutually_exclusive_group()
    sync_target.add_argument("--annotator", help="Target a single annotator by username")
    sync_target.add_argument("--all", action="store_true", help="Target every annotator (default)")

    rm = sub.add_parser(
        "remove",
        help="Stop and remove an annotator container. Keeps its data volume and "
        "register entry by default so `add` can restore it; use --purge to forget it.",
    )
    rm.add_argument("username")
    rm.add_argument(
        "--purge",
        action="store_true",
        help="Also delete the data volume and drop the annotator from the register",
    )

    args = parser.parse_args()

    if args.command == "add":
        if args.username is None:
            records = provision_all()
            for record in records:
                print(f"{record['ls_url']:25} [{record['container_name']}]")
            if records:
                print(f"State saved to {FLEET_CONFIG}")
            return
        record = add_annotator(args.username, email=args.email)
        print(f"Annotator: {args.username}")
        print(f"URL: {record['ls_url']}")
        print(f"Email: {record['email']}")
        print(f"Container: {record['container_name']}")
        print(f"Port: {record['port']}")
        print(f"Token: {record['token']}")
        print(f"State saved to {FLEET_CONFIG}")
        return

    if args.command == "list":
        fleet = load_fleet()
        register = load_register()
        live = fleet["annotators"]
        if not live and not register:
            print("No annotators provisioned yet.")
            return
        for username, record in live.items():
            running = ls.container_running(record["container_name"])
            status = "running" if running else "stopped"
            print(f"{username:20} {record['ls_url']:25} {record['container_name']:28} [{status}]")
        # Annotators remembered by the register but not currently in the fleet:
        # removed without --purge, restorable with `add <username>`.
        for username, record in register.items():
            if username in live:
                continue
            print(
                f"{username:20} {record.get('ls_url',''):25} "
                f"{record.get('container_name',''):28} [retired]"
            )
        return

    if args.command == "setup-dataset":
        setup_dataset(args.dataset, username=args.annotator, all_=args.all)
        return

    if args.command == "sync":
        # Default to the whole fleet unless a single annotator was named.
        all_ = args.all or not args.annotator
        sync(dataset=args.dataset, username=args.annotator, all_=all_)
        return

    if args.command == "remove":
        remove_annotator(args.username, purge=args.purge)
        if args.purge:
            print(f"Purged annotator: {args.username} (volume deleted, dropped from register)")
        else:
            print(
                f"Removed annotator: {args.username} "
                "(volume + register entry kept; `add` will restore it)"
            )
        return


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
