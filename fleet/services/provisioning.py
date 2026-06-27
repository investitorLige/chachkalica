"""Provision and tear down per-annotator Label Studio containers.

The orchestration boundary the queue jobs (and management commands) call. The
``Annotator`` row already exists and holds the reserved port + generated
secrets *before* we touch Docker — the model's save() guarantees that — so a
crash mid-bootstrap never orphans a container whose credentials we've lost.
"""

from pathlib import Path

from django.conf import settings

from fleet.models import Annotator
from fleet.services import lsapi, tokens
from fleet.services.paths import source_root, target_root
from fleet.models import FleetSettings

CONTAINER_DOC_ROOT = "/label-studio/data/local"


def reachable_url(annotator: Annotator) -> str:
    """URL the server (worker/webhook) uses to reach the instance.

    The stored ls_url (http://localhost:<port>) is for the operator's browser on
    the host. Server-side we go through LS_HOST — `localhost` when running on the
    host, `host.docker.internal` when running inside docker compose.
    """
    return f"http://{settings.LS_HOST}:{annotator.port}"


def host_mount_path(path: Path) -> str:
    """Translate a path under our data dir to its location on the docker HOST.

    The worker may run inside a container where data is at /app/data, but
    `docker run -v` is interpreted by the host daemon, so the mount source must
    be the host path. FLEET_HOST_DATA_DIR (set in compose to the host ./data)
    enables the translation; unset (host run) → the resolved path is already
    correct.
    """
    resolved = path.resolve()
    host_data = getattr(settings, "FLEET_HOST_DATA_DIR", "")
    if host_data:
        data_root = (Path(settings.BASE_DIR) / "data").resolve()
        try:
            return str(Path(host_data) / resolved.relative_to(data_root))
        except ValueError:
            pass  # path lives outside ./data; can't translate, use as-is
    return str(resolved)


def run_container(*, annotator: Annotator, image_name: str, source_path: Path, target_path: Path):
    cmd = [
        "docker", "run", "-d",
        "--name", annotator.container_name,
        "-p", f"{annotator.port}:8080",
        # Reach the host-published webhook receiver from inside the container
        # (host.docker.internal does not resolve on Linux otherwise).
        "--add-host", "host.docker.internal:host-gateway",
        "-v", f"{annotator.volume_name}:/label-studio/data",
        "-v", f"{host_mount_path(source_path)}:{CONTAINER_DOC_ROOT}/source",
        "-v", f"{host_mount_path(target_path)}:{CONTAINER_DOC_ROOT}/target",
        "-e", "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true",
        "-e", f"LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={CONTAINER_DOC_ROOT}",
        "-e", f"LABEL_STUDIO_USERNAME={annotator.email}",
        "-e", f"LABEL_STUDIO_PASSWORD={annotator.password}",
        "-e", f"LABEL_STUDIO_USER_TOKEN={annotator.token}",
        image_name,
    ]
    lsapi.run(cmd)


def add_annotator(annotator: Annotator) -> dict:
    """Create/start the annotator's container and bootstrap its API token.

    Idempotent: a running container with a working token is a no-op. Restoring
    a retired annotator reuses its kept volume (projects + annotations intact)
    because the row carries the same volume/port/token.
    """
    if not lsapi.docker_available():
        raise RuntimeError("Docker is not installed or is not available on PATH")

    fs = FleetSettings.load()
    source_path = source_root(fs)
    target_path = target_root(fs)
    source_path.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)

    url = reachable_url(annotator)

    # Fast path: already provisioned and reachable.
    if (
        lsapi.container_exists(annotator.container_name)
        and lsapi.container_running(annotator.container_name)
        and tokens.token_works(url, annotator.token)
    ):
        if annotator.status != Annotator.ACTIVE:
            annotator.status = Annotator.ACTIVE
            annotator.save(update_fields=["status", "updated_at"])
        return {"username": annotator.username, "outcome": "already provisioned"}

    if not lsapi.container_exists(annotator.container_name):
        run_container(
            annotator=annotator,
            image_name=fs.image_name,
            source_path=source_path,
            target_path=target_path,
        )
        outcome = "container created"
    elif not lsapi.container_running(annotator.container_name):
        lsapi.run(["docker", "start", annotator.container_name])
        outcome = "container started"
    else:
        outcome = "container running"

    tokens.wait_until_http(url)
    tokens.enable_legacy_tokens(url, annotator.email, annotator.password)
    annotator.token = tokens.resolve_token(
        url,
        candidate_token=annotator.token,
        email=annotator.email,
        password=annotator.password,
    )
    annotator.status = Annotator.ACTIVE
    annotator.save(update_fields=["token", "status", "updated_at"])

    return {
        "username": annotator.username,
        "ls_url": annotator.ls_url,
        "port": annotator.port,
        "outcome": outcome,
    }


def remove_annotator(annotator: Annotator, *, purge: bool = False) -> dict:
    """Stop and remove the container.

    Default keeps the data volume and the row (status -> retired) so a later
    re-provision restores the same identity. ``purge`` also deletes the volume
    and the row — a full, irreversible teardown.
    """
    container_name = annotator.container_name
    if lsapi.container_exists(container_name):
        if lsapi.container_running(container_name):
            lsapi.run(["docker", "stop", container_name])
        lsapi.run(["docker", "rm", container_name])

    if purge:
        try:
            lsapi.run(["docker", "volume", "rm", annotator.volume_name])
        except RuntimeError:
            pass  # volume may not exist
        username = annotator.username
        annotator.delete()
        return {"username": username, "outcome": "purged (volume + row deleted)"}

    annotator.status = Annotator.RETIRED
    annotator.save(update_fields=["status", "updated_at"])
    return {"username": annotator.username, "outcome": "removed (volume + row kept; retired)"}
