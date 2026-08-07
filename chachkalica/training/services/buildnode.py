"""HTTP client for a remote TensorRT build node.

Sibling of :mod:`training.services.runner`, and deliberately shaped like it —
same error-detail extraction, same "404 means the far side has no record" rule.
The difference is what crosses the wire. The trainer shares a filesystem with us
and takes paths (see the SHARED FILESYSTEM note in the root ``docker-compose.yml``);
a build node is another machine, so artifacts go up as uploads and come back as a
tarball.

The node's contract lives in ``buildnode/service.py``; the operator-facing story
is ``docs/build-nodes.md``.
"""

import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import requests
from django.utils import timezone

# A ping is a liveness check rendered in an admin page — it must fail fast rather
# than hold the request open.
PING_TIMEOUT = 5
# An upload is a ~100-250 MB ONNX over a LAN.
SUBMIT_TIMEOUT = 1800
POLL_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 1800

# Poll cadence + ceiling for a remote build, mirroring training.jobs' trainer
# poll loop. A cold node compiles the person detector on its first person-crop
# build (a couple of minutes) on top of the model's own compile.
POLL_INTERVAL = 10
MAX_WAIT = 3600


class BuildNodeError(RuntimeError):
    """Anything that went wrong talking to, or on, a build node."""


def _headers(node) -> dict:
    return {"Authorization": f"Bearer {node.token}"} if node.token else {}


def _detail(resp) -> str:
    """The node's error message, unwrapped from FastAPI's ``{"detail": ...}``."""
    detail = resp.text
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("detail"):
        detail = str(body["detail"])
    return detail


def _check(resp, node, what: str):
    if resp.status_code >= 400:
        raise BuildNodeError(
            f"build node {node.name} {what} returned HTTP {resp.status_code}: {_detail(resp)}"
        )
    return resp


# ── liveness ─────────────────────────────────────────────────────────────────


def health(node) -> dict:
    resp = requests.get(
        f"{node.api_url}/health", headers=_headers(node), timeout=PING_TIMEOUT)
    return _check(resp, node, "/health").json()


def record_health(node) -> dict:
    """Ping ``node`` and write the snapshot onto the row. Never raises.

    Returns the health body on success, or ``{}`` on failure — the row carries the
    outcome either way, which is what the admin renders. A node being down is an
    ordinary state to display, not an exception to propagate into a page render.
    """
    from training.models import BuildNode

    fields = [
        "last_status", "last_error", "last_checked_at", "last_seen_at",
        "last_health", "gpu_name", "compute_capability", "tensorrt_version",
        "driver_version",
    ]
    node.last_checked_at = timezone.now()

    try:
        body = health(node)
    except Exception as exc:  # noqa: BLE001 - unreachable/refusing is a reportable state
        node.last_status = BuildNode.ERROR
        node.last_error = str(exc)[:2000]
        node.last_health = None
        node.save(update_fields=fields)
        return {}

    node.last_status = BuildNode.OK if body.get("status") == "ok" else BuildNode.ERROR
    node.last_error = "" if node.last_status == BuildNode.OK else (
        f"node reports status={body.get('status')!r} — "
        f"gpu={body.get('gpu_name')} tensorrt={body.get('tensorrt_version')}"
    )
    node.last_seen_at = node.last_checked_at
    node.last_health = body
    node.gpu_name = (body.get("gpu_name") or "")[:128]
    node.compute_capability = (body.get("compute_capability") or "")[:16]
    node.tensorrt_version = (body.get("tensorrt_version") or "")[:32]
    node.driver_version = (body.get("driver_version") or "")[:32]
    node.save(update_fields=fields)
    return body


# ── running a build ──────────────────────────────────────────────────────────


def submit_build(node, spec: dict, files: dict) -> str:
    """Upload a build and return its id. ``files`` maps part name -> path.

    Part names are the ones ``buildnode.service.create_build`` declares:
    ``model``, ``model_meta``, and optionally ``detector``/``detector_meta``.
    """
    handles = []
    try:
        parts = []
        for field, path in files.items():
            handle = Path(path).open("rb")
            handles.append(handle)
            parts.append((field, (Path(path).name, handle, "application/octet-stream")))
        resp = requests.post(
            f"{node.api_url}/builds",
            headers=_headers(node),
            data={"spec": json.dumps(spec)},
            files=parts,
            timeout=SUBMIT_TIMEOUT,
        )
    finally:
        for handle in handles:
            handle.close()

    body = _check(resp, node, "/builds").json()
    build_id = body.get("build_id")
    if not build_id:
        raise BuildNodeError(f"build node {node.name} accepted the build but returned no build_id")
    return build_id


def poll(node, build_id: str) -> dict:
    """The node's view of a build.

    A 404 comes back as ``{"status": "unknown"}`` rather than raising — the same
    convention ``runner.fetch_status`` uses. It means the node restarted or reaped
    the build, which is terminal for us (there is no artifact to collect) but is
    the caller's decision to make, not this function's.
    """
    resp = requests.get(
        f"{node.api_url}/builds/{build_id}", headers=_headers(node), timeout=POLL_TIMEOUT)
    if resp.status_code == 404:
        return {"status": "unknown"}
    return _check(resp, node, f"/builds/{build_id}").json()


def wait(node, build_id: str, *, on_progress=None) -> dict:
    """Block until the build finishes. Returns its final status body.

    Raises on timeout or on the node losing the build; a build that finished with
    ``status == "error"`` is returned normally, because the caller wants the log
    tail off it for the error message.
    """
    waited = 0
    while waited < MAX_WAIT:
        status = poll(node, build_id)
        state = status.get("status")
        if state in ("ok", "error"):
            return status
        if state == "unknown":
            raise BuildNodeError(
                f"build node {node.name} lost build {build_id} (restarted, or the build "
                f"was reaped). Nothing to collect; re-queue the export."
            )
        if on_progress:
            on_progress(status)
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
    raise BuildNodeError(
        f"build node {node.name} did not finish build {build_id} within {MAX_WAIT}s"
    )


def download_artifact(node, build_id: str, dest_tar: Path) -> Path:
    """Stream the finished bundle tarball to ``dest_tar``."""
    dest_tar = Path(dest_tar)
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(
        f"{node.api_url}/builds/{build_id}/artifact",
        headers=_headers(node),
        timeout=DOWNLOAD_TIMEOUT,
        stream=True,
    ) as resp:
        _check(resp, node, f"/builds/{build_id}/artifact")
        with dest_tar.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return dest_tar


def cleanup(node, build_id: str) -> None:
    """Ask the node to drop the build's scratch dir. Best effort.

    A node reaps on its own TTL, so failing to clean up costs disk for a day, not
    correctness — never worth failing an otherwise-successful export over.
    """
    try:
        requests.delete(
            f"{node.api_url}/builds/{build_id}", headers=_headers(node), timeout=POLL_TIMEOUT)
    except Exception:  # noqa: BLE001 - best effort by design
        pass


# ── unpacking what came back ─────────────────────────────────────────────────


def _safe_members(tar: tarfile.TarFile, root: Path):
    """Yield members that stay inside ``root``, rejecting anything that escapes.

    The tarball came off another machine over the network, so it is untrusted
    input. Absolute paths, ``..`` traversal, and links are all refused outright —
    a bundle is plain files and directories, so a link in one is either a mistake
    or an attack. Mirrors the containment check in ``exports.resolve``.
    """
    for member in tar.getmembers():
        if member.islnk() or member.issym():
            raise BuildNodeError(
                f"bundle archive contains a link ({member.name}); refusing to extract"
            )
        if not (member.isfile() or member.isdir()):
            raise BuildNodeError(
                f"bundle archive contains a special file ({member.name}); refusing to extract"
            )
        target = (root / member.name).resolve()
        if not str(target).startswith(str(root.resolve()) + "/") and target != root.resolve():
            raise BuildNodeError(
                f"bundle archive member {member.name!r} would extract outside the "
                f"destination; refusing"
            )
        yield member


def extract_bundle(tar_path: Path, dest_parent: Path) -> Path:
    """Extract the archive under ``dest_parent`` and return the bundle directory.

    The archive holds exactly one top-level directory (the bundle). Extraction
    goes to a temp dir first and is moved into place only once it is whole, so an
    interrupted download cannot leave a half-bundle that ``list_bundles`` would
    then offer as if it were usable.
    """
    tar_path, dest_parent = Path(tar_path), Path(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=".unpack-", dir=dest_parent))
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = list(_safe_members(tar, staging))
            tar.extractall(staging, members=members)

        roots = [child for child in staging.iterdir()]
        if len(roots) != 1 or not roots[0].is_dir():
            raise BuildNodeError(
                f"expected exactly one bundle directory in the archive, found "
                f"{[r.name for r in roots]}"
            )
        final = dest_parent / roots[0].name
        if final.exists():
            shutil.rmtree(final)
        roots[0].replace(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return final
