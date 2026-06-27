"""Minimal Label Studio API access for the webhook receiver.

The webhook payload identifies the task only by id, so we fetch the task to
learn its image filename (and, on delete, its surviving annotations). Tokens
and ports come from the gitignored fleet state, mounted read-only into the
container.

Networking: fleet state records `http://localhost:<port>`, which is wrong from
inside this container. We rebuild the URL as `http://<LS_HOST>:<port>`, where
LS_HOST defaults to `host.docker.internal` (the docker host).
"""

import os
from pathlib import Path

import requests
import yaml

FLEET_CONFIG = Path(os.getenv("FLEET_CONFIG", "/configs/fleet.local.yaml"))
LS_HOST = os.getenv("LS_HOST", "host.docker.internal")
_TIMEOUT = 10


def _load_fleet() -> dict:
    # Re-read every call: tokens can be rotated by fleet.py while we run.
    return yaml.safe_load(FLEET_CONFIG.read_text(encoding="utf-8")) or {}


def annotator_endpoint(username: str) -> tuple[str, str]:
    """Return (base_url, api_token) for an annotator from fleet state."""
    fleet = _load_fleet()
    record = (fleet.get("annotators") or {}).get(username)
    if not record:
        raise KeyError(f"annotator {username!r} not found in {FLEET_CONFIG}")
    port = record["port"]
    return f"http://{LS_HOST}:{port}", record["token"]


def get_task(username: str, task_id: int) -> dict:
    base, token = annotator_endpoint(username)
    response = requests.get(
        f"{base}/api/tasks/{task_id}",
        headers={"Authorization": f"Token {token}"},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def list_project_tasks(username: str, project_id: int) -> list[dict]:
    """Return every task in a project, each with its inline `annotations`.

    Used by the delete handler: Label Studio's ANNOTATIONS_DELETED payload only
    names the deleted annotation by id (no task), so the only way to learn which
    image to rewrite is to re-read the project's current tasks and reconcile.
    `fields=all` makes each task carry its surviving annotations' `result`.
    """
    base, token = annotator_endpoint(username)
    headers = {"Authorization": f"Token {token}"}
    page_size = 100
    tasks: list[dict] = []
    page = 1
    while True:
        response = requests.get(
            f"{base}/api/tasks/",
            headers=headers,
            params={"project": project_id, "fields": "all", "page": page, "page_size": page_size},
            timeout=_TIMEOUT,
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
