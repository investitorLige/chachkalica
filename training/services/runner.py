"""HTTP client for the friendy_mercury trainer service.

The trainer runs in its own torch/CUDA environment behind the small FastAPI
service (``friendy_mercury/service.py``). We talk to it over HTTP, pointing it at
config YAMLs we wrote to the shared filesystem and reading back run liveness.
"""

import requests

from training.models import TrainingSettings

TIMEOUT = 30


def base_url(ts: TrainingSettings | None = None) -> str:
    ts = ts or TrainingSettings.load()
    return ts.service_base_url.rstrip("/")


def health(ts: TrainingSettings | None = None) -> dict:
    resp = requests.get(f"{base_url(ts)}/health", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def launch(run, resume: bool = False, ts: TrainingSettings | None = None) -> dict:
    """Ask the service to start training ``run`` from its generated config."""
    payload = {"run_id": run.pk, "config_path": run.config_yaml_path, "resume": resume}
    resp = requests.post(f"{base_url(ts)}/train", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_status(run, ts: TrainingSettings | None = None) -> dict:
    """Return the service's view of ``run``.

    A 404 means the service has no record (e.g. it was restarted) — reported as
    ``{"status": "unknown"}`` so the caller can fall back to the shared
    filesystem (a finished run leaves run_summary.yaml behind).
    """
    resp = requests.get(f"{base_url(ts)}/runs/{run.pk}", timeout=TIMEOUT)
    if resp.status_code == 404:
        return {"status": "unknown"}
    resp.raise_for_status()
    return resp.json()


def launch_eval(eval_run, ts: TrainingSettings | None = None) -> dict:
    """Ask the service to evaluate a trained model from its generated request."""
    payload = {"eval_id": eval_run.pk, "request_path": eval_run.request_yaml_path}
    resp = requests.post(f"{base_url(ts)}/eval", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_eval_status(eval_run, ts: TrainingSettings | None = None) -> dict:
    resp = requests.get(f"{base_url(ts)}/evals/{eval_run.pk}", timeout=TIMEOUT)
    if resp.status_code == 404:
        return {"status": "unknown"}
    resp.raise_for_status()
    return resp.json()
