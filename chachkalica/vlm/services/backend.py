"""HTTP client for the vlm-backend service.

The VLM runs in its own torch/CUDA container behind a small FastAPI service
(``chachkalica/ml_backends/vlm/service.py``), the same arrangement the trainer
and the SAM backend already use. Frames are handed over as *paths* on the shared
``data/`` mount rather than base64 — the convention
``docs/camera-live-inference.md`` explains under "Why frames go through the
filesystem", and the same way ``training.services.runner.predict_image`` works.
"""

import os

import requests

HEALTH_TIMEOUT = 10

# A wedged backend must not pin an rq worker forever, so every inference call is
# bounded. Generous, because a cold first call also pays for loading the model.
INFER_TIMEOUT = 600


def base_url() -> str:
    # Service name inside the unified compose stack; VLM_BACKEND_URL lets the
    # standalone stack point at a host-published port instead.
    return os.getenv("VLM_BACKEND_URL", "http://vlm-backend:9091").rstrip("/")


def health() -> dict:
    resp = requests.get(f"{base_url()}/health", timeout=HEALTH_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def infer(image_path: str, config: dict, prompt: str) -> dict:
    """Ask the backend what it sees in one frame.

    ``config`` is a :meth:`vlm.models.VlmModel.config` payload — the snapshot
    stored on the run, so a run keeps calling the model it started with even if
    the catalogue row is edited mid-flight.

    Returns ``{"text": ..., "latency_ms": ...}``.
    """
    payload = {
        "image_path": image_path,
        "prompt": prompt,
        "backend": config.get("backend", "transformers"),
        "family": config.get("family", ""),
        "weights": config.get("weights", ""),
        "quantization": config.get("quantization", ""),
        "max_new_tokens": config.get("max_new_tokens", 128),
    }
    resp = requests.post(f"{base_url()}/infer", json=payload, timeout=INFER_TIMEOUT)
    resp.raise_for_status()
    return resp.json()
