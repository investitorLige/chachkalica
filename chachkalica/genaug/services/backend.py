"""HTTP client for the genaug-backend service (FireRed)."""

import os

import requests

HEALTH_TIMEOUT = 5

#: A cold first call loads ~30 GB of weights from disk and quantizes the text
#: encoder before it edits anything, so it is allowed a long time. Every
#: later call is one image.
EDIT_TIMEOUT = 1800


class GpuBusy(RuntimeError):
    """The backend declined to load because the GPU is in use. Retry later."""


class BackendError(RuntimeError):
    pass


def base_url() -> str:
    return os.getenv("GENAUG_BACKEND_URL", "http://genaug-backend:9092").rstrip("/")


def health() -> dict:
    resp = requests.get(f"{base_url()}/health", timeout=HEALTH_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def edit(*, image_path, output_path, generation: dict, editor_config: dict, seed: int) -> dict:
    """Ask the backend to edit one image and write the result to ``output_path``."""
    payload = {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "prompt": generation["text"],
        "negative_prompt": generation.get("negative_prompt") or " ",
        "seed": int(seed),
        "num_inference_steps": int(generation.get("num_inference_steps") or 40),
        "true_cfg_scale": float(generation.get("true_cfg_scale") or 4.0),
        "editor": editor_config.get("editor", "firered"),
        "transformer": editor_config.get("transformer", "q4_k_m"),
        "lightning": bool(editor_config.get("lightning")),
    }
    try:
        resp = requests.post(f"{base_url()}/edit", json=payload, timeout=EDIT_TIMEOUT)
    except requests.RequestException as exc:
        raise BackendError(f"genaug-backend unreachable at {base_url()}: {exc}") from exc
    if resp.status_code == 503:
        raise GpuBusy(_detail(resp))
    if resp.status_code >= 400:
        raise BackendError(_detail(resp))
    return resp.json()


def _detail(resp) -> str:
    try:
        return str(resp.json().get("detail") or resp.text)
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"
