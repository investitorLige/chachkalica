"""FastAPI service wrapping a vision-language model.

Runs in its own container with its own torch/CUDA environment, the same
arrangement ``friendy_chachkalica/service.py`` (the trainer) and the SAM backend
already use. The Django worker hands it a frame *path* on the shared ``data/``
mount and gets back a line of text.

**One model is warm at a time.** The cache holds exactly one entry and evicts on
any configuration change — the same trade-off the trainer's ``_predict_cache``
makes, and for the same reason: these are multi-gigabyte models on a GPU already
shared with the trainer and the SAM backend. Alternating two VLM rows against
one video will reload on every call and be unusably slow; that is a deliberate
limit, not a bug.

A lock serializes generation, so concurrent callers queue rather than racing
each other's CUDA context.

Weights come from the offline HF cache (``HF_HOME`` + ``HF_HUB_OFFLINE=1``):
huggingface.co is not reachable from this network, so anything not already
copied into ``data/hf_cache/hub`` cannot load here at all.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml_backends.vlm.registry import build_adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vlm-backend")

app = FastAPI(title="VLM backend")

_lock = threading.Lock()
_cache_key: tuple | None = None
_cache_adapter = None


class InferRequest(BaseModel):
    image_path: str
    prompt: str
    backend: str = "transformers"
    family: str = ""
    weights: str = ""
    quantization: str = ""
    max_new_tokens: int = 128


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - health must answer even on a broken install
        return "unknown"


def _cache_miss(weights: str) -> str | None:
    """Why this repo cannot load, if the offline cache does not have it.

    transformers' own message for a cache miss is a generic "couldn't connect to
    huggingface.co", which on this network is true of *everything* and so says
    nothing. Checking first lets the error name the actual problem and the
    directory that has to appear.
    """
    if not weights or "/" not in weights or weights.startswith((".", "/", "~")):
        return None  # a local path — the loader's own error will be accurate
    hub = os.path.join(os.environ.get("HF_HOME", ""), "hub")
    if not os.path.isdir(hub):
        return f"the HF cache at {hub} does not exist"
    snapshot = os.path.join(hub, "models--" + weights.replace("/", "--"))
    if not os.path.isdir(snapshot):
        return (
            f"{weights} is not in the offline HF cache — huggingface.co is not "
            f"reachable from here, so its snapshot has to be copied to {snapshot} "
            f"first (see `manage.py fetch_vlm_weights`)"
        )
    return None


def _get_adapter(req: InferRequest):
    """Return the warm adapter for this configuration, loading it if needed."""
    global _cache_key, _cache_adapter

    key = (req.backend, req.family, req.weights, req.quantization)
    if _cache_key == key and _cache_adapter is not None:
        return _cache_adapter

    if _cache_adapter is not None:
        logger.info("evicting %s to make room for %s", _cache_key, key)
        _cache_adapter = None
        _cache_key = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            logger.exception("failed to release the previous model's memory")

    logger.info("loading %s", key)
    adapter = build_adapter(req.backend, req.family, req.weights, req.quantization)
    adapter.load()
    _cache_adapter = adapter
    _cache_key = key
    logger.info("loaded %s", adapter.label)
    return adapter


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": _device(),
        "loaded": _cache_adapter.label if _cache_adapter else None,
        "hf_home": os.environ.get("HF_HOME"),
        "hf_offline": os.environ.get("HF_HUB_OFFLINE"),
    }


@app.post("/infer")
def infer(req: InferRequest) -> dict:
    if not os.path.isfile(req.image_path):
        # The caller and this service share the data/ mount, so a missing file
        # here almost always means the mount is wrong rather than the frame is.
        raise HTTPException(
            status_code=400,
            detail=f"frame not found at {req.image_path} — is data/ mounted here?",
        )

    if req.backend == "transformers":
        miss = _cache_miss(req.weights)
        if miss:
            raise HTTPException(status_code=400, detail=miss)

    started = time.monotonic()
    with _lock:
        try:
            adapter = _get_adapter(req)
            text = adapter.infer(req.image_path, req.prompt, req.max_new_tokens)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("inference failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"text": text, "latency_ms": int((time.monotonic() - started) * 1000)}
