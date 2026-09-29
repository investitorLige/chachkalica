"""FastAPI service wrapping a generative image editor (FireRed-Image-Edit).

Same arrangement as ``ml_backends/vlm/service.py``: its own container and
torch/CUDA environment, one model warm at a time, a lock serializing
generation, and images passed as *paths* on the shared ``data/`` mount. The
Django worker says "edit this file with this prompt and write the result
there", and gets back timing and what ran.

Two things differ from the VLM backend, both because FireRed is ~14 GB of GPU
even at 4-bit:

* **It unloads itself when idle** (``GENAUG_IDLE_UNLOAD_S``, default 300 s),
  so a finished preview session hands the GPU back to training without anyone
  remembering to. The next request pays the reload.
* **It refuses to load onto a busy GPU** and answers 503 with how much memory
  was free, instead of OOMing mid-load. Callers treat 503 as "try later".

Weights come from the offline HF cache (``HF_HOME`` + ``HF_HUB_OFFLINE=1``);
``manage.py fetch_genaug_weights`` lists what must be copied in.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel

from ml_backends.genaug.editors.base import EditParams
from ml_backends.genaug.registry import build_editor, required_repos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genaug-backend")

app = FastAPI(title="Generative augmentation backend")

DATA_ROOT = os.environ.get("GENAUG_DATA_ROOT", "/app/data")
IDLE_UNLOAD_S = float(os.environ.get("GENAUG_IDLE_UNLOAD_S", "300"))

_lock = threading.Lock()
_cache_key: tuple | None = None
_editor = None
_last_used = 0.0


class EditRequest(BaseModel):
    image_path: str
    output_path: str
    prompt: str
    seed: int = 42
    negative_prompt: str = " "
    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    # Editor configuration — see registry.build_editor.
    editor: str = "firered"
    transformer: str = "q4_k_m"
    lightning: bool = False
    offload: str = "model"


def _inside_data_root(path: str) -> bool:
    real = os.path.realpath(path)
    root = os.path.realpath(DATA_ROOT)
    return real == root or real.startswith(root + os.sep)


def _cache_miss(repo: str, filename: str | None) -> str | None:
    hub = os.path.join(os.environ.get("HF_HOME", ""), "hub")
    snapshot = os.path.join(hub, "models--" + repo.replace("/", "--"))
    if not os.path.isdir(snapshot):
        return (f"{repo} is not in the offline HF cache ({snapshot}) — see "
                f"`manage.py fetch_genaug_weights` for what to copy in")
    if filename:
        snapshots = os.path.join(snapshot, "snapshots")
        found = any(os.path.exists(os.path.join(snapshots, rev, filename))
                    for rev in (os.listdir(snapshots) if os.path.isdir(snapshots) else []))
        if not found:
            return (f"{repo} is cached but {filename} is not in it — see "
                    f"`manage.py fetch_genaug_weights`")
    return None


def _release() -> None:
    global _editor, _cache_key
    if _editor is None:
        return
    logger.info("unloading %s", _editor.label)
    _editor.unload()
    _editor = None
    _cache_key = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        logger.exception("failed to release GPU memory")


def _get_editor(req: EditRequest):
    global _editor, _cache_key
    key = (req.editor, req.transformer, req.lightning, req.offload)
    if _cache_key == key and _editor is not None:
        return _editor
    _release()
    editor = build_editor(req.editor, {
        "transformer": req.transformer, "lightning": req.lightning, "offload": req.offload,
    })
    logger.info("loading %s", editor.label)
    editor.load()
    _editor, _cache_key = editor, key
    return editor


def _idle_reaper() -> None:
    while True:
        time.sleep(15)
        if IDLE_UNLOAD_S <= 0 or _editor is None:
            continue
        if time.monotonic() - _last_used < IDLE_UNLOAD_S:
            continue
        # Non-blocking: an edit in progress means "not idle".
        if _lock.acquire(blocking=False):
            try:
                if _editor is not None and time.monotonic() - _last_used >= IDLE_UNLOAD_S:
                    _release()
            finally:
                _lock.release()


threading.Thread(target=_idle_reaper, daemon=True).start()


def _gpu() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"device": "cpu"}
        free, total = torch.cuda.mem_get_info()
        return {"device": torch.cuda.get_device_name(0),
                "free_gib": round(free / 2**30, 2), "total_gib": round(total / 2**30, 2)}
    except Exception:  # noqa: BLE001 - health must answer even on a broken install
        return {"device": "unknown"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "gpu": _gpu(),
        "loaded": _editor.describe() if _editor else None,
        "idle_unload_s": IDLE_UNLOAD_S,
        "hf_home": os.environ.get("HF_HOME"),
        "hf_offline": os.environ.get("HF_HUB_OFFLINE"),
    }


@app.post("/unload")
def unload() -> dict:
    with _lock:
        _release()
    return {"status": "ok"}


@app.post("/edit")
def edit(req: EditRequest) -> dict:
    global _last_used
    for path in (req.image_path, req.output_path):
        if not _inside_data_root(path):
            raise HTTPException(400, f"{path} is outside {DATA_ROOT}")
    if not os.path.isfile(req.image_path):
        raise HTTPException(400, f"image not found at {req.image_path} — is data/ mounted here?")

    for repo, filename in required_repos(req.editor, req.transformer, req.lightning):
        miss = _cache_miss(repo, filename)
        if miss:
            raise HTTPException(400, miss)

    from ml_backends.genaug.editors.firered import GpuBusyError

    with _lock:
        started = time.monotonic()
        try:
            editor = _get_editor(req)
            load_ms = int((time.monotonic() - started) * 1000)
            # Labels are relative to the displayed (EXIF-rotated) image — the
            # orientation the trainer reads too — so edit that one.
            image = ImageOps.exif_transpose(Image.open(req.image_path)).convert("RGB")
            edit_started = time.monotonic()
            result = editor.edit(image, req.prompt, req.seed, EditParams(
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.num_inference_steps,
                true_cfg_scale=req.true_cfg_scale,
            ))
            edit_ms = int((time.monotonic() - edit_started) * 1000)
        except GpuBusyError as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("edit failed")
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        finally:
            _last_used = time.monotonic()

    os.makedirs(os.path.dirname(req.output_path), exist_ok=True)
    # Written to a temp name and renamed, so a reader never sees half a PNG.
    tmp = req.output_path + ".part"
    result.save(tmp, format="PNG")
    os.replace(tmp, req.output_path)
    logger.info("edited %s in %d ms (+%d ms load)", os.path.basename(req.image_path),
                edit_ms, load_ms if load_ms > 50 else 0)
    return {"output_path": req.output_path, "edit_ms": edit_ms, "load_ms": load_ms,
            "editor": editor.describe()}
