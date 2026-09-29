"""Generate one variant (or reuse a cached one), and validate it.

The single step both the studio's previews and a dataset build are made of:
``(source image, prompt, seed, editor) -> cached PNG -> ValidationResult``.

**Cache key.** A variant is identified by everything that determines its
pixels: the source file (path, size, mtime — a re-exported image is a new
source), the editor configuration, the prompt text, negative prompt, steps,
CFG and seed. Validation settings are deliberately *not* in it: validating is
cheap and CPU-only, so it is recomputed every time, which is what lets the
studio re-check old previews under new thresholds for free.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from genaug.services import backend, sources
from genaug.services.validation import build_validator

#: How long to wait between retries while the GPU is busy.
GPU_BUSY_RETRY_S = 60


def variant_key(source: Path, generation: dict, editor_config: dict, seed: int) -> str:
    stat = source.stat()
    payload = {
        "source": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "editor": editor_config,
        "text": generation["text"], "negative_prompt": generation.get("negative_prompt") or "",
        "num_inference_steps": int(generation.get("num_inference_steps") or 40),
        "true_cfg_scale": float(generation.get("true_cfg_scale") or 4.0),
        "seed": int(seed),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_path(key: str) -> Path:
    return sources.genaug_root() / "cache" / key[:2] / f"{key}.png"


def generate(source: Path, generation: dict, editor_config: dict, seed: int, *,
             force: bool = False, on_busy=None, should_stop=None) -> tuple[Path, bool, int | None]:
    """Return ``(variant_path, was_cached, edit_ms)``.

    While the backend answers "GPU busy", waits and retries rather than
    failing: a long build that runs into someone starting a training run should
    pause, not throw away every remaining image. ``on_busy(message)`` is told
    each time; ``should_stop()`` returning true abandons the wait.
    """
    key = variant_key(source, generation, editor_config, seed)
    path = cache_path(key)
    if path.is_file() and not force:
        return path, True, None
    while True:
        try:
            result = backend.edit(image_path=source, output_path=path, generation=generation,
                                  editor_config=editor_config, seed=seed)
            break
        except backend.GpuBusy as exc:
            if on_busy is None or (should_stop is not None and should_stop()):
                raise
            on_busy(str(exc))
            time.sleep(GPU_BUSY_RETRY_S)
            if should_stop is not None and should_stop():
                raise
    if on_busy is not None:
        on_busy("")
    return path, False, result.get("edit_ms")


def validate(source: Path, variant: Path, objects: list[dict], settings: dict) -> dict:
    """Validation verdict as a JSON-able dict (``valid`` plus the evidence)."""
    validator = build_validator(settings)
    if validator is None:
        return {"valid": True, "checked": False, "reason": None, "bbox_drift": None}
    # cv2.imread applies EXIF orientation, matching the trainer's and the
    # backend's exif_transpose, so the boxes (relative to the displayed
    # orientation) line up.
    original, generated = read_rgb(source), read_rgb(variant)
    height, width = original.shape[:2]
    result = validator.validate(original, generated,
                                sources.pixel_boxes(objects, width, height))
    data = result.to_dict()
    data["checked"] = True
    data["validator"] = validator.name
    return data


def read_rgb(path: Path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
