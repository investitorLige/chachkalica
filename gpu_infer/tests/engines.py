"""Finding real engines to test against, and skipping cleanly when there are none.

A TensorRT engine plan only deserializes on the exact runtime version that built it. This repo
therefore accumulates engines that are simply *stale* — built by a TensorRT the current image no
longer has — and a test that tries to load one dies with ``Error Code 6 ... expecting library
version`` (or a bare "failed to deserialize", when there is no ``.engine.json`` sidecar to explain
it). That reads like a code regression and is not one.

So discovery here filters on the sidecar's recorded ``tensorrt_version`` **before** attempting a
load, and a test with no version-matching engine *skips* with a message naming what it wanted.
That keeps a stale checkout honest: green because nothing ran, and it says so.

Random-init engines are not an acceptable substitute for an exactness gate. A randomly
initialized head collapses so every anchor carries one bias-only score, and EfficientNMS's top-K
over those ties differs between calls — ``run(x)`` twice disagrees with itself. Anything
asserting bit-identical output must use trained weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where bundles and exports land. ``/app/data`` covers the container's bind mount.
_SEARCH_ROOTS = (
    _REPO_ROOT / "chachkalica" / "data" / "bundles",
    _REPO_ROOT / "chachkalica" / "data" / "training" / "runs" / "exports",
    _REPO_ROOT / "dist",
    Path("/app/data/bundles"),
    Path("/app/data/training/runs/exports"),
)


def runtime_version() -> Optional[str]:
    """The installed TensorRT version, or ``None`` if tensorrt is not importable."""
    try:
        import tensorrt

        return str(tensorrt.__version__)
    except Exception:
        return None


def engine_build_version(engine_path: Path) -> Optional[str]:
    """The TensorRT version recorded beside ``engine_path``, or ``None`` if unrecorded.

    The builder stamps this into ``<engine>.json``. An engine without the sidecar cannot be
    vetted up front, so :func:`find_bundles` treats it as unusable rather than gambling a
    confusing deserialization failure on it.
    """
    sidecar = Path(str(engine_path) + ".json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text()).get("tensorrt_version")
    except (json.JSONDecodeError, OSError):
        return None


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def find_bundles(*, require_detector: bool = False) -> List[Path]:
    """Bundle directories whose engines this runtime can actually load.

    A bundle qualifies when ``models/model.engine`` exists and its sidecar records exactly the
    running TensorRT version. ``require_detector`` additionally demands a loadable
    ``models/detector.engine``, which the two-stage pipelines need.

    ``GPU_INFER_TEST_BUNDLE`` overrides discovery entirely, for pointing a run at one specific
    freshly-built bundle.
    """
    override = os.environ.get("GPU_INFER_TEST_BUNDLE")
    if override:
        candidates = [Path(override)]
    else:
        candidates = []
        for root in _SEARCH_ROOTS:
            if root.is_dir():
                candidates.extend(sorted(root.glob("**/models/model.engine")))
        candidates = [path.parent.parent for path in candidates]

    version = runtime_version()
    usable = []
    for bundle in candidates:
        model = bundle / "models" / "model.engine"
        if not model.exists() or engine_build_version(model) != version:
            continue
        if require_detector:
            detector = bundle / "models" / "detector.engine"
            if not detector.exists() or engine_build_version(detector) != version:
                continue
        usable.append(bundle)
    return usable


def find_engines(*, efficientnms: Optional[bool] = None) -> List[Path]:
    """Individual loadable ``.engine`` paths, optionally filtered by graph layout.

    ``efficientnms=True`` returns only engines carrying the fused-NMS plugin outputs (the
    detector engines here), ``False`` only passthrough ones (rtdetr/rfdetr), ``None`` both. The
    layout is read from the engine's own tensor names, so this needs a real load — used by tests
    that want to cover both output paths without hard-coding which artifact is which.
    """
    version = runtime_version()
    found = []
    for root in _SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("**/models/*.engine")):
            if engine_build_version(path) == version:
                found.append(path)
    if efficientnms is None:
        return found
    return [path for path in found if _is_efficientnms(path) == efficientnms]


def _is_efficientnms(engine_path: Path) -> bool:
    """Whether this engine's graph emits the EfficientNMS_TRT plugin's four outputs.

    Read from the deserialized engine's tensor names rather than guessed from the filename or
    the arch string, matching how ``trt_infer.session.TrtModel`` decides.
    """
    from onnx_infer.meta import ModelMeta

    from gpu_infer.engine import AsyncEngine

    meta = ModelMeta.load(engine_path.with_suffix(".meta.json"))
    engine = AsyncEngine(engine_path, meta, warmup=False)
    return engine.efficientnms


def skip_reason(*, require_detector: bool = False) -> Optional[str]:
    """Why the GPU engine tests cannot run here, or ``None`` if they can.

    Reports the *specific* obstacle — no CUDA, no tensorrt, or engines present but built by a
    different version — because "skipped" with no reason is how a stale checkout quietly stops
    testing the thing it claims to.
    """
    if not cuda_available():
        return "needs a CUDA GPU"
    version = runtime_version()
    if version is None:
        return "needs tensorrt installed"
    if find_bundles(require_detector=require_detector):
        return None

    stale = []
    for root in _SEARCH_ROOTS:
        if root.is_dir():
            for path in root.glob("**/models/model.engine"):
                stale.append(engine_build_version(path) or "unrecorded")
    if stale:
        return (
            f"no bundle whose engines were built by the running TensorRT {version} "
            f"(found engines built by: {sorted(set(stale))}). Rebuild them, or set "
            f"GPU_INFER_TEST_BUNDLE to one that matches."
        )
    return f"no exported .engine bundle found to test against (TensorRT {version})"


def load_pair(bundle: Path) -> Tuple:
    """``(model_engine, model_meta, detector_engine, detector_meta)`` for a bundle."""
    from onnx_infer.meta import ModelMeta

    models = bundle / "models"
    model_path = models / "model.engine"
    detector_path = models / "detector.engine"
    return (
        model_path,
        ModelMeta.load(model_path.with_suffix(".meta.json")),
        detector_path if detector_path.exists() else None,
        ModelMeta.load(detector_path.with_suffix(".meta.json"))
        if detector_path.exists()
        else None,
    )
