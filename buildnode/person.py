"""The person detector this node builds for itself.

Every person-crop pipeline (``people_detect_first``, ``batch_people``, and chains
containing them) needs a person detector inside the bundle. On the training box
that detector ships as a prebuilt ``models/people/best_ckpt.engine``, and
``bundle_export`` copies it straight in — which is exactly what cannot travel,
because it is an engine compiled for *that* GPU. A bundle built here carrying that
file would fail to deserialize the moment it ran.

So the node carries the detector's **graphs** instead, baked into the image, and
compiles its own engine. Two graphs, because the two bundle formats need different
things and neither one covers both:

``detector.onnx``
    The standard exported graph. Goes into an ONNX bundle as-is, where it runs
    under onnxruntime on whatever machine the bundle lands on.

``detector.trt.onnx``
    The same model re-exported as raw outputs plus an ``EfficientNMS_TRT`` node.
    This is the only form TensorRT can compile for a YOLOX graph — the standard
    one bakes a data-dependent NMS the compiler rejects — and producing it needs
    the ``.pt`` and torch, which is why it is baked rather than derived here.
    onnxruntime cannot run it: ``EfficientNMS_TRT`` is a TensorRT plugin, so this
    graph is only ever an input to a compile, never an artifact in an ONNX bundle.

Regenerate both with ``buildnode/scripts/make_person_graphs.py``.

The meta sidecar is copied verbatim from the repo and must stay that way. It is a
hand-patched Megvii checkpoint — ``layout: bgr``, ``input_scale: byte``,
``pad_value: 114`` — where friendy's own YOLOX exporter writes ``rgb``/``unit``.
Regenerating it from the exporter silently mis-preprocesses every frame.
"""

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, Optional

PERSON_DIR = Path(os.environ.get("BUILDNODE_PERSON_DIR", "/opt/buildnode/person"))

# Standard graph — for an ONNX bundle.
PERSON_ONNX = PERSON_DIR / "detector.onnx"
PERSON_META = PERSON_DIR / "detector.meta.json"

# EfficientNMS-prepared graph — the compile input for an engine bundle.
PERSON_TRT_ONNX = PERSON_DIR / "detector.trt.onnx"
PERSON_TRT_META = PERSON_DIR / "detector.trt.meta.json"

CACHE_DIR = Path(os.environ.get("BUILDNODE_CACHE_DIR", "/var/lib/buildnode/person-cache"))

# The profile and batch range the SHIPPED person engine
# (``models/people/best_ckpt.engine``) was built with. Reproduced here on purpose:
# left to derive itself from the meta, this graph compiles to a static 640 profile
# at batch 1, and both of those are real regressions against the detector every
# existing bundle carries. The detector runs once per frame over many person crops,
# so batch 1 costs throughput directly; a static 640 caps the frame size the engine
# will accept, where the shipped one takes up to 1024 and rescales above that.
# Override per node if its GPU cannot hold the wider profile.
PROFILE_MIN_HW = (64, 64)
PROFILE_OPT_HW = (640, 640)
PROFILE_MAX_HW = (1024, 1024)
BATCH_MIN, BATCH_OPT, BATCH_MAX = 1, 8, 16

_cache_lock = threading.Lock()


def available() -> bool:
    """Whether this node can supply a detector for *either* bundle format."""
    return has_onnx() or has_trt_graph()


def has_onnx() -> bool:
    return PERSON_ONNX.is_file() and PERSON_META.is_file()


def has_trt_graph() -> bool:
    return PERSON_TRT_ONNX.is_file() and PERSON_TRT_META.is_file()


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256() -> Optional[str]:
    """Digest of the compile-input graph — the cache key, and what ``/health`` reports.

    Django surfaces it so an operator can see at a glance whether two nodes are
    building against the same detector.
    """
    return _sha256(PERSON_TRT_ONNX) or _sha256(PERSON_ONNX)


def describe() -> Dict[str, Any]:
    return {
        "present": available(),
        "onnx": has_onnx(),
        "trt_graph": has_trt_graph(),
        "sha256": sha256(),
        "arch": meta().get("arch"),
    }


def meta() -> Dict[str, Any]:
    """The baked detector's Contract-B sidecar, for diagnostics."""
    for path in (PERSON_META, PERSON_TRT_META):
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
    return {}


def _cached_dir(digest: str, precision: str) -> Path:
    # The profile and batch are in the key, not just the graph and precision: the
    # cache lives in a volume that outlives the image, so changing the constants
    # above must produce a new engine rather than silently reuse the old one.
    shape = f"{PROFILE_MIN_HW[0]}-{PROFILE_OPT_HW[0]}-{PROFILE_MAX_HW[0]}-b{BATCH_MAX}"
    return CACHE_DIR / f"{digest[:16]}-{precision}-{shape}"


def ensure_engine(precision: str, *, log=None) -> Path:
    """Compile the baked person graph to an engine on this GPU, once.

    The detector never changes between builds, so the result is cached by (graph
    digest, precision) and copied into every subsequent bundle. Returns the
    ``.engine`` path with its ``.meta.json`` and ``.engine.json`` sidecars beside
    it — the shape ``bundle_export._export_role`` expects of a prebuilt engine.
    """
    if not has_trt_graph():
        raise FileNotFoundError(
            f"This node has no TensorRT-ready person graph ({PERSON_TRT_ONNX}). An "
            f"engine bundle for a person-crop pipeline needs one — rebuild the image "
            f"with buildnode/scripts/make_person_graphs.py, or send a detector with "
            f"the build."
        )

    digest = _sha256(PERSON_TRT_ONNX) or "unknown"
    target = _cached_dir(digest, precision)
    engine_path = target / "detector.engine"

    with _cache_lock:
        if engine_path.is_file():
            if log:
                log(f"[person] reusing cached detector engine ({precision})")
            return engine_path

        # Build into a sibling and move into place, so an interrupted build cannot
        # leave a half-written engine that later looks cached.
        staging = target.with_name(target.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        graph = staging / "detector.trt.onnx"
        shutil.copy2(PERSON_TRT_ONNX, graph)
        shutil.copy2(PERSON_TRT_META, staging / "detector.trt.meta.json")

        if log:
            log(f"[person] compiling the person detector ({precision}) — first time on this node")

        from .compile import compile_engine

        compile_engine(
            graph,
            staging / "detector.engine",
            precision=precision,
            prepared=True,
            min_hw=PROFILE_MIN_HW,
            opt_hw=PROFILE_OPT_HW,
            max_hw=PROFILE_MAX_HW,
            min_batch=BATCH_MIN,
            opt_batch=BATCH_OPT,
            max_batch=BATCH_MAX,
        )

        # The graph was only a compile input; the cache holds the engine.
        graph.unlink(missing_ok=True)
        (staging / "detector.trt.meta.json").unlink(missing_ok=True)
        staging.replace(target)

    if log:
        log(f"[person] cached detector engine at {engine_path}")
    return engine_path


def detector_source(fmt: str, precision: str, *, log=None) -> Path:
    """The baked detector in the form this bundle format wants.

    An ONNX bundle gets the standard graph, so it stays portable and whoever runs
    it can compile their own engine. An engine bundle gets this node's engine.
    """
    if fmt == "onnx":
        if not has_onnx():
            raise FileNotFoundError(
                f"This node has no standard person ONNX ({PERSON_ONNX}), which an ONNX "
                f"bundle needs. The TensorRT-ready graph is not a substitute: its "
                f"EfficientNMS_TRT node is a TensorRT plugin onnxruntime cannot run."
            )
        return PERSON_ONNX
    return ensure_engine(precision, log=log)
