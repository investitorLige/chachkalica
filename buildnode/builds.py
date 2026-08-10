"""Running one build: scratch dir, path rewriting, compile, bundle, tarball.

The hard part here is not the compile — it is that this node shares no filesystem
with whoever asked for the build. The rest of the stack passes absolute paths and
opens them (see the SHARED FILESYSTEM note in the root ``docker-compose.yml``);
a chachak pipeline request is full of them. So every path in an incoming request
is rewritten to point at the files that arrived with it, and any request naming a
path this node did not receive is rejected rather than quietly resolved against
whatever happens to be on this disk.
"""

import json
import shutil
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import NODE_VERSION, compile as compile_mod, gpu, person, settings

QUEUED, RUNNING, OK, ERROR = "queued", "running", "ok", "error"

# One build at a time. TensorRT compiles are GPU- and memory-hungry, and the whole
# point of a node is that an operator dispatched work here on purpose — mirrors the
# trainer's _trt_build_lock.
_gpu_lock = threading.Lock()
_registry_lock = threading.Lock()
_builds: Dict[str, "BuildRecord"] = {}


@dataclass
class BuildRecord:
    build_id: str
    name: str
    scratch: Path
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: str = ""
    result: Optional[Dict[str, Any]] = None
    log: List[str] = field(default_factory=list)

    def append(self, line: str) -> None:
        self.log.append(line)
        del self.log[: max(0, len(self.log) - settings.LOG_LINES)]

    def as_dict(self, *, with_log: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "build_id": self.build_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
        }
        if with_log:
            payload["log_tail"] = "\n".join(self.log)
        return payload

    @property
    def artifact(self) -> Path:
        return self.scratch / "bundle.tar.gz"


class SpecError(ValueError):
    """A build request this node will not accept. Surfaces as HTTP 400."""


# ── the request's paths ──────────────────────────────────────────────────────


def _basename(value: Any, field_name: str) -> str:
    """Accept only a bare filename.

    A request arrives from another machine, so any path component in it is either
    a mistake or an attempt to read this node's disk. ``models/model.onnx`` and
    ``../../etc/passwd`` are both rejected the same way.
    """
    text = str(value or "").strip()
    if not text:
        raise SpecError(f"{field_name} is empty")
    if text != Path(text).name or text in (".", ".."):
        raise SpecError(
            f"{field_name} must be a bare filename, got {text!r} — a build request "
            f"may only name files uploaded with it."
        )
    return text


def _uploaded(inputs: Path, filename: str, field_name: str) -> Path:
    path = inputs / _basename(filename, field_name)
    if not path.is_file():
        raise SpecError(f"{field_name} names {filename!r}, which was not uploaded")
    return path


def materialize_request(
    request: Dict[str, Any],
    *,
    inputs: Path,
    scratch: Path,
    model_path: Path,
    detector_path: Optional[Path],
) -> Dict[str, Any]:
    """Rewrite a chachak request so every path points inside this build.

    ``model_path``/``detector_path`` are already-resolved absolute paths (the model
    may have been compiled to an engine first). Extra checkpoints are resolved
    against the uploaded inputs by basename.
    """
    if not isinstance(request, dict):
        raise SpecError("request must be a mapping")

    rewritten = dict(request)
    rewritten["model_checkpoint"] = str(model_path)

    extras = rewritten.get("extra_checkpoints") or []
    if extras:
        if not isinstance(extras, list):
            raise SpecError("extra_checkpoints must be a list")
        rewritten["extra_checkpoints"] = [
            str(_uploaded(inputs, name, f"extra_checkpoints[{index}]"))
            for index, name in enumerate(extras)
        ]

    detector = rewritten.get("detector")
    if isinstance(detector, dict):
        detector = dict(detector)
        if detector_path is None:
            raise SpecError(
                "this request needs a person detector but none was supplied and the "
                "node has no baked one"
            )
        detector["checkpoint"] = str(detector_path)
        rewritten["detector"] = detector

    # chachak requires these to parse; nothing here reads a dataset. Anchoring
    # output_dir inside the scratch dir keeps a build's writes contained.
    rewritten["images"] = "."
    rewritten["labels"] = None
    rewritten["output_dir"] = str(scratch / "runs")
    return rewritten


# ── the build itself ─────────────────────────────────────────────────────────


def _needs_detector(request: Dict[str, Any]) -> bool:
    from chachak.config import _needs_detector as needs

    class _View:
        pipeline = request.get("pipeline")
        chain = request.get("chain") or []

    return needs(_View)


def _run(record: BuildRecord, spec: Dict[str, Any]) -> None:
    """Execute one build. Runs on a worker thread, holding the GPU lock."""
    log = record.append
    inputs = record.scratch / "inputs"
    work = record.scratch / "work"
    work.mkdir(parents=True, exist_ok=True)

    fmt = spec["fmt"]
    precision = spec["precision"]
    request = spec["request"]

    model_onnx = _uploaded(inputs, spec["model_file"], "model_file")
    model_source = model_onnx
    engine_precision = None

    if fmt == "engine":
        # Compile the model FIRST and hand bundle_export the finished engine, rather
        # than letting it compile inside the bundle. Two reasons: it is the same order
        # the local path runs in (export_trt then bundle the artifact), so input_hw is
        # honoured the same way; and bundle_export's own engine build refuses an
        # EfficientNMS arch that isn't a .pt, which is every arch this node handles
        # via a prepared graph.
        log(f"[build] compiling {model_onnx.name} -> model.engine ({precision})")
        model_source = compile_mod.compile_engine(
            model_onnx,
            work / "model.engine",
            precision=precision,
            prepared=bool(spec["model_prepared"]),
            input_hw=spec.get("input_hw"),
        )
        engine_precision = compile_mod.built_precision(model_source, precision)
        if engine_precision != precision:
            log(f"[build] requested {precision}, engine actually built {engine_precision}")

    detector_source: Optional[Path] = None
    if _needs_detector(request):
        if spec["detector_file"]:
            detector_onnx = _uploaded(inputs, spec["detector_file"], "detector_file")
            if fmt == "engine":
                log(f"[build] compiling supplied detector {detector_onnx.name}")
                detector_source = compile_mod.compile_engine(
                    detector_onnx,
                    work / "detector.engine",
                    precision=precision,
                    prepared=bool(spec["detector_prepared"]),
                )
            else:
                detector_source = detector_onnx
        else:
            log("[build] using this node's baked person detector")
            detector_source = person.detector_source(fmt, precision, log=log)

    resolved = materialize_request(
        request,
        inputs=inputs,
        scratch=work,
        model_path=model_source,
        detector_path=detector_source,
    )

    from chachak.bundle_export.cli import export_bundle_for_config
    from chachak.config import pipeline_config_from_dict

    config = pipeline_config_from_dict(resolved, work)
    bundle_dir = work / f"{record.name}-bundle"
    log(f"[build] assembling bundle ({fmt})")
    bundle_dir = Path(
        export_bundle_for_config(
            config,
            bundle_dir,
            fmt=fmt,
            conf=spec.get("conf"),
            precision=precision,
            overwrite=True,
            gpu_infer=bool(spec.get("gpu_infer")),
        )
    )

    log(f"[build] packing {bundle_dir.name}")
    _pack(bundle_dir, record.artifact)

    identity = gpu.describe()
    record.result = {
        "bundle_name": bundle_dir.name,
        "artifact_bytes": record.artifact.stat().st_size,
        "fmt": fmt,
        "requested_precision": precision,
        "precision": engine_precision or precision,
        "node_version": NODE_VERSION,
        # A positive acknowledgement, so the caller can tell "this node honoured the flag"
        # from "this node is too old to know it" without inspecting the bundle.
        "gpu_infer": bool(spec.get("gpu_infer")),
        "gpu_name": identity["gpu_name"],
        "compute_capability": identity["compute_capability"],
        "driver_version": identity["driver_version"],
        "tensorrt_version": identity["tensorrt_version"],
        "person_detector": person.describe() if detector_source else None,
    }
    log(f"[build] done — {record.result['artifact_bytes']} bytes")


def _pack(bundle_dir: Path, target: Path) -> None:
    """tar.gz the bundle, keeping its own directory name as the single root entry."""
    with tarfile.open(target, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)


def _worker(record: BuildRecord, spec: Dict[str, Any]) -> None:
    with _gpu_lock:
        record.status, record.started_at = RUNNING, time.time()
        try:
            _run(record, spec)
            record.status = OK
        except Exception as exc:  # noqa: BLE001 - every failure belongs on the record
            record.status = ERROR
            record.error = f"{type(exc).__name__}: {exc}"
            record.append(f"[build] FAILED — {record.error}")
        finally:
            record.finished_at = time.time()


# ── registry ─────────────────────────────────────────────────────────────────


def submit(name: str, spec: Dict[str, Any], scratch: Path) -> BuildRecord:
    record = BuildRecord(build_id=uuid.uuid4().hex[:16], name=name, scratch=scratch)
    with _registry_lock:
        _builds[record.build_id] = record
    threading.Thread(target=_worker, args=(record, spec), daemon=True).start()
    return record


def get(build_id: str) -> Optional[BuildRecord]:
    with _registry_lock:
        return _builds.get(build_id)


def remove(build_id: str) -> bool:
    with _registry_lock:
        record = _builds.pop(build_id, None)
    if record is None:
        return False
    shutil.rmtree(record.scratch, ignore_errors=True)
    return True


def stats() -> Dict[str, Any]:
    with _registry_lock:
        records = list(_builds.values())
    return {
        "busy": _gpu_lock.locked(),
        "queue_len": sum(1 for r in records if r.status == QUEUED),
        "builds": len(records),
    }


def reap(now: Optional[float] = None) -> int:
    """Drop finished builds past their TTL. Returns how many were removed.

    A bundle is ~100-200 MB and a node is long-lived, so without this a build host
    fills its disk with tarballs nobody is coming back for.
    """
    now = now if now is not None else time.time()
    with _registry_lock:
        stale = [
            build_id
            for build_id, record in _builds.items()
            if record.finished_at is not None and now - record.finished_at > settings.BUILD_TTL
        ]
    return sum(1 for build_id in stale if remove(build_id))


def new_scratch() -> Path:
    scratch = settings.WORK_DIR / f"build-{uuid.uuid4().hex[:12]}"
    (scratch / "inputs").mkdir(parents=True)
    return scratch


def describe_spec(spec: Dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in spec.items() if k != "request"}, sort_keys=True)
