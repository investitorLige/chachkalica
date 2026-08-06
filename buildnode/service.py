"""HTTP surface of a build node.

    GET    /health               -> what this node is and whether it can build
    POST   /builds               -> multipart: spec + model.onnx + model.meta.json
                                    [+ detector.onnx + detector.meta.json]
    GET    /builds/{id}          -> status, log tail, result
    GET    /builds/{id}/artifact -> the bundle, as a tar.gz stream
    DELETE /builds/{id}          -> drop the scratch dir

Every endpoint requires the node's bearer token (see ``auth.py``). Builds are
asynchronous: ``POST /builds`` returns as soon as the uploads are on disk, and the
caller polls — a TensorRT compile plus a bundle assembly runs for minutes, well
past any sensible HTTP timeout.

Run it:  uvicorn buildnode.service:app --host 0.0.0.0 --port 8300
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import NODE_VERSION, builds, gpu, person, settings
from .auth import require_token
from .compile import CompileError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [buildnode] %(levelname)s %(message)s"
)
log = logging.getLogger("buildnode")

app = FastAPI(title="chachkalica build node", version=NODE_VERSION)

_VALID_FMT = ("onnx", "engine")
_VALID_PRECISION = ("fp16", "fp32", "auto")


@app.on_event("startup")
def _startup() -> None:
    settings.WORK_DIR.mkdir(parents=True, exist_ok=True)
    identity = gpu.describe()
    log.info(
        "node up — gpu=%s tensorrt=%s person_detector=%s",
        identity["gpu_name"],
        identity["tensorrt_version"],
        person.available(),
    )
    if not gpu.usable():
        log.warning(
            "no usable GPU/TensorRT visible — /health will report degraded and builds "
            "will fail. Check the nvidia container runtime."
        )
    builds.reap()


# ── health ───────────────────────────────────────────────────────────────────


@app.get("/health", dependencies=[Depends(require_token)])
def health() -> Dict[str, Any]:
    """Identity, capability and current load.

    ``status`` is ``ok`` only when this node could actually take a build right now:
    a visible GPU and an importable TensorRT. Django stores the whole body on the
    BuildNode row, so anything added here shows up in the admin without a change
    on that side.
    """
    identity = gpu.describe()
    load = builds.stats()
    return {
        "status": "ok" if gpu.usable() else "degraded",
        "node_version": NODE_VERSION,
        **identity,
        **load,
        "person_detector": person.describe(),
        "work_dir_free_mb": _free_mb(settings.WORK_DIR),
    }


def _free_mb(path: Path) -> Optional[int]:
    try:
        return shutil.disk_usage(path).free // (1024 * 1024)
    except OSError:
        return None


# ── submitting a build ───────────────────────────────────────────────────────


def _parse_spec(raw: str) -> Dict[str, Any]:
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"spec is not valid JSON: {exc}")
    if not isinstance(spec, dict):
        raise HTTPException(status_code=400, detail="spec must be a JSON object")

    name = str(spec.get("name") or "").strip()
    if not name or name != Path(name).name or name in (".", ".."):
        raise HTTPException(
            status_code=400, detail=f"spec.name must be a plain bundle name, got {name!r}"
        )

    fmt = str(spec.get("fmt") or "engine")
    if fmt not in _VALID_FMT:
        raise HTTPException(status_code=400, detail=f"fmt must be one of {_VALID_FMT}, got {fmt!r}")

    precision = str(spec.get("precision") or "fp16")
    if precision not in _VALID_PRECISION:
        raise HTTPException(
            status_code=400, detail=f"precision must be one of {_VALID_PRECISION}, got {precision!r}"
        )

    input_hw = spec.get("input_hw")
    if input_hw is not None:
        if not isinstance(input_hw, (list, tuple)) or len(input_hw) != 2:
            raise HTTPException(status_code=400, detail="input_hw must be [H, W]")
        try:
            input_hw = (int(input_hw[0]), int(input_hw[1]))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="input_hw must be two integers")
        if any(v <= 0 for v in input_hw):
            raise HTTPException(status_code=400, detail="input_hw values must be positive")

    request = spec.get("request")
    if not isinstance(request, dict):
        raise HTTPException(status_code=400, detail="spec.request must be a chachak request object")

    return {
        "name": name,
        "fmt": fmt,
        "precision": precision,
        "conf": spec.get("conf"),
        "input_hw": input_hw,
        "request": request,
        "model_file": "model.onnx",
        "model_prepared": bool(spec.get("model_prepared", False)),
        "detector_file": None,
        "detector_prepared": bool(spec.get("detector_prepared", False)),
    }


async def _save(upload: UploadFile, target: Path) -> int:
    """Stream an upload to disk, refusing anything past the size cap."""
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with target.open("wb") as handle:
        while chunk := await upload.read(4 * 1024 * 1024):
            written += len(chunk)
            if written > limit:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"{target.name} exceeds BUILDNODE_MAX_UPLOAD_MB "
                    f"({settings.MAX_UPLOAD_MB} MB)",
                )
            handle.write(chunk)
    return written


@app.post("/builds", dependencies=[Depends(require_token)])
async def create_build(
    spec: str = Form(...),
    model: UploadFile = File(...),
    model_meta: UploadFile = File(...),
    detector: Optional[UploadFile] = File(None),
    detector_meta: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Accept a build and start it. Returns immediately; poll ``/builds/{id}``.

    ``model``/``detector`` are ONNX graphs, each with its Contract-B ``.meta.json``.
    A detector is optional: omit it to use the node's baked person detector, which
    is the normal case for a person-crop pipeline.
    """
    if not gpu.usable():
        raise HTTPException(
            status_code=503,
            detail="This node has no usable GPU/TensorRT — see /health. Refusing the "
            "build rather than failing several minutes in.",
        )

    parsed = _parse_spec(spec)
    scratch = builds.new_scratch()
    inputs = scratch / "inputs"

    try:
        # Fixed names on disk: the spec may only name files by basename, and these
        # are the names materialize_request resolves against.
        await _save(model, inputs / "model.onnx")
        await _save(model_meta, inputs / "model.meta.json")
        if detector is not None:
            await _save(detector, inputs / "detector.onnx")
            if detector_meta is None:
                raise HTTPException(
                    status_code=400, detail="a detector upload needs its detector_meta sidecar"
                )
            await _save(detector_meta, inputs / "detector.meta.json")
            parsed["detector_file"] = "detector.onnx"

        # Fail fast on an arch this node can't compile, before taking the GPU.
        if parsed["fmt"] == "engine":
            from .compile import check_buildable, read_meta

            try:
                check_buildable(
                    read_meta(inputs / "model.onnx"), prepared=parsed["model_prepared"]
                )
            except CompileError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(scratch, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"could not accept the upload: {exc}")

    builds.reap()
    record = builds.submit(parsed["name"], parsed, scratch)
    log.info("accepted build %s (%s) %s", record.build_id, record.name, builds.describe_spec(parsed))
    return {"build_id": record.build_id, "status": record.status}


# ── polling, download, cleanup ───────────────────────────────────────────────


def _require(build_id: str) -> builds.BuildRecord:
    record = builds.get(build_id)
    if record is None:
        # 404 is meaningful to the caller: the Django job treats "the node has no
        # record of this" as a terminal unknown rather than retrying forever.
        raise HTTPException(status_code=404, detail=f"no build {build_id!r} on this node")
    return record


@app.get("/builds/{build_id}", dependencies=[Depends(require_token)])
def build_status(build_id: str) -> Dict[str, Any]:
    return _require(build_id).as_dict()


@app.get("/builds/{build_id}/artifact", dependencies=[Depends(require_token)])
def build_artifact(build_id: str):
    record = _require(build_id)
    if record.status != builds.OK:
        raise HTTPException(
            status_code=409,
            detail=f"build {build_id} is {record.status}, not ok — nothing to download",
        )
    if not record.artifact.is_file():
        raise HTTPException(
            status_code=410, detail=f"build {build_id} completed but its artifact was reaped"
        )
    return FileResponse(
        record.artifact,
        media_type="application/gzip",
        filename=f"{record.name}-bundle.tar.gz",
    )


@app.delete("/builds/{build_id}", dependencies=[Depends(require_token)])
def delete_build(build_id: str) -> Dict[str, Any]:
    record = builds.get(build_id)
    if record is not None and record.status == builds.RUNNING:
        raise HTTPException(
            status_code=409, detail=f"build {build_id} is still running; wait for it to finish"
        )
    return {"removed": builds.remove(build_id)}


@app.get("/builds", dependencies=[Depends(require_token)])
def list_builds() -> Dict[str, Any]:
    with builds._registry_lock:  # noqa: SLF001 - same package
        records = [r.as_dict(with_log=False) for r in builds._builds.values()]
    return {"builds": sorted(records, key=lambda r: r["created_at"], reverse=True)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
