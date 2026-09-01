"""Thin HTTP service wrapping the friendy_chachkalica training pipeline.

The trainer is a heavy torch/CUDA library that must run in *this* environment
(its own ``.venv``), so a separate process (e.g. a Django app) can't import it.
This service is the bridge: it accepts a request naming a config YAML already on
the shared filesystem, launches ``run.py`` against it as a subprocess, and
reports status. All real outputs (checkpoints, ``results.yaml``,
``run_summary.yaml``) are written by the trainer to the config's ``output_dir``
— this service only owns *launching* and *liveness*.

Run it from the repo root:

    .venv/bin/uvicorn service:app --host 0.0.0.0 --port 8200

Endpoints:
    GET  /health              -> {"status": "ok"}
    POST /train               -> {run_id, status, pid}    (body: {run_id, config_path, resume?})
    GET  /runs/{run_id}       -> {run_id, status, returncode, started_at, finished_at, log_tail}
    POST /runs/{run_id}/stop  -> {run_id, status, outcome} (gracefully terminate a running train)
    POST /eval                -> {eval_id, status, pid}   (body: {eval_id, request_path})
    GET  /evals/{eval_id}     -> {eval_id, status, returncode, started_at, finished_at, log_tail}
    POST /pipeline            -> {pipeline_id, status, pid} (body: {pipeline_id, request_path})
    GET  /pipelines/{id}      -> {pipeline_id, status, returncode, started_at, finished_at, log_tail}
    POST /predict_image       -> {boxes, classes}          (synchronous 1-image inference; warm model)
    POST /checkpoint_info     -> {arch, trained_size}      (synchronous, cheap: no export)
    POST /export_onnx         -> {onnx_path, meta_path}    (synchronous ONNX export of one checkpoint)
    POST /export_trt_onnx     -> {onnx_path, meta_path, arch, prepared}  (TRT-ready ONNX; CPU-only)
    POST /promote_labels      -> {labels_written, ...}     (synchronous: write a run's predictions as source labels)

Operational logging (what the service itself does — launches, stops, rejections,
errors) is written under ``<repo_root>/logs/``, split three ways: ``train/`` (one
file per run, ``train-{run_id}.log``), ``eval/`` (``eval-{eval_id}.log``), and
``other/service.log`` for everything not tied to a specific job. This is separate
from each subprocess's own stdout, which the trainer keeps writing to
``output_dir/service.log``.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
# Single-GPU safety: refuse a new launch while one is running unless overridden.
MAX_CONCURRENT = int(os.environ.get("FC_MAX_CONCURRENT", "1"))

# The bundle benchmark's result file, written inside the request's output_dir. Named
# here because it is a contract with the caller: Django's ingest job reads exactly
# this path back out of the shared data mount.
BENCHMARK_RESULT_NAME = "benchmark.json"

# --- Operational logging -----------------------------------------------------
# Overridable so containers can point logs at a mounted volume; defaults to the
# repo root (HERE.parent) since the service runs with cwd=HERE.
LOGS_DIR = Path(os.environ.get("FC_LOGS_DIR", HERE.parent / "logs"))
TRAIN_LOG_DIR = LOGS_DIR / "train"
EVAL_LOG_DIR = LOGS_DIR / "eval"
PIPELINE_LOG_DIR = LOGS_DIR / "pipeline"
BENCHMARK_LOG_DIR = LOGS_DIR / "benchmark"
OTHER_LOG_DIR = LOGS_DIR / "other"
_LOG_FMT = "%(asctime)s [%(name)s] %(levelname)s %(message)s"


def _configure_logging() -> None:
    """Create the log dirs and attach a console handler to the ``fc`` root.

    Per-job loggers (``fc.train.*``/``fc.eval.*``) and the service logger
    (``fc.service``) each add their own FileHandler but propagate up to ``fc``
    for the shared console output. ``fc`` itself does not propagate, so records
    don't get duplicated by uvicorn's root logger.
    """
    for d in (TRAIN_LOG_DIR, EVAL_LOG_DIR, PIPELINE_LOG_DIR, BENCHMARK_LOG_DIR,
              OTHER_LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("fc")
    root.setLevel(logging.INFO)
    root.propagate = False
    if not root.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter(_LOG_FMT))
        root.addHandler(stream)


def _logger(name: str, log_path: Path) -> logging.Logger:
    """Return the named logger, attaching a FileHandler to ``log_path`` once.

    Loggers are cached by name by the logging module, so repeated calls (e.g.
    every status poll for the same run) reuse the same handler rather than
    piling up duplicates. FileHandler opens in append mode, so a run's log
    accumulates across relaunches instead of being clobbered.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not any(getattr(h, "_fc_path", None) == str(log_path) for h in logger.handlers):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        setattr(handler, "_fc_path", str(log_path))  # marker so we don't re-add on reuse
        handler.setFormatter(logging.Formatter(_LOG_FMT))
        logger.addHandler(handler)
    return logger


def _service_log() -> logging.Logger:
    return _logger("fc.service", OTHER_LOG_DIR / "service.log")


def _train_log(run_id: int) -> logging.Logger:
    return _logger(f"fc.train.{run_id}", TRAIN_LOG_DIR / f"train-{run_id}.log")


def _eval_log(eval_id: int) -> logging.Logger:
    return _logger(f"fc.eval.{eval_id}", EVAL_LOG_DIR / f"eval-{eval_id}.log")


def _pipeline_log(pipeline_id: int) -> logging.Logger:
    return _logger(f"fc.pipeline.{pipeline_id}", PIPELINE_LOG_DIR / f"pipeline-{pipeline_id}.log")


def _benchmark_log(benchmark_id: int) -> logging.Logger:
    return _logger(f"fc.benchmark.{benchmark_id}",
                   BENCHMARK_LOG_DIR / f"benchmark-{benchmark_id}.log")


_configure_logging()

app = FastAPI(title="friendy_chachkalica trainer")

_lock = threading.Lock()
_jobs: dict[str, dict] = {}  # "train-{id}"/"eval-{id}" -> {proc, pid, started_at, ...}

# Synchronous single-image inference (the interactive model preview) keeps ONE
# built model warm in GPU memory between clicks. _predict_lock serializes both
# the cache and the CUDA call so overlapping preview requests never run on the
# GPU at once; the cache holds a single entry (rebuilding on any key change and
# freeing the old model) to bound GPU memory.
_predict_lock = threading.Lock()
_predict_cache: dict = {}  # {key: {"kind", "obj", "info", "device"}}, size 1

# ONNX export is CPU-only and stateless, but the RT-DETR exporter monkeypatches a
# transformers class attribute for the duration of its trace — serialize exports
# (and keep them off the predict path's toes) with a dedicated lock.
_export_lock = threading.Lock()

# TensorRT engine builds run ON THE GPU (unlike ONNX export) and are memory- and
# compute-heavy, so they contend with active training. A dedicated lock serializes
# builds; operators should avoid building while a training run is using the GPU.
_trt_build_lock = threading.Lock()

_service_log().info("service starting (max_concurrent=%s, logs=%s)", MAX_CONCURRENT, LOGS_DIR)


class TrainRequest(BaseModel):
    run_id: int
    config_path: str
    resume: bool = False


class EvalRequest(BaseModel):
    eval_id: int
    request_path: str


class PipelineRequest(BaseModel):
    pipeline_id: int
    request_path: str


class CheckpointInfoRequest(BaseModel):
    """Inspect a ``.pt`` checkpoint without exporting anything."""

    checkpoint_path: str


class ExportOnnxRequest(BaseModel):
    """Export one ``.pt`` checkpoint to ``<onnx_path>`` + ``<onnx_path>.meta.json``."""

    checkpoint_path: str
    onnx_path: str


class ExportTrtOnnxRequest(BaseModel):
    """Re-export one ``.pt`` as a TensorRT-*ready* ONNX graph at ``<onnx_path>``.

    For the archs whose standard ONNX bakes a data-dependent NMS that TensorRT
    cannot compile (yolox, retinanet, fasterrcnn), this produces the raw-output +
    ``EfficientNMS_TRT`` graph the compiler does accept. Everything else already
    compiles from its standard export and does not need this.
    """

    checkpoint_path: str
    onnx_path: str


class ExportTrtRequest(BaseModel):
    """Build a TensorRT engine at ``<engine_path>`` from one ``.pt`` checkpoint.

    The ONNX export runs first when no sibling ``.onnx`` exists yet (the engine is
    compiled from the ONNX graph). ``precision`` is ``"fp16"`` (default) or ``"fp32"``.
    """

    checkpoint_path: str
    engine_path: str
    precision: str = "fp16"
    # Optional static input size (H, W). When set, the engine is built at a fixed
    # profile min==opt==max==(H,W). Required to get FP16 on Faster R-CNN, whose graph
    # only compiles FP16 at a static size (a dynamic profile falls back to FP32).
    input_hw: Optional[List[int]] = None
    # TensorRT batch profile. Default 1/1/1 (today's behavior for every arch).
    # max_batch > 1 is only accepted for a batch-aware arch (see
    # ml/checkpoint_info.py's inspect_checkpoint) — build_engine raises otherwise
    # rather than silently building a profile the graph can't back correctly.
    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 1


class ExportBundleRequest(BaseModel):
    """Bundle an already-exported model (+ detector, if the pipeline needs one)
    as a self-contained, runnable ``chachak.bundle_export`` pipeline.

    ``request`` is a chachak pipeline request in dict form — same shape as a
    request YAML (``pipeline``, ``model_checkpoint``, ``classes``, ``tiling``,
    ``detector``, ``chain``, ``merge_nms_iou``, ...) — with ``model_checkpoint``
    already pointing at the artifact this call bundles, not the original ``.pt``.
    ``detector.checkpoint``, if present, may still be a raw ``.pt``: this export
    it fresh, since the admin export actions never export the detector on their
    own.
    """

    request: dict
    output_dir: str
    fmt: str = "onnx"
    conf: Optional[float] = None
    precision: str = "auto"
    overwrite: bool = True
    # Opt-in extra: also ship the GPU-only infer_gpu.py entrypoint and the vendored
    # gpu_infer package alongside the ordinary infer.py. Absent from an older caller's
    # payload, which pydantic reads as False -- so the bundle it gets is exactly the
    # bundle it got before this field existed.
    gpu_infer: bool = False


class BenchmarkBundleRequest(BaseModel):
    """Benchmark an exported bundle through the bundle's own vendored runtime.

    Runs here rather than in Django's env for the obvious reason — web/worker have no
    GPU, the trainer owns it — and as a tracked subprocess rather than synchronously,
    because a sweep with warmup, three timing passes and a stage pass per cell takes
    minutes, not seconds.

    ``benchmark_id`` is the caller's row id; it keys the job so status can be polled
    and the run stopped, exactly like ``run_id`` does for ``/train``.
    """

    benchmark_id: int
    bundle_dir: str
    images: str
    output_dir: str
    max_images: Optional[int] = 24
    batch_sizes: str = "1"
    concurrency: str = "1"
    warmup: int = 30
    calls: int = 200
    min_duration_s: float = 2.0
    measure_stages: bool = True
    skip_bare_model: bool = False
    # Opt-in: an extra sequential pass that re-runs one cell under Nsight Systems in a
    # child process and leaves a .nsys-rep plus per-kernel CSVs in the output dir.
    # Absent from an older caller's payload, which pydantic reads as False.
    profile_nsys: bool = False
    device: Optional[str] = None
    fmt: Optional[str] = None
    conf: Optional[float] = None
    detector_conf: Optional[float] = None


class PromoteLabelsRequest(BaseModel):
    """Write a finished run's predictions into a dataset's source ``labels/``.

    ``predictions_path`` is the run's ``*_predictions.pt``. Exactly one of
    ``checkpoint_path`` (the model that produced it — its train classes drive
    the name-based remap onto ``dataset_classes``) or ``prediction_classes`` (a
    combined multi-model run, whose predictions are already indexed in
    ``dataset_classes``' own space, so the remap is an identity lookup and no
    checkpoint needs loading) should be set. Existing ``labels_dir/*.txt`` are
    moved to ``backup_dir`` before the new labels are written.
    """

    predictions_path: str
    checkpoint_path: Optional[str] = None
    prediction_classes: Optional[list[str]] = None
    images_dir: str
    dataset_classes: list[str]
    labels_dir: str
    backup_dir: str
    score_threshold: float = 0.25


class PredictImageRequest(BaseModel):
    """One-image, synchronous inference for the admin preview viewer."""

    model_checkpoint: str
    image_path: str
    pipeline: str = "raw"  # "raw" (adapter only) or a chachak PIPELINE_NAMES value
    detector_checkpoint: Optional[str] = None
    detector_expand_ratio: Optional[float] = None
    detector_min_box_size: Optional[float] = None
    tile_size_px: Optional[int] = None
    tile_width_pct: Optional[float] = None
    tile_height_pct: Optional[float] = None
    overlap: Optional[float] = None
    # Class-aware NMS IoU used to merge predictions across tiles/crops. None leaves
    # chachak's own default (PipelineConfig.merge_nms_iou) in place — the field is
    # parsed with an unconditional float(), so it must be omitted rather than sent
    # as null.
    merge_nms_iou: Optional[float] = None
    chain: Optional[list[str]] = None
    score_threshold: Optional[float] = 0.05
    device: str = "auto"


def _job_status(job: dict) -> str:
    proc: subprocess.Popen = job["proc"]
    rc = proc.poll()
    if rc is None:
        return "running"
    if job.get("finished_at") is None:
        job["finished_at"] = time.time()
    job["returncode"] = rc
    return "ok" if rc == 0 else "error"


def _active_count() -> int:
    return sum(1 for job in _jobs.values() if _job_status(job) == "running")


def _log_tail(log_path: Optional[str], limit: int = 4000) -> str:
    if not log_path or not Path(log_path).exists():
        return ""
    data = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return data[-limit:]


@app.get("/health")
def health():
    return {"status": "ok", "active": _active_count()}


def _spawn(key: str, cmd: list[str], output_dir: Path,
           extra_env: Optional[dict] = None) -> dict:
    """Launch ``cmd`` as a tracked subprocess logging into output_dir/service.log.

    ``extra_env`` is layered over this process's environment — used by the bundle
    benchmark, which runs as ``python -m inferlica.…`` and so needs the repo root on
    ``PYTHONPATH`` (``cwd`` is HERE, one level below it). Absent for every other
    caller, whose environment is unchanged.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "service.log"
    log_file = open(log_path, "w", encoding="utf-8")
    env = None
    if extra_env:
        env = {**os.environ, **{str(k): str(v) for k, v in extra_env.items()}}
    # start_new_session=True puts the child in its own process group so a later
    # stop can signal the whole group (run.py may spawn dataloader/worker procs).
    proc = subprocess.Popen(
        cmd, cwd=str(HERE), stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
    job = {
        "proc": proc,
        "pid": proc.pid,
        "started_at": time.time(),
        "finished_at": None,
        "returncode": None,
        "log_path": str(log_path),
        "output_dir": str(output_dir),
        "_log_file": log_file,
    }
    _jobs[key] = job
    return job


def _signal_proc(proc: subprocess.Popen, sig: int) -> None:
    """Deliver ``sig`` to the child's whole process group when it leads its own
    session (see ``start_new_session`` in _spawn), else to just the child.

    The single-process fallback is a safety net for jobs launched before that
    flag existed: signalling their group would be this service's own group.
    """
    try:
        pgid = os.getpgid(proc.pid)
        if pgid == proc.pid:
            os.killpg(pgid, sig)
            return
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.send_signal(sig)
    except ProcessLookupError:
        pass


def _stop_job(job: dict, grace: float = 10.0) -> str:
    """Gracefully stop a job: SIGTERM, then SIGKILL if it outlasts ``grace``."""
    proc: subprocess.Popen = job["proc"]
    if proc.poll() is not None:
        return "already exited"

    _signal_proc(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
        outcome = "terminated"
    except subprocess.TimeoutExpired:
        _signal_proc(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        outcome = "killed"

    if job.get("finished_at") is None:
        job["finished_at"] = time.time()
    job["returncode"] = proc.returncode
    log_file = job.get("_log_file")
    if log_file and not log_file.closed:
        log_file.close()
    return outcome


def _status_payload(job: dict) -> dict:
    return {
        "status": _job_status(job),
        "pid": job["pid"],
        "returncode": job.get("returncode"),
        "started_at": job["started_at"],
        "finished_at": job.get("finished_at"),
        "output_dir": job["output_dir"],
        "log_tail": _log_tail(job.get("log_path")),
    }


@app.post("/train")
def train(req: TrainRequest):
    log = _train_log(req.run_id)
    config_path = Path(req.config_path)
    if not config_path.exists():
        log.error("train rejected: config not found: %s", config_path)
        raise HTTPException(status_code=400, detail=f"config not found: {config_path}")

    with _lock:
        key = f"train-{req.run_id}"
        existing = _jobs.get(key)
        if existing and _job_status(existing) == "running":
            log.info("train already running (pid=%s); ignoring duplicate launch", existing["pid"])
            return {"run_id": req.run_id, "status": "running", "pid": existing["pid"]}
        if _active_count() >= MAX_CONCURRENT:
            log.warning("train rejected: trainer busy (%d active >= %d max)",
                        _active_count(), MAX_CONCURRENT)
            raise HTTPException(status_code=409, detail="trainer busy: another run is active")

        try:
            output_dir = Path(yaml.safe_load(config_path.read_text())["output_dir"])
        except Exception as exc:  # noqa: BLE001
            log.error("train rejected: bad config %s: %s", config_path, exc)
            raise HTTPException(status_code=400, detail=f"bad config: {exc}")

        cmd = [sys.executable, "run.py", str(config_path)]
        if req.resume:
            cmd.append("--resume")
        job = _spawn(key, cmd, output_dir)
        log.info("launched train: pid=%s config=%s resume=%s output_dir=%s",
                 job["pid"], config_path, req.resume, output_dir)
        return {"run_id": req.run_id, "status": "running", "pid": job["pid"]}


@app.get("/runs/{run_id}")
def run_status(run_id: int):
    job = _jobs.get(f"train-{run_id}")
    if job is None:
        raise HTTPException(status_code=404, detail="unknown run_id")
    return {"run_id": run_id, **_status_payload(job)}


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: int, grace: float = 10.0):
    log = _train_log(run_id)
    with _lock:
        job = _jobs.get(f"train-{run_id}")
        if job is None:
            log.warning("stop requested for unknown run")
            raise HTTPException(status_code=404, detail="unknown run_id")
        log.info("stop requested (grace=%ss)", grace)
        outcome = _stop_job(job, grace=grace)
    log.info("stop outcome=%s returncode=%s", outcome, job.get("returncode"))
    return {"run_id": run_id, "status": _job_status(job), "outcome": outcome,
            "returncode": job.get("returncode")}


@app.post("/eval")
def evaluate(req: EvalRequest):
    log = _eval_log(req.eval_id)
    request_path = Path(req.request_path)
    if not request_path.exists():
        log.error("eval rejected: request not found: %s", request_path)
        raise HTTPException(status_code=400, detail=f"eval request not found: {request_path}")

    with _lock:
        key = f"eval-{req.eval_id}"
        existing = _jobs.get(key)
        if existing and _job_status(existing) == "running":
            log.info("eval already running (pid=%s); ignoring duplicate launch", existing["pid"])
            return {"eval_id": req.eval_id, "status": "running", "pid": existing["pid"]}
        if _active_count() >= MAX_CONCURRENT:
            log.warning("eval rejected: trainer busy (%d active >= %d max)",
                        _active_count(), MAX_CONCURRENT)
            raise HTTPException(status_code=409, detail="trainer busy: another job is active")

        try:
            output_dir = Path(yaml.safe_load(request_path.read_text())["output_dir"])
        except Exception as exc:  # noqa: BLE001
            log.error("eval rejected: bad eval request %s: %s", request_path, exc)
            raise HTTPException(status_code=400, detail=f"bad eval request: {exc}")

        cmd = [sys.executable, "ml/eval_checkpoint.py", str(request_path)]
        job = _spawn(key, cmd, output_dir)
        log.info("launched eval: pid=%s request=%s output_dir=%s",
                 job["pid"], request_path, output_dir)
        return {"eval_id": req.eval_id, "status": "running", "pid": job["pid"]}


@app.get("/evals/{eval_id}")
def eval_status(eval_id: int):
    job = _jobs.get(f"eval-{eval_id}")
    if job is None:
        raise HTTPException(status_code=404, detail="unknown eval_id")
    return {"eval_id": eval_id, **_status_payload(job)}


@app.post("/pipeline")
def run_pipeline(req: PipelineRequest):
    """Run a chachak inference/eval pipeline against a generated request YAML.

    chachak lives beside this package (``<repo_root>/chachak``) and shares this
    torch/CUDA environment; ``run.py`` bootstraps its own imports, so we spawn it
    exactly like the eval subprocess.
    """
    log = _pipeline_log(req.pipeline_id)
    request_path = Path(req.request_path)
    if not request_path.exists():
        log.error("pipeline rejected: request not found: %s", request_path)
        raise HTTPException(status_code=400, detail=f"pipeline request not found: {request_path}")

    with _lock:
        key = f"pipeline-{req.pipeline_id}"
        existing = _jobs.get(key)
        if existing and _job_status(existing) == "running":
            log.info("pipeline already running (pid=%s); ignoring duplicate launch",
                     existing["pid"])
            return {"pipeline_id": req.pipeline_id, "status": "running", "pid": existing["pid"]}
        if _active_count() >= MAX_CONCURRENT:
            log.warning("pipeline rejected: trainer busy (%d active >= %d max)",
                        _active_count(), MAX_CONCURRENT)
            raise HTTPException(status_code=409, detail="trainer busy: another job is active")

        try:
            output_dir = Path(yaml.safe_load(request_path.read_text())["output_dir"])
        except Exception as exc:  # noqa: BLE001
            log.error("pipeline rejected: bad request %s: %s", request_path, exc)
            raise HTTPException(status_code=400, detail=f"bad pipeline request: {exc}")

        chachak_run = HERE.parent / "chachak" / "run.py"
        cmd = [sys.executable, str(chachak_run), str(request_path)]
        job = _spawn(key, cmd, output_dir)
        log.info("launched pipeline: pid=%s request=%s output_dir=%s",
                 job["pid"], request_path, output_dir)
        return {"pipeline_id": req.pipeline_id, "status": "running", "pid": job["pid"]}


@app.get("/pipelines/{pipeline_id}")
def pipeline_status(pipeline_id: int):
    job = _jobs.get(f"pipeline-{pipeline_id}")
    if job is None:
        raise HTTPException(status_code=404, detail="unknown pipeline_id")
    return {"pipeline_id": pipeline_id, **_status_payload(job)}


def _ensure_chachak_importable() -> None:
    """Put the repo root on sys.path so ``import chachak`` resolves in-process.

    chachak lives at ``<repo_root>/chachak`` and shares this torch/CUDA env (the
    ``/pipeline`` endpoint already spawns ``chachak/run.py`` in it). For the
    synchronous preview we import it here instead of spawning, to keep the model
    warm across requests.
    """
    root = str(HERE.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def _predict_key(req: "PredictImageRequest") -> tuple:
    return (
        req.model_checkpoint,
        req.detector_checkpoint or "",
        req.detector_expand_ratio,
        req.detector_min_box_size,
        req.pipeline,
        req.tile_size_px,
        req.tile_width_pct,
        req.tile_height_pct,
        req.overlap,
        req.merge_nms_iou,
        tuple(req.chain or []),
        req.score_threshold,
        req.device,
    )


def _build_predict_runtime(req: "PredictImageRequest", device) -> dict:
    """Load the model (raw adapter or full chachak pipeline) for ``req``."""
    from chachak.config import pipeline_config_from_dict
    from chachak.infer import load_checkpoint_adapter
    from chachak.run import build_pipeline_runtime

    if req.pipeline == "raw":
        adapter, info = load_checkpoint_adapter(req.model_checkpoint, device)
        return {"kind": "raw", "obj": adapter, "info": info}

    # chachak's PipelineConfig requires images/labels/output_dir/classes, but the
    # single-image predict path never touches the dataloader and reads class
    # names from the checkpoint — so these are harmless placeholders.
    raw = {
        "pipeline": req.pipeline,
        "model_checkpoint": req.model_checkpoint,
        "images": ".",
        "labels": ".",
        "output_dir": ".",
        "classes": ["_"],
        "device": req.device,
    }
    if req.score_threshold is not None:
        raw["score_threshold"] = req.score_threshold
    if req.detector_checkpoint:
        detector: dict = {"checkpoint": req.detector_checkpoint}
        if req.detector_expand_ratio is not None:
            detector["expand_ratio"] = req.detector_expand_ratio
        if req.detector_min_box_size is not None:
            detector["min_box_size"] = req.detector_min_box_size
        raw["detector"] = detector
    tiling: dict = {}
    if req.tile_size_px:
        tiling["tile_size_px"] = req.tile_size_px
    if req.tile_width_pct:
        tiling["tile_width_pct"] = req.tile_width_pct
    if req.tile_height_pct:
        tiling["tile_height_pct"] = req.tile_height_pct
    if req.overlap is not None:
        tiling["overlap"] = req.overlap
    if tiling:
        raw["tiling"] = tiling
    if req.merge_nms_iou is not None:
        raw["merge_nms_iou"] = req.merge_nms_iou
    if req.chain:
        raw["chain"] = list(req.chain)

    config = pipeline_config_from_dict(raw, HERE.parent)
    pipeline, info = build_pipeline_runtime(config, device)
    return {"kind": "pipeline", "obj": pipeline, "info": info}


def _get_predict_runtime(req: "PredictImageRequest") -> dict:
    """Return the warm runtime for ``req``, (re)building on a key change.

    Caller must hold ``_predict_lock``.
    """
    key = _predict_key(req)
    entry = _predict_cache.get(key)
    if entry is not None:
        return entry

    import torch

    from chachak._friendy import resolve_device

    _predict_cache.clear()  # size 1: drop the previous model before loading a new one
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = resolve_device(req.device)
    entry = _build_predict_runtime(req, device)
    entry["device"] = device
    _predict_cache[key] = entry
    return entry


@app.post("/predict_image")
def predict_image(req: PredictImageRequest):
    """Run one image through the model synchronously and return its boxes.

    Boxes are Friendy normalized center-xywh dicts
    (``{cx, cy, w, h, confidence, class_id, class_name}``). The first call for a
    given model/pipeline loads it (slow); subsequent calls reuse the warm model.
    """
    log = _service_log()
    image_path = Path(req.image_path)
    if not image_path.exists():
        log.error("predict rejected: image not found: %s", image_path)
        raise HTTPException(status_code=400, detail=f"image not found: {image_path}")
    if not Path(req.model_checkpoint).exists():
        log.error("predict rejected: checkpoint not found: %s", req.model_checkpoint)
        raise HTTPException(
            status_code=400, detail=f"checkpoint not found: {req.model_checkpoint}")

    _ensure_chachak_importable()
    with _predict_lock:
        try:
            entry = _get_predict_runtime(req)
            from chachak.preview import predict_one, predict_one_raw

            if entry["kind"] == "raw":
                boxes = predict_one_raw(
                    entry["obj"], entry["info"], image_path,
                    entry["device"], req.score_threshold)
            else:
                boxes = predict_one(
                    entry["obj"], entry["info"], image_path,
                    entry["device"], req.score_threshold)
        except HTTPException:
            raise
        except ValueError as exc:
            log.warning("predict rejected: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("predict failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"predict failed: {exc}")

    return {"boxes": boxes, "classes": entry["info"].get("train_classes", {})}


@app.post("/checkpoint_info")
def checkpoint_info(req: CheckpointInfoRequest):
    """Inspect a checkpoint's arch + trained input size, without exporting anything.

    Used to prefill the ONNX/TensorRT export forms' static input size — cheap
    (``torch.load`` only, see ``ml/checkpoint_info.py``), so it's safe to call
    synchronously while rendering a form. No lock: read-only and CPU-only.
    """
    log = _service_log()
    checkpoint = Path(req.checkpoint_path)
    if not checkpoint.exists():
        log.error("checkpoint_info rejected: checkpoint not found: %s", checkpoint)
        raise HTTPException(status_code=400, detail=f"checkpoint not found: {checkpoint}")

    try:
        from ml.checkpoint_info import inspect_checkpoint

        return inspect_checkpoint(checkpoint)
    except Exception as exc:  # noqa: BLE001 - surface inspection failures to the caller
        log.exception("checkpoint_info failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"checkpoint_info failed: {exc}")


@app.post("/export_onnx")
def export_onnx(req: ExportOnnxRequest):
    """Export a checkpoint to ONNX + meta.json synchronously.

    Rebuilds the adapter from the ``.pt`` and dispatches to the arch's exporter
    (see ``ml/onnx_export/cli.py``), writing ``<onnx_path>`` and its sibling
    ``.meta.json``. Runs on CPU — safe to call while the GPU is training.
    """
    log = _service_log()
    checkpoint = Path(req.checkpoint_path)
    if not checkpoint.exists():
        log.error("export rejected: checkpoint not found: %s", checkpoint)
        raise HTTPException(status_code=400, detail=f"checkpoint not found: {checkpoint}")

    with _export_lock:
        try:
            from ml.onnx_export.cli import export_checkpoint

            onnx_path = export_checkpoint(req.checkpoint_path, req.onnx_path)
        except HTTPException:
            raise
        except ValueError as exc:
            log.warning("export rejected: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - surface export failures to the caller
            log.exception("export failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"export failed: {exc}")

    meta_path = onnx_path.with_suffix(".meta.json")
    log.info("exported %s -> %s", checkpoint, onnx_path)
    return {"onnx_path": str(onnx_path), "meta_path": str(meta_path)}


@app.post("/export_trt_onnx")
def export_trt_onnx(req: ExportTrtOnnxRequest):
    """Produce the TensorRT-ready ONNX for a checkpoint, without building anything.

    This exists for the remote build nodes (``buildnode/``). A node compiles
    engines without torch, which works for rtdetr/rfdetr/ecdet — they compile
    straight from their standard ONNX — but not for yolox, retinanet or
    fasterrcnn, whose graphs must first be re-exported as raw outputs plus an
    ``EfficientNMS_TRT`` node. That re-export runs off the torch model, so it has
    to happen here.

    CPU-only and GPU-free, exactly like ``/export_onnx``: this is a torch ONNX
    export, not a compile. It takes ``_export_lock`` rather than
    ``_trt_build_lock`` so it does not queue behind an engine build, and is safe
    to call while the GPU is training.

    Returns ``{onnx_path, meta_path, arch, prepared}``. ``prepared`` is False for
    an arch that needs no re-export — the standard ONNX was copied through
    unchanged, and the caller can send it to a node as-is.
    """
    log = _service_log()
    checkpoint = Path(req.checkpoint_path)
    if not checkpoint.exists():
        log.error("trt-onnx export rejected: checkpoint not found: %s", checkpoint)
        raise HTTPException(status_code=400, detail=f"checkpoint not found: {checkpoint}")

    with _export_lock:
        try:
            from ml.checkpoint_info import inspect_checkpoint
            from ml.onnx_export.cli import export_checkpoint
            from ml.trt_export.arch import get_trt_prep, has_trt_prep
            from ml.trt_export.cli import _load_adapter

            onnx_path = Path(req.onnx_path)
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            arch = inspect_checkpoint(checkpoint).get("arch")

            if not has_trt_prep(arch):
                # Nothing to prepare — the standard export is already compilable.
                written = export_checkpoint(str(checkpoint), str(onnx_path))
                return {
                    "onnx_path": str(written),
                    "meta_path": str(Path(written).with_suffix(".meta.json")),
                    "arch": arch,
                    "prepared": False,
                }

            # The prep needs the meta the standard exporter writes, so export once
            # to a scratch name first and carry its sidecar over verbatim.
            staging = onnx_path.with_name(onnx_path.stem + ".standard.onnx")
            export_checkpoint(str(checkpoint), str(staging))
            meta = json.loads(staging.with_suffix(".meta.json").read_text())

            get_trt_prep(arch)(_load_adapter(checkpoint), meta, onnx_path)
            onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

            # The scratch standard export and the prep's own raw intermediate are
            # both inputs, not outputs; leaving them would double the disk cost of
            # every remote build.
            staging.unlink(missing_ok=True)
            staging.with_suffix(".meta.json").unlink(missing_ok=True)
            raw = onnx_path.with_name(onnx_path.name.replace(".onnx", ".raw.onnx"))
            raw.unlink(missing_ok=True)
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as exc:
            log.warning("trt-onnx export rejected: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - surface export failures to the caller
            log.exception("trt-onnx export failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"trt-onnx export failed: {exc}")

    log.info("prepared TRT onnx %s -> %s (%s)", checkpoint, onnx_path, arch)
    return {
        "onnx_path": str(onnx_path),
        "meta_path": str(onnx_path.with_suffix(".meta.json")),
        "arch": arch,
        "prepared": True,
    }


@app.post("/export_trt")
def export_trt(req: ExportTrtRequest):
    """Build a TensorRT engine from a checkpoint synchronously.

    Ensures the ONNX artifact exists (exporting it first if needed), then compiles
    it into ``<engine_path>`` via ``ml/trt_export/cli.py``, alongside a verbatim
    ``.meta.json`` copy and a ``.engine.json`` provenance sidecar.

    UNLIKE ``/export_onnx`` (CPU-only, safe while the GPU is training), this runs
    ON THE GPU and competes with active training — serialized behind
    ``_trt_build_lock``. It also requires TensorRT installed in this env and a
    visible GPU; the produced engine is tied to that exact GPU + TRT version.
    """
    log = _service_log()
    checkpoint = Path(req.checkpoint_path)
    if not checkpoint.exists():
        log.error("trt build rejected: checkpoint not found: %s", checkpoint)
        raise HTTPException(status_code=400, detail=f"checkpoint not found: {checkpoint}")
    if req.precision not in ("fp16", "fp32"):
        raise HTTPException(status_code=400, detail=f"precision must be fp16 or fp32, got {req.precision!r}")
    if not (1 <= req.min_batch <= req.opt_batch <= req.max_batch):
        raise HTTPException(
            status_code=400,
            detail=(
                f"batch profile must satisfy 1 <= min_batch <= opt_batch <= max_batch, "
                f"got min={req.min_batch} opt={req.opt_batch} max={req.max_batch}"
            ),
        )

    static_hw = None
    if req.input_hw is not None:
        if len(req.input_hw) != 2 or any(int(v) <= 0 for v in req.input_hw):
            raise HTTPException(
                status_code=400,
                detail=f"input_hw must be two positive ints [H, W], got {req.input_hw!r}")
        static_hw = (int(req.input_hw[0]), int(req.input_hw[1]))

    with _trt_build_lock:
        try:
            from ml.trt_export.cli import build_engine

            # A static profile (min==opt==max) is what lets Faster R-CNN compile FP16;
            # when input_hw is omitted the profile is derived from the meta (dynamic).
            hw_kwargs = (
                {"min_hw": static_hw, "opt_hw": static_hw, "max_hw": static_hw}
                if static_hw is not None else {}
            )
            engine_path = build_engine(
                req.checkpoint_path, req.engine_path, precision=req.precision,
                min_batch=req.min_batch, opt_batch=req.opt_batch, max_batch=req.max_batch,
                **hw_kwargs,
            )
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as exc:
            log.warning("trt build rejected: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))
        except ModuleNotFoundError as exc:  # tensorrt not installed in this env
            log.error("trt build unavailable: %s", exc)
            raise HTTPException(
                status_code=501,
                detail=f"TensorRT not available in the trainer env: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - surface build failures to the caller
            log.exception("trt build failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"trt build failed: {exc}")

    engine_path = Path(engine_path)
    meta_path = engine_path.with_suffix(".meta.json")
    provenance_path = Path(str(engine_path) + ".json")
    # Report the precision the engine ACTUALLY built at — an fp16 request can fall back
    # to fp32 (e.g. Faster R-CNN with a dynamic profile), and the caller should see that.
    built_precision = req.precision
    try:
        built_precision = json.loads(provenance_path.read_text()).get("precision", req.precision)
    except Exception:  # noqa: BLE001 - provenance is best-effort for the message
        pass
    log.info("built engine %s -> %s (%s)", checkpoint, engine_path, built_precision)
    return {
        "engine_path": str(engine_path),
        "meta_path": str(meta_path),
        "provenance_path": str(provenance_path),
        "precision": built_precision,
    }


@app.post("/export_bundle")
def export_bundle(req: ExportBundleRequest):
    """Assemble a self-contained inference bundle synchronously.

    Runs in this env (not Django's) because it can compile a TensorRT engine for
    the person detector and, for an engine bundle, reads back an engine's batch
    profile — both need the GPU/TensorRT this service owns. Serialized behind
    ``_trt_build_lock`` for an engine bundle, ``_export_lock`` otherwise, same as
    ``/export_trt`` / ``/export_onnx``.
    """
    log = _service_log()
    if req.fmt not in ("onnx", "engine"):
        raise HTTPException(status_code=400, detail=f"fmt must be onnx or engine, got {req.fmt!r}")

    _ensure_chachak_importable()
    lock = _trt_build_lock if req.fmt == "engine" else _export_lock
    with lock:
        try:
            from chachak.bundle_export.cli import export_bundle_for_config
            from chachak.config import pipeline_config_from_dict

            config = pipeline_config_from_dict(dict(req.request), HERE.parent)
            bundle_dir = export_bundle_for_config(
                config, req.output_dir, fmt=req.fmt, conf=req.conf,
                precision=req.precision, overwrite=req.overwrite,
                gpu_infer=req.gpu_infer,
            )
        except HTTPException:
            raise
        except (ValueError, SystemExit) as exc:
            log.warning("bundle export rejected: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))
        except ModuleNotFoundError as exc:  # tensorrt not installed in this env
            log.error("bundle export unavailable: %s", exc)
            raise HTTPException(
                status_code=501,
                detail=f"TensorRT not available in the trainer env: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - surface bundling failures to the caller
            log.exception("bundle export failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"bundle export failed: {exc}")

    log.info("bundled %s -> %s", config.model_checkpoint, bundle_dir)
    return {"bundle_dir": str(bundle_dir)}


@app.post("/benchmark_bundle")
def benchmark_bundle(req: BenchmarkBundleRequest):
    """Launch a bundle benchmark as a tracked subprocess.

    Gated on ``MAX_CONCURRENT`` like ``/train`` and counted in ``_active_count``, and
    that is not mere tidiness: a benchmark sharing the GPU with a training run does
    not fail, it produces plausible numbers that are simply wrong. (The harness also
    records a co-tenant census and stamps the result ``contended`` for whatever it
    cannot prevent — another stack on the same card, say.)
    """
    log = _benchmark_log(req.benchmark_id)
    bundle_dir = Path(req.bundle_dir)
    images = Path(req.images)
    if not (bundle_dir / "pipeline.json").is_file():
        log.error("benchmark rejected: not a bundle: %s", bundle_dir)
        raise HTTPException(status_code=400,
                            detail=f"not a bundle (no pipeline.json): {bundle_dir}")
    if not images.exists():
        log.error("benchmark rejected: images not found: %s", images)
        raise HTTPException(status_code=400, detail=f"images not found: {images}")

    harness = HERE.parent / "inferlica" / "benchmark" / "bundle_bench.py"
    if not harness.is_file():
        log.error("benchmark unavailable: harness missing at %s", harness)
        raise HTTPException(
            status_code=501,
            detail=f"benchmark harness not present in this image: {harness}",
        )

    with _lock:
        key = f"bench-{req.benchmark_id}"
        existing = _jobs.get(key)
        if existing and _job_status(existing) == "running":
            log.info("benchmark already running (pid=%s); ignoring duplicate launch",
                     existing["pid"])
            return {"benchmark_id": req.benchmark_id, "status": "running",
                    "pid": existing["pid"], "output_dir": existing["output_dir"]}
        if _active_count() >= MAX_CONCURRENT:
            log.warning("benchmark rejected: trainer busy (%d active >= %d max)",
                        _active_count(), MAX_CONCURRENT)
            raise HTTPException(status_code=409,
                                detail="trainer busy: another job is active")

        output_dir = Path(req.output_dir)
        cmd = [
            sys.executable, "-m", "inferlica.benchmark.bundle_bench",
            "--bundle", str(bundle_dir),
            "--images", str(images),
            "--out", str(output_dir / BENCHMARK_RESULT_NAME),
            "--batch-sizes", req.batch_sizes,
            "--concurrency", req.concurrency,
            "--warmup", str(req.warmup),
            "--calls", str(req.calls),
            "--min-duration", str(req.min_duration_s),
        ]
        if req.max_images:
            cmd += ["--max-images", str(req.max_images)]
        if not req.measure_stages:
            cmd.append("--no-stages")
        if req.skip_bare_model:
            cmd.append("--no-bare-model")
        if req.profile_nsys:
            cmd.append("--nsys")
        if req.device:
            cmd += ["--device", req.device]
        if req.fmt:
            cmd += ["--format", req.fmt]
        if req.conf is not None:
            cmd += ["--conf", str(req.conf)]
        if req.detector_conf is not None:
            cmd += ["--detector-conf", str(req.detector_conf)]

        # cwd is HERE (see _spawn), whose parent holds inferlica/ -- so `-m` needs
        # the repo root on sys.path, the same relationship chachak already relies on.
        env_path = str(HERE.parent)
        job = _spawn(key, cmd, output_dir, extra_env={"PYTHONPATH": env_path})
        log.info("launched benchmark: pid=%s bundle=%s images=%s batches=%s streams=%s",
                 job["pid"], bundle_dir, images, req.batch_sizes, req.concurrency)
        return {"benchmark_id": req.benchmark_id, "status": "running",
                "pid": job["pid"], "output_dir": str(output_dir)}


@app.get("/benchmarks/{benchmark_id}")
def benchmark_status(benchmark_id: int):
    job = _jobs.get(f"bench-{benchmark_id}")
    if job is None:
        raise HTTPException(status_code=404, detail="unknown benchmark_id")
    payload = {"benchmark_id": benchmark_id, **_status_payload(job)}
    result_path = Path(job["output_dir"]) / BENCHMARK_RESULT_NAME
    payload["result_path"] = str(result_path) if result_path.is_file() else None
    return payload


@app.post("/benchmarks/{benchmark_id}/stop")
def stop_benchmark(benchmark_id: int, grace: float = 10.0):
    log = _benchmark_log(benchmark_id)
    with _lock:
        job = _jobs.get(f"bench-{benchmark_id}")
        if job is None:
            log.warning("stop requested for unknown benchmark")
            raise HTTPException(status_code=404, detail="unknown benchmark_id")
        log.info("stop requested (grace=%ss)", grace)
        outcome = _stop_job(job, grace=grace)
    log.info("stop outcome=%s returncode=%s", outcome, job.get("returncode"))
    return {"benchmark_id": benchmark_id, "status": _job_status(job),
            "outcome": outcome, "returncode": job.get("returncode")}


@app.post("/promote_labels")
def promote_labels(req: PromoteLabelsRequest):
    """Promote a run's predictions into a dataset's source ``labels/`` folder.

    Reads the run's ``*_predictions.pt`` (needs torch — hence trainer-side),
    remaps prediction classes onto ``dataset_classes`` by name, keeps boxes at or
    above ``score_threshold``, backs up any existing labels into ``backup_dir``,
    then writes one YOLO ``.txt`` per image. CPU-only file work — safe while the
    GPU is training — but serialized behind ``_export_lock`` since it torch.loads
    a checkpoint. Returns the write summary from :func:`promote_labels.promote_labels`.
    """
    log = _service_log()
    predictions_path = Path(req.predictions_path)
    if not predictions_path.exists():
        log.error("promote rejected: predictions not found: %s", predictions_path)
        raise HTTPException(status_code=400, detail=f"predictions not found: {predictions_path}")
    if req.prediction_classes is None:
        if not req.checkpoint_path:
            log.error("promote rejected: neither checkpoint_path nor prediction_classes given")
            raise HTTPException(
                status_code=400, detail="either checkpoint_path or prediction_classes is required"
            )
        checkpoint = Path(req.checkpoint_path)
        if not checkpoint.exists():
            log.error("promote rejected: checkpoint not found: %s", checkpoint)
            raise HTTPException(status_code=400, detail=f"checkpoint not found: {checkpoint}")

    with _export_lock:
        try:
            from promote_labels import promote_labels as _promote

            summary = _promote(
                predictions_path=req.predictions_path,
                checkpoint_path=req.checkpoint_path,
                images_dir=req.images_dir,
                dataset_classes=req.dataset_classes,
                labels_dir=req.labels_dir,
                backup_dir=req.backup_dir,
                score_threshold=req.score_threshold,
                prediction_classes=req.prediction_classes,
            )
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as exc:
            log.warning("promote rejected: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - surface promote failures to the caller
            log.exception("promote failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"promote failed: {exc}")

    log.info("promoted %s -> %s (%s labels, %s boxes, %s backed up)",
             predictions_path, summary["labels_dir"], summary["labels_written"],
             summary["boxes_written"], summary["backed_up"])
    return summary


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("FC_PORT", "8200")))
