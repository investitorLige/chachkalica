"""Benchmark an exported infer bundle through the bundle's own vendored runtime.

    python -m inferlica.benchmark.bundle_bench \
        --bundle chachkalica/data/bundles/<name>-bundle \
        --images /app/data/source/dataset1/images \
        --out /tmp/benchmark.json

This is deliberately *not* the sweep in :mod:`inferlica.benchmark.core`. That one
builds architecture variants from random-init weights and feeds them synthetic
tensors, to answer "which arch should we pick". This one answers "how does the thing
we are about to deploy behave": it imports the bundle's own ``infer.py``, calls the
same ``build_runtime`` / ``run_batch`` an operator would, and decodes real images —
so JPEG decode, the person-detector crop path and the class-aware NMS are all in the
measurement, at the detection counts real frames produce.

**Four instruments, and the disagreement is the point.** No single tool covers this:

  T1  ``perf_counter`` per iteration, synchronized      end-to-end latency
  T2  ``torch.cuda.Event`` per iteration                device-side latency
  T3  ``torch.profiler`` / CUPTI, separate pass         device busy time (sees
                                                        onnxruntime *and* TensorRT
                                                        kernels — TRT runs on
                                                        torch's own stream)
  T4  polygraphy (engine) / a bare onnxruntime session  the model artifact alone,
      (onnx), outside this harness entirely             with no bundle around it

:func:`consensus` turns the spread into a verdict rather than an average — T1 ≫ T2
means host-bound, T3 ≪ T2 means the device is idle inside the call, T4 ≪ T2 means the
pipeline around the model costs more than the model.

**Three separate passes, because they measure different things.** Latency and
throughput cannot come from one loop: a per-iteration synchronize is required to make
a latency sample mean anything, and it also destroys the pipelining that sets
throughput. So loop A runs unsynchronized for throughput (the same discipline as
``core.time_predict``'s "the ONLY loop that determines fps"), loop B synchronizes per
iteration for latency percentiles, and the stage breakdown gets a third pass whose
instrumentation overhead is reported rather than hidden.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # module or flat-script, same shim as the rest of this package
    from . import gpu_monitor
    from .bundle_stages import BUNDLE_INFER_MODULE, StageRecorder, load_stage_plan
except ImportError:  # pragma: no cover - flat-script invocation
    import gpu_monitor  # type: ignore
    from bundle_stages import (  # type: ignore
        BUNDLE_INFER_MODULE, StageRecorder, load_stage_plan,
    )

SCHEMA_VERSION = 1

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# The bundle vendors its own copy of these. If the ambient environment already has
# them imported (the trainer container has /app/chachak, /app/onnx_infer,
# /app/trt_infer on the path), a cached module would win over the bundle's copy and
# we would benchmark the wrong code while believing otherwise.
_VENDORED_ROOTS = ("chachak", "onnx_infer", "trt_infer", "gpu_infer", "bundle_manifest")

# Loop A must run long enough for the NVML utilization counter (~1 Hz driver refresh)
# to produce more than a couple of samples. Same floor the console sweep uses.
DEFAULT_MIN_DURATION_S = 2.0

_POLYGRAPHY_AVG_RE = re.compile(r"Average inference time:\s*([0-9.eE+-]+)\s*ms")


# ── bundle loading ──────────────────────────────────────────────────────────────


def _isolate_bundle_runtime(bundle_dir: Path) -> None:
    """Make the bundle's own ``runtime/`` the only source of the vendored packages.

    Popping the cached modules is not optional: ``import chachak`` resolves from
    ``sys.modules`` first, so in the trainer container (which ships its own
    ``/app/chachak``) a previously-imported copy would silently shadow the bundle's
    frozen one — and the whole point of benchmarking a bundle is to measure the code
    that ships inside it.
    """
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in _VENDORED_ROOTS:
            del sys.modules[name]
    runtime = str((bundle_dir / "runtime").resolve())
    while runtime in sys.path:
        sys.path.remove(runtime)
    sys.path.insert(0, runtime)


def _import_bundle_infer(bundle_dir: Path):
    """Import the bundle's ``infer.py`` as a module, under a private name."""
    infer_path = bundle_dir / "infer.py"
    if not infer_path.is_file():
        raise SystemExit(f"[bench] not a bundle: no infer.py in {bundle_dir}")
    _isolate_bundle_runtime(bundle_dir)
    spec = importlib.util.spec_from_file_location(BUNDLE_INFER_MODULE, infer_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[bench] could not load {infer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Bundle:
    """A loaded bundle plus everything needed to build a config at any batch size."""

    directory: Path
    infer: Any
    manifest_contract: Any
    manifest: dict
    fmt: str
    paths: Dict[str, Optional[Path]]
    device: Any
    device_str: str
    conf: Optional[float]
    detector_conf: Optional[float]

    def config_for(self, batch_size: Optional[int]):
        """A validated ``PipelineConfig`` for one batch size.

        ``pipeline_request_from_manifest`` is what applies the artifacts' own batch
        cap, so ``config.infer_batch_size`` is the *effective* batch — read back
        rather than assumed. Over-submitting a TensorRT engine's profile used to
        silently return image 0's boxes for every image in the batch; it raises now,
        but the sweep still has no business attempting it.
        """
        request = self.manifest_contract.pipeline_request_from_manifest(
            self.manifest,
            self.paths,
            fmt=self.fmt,
            images=".",
            output_dir=self.directory,
            conf=self.conf,
            detector_conf=self.detector_conf,
            device=self.device_str,
            batch_size=batch_size,
        )
        from chachak.config import pipeline_config_from_dict

        return pipeline_config_from_dict(request, self.directory)

    @property
    def max_batch(self) -> Optional[int]:
        """The artifacts' own batch ceiling, or ``None`` when there isn't one.

        ONNX is uncapped because ``OnnxAdapter`` runs one image per session call
        regardless — so on an ONNX bundle a larger batch only ever moves the person
        detector's forward pass, never the model's.
        """
        cap = self.manifest_contract.batch_caps(self.manifest).get("model")
        return None if cap is None else max(1, int(cap))


def load_bundle(
    bundle_dir: Path,
    *,
    device: Optional[str] = None,
    fmt: Optional[str] = None,
    conf: Optional[float] = None,
    detector_conf: Optional[float] = None,
) -> Bundle:
    infer = _import_bundle_infer(bundle_dir)
    manifest_contract = importlib.import_module("bundle_manifest")
    manifest = manifest_contract.load_manifest(bundle_dir / "pipeline.json")
    resolved_fmt = manifest_contract.resolve_format(manifest, fmt)
    paths = manifest_contract.artifact_paths(manifest, bundle_dir, resolved_fmt)
    if not infer._needs_detector(manifest.get("pipeline"), manifest.get("chain") or []):
        paths["detector"] = None

    from chachak._friendy import resolve_device

    bundle = Bundle(
        directory=bundle_dir,
        infer=infer,
        manifest_contract=manifest_contract,
        manifest=manifest,
        fmt=resolved_fmt,
        paths=paths,
        device=None,
        device_str=device or "auto",
        conf=conf,
        detector_conf=detector_conf,
    )
    bundle.device = resolve_device(bundle.config_for(None).device)
    return bundle


# ── inputs ──────────────────────────────────────────────────────────────────────


def list_images(source: Path, limit: Optional[int]) -> List[Path]:
    """Images to benchmark on, deterministically ordered.

    Sorted, then truncated: two runs pointed at the same directory must see the same
    frames in the same order or their numbers are not comparable.
    """
    if source.is_file():
        found = [source]
    else:
        found = sorted(
            p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
        )
    if not found:
        raise SystemExit(f"[bench] no images under {source}")
    return found[:limit] if limit else found


def decode_images(bundle: Bundle, paths: Sequence[Path]) -> Tuple[List[Any], float]:
    """Decode every frame once, through the bundle's own loader.

    Decode is timed here and reported as its own stage: at 4K it has been measured at
    44 of 54 ms per frame, which makes it the largest single line item in the budget
    and the one most easily mistaken for model cost.
    """
    load = bundle.infer._load_image_tensor
    started = time.perf_counter()
    images = [load(path, bundle.device) for path in paths]
    return images, time.perf_counter() - started


# ── statistics ──────────────────────────────────────────────────────────────────


def percentiles(samples_ms: Sequence[float]) -> Optional[dict]:
    if not samples_ms:
        return None
    import numpy as np

    array = np.asarray(samples_ms, dtype=float)
    return {
        "p50": round(float(np.percentile(array, 50)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "p99": round(float(np.percentile(array, 99)), 4),
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std()), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
        "samples": int(array.size),
    }


def host_rss_peak_mb() -> Optional[float]:
    """Peak host RSS from ``/proc/self/status`` — psutil is not installed here."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(float(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return None


def consensus(t1_ms: Optional[float], t2_ms: Optional[float],
              t3_ms: Optional[float], t4_ms: Optional[float],
              model_calls_per_frame: Optional[float] = None) -> dict:
    """Turn the four instruments into a verdict instead of an average.

    Averaging them would hide exactly what makes them worth running separately: each
    disagreement has one specific cause, named here so the report can say it.

    ``model_calls_per_frame`` matters for the T4 comparison: a tiling or person-crop
    pipeline invokes the model many times per frame, so comparing a whole frame
    against one bare model call would "discover" a 10x wrapper overhead that is really
    just ten inferences. When it is known, T4 is compared per model call.
    """
    notes: List[str] = []
    present = {k: v for k, v in
               (("t1", t1_ms), ("t2", t2_ms), ("t3", t3_ms), ("t4", t4_ms)) if v}

    if t1_ms and t2_ms:
        if t1_ms > t2_ms * 1.10:
            notes.append(
                f"host-bound: wall {t1_ms:.2f} ms exceeds device {t2_ms:.2f} ms by "
                f"{(t1_ms / t2_ms - 1) * 100:.0f}% — see the stage table"
            )
        else:
            notes.append("gpu-bound: wall and device latency agree, the host adds nothing")
    if t2_ms and t3_ms and t3_ms < t2_ms * 0.75:
        notes.append(
            f"device idle inside the call: {t3_ms:.2f} ms of kernel time in a "
            f"{t2_ms:.2f} ms window — launch overhead, sync bubbles or a kernel swarm"
        )
    if t2_ms and t4_ms:
        calls = model_calls_per_frame if (model_calls_per_frame or 0) > 0 else 1.0
        per_call_ms = t2_ms / calls
        suffix = f" over {calls:g} model call(s) per frame" if calls != 1.0 else ""
        if t4_ms < per_call_ms * 0.50:
            notes.append(
                f"the pipeline around the model costs more than the model "
                f"({t4_ms:.2f} ms bare vs {per_call_ms:.2f} ms per call in-bundle{suffix})"
            )
        else:
            notes.append(
                f"the bundle wrapper adds little over the bare model artifact "
                f"({t4_ms:.2f} ms bare vs {per_call_ms:.2f} ms per call in-bundle{suffix})"
            )

    spread_pct = None
    if len(present) >= 2:
        values = list(present.values())
        spread_pct = round((max(values) - min(values)) / max(values) * 100, 1)

    return {
        "t1_wall_ms": None if t1_ms is None else round(t1_ms, 4),
        "t2_cuda_event_ms": None if t2_ms is None else round(t2_ms, 4),
        "t3_cupti_device_ms": None if t3_ms is None else round(t3_ms, 4),
        "t4_bare_model_ms": None if t4_ms is None else round(t4_ms, 4),
        "spread_pct": spread_pct,
        "disagree": bool(spread_pct is not None and spread_pct > 25),
        "notes": notes,
    }


# ── T4: the outside opinion on the model artifact alone ─────────────────────────


def bare_model_ms(bundle_dir: Path, fmt: str, model_path: Path,
                  iterations: int, warmup: int) -> Tuple[Optional[float], Optional[str]]:
    """Time the model artifact with no bundle around it, or say why we could not.

    Runs *before* the bundle is loaded so it neither contends with the measured
    runtimes for GPU memory nor pollutes the load-time Δmem reading.
    """
    if fmt == "engine":
        return _polygraphy_ms(model_path, iterations, warmup)
    return _bare_ort_ms(model_path, iterations, warmup)


def _cuda_library_path() -> Optional[str]:
    """``LD_LIBRARY_PATH`` for a subprocess that needs the CUDA runtime libraries.

    torch carries its CUDA libraries inside the ``nvidia-*`` wheels rather than in a
    system location, and the in-process case works only because importing torch loads
    them first. A bare ``polygraphy`` subprocess does not import torch, so it dies with
    ``OSError: libcudart.so: cannot open shared object file`` — which surfaced as a
    silently missing cross-check, not as an error anyone would notice.
    """
    import glob
    import site

    roots: List[str] = []
    candidates = list(site.getsitepackages())
    if getattr(site, "getusersitepackages", None):
        candidates.append(site.getusersitepackages())
    for base in candidates:
        roots.extend(sorted(glob.glob(str(Path(base) / "nvidia" / "*" / "lib"))))
    unique = list(dict.fromkeys(roots))
    if not unique:
        return None
    existing = os.environ.get("LD_LIBRARY_PATH")
    return ":".join(unique + ([existing] if existing else []))


def _polygraphy_ms(engine_path: Path, iterations: int,
                   warmup: int) -> Tuple[Optional[float], Optional[str]]:
    """``polygraphy run`` on the plan: a wholly separate TensorRT harness.

    polygraphy ships with nvidia-modelopt, which this image installs for D-FINE's
    AutoCast fp16 — so it is already present. (``trtexec``, the obvious choice, is
    not: the pip ``tensorrt`` wheel carries no binary.)
    """
    # --sequential-runners is not optional company for --warm-up: polygraphy exits 1
    # with "--warm-up requires --sequential-runners" otherwise, and a benchmark whose
    # independent cross-check silently never ran is worse than no cross-check.
    cmd = [
        "polygraphy", "run", str(engine_path), "--model-type", "engine", "--trt",
        "--iterations", str(max(1, iterations)),
        "--warm-up", str(max(0, warmup)), "--sequential-runners",
    ]
    env = None
    library_path = _cuda_library_path()
    if library_path:
        env = {**os.environ, "LD_LIBRARY_PATH": library_path}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    except FileNotFoundError:
        return None, "polygraphy not on PATH"
    except subprocess.TimeoutExpired:
        return None, "polygraphy timed out"
    output = f"{proc.stdout}\n{proc.stderr}"
    match = _POLYGRAPHY_AVG_RE.search(output)
    if match:
        return float(match.group(1)), None
    tail = " ".join(output.strip().splitlines()[-2:])[:300]
    return None, f"polygraphy reported no timing (rc={proc.returncode}): {tail}"


def _bare_input_shape(declared: Sequence[Any], onnx_path: Path) -> List[int]:
    """Concrete input shape for the bare-model reference run.

    A detector graph's spatial dims are usually dynamic, and filling them with 1 is
    the documented way to make one die inside its own backbone
    ("Concat ... Axis 2 has mismatched dimensions of 1 and 2"). The artifact's own
    ``.meta.json`` records the size it was exported for, so use that; only the batch
    axis falls back to 1.
    """
    size = None
    meta_path = onnx_path.with_suffix(".meta.json")
    if meta_path.is_file():
        try:
            size = ((json.loads(meta_path.read_text()).get("input") or {}).get("size"))
        except (OSError, ValueError):
            size = None
    if isinstance(size, (list, tuple)) and len(size) == 2:
        spatial = [int(size[0]), int(size[1])]
    elif isinstance(size, (int, float)) and size:
        spatial = [int(size), int(size)]
    else:
        spatial = [640, 640]

    def concrete(axis: int, dim: Any) -> int:
        if isinstance(dim, int) and dim > 0:
            return dim
        if len(declared) == 4:  # NCHW, the only layout a detector graph uses here
            return {0: 1, 1: 3, 2: spatial[0], 3: spatial[1]}[axis]
        return 1

    return [concrete(axis, dim) for axis, dim in enumerate(declared)]


def _bare_ort_ms(onnx_path: Path, iterations: int,
                 warmup: int) -> Tuple[Optional[float], Optional[str]]:
    """A bare onnxruntime session on the graph — no ``onnx_infer``, no chachak.

    Independence is the whole value here, so the input is synthesized from the
    graph's own declared shape rather than by reusing the bundle's preprocessing.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        return None, f"onnxruntime unavailable: {exc}"
    try:
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        spec = session.get_inputs()[0]
        shape = _bare_input_shape(spec.shape, onnx_path)
        feed = {spec.name: np.random.rand(*shape).astype(np.float32)}
        for _ in range(max(0, warmup)):
            session.run(None, feed)
        started = time.perf_counter()
        for _ in range(max(1, iterations)):
            session.run(None, feed)
        elapsed = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - an unusable graph is an answer, not a crash
        return None, f"bare onnxruntime run failed: {exc}"
    return elapsed * 1000.0 / max(1, iterations), None


# ── one worker's loaded copy of the bundle ──────────────────────────────────────


@dataclass
class Worker:
    """One independent loaded copy of the bundle's pipelines."""

    runtimes: Any
    config: Any
    infer: Any
    caches: List[dict] = field(default_factory=list)

    def clear_caches(self) -> None:
        """Drop the memoized person boxes before every timed iteration.

        ``Pipeline._cached_person_boxes`` keys on the image path, so re-running the
        same frames would skip the person detector entirely from the second
        iteration onward and the benchmark would report a pipeline that never
        detected anybody. This one line is the difference between a real number and
        a fictional one.
        """
        for cache in self.caches:
            cache.clear()

    def run(self, images, paths) -> Any:
        return self.infer.run_batch(self.runtimes, self.config, images, paths)


def build_worker(bundle: Bundle, batch_size: Optional[int]) -> Tuple[Worker, dict]:
    """Load one independent copy of the bundle's pipelines, timing the cold start.

    Engine deserialization alone is ~1.5 s, which swamps any short run — so it is
    reported separately rather than amortized into the loop, and split per artifact
    by briefly wrapping the two loaders the bundle's ``build_runtime`` calls.
    """
    config = bundle.config_for(batch_size)
    infer = bundle.infer
    timings: Dict[str, float] = {}

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                timings[name] = timings.get(name, 0.0) + (time.perf_counter() - started)
        return wrapper

    original_detector = getattr(infer, "load_detector", None)
    original_adapter = getattr(infer, "load_checkpoint_adapter", None)
    if original_detector is not None:
        infer.load_detector = timed("detector_load_ms", original_detector)
    if original_adapter is not None:
        infer.load_checkpoint_adapter = timed("model_load_ms", original_adapter)

    started = time.perf_counter()
    try:
        runtimes = infer.build_runtime(config, bundle.device, bundle.paths.get("detector"))
    finally:
        if original_detector is not None:
            infer.load_detector = original_detector
        if original_adapter is not None:
            infer.load_checkpoint_adapter = original_adapter
    build_ms = (time.perf_counter() - started) * 1000.0

    caches: Dict[int, dict] = {}
    for pipeline, _classes in runtimes:
        cache = getattr(pipeline, "box_cache", None)
        if isinstance(cache, dict):
            caches[id(cache)] = cache

    worker = Worker(runtimes=runtimes, config=config, infer=infer,
                    caches=list(caches.values()))
    coldstart = {
        "build_runtime_ms": round(build_ms, 3),
        "detector_load_ms": round(timings.get("detector_load_ms", 0.0) * 1000.0, 3) or None,
        "model_load_ms": round(timings.get("model_load_ms", 0.0) * 1000.0, 3) or None,
    }
    return worker, coldstart


# ── the three passes ────────────────────────────────────────────────────────────


def _chunks(images: List[Any], paths: List[Path], batch: int) -> List[Tuple[List[Any], List[Path]]]:
    """Split the decoded frames into batch-sized calls, once, up front.

    Decoding is deliberately not inside any timed loop: it is measured once as its own
    stage. Re-decoding per iteration would fold a large, highly variable host cost
    into every latency sample.
    """
    return [
        (images[start:start + batch], paths[start:start + batch])
        for start in range(0, len(images), batch)
    ]


def _sync(device_str: str) -> None:
    if not device_str.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def _call_plan(chunks, calls: int) -> List[int]:
    """Which chunk each of ``calls`` successive inference calls uses.

    Cycling rather than repeating one chunk keeps every frame in the sample, so a
    directory with a mix of easy and crowded images is measured as the mix it is.
    """
    return [index % len(chunks) for index in range(max(1, calls))]


def throughput_pass(workers: List[Worker], chunks, device_str: str, calls: int) -> dict:
    """Loop A — no per-call synchronize, so the pipelining that sets throughput is
    intact. This is the only loop whose wall time becomes ``fps``.

    With more than one worker each gets its own thread and its own loaded copy of the
    bundle: a ``box_cache`` is a plain dict and a TensorRT execution context is not
    safe to share, so "concurrency" here means genuinely independent streams — and
    the memory that costs is reported next to the throughput it buys.
    """
    plan = _call_plan(chunks, calls)
    frames_per_worker = sum(len(chunks[index][0]) for index in plan)
    barrier = threading.Barrier(len(workers) + 1)
    errors: List[BaseException] = []

    def body(worker: Worker) -> None:
        try:
            barrier.wait()
            for index in plan:
                worker.clear_caches()
                batch, batch_paths = chunks[index]
                worker.run(batch, batch_paths)
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=body, args=(worker,), daemon=True)
               for worker in workers]
    for thread in threads:
        thread.start()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    started = time.perf_counter()
    for thread in threads:
        thread.join()
    _sync(device_str)
    elapsed = time.perf_counter() - started
    if errors:
        raise errors[0]

    frames = frames_per_worker * len(workers)
    return {
        "wall_s": round(elapsed, 6),
        "calls": len(plan) * len(workers),
        "frames": frames,
        "fps": round(frames / elapsed, 3) if elapsed > 0 else None,
        "fps_per_stream": round(frames / elapsed / len(workers), 3) if elapsed > 0 else None,
    }


def latency_pass(worker: Worker, chunks, device_str: str, calls: int) -> dict:
    """Loop B — synchronized per call, which is what makes a latency *sample* mean
    anything, and exactly why it cannot also produce the throughput number.

    Both instruments run over the same calls so their percentiles are directly
    comparable: T1 is host wall time around the call, T2 is the device's own elapsed
    time between two CUDA events.
    """
    is_cuda = device_str.startswith("cuda")
    plan = _call_plan(chunks, calls)
    wall_ms: List[float] = []
    device_ms: List[float] = []
    detections: List[int] = []
    if is_cuda:
        import torch

    for position, index in enumerate(plan):
        worker.clear_caches()
        batch, batch_paths = chunks[index]
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
        started = time.perf_counter()
        predictions = worker.run(batch, batch_paths)
        if is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            device_ms.append(start_event.elapsed_time(end_event) / len(batch))
        wall_ms.append((time.perf_counter() - started) * 1000.0 / len(batch))
        if position >= len(plan) - len(chunks):  # the final sweep over every chunk
            detections.extend(int(p.shape[0]) for p in predictions)

    return {
        "wall": percentiles(wall_ms),
        "device": percentiles(device_ms),
        "detections_mean": round(sum(detections) / len(detections), 2) if detections else None,
    }


def cupti_pass(worker: Worker, chunks, device_str: str, calls: int) -> dict:
    """T3 — CUPTI traces the whole *process*, so it sees onnxruntime and TensorRT
    kernels as well as torch's (TensorRT runs on torch's own stream here).

    ``self_device_time_total`` rather than ``device_time_total``: the latter counts a
    nested op's device time again in its parent.
    """
    if not device_str.startswith("cuda"):
        return {"available": False, "reason": "not a cuda device"}
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
    except ImportError as exc:
        return {"available": False, "reason": f"torch.profiler unavailable: {exc}"}

    plan = _call_plan(chunks, calls)
    frames = sum(len(chunks[index][0]) for index in plan)
    try:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            started = time.perf_counter()
            for index in plan:
                worker.clear_caches()
                batch, batch_paths = chunks[index]
                worker.run(batch, batch_paths)
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
        busy_us = sum(event.self_device_time_total for event in prof.key_averages())
    except Exception as exc:  # noqa: BLE001 - profiling is a bonus, never the run
        return {"available": False, "reason": f"cupti pass failed: {exc}"}

    return {
        "available": True,
        "device_ms_per_frame": round(busy_us / 1000.0 / max(1, frames), 4),
        "busy_pct": round(min(busy_us / 1e6 / wall * 100, 100.0), 3) if wall > 0 else None,
        "wall_s": round(wall, 6),
        "method": "cupti",
    }


def stage_pass(bundle: Bundle, worker: Worker, chunks, device_str: str, calls: int,
               decode_ms_per_frame: Optional[float]) -> dict:
    """A third pass, synchronized at every stage boundary.

    The synchronizes are unavoidable — without them a stage's wall time is the time to
    *queue* its work, not to do it — and they inflate the total, so this pass reports
    its own total and the overhead against the headline latency rather than pretending
    the two are the same measurement.
    """
    plan_spec = load_stage_plan(bundle.directory, bundle.manifest)
    recorder = StageRecorder(plan_spec, sync=True, device=device_str)
    plan = _call_plan(chunks, calls)
    frames = sum(len(chunks[index][0]) for index in plan)
    try:
        with recorder:
            started = time.perf_counter()
            for index in plan:
                worker.clear_caches()
                batch, batch_paths = chunks[index]
                worker.run(batch, batch_paths)
            _sync(device_str)
            total_s = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - a probe misfire must not lose the cell
        return {"available": False, "reason": f"stage pass failed: {exc}",
                "diagnostics": recorder.diagnostics()}

    return {
        "available": True,
        "total_ms_per_frame": round(total_s * 1000.0 / max(1, frames), 4),
        "stages": recorder.stages(total_s, frames, decode_ms_per_frame=decode_ms_per_frame),
        "diagnostics": recorder.diagnostics(),
    }


# ── a cell: one (batch, concurrency) pair ───────────────────────────────────────


def runtime_providers(worker: "Worker") -> List[str]:
    """Execution providers the loaded onnxruntime sessions actually got.

    This is not paranoia: in this container onnxruntime's CUDA provider fails to load
    when ``libcublasLt`` is not on the library path, and it does so with a *warning*,
    then runs the whole benchmark on the CPU while the adapter still reports
    ``device=cuda``. Without this read, that produces a plausible-looking report whose
    every number is a CPU number.
    """
    providers: List[str] = []
    for pipeline, _classes in getattr(worker, "runtimes", []) or []:
        for holder in (getattr(pipeline, "model_adapter", None), getattr(pipeline, "detector", None)):
            model = getattr(holder, "_model", None)
            session = getattr(model, "session", None)
            get = getattr(session, "get_providers", None)
            if callable(get):
                try:
                    providers.extend(get())
                except Exception:  # noqa: BLE001 - an introspection failure is not fatal
                    pass
    seen: Dict[str, None] = {}
    for name in providers:
        seen.setdefault(name, None)
    return list(seen)


def _free_workers(workers: List[Worker], device_str: str) -> None:
    workers.clear()
    if device_str.startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best effort
            pass


def run_cell(bundle: Bundle, images: List[Any], paths: List[Path], *,
             batch: int, concurrency: int, warmup: int, calls: int,
             min_duration_s: float, measure_stages: bool,
             decode_ms_per_frame: Optional[float],
             bare_ms: Optional[float]) -> dict:
    """Everything measured for one (batch size, concurrency) pair."""
    device_str = str(bundle.device)
    is_cuda = device_str.startswith("cuda")
    warnings: List[str] = []

    mem_before = gpu_monitor.snapshot_process_mem_mb()
    workers: List[Worker] = []
    coldstart: dict = {}
    for slot in range(concurrency):
        worker, worker_coldstart = build_worker(bundle, batch)
        workers.append(worker)
        if slot == 0:
            coldstart = worker_coldstart
    mem_after_load = gpu_monitor.snapshot_process_mem_mb()
    load_delta_mb = gpu_monitor.own_usage_delta_mb(mem_before, mem_after_load)

    providers = runtime_providers(workers[0])
    effective_batch = int(workers[0].config.infer_batch_size)
    if is_cuda and providers and not any("CUDA" in name or "Tensorrt" in name
                                         for name in providers):
        warnings.append(
            f"device is {device_str} but onnxruntime loaded {providers} — every "
            f"number in this cell is a CPU number"
        )
    if effective_batch != batch:
        warnings.append(
            f"batch {batch} capped to {effective_batch} by the {bundle.fmt} artifacts' "
            f"own profile — throughput only, detections are unaffected"
        )
    chunks = _chunks(images, paths, effective_batch)

    # Warmup, and the first call kept separately: on an engine bundle the first call
    # carries deserialization and TensorRT's own lazy setup, and averaging it in makes
    # a short run look several times slower than the steady state it will actually run
    # at in production.
    first_call_ms = None
    for slot, worker in enumerate(workers):
        for index in range(max(1, warmup)):
            worker.clear_caches()
            chunk_batch, chunk_paths = chunks[index % len(chunks)]
            started = time.perf_counter()
            worker.run(chunk_batch, chunk_paths)
            if is_cuda:
                _sync(device_str)
            if slot == 0 and index == 0:
                first_call_ms = round((time.perf_counter() - started) * 1000.0, 3)
    coldstart["first_call_ms"] = first_call_ms

    # Adaptive growth: NVML's utilization counter refreshes at roughly 1 Hz, so a loop
    # shorter than ~2 s is sampled two or three times and its utilization figure is
    # noise. Calibrated with one unmeasured call, as the console sweep does.
    calibration_start = time.perf_counter()
    workers[0].clear_caches()
    workers[0].run(*chunks[0])
    if is_cuda:
        _sync(device_str)
    per_call_s = max(time.perf_counter() - calibration_start, 1e-6)
    grown_calls = max(calls, math.ceil(min_duration_s / per_call_s))
    if grown_calls > calls:
        warnings.append(
            f"grew {calls} to {grown_calls} calls so the throughput loop spans "
            f"{min_duration_s:g}s (per-call {per_call_s * 1000:.1f} ms)"
        )

    if is_cuda:
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch_baseline = torch.cuda.memory_allocated()

    with gpu_monitor.GpuMonitor(interval_s=0.05) as monitor:
        throughput = throughput_pass(workers, chunks, device_str, grown_calls)
    sample = monitor.result

    torch_peak_mb = None
    if is_cuda:
        # Partial by construction: the torch allocator cannot see TensorRT's or
        # onnxruntime's own arenas, which is why load_delta_mb (NVML) is reported
        # alongside rather than instead.
        torch_peak_mb = round(
            max(torch.cuda.max_memory_allocated() - torch_baseline, 0) / (1024 * 1024), 3)

    latency = latency_pass(workers[0], chunks, device_str, min(grown_calls, max(calls, 30)))
    cupti = cupti_pass(workers[0], chunks, device_str, min(grown_calls, max(calls, 30)))
    stages = (
        stage_pass(bundle, workers[0], chunks, device_str,
                   min(grown_calls, max(calls, 30)), decode_ms_per_frame)
        if measure_stages else {"available": False, "reason": "not requested"}
    )

    t1 = (latency.get("wall") or {}).get("p50")
    t2 = (latency.get("device") or {}).get("p50")
    t3 = cupti.get("device_ms_per_frame") if cupti.get("available") else None
    headline = t2 or t1
    stage_total = stages.get("total_ms_per_frame") if stages.get("available") else None
    model_calls = next(
        (row.get("calls_per_frame") for row in (stages.get("stages") or [])
         if row.get("id") == "model.forward" and row.get("calls_per_frame")),
        None,
    )

    _free_workers(workers, device_str)

    return {
        "batch": effective_batch,
        "batch_requested": batch,
        "providers": providers,
        "concurrency": concurrency,
        "warmup": max(1, warmup),
        "calls": grown_calls,
        "latency_ms": latency.get("device") or latency.get("wall"),
        "latency_source": "cuda_event" if latency.get("device") else "wall_clock",
        "wall_ms": latency.get("wall"),
        "fps": throughput.get("fps"),
        "fps_per_stream": throughput.get("fps_per_stream"),
        "frames": throughput.get("frames"),
        "loop_wall_s": throughput.get("wall_s"),
        "gpu_busy_pct": cupti.get("busy_pct") if cupti.get("available") else sample.mean_util_pct,
        "util_method": "cupti" if cupti.get("available") else (
            "nvml-sampled" if sample.available else "unavailable"),
        "mem": {
            "load_delta_mb": None if load_delta_mb is None else round(load_delta_mb, 2),
            "torch_peak_mb": torch_peak_mb,
            "nvml_peak_mb": None if sample.peak_mem_mb is None else round(sample.peak_mem_mb, 1),
            "host_rss_peak_mb": host_rss_peak_mb(),
        },
        "power": {
            "mean_w": None if sample.mean_power_w is None else round(sample.mean_power_w, 1),
            "peak_w": None if sample.peak_power_w is None else round(sample.peak_power_w, 1),
            "mean_sm_clock_mhz": None if sample.mean_sm_clock_mhz is None
            else round(sample.mean_sm_clock_mhz, 0),
            "throttle_reasons": list(sample.throttle_reasons),
        },
        "coldstart": coldstart,
        "instruments": consensus(t1, t2, t3, bare_ms, model_calls_per_frame=model_calls),
        "model_calls_per_frame": model_calls,
        "stage_pass_total_ms": stage_total,
        "instrumentation_overhead_ms": (
            None if (stage_total is None or headline is None)
            else round(stage_total - headline, 4)
        ),
        "stages": stages.get("stages") or [],
        "stage_diagnostics": stages.get("diagnostics"),
        "stage_error": stages.get("reason") if not stages.get("available") else None,
        "detections_mean": latency.get("detections_mean"),
        "cupti": cupti,
        "warnings": warnings,
    }


# ── the run ─────────────────────────────────────────────────────────────────────


def _parse_int_list(value: str, label: str) -> List[int]:
    try:
        parsed = [int(part) for part in str(value).replace(" ", "").split(",") if part]
    except ValueError:
        raise SystemExit(f"[bench] --{label} must be a comma-separated list of integers")
    if not parsed or any(item < 1 for item in parsed):
        raise SystemExit(f"[bench] --{label} must list positive integers")
    return parsed


def _env_versions() -> dict:
    def version(module_name: str) -> Optional[str]:
        try:
            return getattr(importlib.import_module(module_name), "__version__", None)
        except Exception:  # noqa: BLE001 - absent is an answer
            return None

    return {
        "torch": version("torch"),
        "tensorrt": version("tensorrt"),
        "onnxruntime": version("onnxruntime"),
        "numpy": version("numpy"),
        "polygraphy": version("polygraphy"),
        "python": sys.version.split()[0],
    }


def _artifact_summary(paths: Dict[str, Optional[Path]]) -> List[dict]:
    summary = []
    for role, path in sorted(paths.items()):
        if not path:
            continue
        candidate = Path(path)
        if not candidate.is_file():
            continue
        summary.append({
            "role": role,
            "file": candidate.name,
            "size_mb": round(candidate.stat().st_size / (1024 * 1024), 2),
        })
    return summary


def _directory_size_mb(directory: Path) -> float:
    total = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    return round(total / (1024 * 1024), 2)


def run_benchmark(bundle_dir: Path, images_source: Path, *, max_images: Optional[int],
                  batch_sizes: List[int], concurrency: List[int], warmup: int,
                  calls: int, min_duration_s: float, measure_stages: bool,
                  device: Optional[str], fmt: Optional[str], conf: Optional[float],
                  detector_conf: Optional[float], skip_bare: bool,
                  profile_nsys: bool = False, nsys_path: Optional[str] = None,
                  output_dir: Optional[Path] = None) -> dict:
    bundle_dir = bundle_dir.resolve()
    started_at = time.time()
    warnings: List[str] = []

    context_before = gpu_monitor.device_context()

    # T4 first, deliberately: it is a separate TensorRT/onnxruntime harness holding its
    # own GPU memory, so running it before anything is loaded keeps it from contending
    # with the measured runtimes and from polluting the load-time Δmem reading.
    bare_ms: Optional[float] = None
    bare_error: Optional[str] = None
    manifest_peek = json.loads((bundle_dir / "pipeline.json").read_text())
    model_rel = ((manifest_peek.get("artifacts") or {}).get("model")) or ""
    model_path = bundle_dir / model_rel if model_rel else None
    peek_fmt = fmt or (Path(model_rel).suffix.lstrip(".") if model_rel else "")
    peek_fmt = "engine" if peek_fmt == "engine" else "onnx"
    if skip_bare:
        bare_error = "skipped by --no-bare-model"
    elif model_path is None or not model_path.is_file():
        bare_error = "the manifest names no model artifact on disk"
    else:
        bare_ms, bare_error = bare_model_ms(bundle_dir, peek_fmt, model_path,
                                            iterations=max(20, calls // 4), warmup=10)
    if bare_error:
        warnings.append(f"bare-model reference unavailable: {bare_error}")

    bundle = load_bundle(bundle_dir, device=device, fmt=fmt, conf=conf,
                         detector_conf=detector_conf)
    device_str = str(bundle.device)
    if not device_str.startswith("cuda"):
        warnings.append(
            f"running on {device_str}: CUDA-event latency, CUPTI utilization and GPU "
            f"memory are all unavailable, so only wall-clock numbers are reported"
        )

    image_paths = list_images(images_source.resolve(), max_images)
    images, decode_s = decode_images(bundle, image_paths)
    decode_ms_per_frame = round(decode_s / len(images) * 1000.0, 4) if images else None

    max_batch = bundle.max_batch
    if max_batch is None and bundle.fmt == "onnx" and max(batch_sizes) > 1:
        warnings.append(
            "this ONNX bundle has no batch ceiling, but OnnxAdapter runs one image "
            "per session call — a larger batch moves only the person detector"
        )
    cells: List[dict] = []
    for batch in batch_sizes:
        if max_batch is not None and batch > max_batch:
            warnings.append(
                f"skipped batch {batch}: the artifacts' profile caps this bundle at "
                f"{max_batch}"
            )
            continue
        for streams in concurrency:
            try:
                cells.append(run_cell(
                    bundle, images, image_paths, batch=batch, concurrency=streams,
                    warmup=warmup, calls=calls, min_duration_s=min_duration_s,
                    measure_stages=measure_stages,
                    decode_ms_per_frame=decode_ms_per_frame, bare_ms=bare_ms,
                ))
            except Exception as exc:  # noqa: BLE001 - one bad cell must not lose the rest
                cells.append({
                    "batch": batch, "concurrency": streams,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                warnings.append(f"batch {batch} × {streams} stream(s) failed: {exc}")

    for cell in cells:
        warnings.extend(cell.get("warnings") or [])
    providers = next((cell["providers"] for cell in cells if cell.get("providers")), [])

    # Last, and strictly sequential: it re-runs one cell's loop in a child process under
    # nsys, so it must not overlap anything that is being timed. Once per benchmark
    # rather than once per cell -- the question it answers ("which kernels dominate") is
    # not very sensitive to batch size, and per-cell would multiply an expensive capture.
    nsys_result: dict = {"available": False, "reason": "not requested"}
    if profile_nsys:
        usable = [cell for cell in cells if not cell.get("error")]
        if not usable:
            nsys_result = {"available": False,
                           "reason": "skipped: no cell produced numbers to profile"}
        else:
            target = usable[0]
            nsys_dir = (output_dir or bundle_dir) / "nsys"
            nsys_result = nsys_pass(
                bundle_dir, images_source, nsys_dir,
                batch=int(target["batch"]),
                calls=min(int(target["calls"]), 60),
                warmup=max(3, min(warmup, 10)),
                max_images=max_images, device=device, fmt=fmt, conf=conf,
                detector_conf=detector_conf, nsys_path=nsys_path,
            )
        if not nsys_result.get("available"):
            warnings.append(f"Nsight profile unavailable: {nsys_result.get('reason')}")

    context_after = gpu_monitor.device_context()
    cotenants = (context_before.get("cotenants") or []) + (context_after.get("cotenants") or [])
    unique_cotenants = list({item["pid"]: item for item in cotenants}.values())
    if unique_cotenants:
        warnings.append(
            "another process was on the GPU during this run — these numbers are not "
            "comparable with an idle-GPU baseline"
        )

    return {
        "schema": SCHEMA_VERSION,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)),
        "duration_s": round(time.time() - started_at, 1),
        "bundle": {
            "name": manifest_peek.get("name") or bundle_dir.name,
            "directory": str(bundle_dir),
            "fmt": bundle.fmt,
            "pipeline": manifest_peek.get("pipeline"),
            "classes": manifest_peek.get("classes") or {},
            "max_batch": max_batch,
            "size_mb": _directory_size_mb(bundle_dir),
            "artifacts": _artifact_summary(bundle.paths),
            "provenance": manifest_peek.get("provenance") or {},
        },
        "env": {
            "device": device_str,
            "gpu": context_before.get("name"),
            "driver": context_before.get("driver"),
            "total_mem_mb": context_before.get("total_mem_mb"),
            "power_limit_w": context_before.get("power_limit_w"),
            "max_sm_clock_mhz": context_before.get("max_sm_clock_mhz"),
            "onnx_providers": providers,
            **_env_versions(),
        },
        "config": {
            "images": str(images_source),
            "image_count": len(images),
            "max_images": max_images,
            "warmup": warmup,
            "calls": calls,
            "min_duration_s": min_duration_s,
            "batch_sizes": batch_sizes,
            "concurrency": concurrency,
            "stages_measured": measure_stages,
            "decode_ms_per_frame": decode_ms_per_frame,
        },
        "trust": {
            "contended": bool(unique_cotenants),
            "cotenants": unique_cotenants,
            "gpu_util_before_pct": context_before.get("utilization_pct"),
            "gpu_used_mem_before_mb": context_before.get("used_mem_mb"),
        },
        "bare_model": {"ms": None if bare_ms is None else round(bare_ms, 4),
                       "tool": "polygraphy" if peek_fmt == "engine" else "onnxruntime",
                       "error": bare_error},
        "cells": cells,
        "nsys": nsys_result,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark an exported infer bundle through its own runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bundle", required=True, help="Bundle directory (holds pipeline.json)")
    parser.add_argument("--images", required=True, help="Image file or directory (recursive)")
    parser.add_argument("--out", required=True, help="Where to write the result JSON")
    parser.add_argument("--max-images", type=int, default=24,
                        help="Cap the frame set; sorted then truncated, so two runs "
                             "on the same directory see the same frames")
    parser.add_argument("--batch-sizes", default="1",
                        help="Comma-separated; each is capped by the artifacts' profile")
    parser.add_argument("--concurrency", default="1",
                        help="Comma-separated worker-thread counts; each worker loads "
                             "its own copy of the bundle")
    parser.add_argument("--warmup", type=int, default=30, help="Untimed calls before each cell")
    parser.add_argument("--calls", type=int, default=200,
                        help="Timed inference calls per cell (grown to meet --min-duration)")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S,
                        help="Floor on the throughput loop's wall time, in seconds")
    parser.add_argument("--no-stages", action="store_true", help="Skip the stage-breakdown pass")
    parser.add_argument("--no-bare-model", action="store_true",
                        help="Skip the polygraphy/onnxruntime reference (T4)")
    parser.add_argument("--nsys", action="store_true",
                        help="Also capture an Nsight Systems trace of one cell's loop "
                             "(sequential, in a child process) and summarize its "
                             "per-kernel table")
    parser.add_argument("--nsys-path", default=None,
                        help="nsys binary (default: $NSYS_PATH, /opt/nsight/bin/nsys, PATH)")
    parser.add_argument("--nsys-worker", action="store_true",
                        help=argparse.SUPPRESS)  # internal: the child nsys profiles
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:1, ...")
    parser.add_argument("--format", default=None, choices=("onnx", "engine"),
                        help="Assert the bundle carries this format")
    parser.add_argument("--conf", type=float, default=None, help="Detection threshold override")
    parser.add_argument("--detector-conf", type=float, default=None,
                        help="Person-detector threshold override")
    args = parser.parse_args()

    if args.nsys_worker:
        batches = _parse_int_list(args.batch_sizes, "batch-sizes")
        nsys_worker(
            Path(args.bundle), Path(args.images), batch=batches[0], calls=args.calls,
            warmup=args.warmup, max_images=args.max_images, device=args.device,
            fmt=args.format, conf=args.conf, detector_conf=args.detector_conf,
        )
        return

    result = run_benchmark(
        Path(args.bundle), Path(args.images),
        max_images=args.max_images,
        batch_sizes=_parse_int_list(args.batch_sizes, "batch-sizes"),
        concurrency=_parse_int_list(args.concurrency, "concurrency"),
        warmup=args.warmup, calls=args.calls, min_duration_s=args.min_duration,
        measure_stages=not args.no_stages, device=args.device, fmt=args.format,
        conf=args.conf, detector_conf=args.detector_conf, skip_bare=args.no_bare_model,
        profile_nsys=args.nsys, nsys_path=args.nsys_path,
        output_dir=Path(args.out).parent,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    for warning in result["warnings"]:
        print(f"[bench] warning: {warning}")
    nsys_info = result.get("nsys") or {}
    if nsys_info.get("available"):
        print(f"[bench] nsight report: {nsys_info['directory']}/"
              f"{nsys_info['report_file']} ({nsys_info.get('kernel_count')} kernels)")
    for cell in result["cells"]:
        if cell.get("error"):
            print(f"[bench] batch {cell['batch']}x{cell['concurrency']}: {cell['error']}")
            continue
        latency = cell.get("latency_ms") or {}
        print(f"[bench] batch {cell['batch']} x{cell['concurrency']}: "
              f"p50 {latency.get('p50')} ms, {cell.get('fps')} fps, "
              f"{(cell.get('mem') or {}).get('load_delta_mb')} MB")
    print(f"[bench] wrote {out_path}")




# ── T5 (opt-in): Nsight Systems kernel profile ──────────────────────────────────
#
# A wholly separate, sequential pass whose product is a file rather than a number.
# What it adds over T3 is the part CUPTI's aggregate cannot give: a per-kernel
# breakdown, so "the engine forward costs 28 ms" becomes "these five kernels cost it".
#
# It profiles a *re-invocation of this module* in --nsys-worker mode rather than the
# live process: nsys and torch.profiler both drive CUPTI and must not be in the same
# process, and a dedicated child running only the timed loop keeps the .nsys-rep small
# and attributable to one cell instead of spanning every pass.
#
# GPU performance counters (SM occupancy, tensor-core active) are deliberately NOT
# requested. They need CAP_SYS_ADMIN and fail with ERR_NVGPUCTRPERM in an unprivileged
# container; asking for them would make the whole capture fail rather than degrade.

NSYS_REPORTS = ("cuda_gpu_kern_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum")
NSYS_TOP_KERNELS = 15


def resolve_nsys(explicit: Optional[str] = None) -> Optional[str]:
    """Path to an ``nsys`` binary, or ``None``.

    Nsight Systems is ~1.1 GB, so it is bind-mounted into the container rather than
    baked into the image (see docker-compose.yml) — hence the explicit search instead
    of relying on PATH.
    """
    import shutil

    candidates = [explicit, os.environ.get("NSYS_PATH"), "/opt/nsight/bin/nsys"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("nsys")


def _nsys_stats(nsys: str, report_path: Path, report: str,
                out_dir: Path) -> Tuple[Optional[Path], List[dict]]:
    """Run one ``nsys stats`` report, save the CSV, and parse it into rows.

    ``nsys stats`` prints progress lines ("Generating SQLite file…", "Processing…")
    before the CSV, so the header is found by scanning rather than assumed to be line 1.
    """
    import csv
    import io

    # --force-export is not optional: the first report writes a profile.sqlite beside the
    # .nsys-rep, and every later report then refuses with "Existing SQLite export found"
    # plus a usage dump — so without it exactly one of the three reports ever lands.
    try:
        proc = subprocess.run(
            [nsys, "stats", "--force-export=true", "--report", report,
             "--format", "csv", str(report_path)],
            capture_output=True, text=True, timeout=600,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, []

    lines = proc.stdout.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Time (%)")), None)
    if start is None:
        return None, []
    body = "\n".join(lines[start:])

    csv_path = out_dir / f"{report}.csv"
    csv_path.write_text(body)
    rows = list(csv.DictReader(io.StringIO(body)))
    return csv_path, rows


def _kernel_rows(rows: List[dict], limit: int) -> List[dict]:
    """Top kernels by total time, with the names kept readable.

    A templated CUDA kernel name runs to several hundred characters; the full name stays
    in the CSV, and the table gets a truncated form plus the whole thing as a tooltip.
    """
    def number(value, cast=float):
        try:
            return cast(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    def first_of(row, *keys):
        for key in keys:
            if row.get(key) not in (None, ""):
                return row[key]
        return None

    parsed = []
    for row in rows:
        # cuda_gpu_kern_sum says Name/Instances, cuda_api_sum says Name/Num Calls, and
        # cuda_gpu_mem_time_sum says Operation/Count. Reading only the kernel spelling
        # leaves the other two tables nameless and countless.
        name = str(first_of(row, "Name", "Operation") or "").strip()
        parsed.append({
            "name": name if len(name) <= 90 else name[:87] + "…",
            "full_name": name,
            "time_pct": number(row.get("Time (%)")),
            "total_ms": (lambda ns: None if ns is None else round(ns / 1e6, 4))(
                number(row.get("Total Time (ns)"))),
            "instances": number(first_of(row, "Instances", "Num Calls", "Count"), int),
            "avg_us": (lambda ns: None if ns is None else round(ns / 1e3, 3))(
                number(row.get("Avg (ns)"))),
            "med_us": (lambda ns: None if ns is None else round(ns / 1e3, 3))(
                number(row.get("Med (ns)"))),
        })
    parsed.sort(key=lambda item: item["total_ms"] or 0.0, reverse=True)
    return parsed[:limit]


def nsys_pass(bundle_dir: Path, images_source: Path, out_dir: Path, *,
              batch: int, calls: int, warmup: int, max_images: Optional[int],
              device: Optional[str], fmt: Optional[str], conf: Optional[float],
              detector_conf: Optional[float],
              nsys_path: Optional[str] = None) -> dict:
    """Capture an Nsight Systems trace of one cell's timed loop.

    Returns a dict that always explains itself: ``available`` false plus a ``reason``
    when nsys is absent or the capture failed, so a missing profile reads as a missing
    profile rather than as a bundle with no kernels.
    """
    nsys = resolve_nsys(nsys_path)
    if nsys is None:
        return {"available": False,
                "reason": "nsys not found — mount Nsight Systems at /opt/nsight "
                          "(see docker-compose.yml) or set NSYS_PATH"}

    out_dir.mkdir(parents=True, exist_ok=True)
    report_stem = out_dir / "profile"
    child = [
        sys.executable, "-m", "inferlica.benchmark.bundle_bench", "--nsys-worker",
        "--bundle", str(bundle_dir), "--images", str(images_source),
        # --out is required by the parser but unused in worker mode (the child writes no
        # result — nsys's trace is the whole product), so nothing appears at this path.
        "--out", str(out_dir / "worker-unused.json"),
        "--batch-sizes", str(batch), "--calls", str(calls), "--warmup", str(warmup),
    ]
    if max_images:
        child += ["--max-images", str(max_images)]
    if device:
        child += ["--device", device]
    if fmt:
        child += ["--format", fmt]
    if conf is not None:
        child += ["--conf", str(conf)]
    if detector_conf is not None:
        child += ["--detector-conf", str(detector_conf)]

    cmd = [
        nsys, "profile",
        # cuda gives the kernel and memcpy timeline (the point); osrt/nvtx are cheap and
        # make the host side legible in the GUI. Sampling and context switches are off:
        # they multiply the trace size for a question this pass is not asking.
        "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
        "--force-overwrite", "true", "-o", str(report_stem),
        *child,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "nsys capture timed out"}

    report_path = Path(f"{report_stem}.nsys-rep")
    if not report_path.is_file():
        tail = " ".join(f"{proc.stdout}\n{proc.stderr}".strip().splitlines()[-3:])[:400]
        return {"available": False,
                "reason": f"nsys wrote no report (rc={proc.returncode}): {tail}"}

    summaries: Dict[str, List[dict]] = {}
    files = [report_path.name]
    for report in NSYS_REPORTS:
        csv_path, rows = _nsys_stats(nsys, report_path, report, out_dir)
        if csv_path is not None:
            files.append(csv_path.name)
        if rows:
            summaries[report] = rows

    all_kernels = _kernel_rows(summaries.get("cuda_gpu_kern_sum", []), 10_000)
    kernels = all_kernels[:NSYS_TOP_KERNELS]
    sqlite_path = report_path.with_suffix(".sqlite")
    if sqlite_path.is_file():
        try:
            sqlite_path.unlink()
        except OSError:  # noqa: PERF203 - a leftover export is untidy, not fatal
            pass
    api = _kernel_rows(summaries.get("cuda_api_sum", []), 8)
    memory = _kernel_rows(summaries.get("cuda_gpu_mem_time_sum", []), 8)

    return {
        "available": True,
        "version": _nsys_version(nsys),
        "report_file": report_path.name,
        "files": files,
        "directory": str(out_dir),
        "profiled_cell": {"batch": batch, "calls": calls, "warmup": warmup},
        "kernels": kernels,
        "api": api,
        "memory": memory,
        # Over every kernel, not just the ones shown: the table is a top-N and its sum
        # would understate the total it is a share of.
        "kernel_count": len(all_kernels),
        "kernel_total_ms": round(sum(k["total_ms"] or 0.0 for k in all_kernels), 4),
        # Stated rather than silently missing: this is the one thing Nsight could add
        # that it cannot add here.
        "counters": "not requested: GPU performance counters (SM/tensor-core occupancy) "
                    "need CAP_SYS_ADMIN and fail with ERR_NVGPUCTRPERM in this "
                    "unprivileged container",
    }


def _nsys_version(nsys: str) -> Optional[str]:
    try:
        proc = subprocess.run([nsys, "--version"], capture_output=True, text=True,
                              timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for line in proc.stdout.splitlines():
        if "version" in line.lower():
            return line.strip()
    return None


def nsys_worker(bundle_dir: Path, images_source: Path, *, batch: int, calls: int,
                warmup: int, max_images: Optional[int], device: Optional[str],
                fmt: Optional[str], conf: Optional[float],
                detector_conf: Optional[float]) -> None:
    """The child nsys profiles: load the bundle, then run the timed loop and nothing else.

    No CUPTI pass, no stage wrappers, no percentile bookkeeping — anything else in here
    would end up in the trace and be mistaken for the pipeline's own cost.
    """
    bundle = load_bundle(bundle_dir, device=device, fmt=fmt, conf=conf,
                         detector_conf=detector_conf)
    image_paths = list_images(images_source.resolve(), max_images)
    images, _decode_s = decode_images(bundle, image_paths)
    worker, _coldstart = build_worker(bundle, batch)
    chunks = _chunks(images, image_paths, int(worker.config.infer_batch_size))
    device_str = str(bundle.device)

    for index in _call_plan(chunks, warmup):
        worker.clear_caches()
        worker.run(*chunks[index])
    _sync(device_str)
    for index in _call_plan(chunks, calls):
        worker.clear_caches()
        worker.run(*chunks[index])
    _sync(device_str)
    print(f"[bench] nsys worker ran {calls} call(s) at batch "
          f"{worker.config.infer_batch_size}")


if __name__ == "__main__":
    main()
