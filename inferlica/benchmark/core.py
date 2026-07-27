"""Sweep engine: builds/exports/loads one (arch, variant, format) at a time,
times synthetic inference under GPU monitoring, and assembles one pandas
DataFrame per arch (columns = "<variant>__<format>", rows = metrics).

No real checkpoints, no dataset: every adapter is built random-init via
friendy_chachkalica.registry.build_model and fed synthetic torch.rand images,
so timing/GPU numbers cover every arch+variant in the registry today without
needing a trained checkpoint for each of them (compute cost doesn't depend on
weight values).
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

try:
    from .gpu_monitor import GpuMonitor, own_usage_delta_mb, read_current_mem_mb, snapshot_process_mem_mb
    from .kernel_util import native_kernel_busy_pct
    from .variants import ARCH_VARIANTS, NMS_STATUS, NO_TRT_ARCHS, VariantSpec, rfdetr_variant_specs, yolox_variant_specs
except ImportError:  # run as a flat script
    from gpu_monitor import GpuMonitor, own_usage_delta_mb, read_current_mem_mb, snapshot_process_mem_mb  # type: ignore
    from kernel_util import native_kernel_busy_pct  # type: ignore
    from variants import ARCH_VARIANTS, NMS_STATUS, NO_TRT_ARCHS, VariantSpec, rfdetr_variant_specs, yolox_variant_specs  # type: ignore

METRIC_ROW_ORDER = [
    "status",
    "status_reason",
    "nms_status",
    "precision",
    "eval_seconds",
    "inference_seconds",
    "fps",
    "gpu_mem_mb_peak",
    "gpu_util_pct_mean",
    "gpu_util_pct_peak",
    "gpu_util_method",
]

# Archs whose default TensorRT profile is dynamic (a wide min/opt/max range) --
# fasterrcnn (resize_mode "none") and rtdetr (resize_mode "longest_side") both
# get a wide, non-degenerate min!=max profile from profile.py unless overridden,
# which is the one combination TensorRT's fp16 cast can't compile (a Myelin
# dynamic-shape/fp16 type mismatch). A static profile (min==opt==max) builds
# fp16 clean. Since every benchmark cell only ever times ONE fixed image_size
# anyway, pinning the engine profile to it is strictly more representative of
# what's timed -- not just a fp16 workaround.
#
# rtdetr was added here after the sweep showed it falling back to fp32 on
# EVERY variant despite trt_export/arch/__init__.py's UNTRUSTED_FP16 comment
# claiming "rtdetr is unaffected (its longest_side profile builds fp16
# as-is)" -- that comment is stale relative to observed behavior; verify here
# whether pinning the profile actually fixes it.
#
# retinanet was added later for the same two payoffs: its default dynamic
# 256-896² profile (a) stayed fp32 (same Myelin issue) and (b) can't run an
# input larger than 896 (a 960 sweep errored with execute_async_v3 -> False).
# Pinning the profile to the cell's image_size fixes both -- retinanet now
# builds fp16 and follows --image-size like the others.
ARCHS_NEEDING_STATIC_FP16_PROFILE = {"fasterrcnn", "rtdetr", "retinanet"}


def _build_rtdetr_variant(repo_id: str, num_classes: int):
    """Config-only fetch (small ``config.json``) + fresh random-init build.

    Bypasses ``friendy_chachkalica.ml.adapters.rtdetr.build_rtdetr``'s
    ``weights=repo_id`` branch, which calls
    ``RTDetrForObjectDetection.from_pretrained(repo_id)`` and downloads the
    full pretrained weight tensors (100s of MB) -- unlike every other arch in
    this sweep, which is random-init with no full-weight download. A
    randomly-initialized model at a given backbone size has the same
    latency/memory profile as its pretrained counterpart, so there is no
    benchmarking value in paying for that download; any network hiccup here
    (no internet, HF hub down) surfaces as a normal per-cell error/skip, not a
    crash.
    """
    from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor

    from friendy_chachkalica.ml.adapters.rtdetr import RTDETRAdapter

    id2label = {class_id: str(class_id) for class_id in range(num_classes)}
    label2id = {name: class_id for class_id, name in id2label.items()}

    config = RTDetrConfig.from_pretrained(repo_id)  # network: config.json only
    config.id2label = id2label
    config.label2id = label2id
    config.num_labels = num_classes
    model = RTDetrForObjectDetection(config)  # fresh random init, no weight download

    return RTDETRAdapter(
        model=model,
        image_processor=RTDetrImageProcessor(),
        num_classes=num_classes,
    )


def build_pt_adapter(arch: str, variant: VariantSpec, num_classes: int):
    """Build one arch/variant's adapter in memory, random-init, left on CPU.

    Stays on CPU regardless of the target benchmark device: the ONNX/TRT
    exporters (``export_detection_wrapper``, ``export_raw_boxes_scores``)
    trace with a hardcoded CPU dummy tensor, so an adapter destined for
    ``.onnx``/``.engine`` export must not be moved to CUDA first. Only the
    ``.pt`` timing path (and the loaded onnx/engine runtime, which manages its
    own device placement) actually runs on ``device``.
    """
    if arch == "rtdetr":
        repo_id = variant.build_kwargs["_rtdetr_repo"]
        adapter = _build_rtdetr_variant(repo_id, num_classes)
    else:
        from friendy_chachkalica.registry import build_model

        adapter = build_model(arch, num_classes=num_classes, **variant.build_kwargs)
    adapter.eval()
    return adapter


def export_onnx_artifact(
    arch: str,
    variant: VariantSpec,
    adapter,
    num_classes: int,
    output_dir: Path,
    force_rebuild: bool,
) -> Path:
    onnx_path = output_dir / arch / f"{variant.name}.onnx"
    meta_path = onnx_path.with_suffix(".meta.json")
    if onnx_path.exists() and meta_path.exists() and not force_rebuild:
        return onnx_path

    from friendy_chachkalica.ml.onnx_export.registry import get_exporter

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    class_map = {i: str(i) for i in range(num_classes)}
    exporter = get_exporter(arch)
    # Mirrors onnx_export/cli.py::export_checkpoint's tail (build + write
    # meta.json sidecar), minus the checkpoint-loading step -- we already
    # have `adapter` in memory.
    meta = exporter(
        adapter,
        num_classes=num_classes,
        params=dict(variant.build_kwargs),
        class_map=class_map,
        onnx_path=onnx_path,
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    return onnx_path


def _engine_cache_matches(
    engine_path: Path, onnx_path: Path, precision: str, static_hw: Optional[tuple]
) -> bool:
    """True if a cached ``.engine`` was already built for this exact precision +
    profile request, so it's safe to reuse without rebuilding.

    ``export_engine_artifact`` used to treat "the file exists" as "the cache is
    valid" -- but a ``bench_out/`` reused across runs with a different
    ``--image-size``/``--precision`` would then silently keep serving an engine
    built for the OLD request, mislabeling the CSV's ``precision`` row
    with no warning (this is exactly how one run's ``fasterrcnn/
    mobilenet_v3_large_320_fpn`` cell reported a lone stale fp32 while its
    sibling variants, rebuilt fresh, reported fp16). Compare against the
    ``.engine.json`` provenance sidecar instead of just checking existence.
    """
    provenance_path = Path(str(engine_path) + ".json")
    if not provenance_path.exists():
        return False
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    if provenance.get("requested_precision") != precision:
        return False

    from friendy_chachkalica.ml.trt_export.profile import profile_from_meta

    meta_path = onnx_path.with_suffix(".meta.json")
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected_min, expected_opt, expected_max = profile_from_meta(
        meta, min_hw=static_hw, opt_hw=static_hw, max_hw=static_hw
    )
    recorded_profile = provenance.get("profile", {})
    return (
        list(recorded_profile.get("min", [])) == list(expected_min)
        and list(recorded_profile.get("opt", [])) == list(expected_opt)
        and list(recorded_profile.get("max", [])) == list(expected_max)
    )


def export_engine_artifact(
    arch: str,
    variant: VariantSpec,
    adapter,
    onnx_path: Path,
    output_dir: Path,
    precision: str,
    workspace_gb: float,
    image_size: int,
    force_rebuild: bool,
) -> Optional[Path]:
    if arch in NO_TRT_ARCHS:
        return None

    engine_path = output_dir / arch / f"{variant.name}.engine"
    static_hw = (image_size, image_size) if arch in ARCHS_NEEDING_STATIC_FP16_PROFILE else None

    if engine_path.exists() and not force_rebuild:
        if _engine_cache_matches(engine_path, onnx_path, precision, static_hw):
            return engine_path
        print(
            f"[benchmark] {arch}/{variant.name}: cached .engine was built for a "
            "different precision/profile request -- rebuilding"
        )

    from friendy_chachkalica.ml.trt_export.cli import build_engine

    # adapter= is required since there's no .pt checkpoint file backing this
    # onnx_path; it's used internally only for retinanet/yolox's EfficientNMS
    # raw re-export and harmlessly ignored for passthrough archs.
    return build_engine(
        onnx_path,
        engine_path,
        precision=precision,
        adapter=adapter,
        workspace_gb=workspace_gb,
        min_hw=static_hw,
        opt_hw=static_hw,
        max_hw=static_hw,
    )


def export_onnx_fp16_artifact(onnx_path: Path, arch: str, force_rebuild: bool) -> Optional[Path]:
    """Cast the standard (fp32) ONNX export to fp16 for a GPU onnxruntime run.

    Reuses the same graph-level caster AND the same per-arch fp16 policy the
    export CLI's ``_write_fp16_sidecar`` and the TRT builder use -- the arch's
    ``get_fp16_op_block``/``get_fp16_node_block`` keep-lists are passed through
    so an arch with fp16-fragile ops gets the identical fp32 treatment here as
    everywhere else. Without them the benchmark would silently cast those ops to
    fp16 and report 'fp16' numbers for a graph the rest of the codebase keeps
    partly in fp32, making the cross-format comparison apples-to-oranges. I/O
    dtypes stay fp32, so the ORIGINAL meta.json applies verbatim.

    Deliberately does NOT apply the CLI sidecar's ``is_fp16_trusted`` gate: that
    is a shipping-safety floor, whereas this is a measurement tool. The benchmark
    times whatever precision the sweep requests (its TRT path likewise passes an
    explicit precision straight to ``build_engine``, bypassing the auto-only
    trust floor in ``trt_export.cli``), so gating the ONNX path on trust would
    make it inconsistent with the engine path in the same sweep. Degrades
    gracefully (returns ``None`` instead of raising) so one arch without an
    importable converter never kills the sweep.
    """
    fp16_path = onnx_path.with_name(onnx_path.stem + "_fp16.onnx")
    meta_path = onnx_path.with_suffix(".meta.json")
    fp16_meta_path = fp16_path.with_suffix(".meta.json")
    if fp16_path.exists() and fp16_meta_path.exists() and not force_rebuild:
        return fp16_path

    from friendy_chachkalica.ml.trt_export.arch import get_fp16_node_block, get_fp16_op_block
    from friendy_chachkalica.ml.trt_export.fp16_cast import cast_onnx_bytes_to_fp16

    fp16_bytes, _method = cast_onnx_bytes_to_fp16(
        onnx_path.read_bytes(),
        extra_fp32_ops=get_fp16_op_block(arch),
        node_block_substrings=get_fp16_node_block(arch),
    )
    if fp16_bytes is None:
        return None
    fp16_path.write_bytes(fp16_bytes)
    fp16_meta_path.write_text(meta_path.read_text())
    return fp16_path


def _synthetic_images(batch_size: int, image_size: int, device: torch.device) -> List[torch.Tensor]:
    return [torch.rand(3, image_size, image_size, device=device) for _ in range(batch_size)]


# Once per process per runtime kind ("onnx" | "engine") -- see _prime_gpu_runtime.
_runtime_primed: Dict[str, bool] = {}


def _prime_gpu_runtime(fmt: str, path: Path, device: torch.device, image_size: int) -> None:
    """Absorb TensorRT/onnxruntime's one-time CUDA runtime initialization
    cost once per process, unmeasured, before any cell's real memory
    baseline is taken.

    Verified directly: loading three TensorRT engines back to back in one
    process measured deltas of 402 / 110 / 112 MB -- the first alone ate
    ~290MB of cuDNN/cuBLAS/plugin-registry setup that has nothing to do with
    that specific model (the second and third, model-specific-only, land
    much closer together). Without this, whichever cell happens to run
    first in the sweep unfairly inherits that fixed tax in its reported
    Δmem. Uses THIS cell's own already-built artifact rather than a
    separate dummy network -- simpler, and the memory it uses is fully
    freed (``del`` + ``gc.collect`` + ``empty_cache``) before returning, so
    it doesn't leak into the real measurement that follows.
    """
    if _runtime_primed.get(fmt):
        return
    primer = None
    try:
        images = _synthetic_images(1, image_size, device)
        if fmt == "engine":
            from trt_infer import load_trt_adapter

            primer, _ = load_trt_adapter(path, device)
        else:
            from onnx_infer import load_onnx_adapter

            primer, _ = load_onnx_adapter(path, device)
        from chachak.infer import predict_adapter as _predict_for_prime

        _predict_for_prime(primer, images)
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - priming is best-effort, never worth failing a cell over
        pass
    finally:
        del primer
        gc.collect()
        torch.cuda.empty_cache()
        _runtime_primed[fmt] = True


def time_predict(
    adapter,
    images: List[torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
    amp_enabled: bool = False,
    min_duration_s: float = 0.0,
    profile_cuda: bool = False,
) -> Dict[str, float]:
    """Warmup + timed predict() loop, bracketed with torch.cuda.synchronize()
    so torch/onnxruntime-CUDA timings aren't optimistic the way
    inferlica/eval.py::run_eval's timer is (it stops before the
    .detach().cpu() call that would force a sync).

    ``amp_enabled`` wraps the loop in ``torch.autocast`` -- the same mechanism
    (and the same ``supports_amp`` gate) ``train.py``'s AMP path already uses,
    rather than a blanket ``.half()`` cast, so a model with a documented fp16
    instability (e.g. rtdetr's decoder/GIoU overflow, ``supports_amp=False``
    on its adapter) is never forced through it here either.

    ``min_duration_s`` grows ``iterations`` (never shrinks it) so the timed
    loop reliably clears this GPU's ~1-second NVML utilization refresh
    window -- verified directly that querying it 1.28M times in one second
    only sees the value change once, so any cell shorter than that window
    gets a stale/lucky snapshot rather than a real reading, no matter how
    it's sampled. One calibration call (after warmup, unmeasured towards the
    real timing) estimates per-iteration cost, then the loop runs enough
    iterations to clear the target with margin.

    ``profile_cuda`` runs one ADDITIONAL pass (same iteration count) wrapped
    in ``torch.profiler`` (CUPTI-backed) after the real timed loop, and
    returns ``cuda_busy_pct`` from it -- real kernel-level GPU busy time as a
    percent of wall clock, not an NVML sample. Deliberately a *separate*
    pass, not wrapped around the timed loop above: verified directly that
    profiling has real, substantial overhead (same model, same iteration
    count, unprofiled 241.5 fps vs profiled 182.7 fps -- a genuine ~24%
    slowdown, not noise, roughly proportional to how many small ops a model
    dispatches per call). Wrapping the real timing loop in the profiler would
    have silently made every profiled cell's own fps/latency pessimistic by
    exactly that amount -- which is what an earlier version of this function
    did, undetected until a before/after comparison surfaced fps dropping
    5-28% for PT specifically (smaller/faster models hit hardest, the
    signature of a roughly fixed per-op instrumentation cost) while
    ONNX/ENGINE (never profiled) stayed flat.

    Now requested for ALL cuda formats, not just pt: CUPTI traces the CUDA
    driver timeline for the whole process, so it also captures kernels
    launched by onnxruntime and TensorRT (an earlier version wrongly assumed
    it could only see torch's own dispatcher). When it does, that exact
    per-process kernel-busy% replaces NVML's whole-device ~1Hz sample for
    those formats too; when CUPTI reports nothing for a cell,
    core.benchmark_cell falls back to a runtime-native profiler
    (kernel_util.py) and only then to GpuMonitor's NVML sampling.
    """
    from chachak.infer import predict_adapter

    is_cuda = str(device).startswith("cuda")
    iterations = max(1, iterations)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        for _ in range(max(0, warmup)):
            predict_adapter(adapter, images)

        if is_cuda:
            torch.cuda.synchronize()

        if min_duration_s > 0:
            calib_start = time.perf_counter()
            predict_adapter(adapter, images)
            if is_cuda:
                torch.cuda.synchronize()
            per_iter_s = max(time.perf_counter() - calib_start, 1e-6)
            iterations = max(iterations, math.ceil(min_duration_s / per_iter_s))

        # The ONLY loop that determines eval_seconds/fps -- never profiled.
        start = time.perf_counter()
        for _ in range(iterations):
            predict_adapter(adapter, images)
        if is_cuda:
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start

        cuda_busy_pct = None
        if profile_cuda and is_cuda:
            # CUPTI traces the CUDA driver timeline for the whole PROCESS, not
            # just torch's dispatcher -- so this same pass captures kernels
            # launched by onnxruntime and TensorRT too (TRT runs on torch's own
            # current stream; ORT shares the process's CUDA context). It returns
            # 0 only if CUPTI genuinely saw no device activity for this cell,
            # which core.benchmark_cell treats as the signal to fall back to a
            # runtime-native profiler (kernel_util.py) and then to NVML. Wrapped
            # so any CUPTI unavailability (missing libcupti, a co-attached
            # profiler) degrades to None rather than failing the cell.
            try:
                from torch.profiler import ProfilerActivity, profile

                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    prof_start = time.perf_counter()
                    for _ in range(iterations):
                        predict_adapter(adapter, images)
                    torch.cuda.synchronize()
                    prof_wall_s = time.perf_counter() - prof_start
                # self_device_time_total (not device_time_total) -- the latter
                # double/triple-counts the same kernel across nested op-level
                # events (e.g. aten::conv2d -> aten::convolution ->
                # aten::cudnn_convolution all reporting overlapping "total" time
                # for one underlying kernel); "self" time is the non-overlapping
                # figure, verified directly against the profiler's own printed
                # "Self CUDA time total".
                cuda_busy_us = sum(e.self_device_time_total for e in prof.key_averages())
                if prof_wall_s > 0:
                    cuda_busy_pct = round(min(cuda_busy_us / 1e6 / prof_wall_s * 100, 100.0), 3)
            except Exception:  # noqa: BLE001 - profiling is a measurement extra, never fatal
                cuda_busy_pct = None

    num_images = iterations * len(images)
    fps = num_images / inference_seconds if inference_seconds > 0 else 0.0
    result = {
        "eval_seconds": round(inference_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "fps": round(fps, 3),
    }
    if cuda_busy_pct is not None:
        result["cuda_busy_pct"] = cuda_busy_pct
    return result


def benchmark_cell(
    arch: str,
    variant: VariantSpec,
    fmt: str,
    *,
    num_classes: int,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    iterations: int,
    warmup: int,
    image_size: int,
    precision: str,
    workspace_gb: float,
    gpu_poll_interval_s: float,
    force_rebuild: bool,
    min_duration_s: float = 0.0,
) -> Dict[str, Any]:
    """Benchmark one (arch, variant, format) cell.

    Never raises -- every failure (unsupported variant, missing optional
    package, TRT build failure, no network for the rtdetr config fetch, etc.)
    is caught and turned into status="error"/"skipped" plus a short reason, so
    one broken cell never kills the sweep.
    """
    result: Dict[str, Any] = {
        "status": "error",
        "status_reason": None,
        "nms_status": NMS_STATUS[arch],
        "precision": None,
    }

    if fmt == "engine" and arch in NO_TRT_ARCHS:
        result["status"] = "skipped"
        result["status_reason"] = "no TRT export path for this arch (data-dependent NMS/RPN ops)"
        return result

    # Force the previous cell's runnable (onnxruntime session / TRT execution
    # context / torch model) out of memory before this cell allocates anything.
    # Their CUDA memory isn't managed by torch's caching allocator, so it's only
    # released once their Python wrapper is actually garbage collected -- plain
    # refcounting should already do that once the prior benchmark_cell() call
    # returned, but gc.collect() is cheap insurance against a reference cycle
    # either library holds internally (this is what makes the baseline taken
    # below meaningful; see gpu_monitor.py's module docstring).
    gc.collect()

    is_cuda = str(device).startswith("cuda")
    amp_enabled = False

    try:
        pt_adapter = build_pt_adapter(arch, variant, num_classes)

        # onnx/engine pre-allocate essentially everything they'll ever need at
        # LOAD time -- TensorRT's execution-context workspace is fixed at
        # build time (its own `device_memory_size_v2` property, checked
        # directly, turned out to only cover the context's own scratch space,
        # not the separately-allocated engine weights -- so it undercounts the
        # real total and doesn't vary sensibly across model sizes either);
        # onnxruntime's arena allocator (mem-pattern, the default) reuses its
        # first-call allocation on every subsequent Run(). Verified neither
        # allocates further during the timed loop. So for those two formats
        # the informative number is a load-time before/after delta, not a
        # during-loop peak -- only pt's forward pass genuinely spikes per-call
        # (handled separately below via torch's own allocator tracker).
        #
        # Uses a full per-pid snapshot diff (snapshot_process_mem_mb /
        # own_usage_delta_mb), NOT read_current_mem_mb's residual subtraction
        # -- verified directly that this process's own CUDA activity gets
        # listed by nvml under a pid that doesn't match os.getpid() at all
        # (not a namespace artifact), so residual mode's "subtract every pid
        # except ours" was accidentally subtracting our OWN entry too,
        # netting to ~zero regardless of real usage. Diffing two full
        # snapshots and summing whichever pids grew sidesteps needing to
        # identify "our" pid at all.
        #
        # The baseline is taken PER FORMAT, after that format's runtime has
        # been primed (see _prime_gpu_runtime) -- not once up front -- because
        # TensorRT/onnxruntime's CUDA libraries pay a large, ~constant
        # one-time initialization cost (cuDNN/cuBLAS/plugin-registry setup)
        # the first time EITHER is used in this process, independent of model
        # size: verified directly by loading three engines back to back in
        # one process (402 / 110 / 112 MB) -- the first alone ate ~290MB of
        # pure runtime setup that has nothing to do with that specific model.
        # Priming once per format, per process, before any cell's real
        # baseline, means no cell unfairly inherits that tax just for having
        # run first in the sweep.
        load_baseline_snapshot = None

        if fmt == "pt":
            runnable = pt_adapter.to(device)
            # Same mechanism + same supports_amp gate as train.py's AMP path
            # (see time_predict's docstring) -- not every arch's forward pass
            # is safe under fp16 (rtdetr's adapter documents a decoder/GIoU
            # NaN overflow), so this only ever enables autocast where that
            # flag doesn't say otherwise.
            amp_enabled = precision == "fp16" and is_cuda and getattr(pt_adapter, "supports_amp", True)
            if precision == "fp16" and is_cuda and not amp_enabled:
                result["status_reason"] = "adapter has supports_amp=False (documented fp16 instability) -- ran fp32"
            result["precision"] = "fp16" if amp_enabled else "fp32"
        elif fmt == "onnx":
            onnx_path = export_onnx_artifact(arch, variant, pt_adapter, num_classes, output_dir, force_rebuild)
            load_path = onnx_path
            result["precision"] = "fp32"
            if precision == "fp16" and is_cuda:
                fp16_path = export_onnx_fp16_artifact(onnx_path, arch, force_rebuild)
                if fp16_path is None:
                    result["status_reason"] = "no ONNX fp16 converter importable -- ran fp32"
                else:
                    load_path = fp16_path
                    result["precision"] = "fp16"

            if is_cuda:
                # Prime with onnx_path (the plain fp32 export), never
                # load_path -- load_path may be the fp16-cast graph, which
                # can be genuinely invalid for some archs (the same
                # dangling-node/duplicate-node converter bug documented
                # elsewhere), and priming only needs SOME valid graph to
                # warm up onnxruntime's CUDA runtime, not this cell's own.
                _prime_gpu_runtime("onnx", onnx_path, device, image_size)
                torch.cuda.empty_cache()
                load_baseline_snapshot = snapshot_process_mem_mb()

            from onnx_infer import load_onnx_adapter

            try:
                runnable, _ = load_onnx_adapter(load_path, device)
            except Exception as fp16_load_exc:  # noqa: BLE001 - graceful fp16->fp32 fallback, not a cell failure
                if load_path is onnx_path:
                    raise  # not an fp16-cast issue -- a genuine failure, let the outer handler report it
                # The shared fp16 caster (also used by the .engine path) can emit a
                # graph TensorRT's parser tolerates but onnxruntime's stricter
                # session loader rejects outright (observed: a dangling node input
                # reference on yolox, survives an onnx_graphsurgeon toposort --
                # a real bug in the converter, not an ordering artifact). Fall back
                # to the plain fp32 graph rather than failing the whole cell.
                result["status_reason"] = (
                    f"ONNX fp16 graph failed to load ({type(fp16_load_exc).__name__}) -- ran fp32"
                )
                result["precision"] = "fp32"
                load_path = onnx_path
                runnable, _ = load_onnx_adapter(load_path, device)

            # onnxruntime only actually uses CUDA if onnxruntime-gpu (with a
            # matching CUDA/cuDNN runtime on LD_LIBRARY_PATH) is what's
            # importable in this process -- onnx_infer/session.py already
            # requests CUDAExecutionProvider first when device is cuda, but a
            # provider request silently degrades to CPU if the GPU provider
            # isn't actually available. Surface that instead of silently
            # reporting a GPU/fp16 result that was actually CPU/fp32.
            try:
                actual_providers = runnable._model.session.get_providers()
            except Exception:  # noqa: BLE001 - introspection is best-effort
                actual_providers = None
            if is_cuda and actual_providers is None:
                # Couldn't confirm the execution provider (wrapper internals
                # moved). Fail closed: flag that the reported precision/device
                # is unverified rather than letting a possible silent CPU
                # degrade pass as a clean fp16/CUDA result.
                unverified = (
                    "could not verify onnx execution provider "
                    "-- reported precision/device unconfirmed"
                )
                result["status_reason"] = (
                    f"{result['status_reason']}; {unverified}"
                    if result["status_reason"]
                    else unverified
                )
            elif is_cuda and "CUDAExecutionProvider" not in actual_providers:
                # Preserve any earlier fp16-converter/-load reason instead of
                # overwriting it -- the CPU degrade is additional context, not a
                # replacement for a real fp16-export defect the operator needs.
                cpu_reason = (
                    f"onnx requested cuda but CUDAExecutionProvider unavailable "
                    f"(providers={actual_providers}) -- ran CPU fp32"
                )
                result["status_reason"] = (
                    f"{result['status_reason']}; {cpu_reason}"
                    if result["status_reason"]
                    else cpu_reason
                )
                # If we'd loaded the fp16-cast graph, reload the real fp32 graph
                # before timing: an fp16-interior graph emulated on the CPU EP is
                # not fp32, so timing/labelling it as fp32 (as the old code did)
                # is exactly the mislabel this block exists to prevent.
                if load_path is not onnx_path:
                    load_path = onnx_path
                    runnable, _ = load_onnx_adapter(load_path, device)
                result["precision"] = "fp32"
        elif fmt == "engine":
            onnx_path = export_onnx_artifact(arch, variant, pt_adapter, num_classes, output_dir, force_rebuild)
            engine_path = export_engine_artifact(
                arch, variant, pt_adapter, onnx_path, output_dir, precision, workspace_gb, image_size, force_rebuild
            )
            from trt_infer import load_trt_adapter

            # build_engine writes the precision it ACTUALLY built at (an fp16
            # request can silently fall back to fp32) into this sidecar --
            # surface it in the table so a fallback is visible in the CSV
            # itself, not just buried in a per-engine .engine.json on disk.
            provenance_path = Path(str(engine_path) + ".json")
            if provenance_path.exists():
                result["precision"] = json.loads(provenance_path.read_text()).get("precision")

            if is_cuda:
                _prime_gpu_runtime("engine", engine_path, device, image_size)
                torch.cuda.empty_cache()
                load_baseline_snapshot = snapshot_process_mem_mb()

            runnable, _ = load_trt_adapter(engine_path, device)
        else:
            raise ValueError(f"unknown format {fmt!r}")

        images = _synthetic_images(batch_size, image_size, device)

        # One settling call before the "after" snapshot, for the same reason
        # noted above: ORT's arena and TRT's internal setup should already be
        # steady-state by the time the timed loop's own warmup runs, not
        # still catching up mid-measurement. Uses the same warmup count as
        # the timed loop for consistency, not because more than ~1 call is
        # expected to matter.
        load_delta_mem_mb = None
        if fmt in ("onnx", "engine") and is_cuda and load_baseline_snapshot is not None:
            from chachak.infer import predict_adapter as _predict_for_load

            for _ in range(max(1, warmup)):
                _predict_for_load(runnable, images)
            torch.cuda.synchronize()
            load_after_snapshot = snapshot_process_mem_mb()
            load_delta_mem_mb = own_usage_delta_mb(load_baseline_snapshot, load_after_snapshot)

        # torch.cuda.empty_cache() never frees a live tensor/model -- it only
        # returns already-unreferenced cached blocks to the driver -- so this
        # is safe to call with the adapter/images already built, and it gives
        # the tightest possible floor: it flushes both this cell's own
        # transient build garbage AND anything gc.collect() just freed from
        # the previous cell above.
        if is_cuda:
            torch.cuda.empty_cache()
        baseline_mem_mb = read_current_mem_mb() if is_cuda else None
        # pt gets exact (not sampled) treatment via torch's own allocator
        # bookkeeping -- updated synchronously on every alloc/free, so it
        # cannot miss a spike between two polls the way GpuMonitor's NVML
        # sampling can (confirmed: NVML sampling read 0.0 MB on a cell torch's
        # own tracker measured at ~479 MB peak allocated). pt's forward pass
        # genuinely spikes per-call (unlike onnx/engine, handled above via a
        # load-time delta instead), so this brackets the timed loop itself
        # rather than just the load.
        pt_baseline_alloc_bytes = None
        if fmt == "pt" and is_cuda:
            torch.cuda.synchronize()
            pt_baseline_alloc_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        # profile_cuda for ALL cuda formats now: CUPTI sees onnxruntime's and
        # TensorRT's kernels too (whole-process driver trace), so it can give
        # the same exact per-process busy% for them that it always gave pt.
        # GpuMonitor still runs in parallel so its NVML sample is available as a
        # last-resort fallback for any cell where CUPTI (and the native
        # profiler) come up empty.
        with GpuMonitor(interval_s=gpu_poll_interval_s) as mon:
            timing = time_predict(
                runnable, images, device, warmup, iterations,
                amp_enabled=amp_enabled, min_duration_s=min_duration_s, profile_cuda=is_cuda,
            )
        gpu = mon.result

        result.update(timing)
        if fmt == "pt" and pt_baseline_alloc_bytes is not None:
            peak_alloc_bytes = torch.cuda.max_memory_allocated()
            result["gpu_mem_mb_peak"] = round(max(peak_alloc_bytes - pt_baseline_alloc_bytes, 0) / (1024 * 1024), 4)
        elif load_delta_mem_mb is not None:
            # onnx/engine: load-time before/after delta (see above) --
            # includes weights + workspace/arena, not just execution-context
            # scratch, and isn't vulnerable to missing a during-loop spike
            # since there isn't one for either format once loaded.
            result["gpu_mem_mb_peak"] = round(load_delta_mem_mb, 4)
        elif gpu.peak_mem_mb is not None and baseline_mem_mb is not None:
            # Fallback only: shouldn't normally fire for cuda pt/onnx/engine
            # given the branches above, but kept for a CPU-provider-only
            # environment or any future format.
            result["gpu_mem_mb_peak"] = round(max(gpu.peak_mem_mb - baseline_mem_mb, 0.0), 4)
        else:
            result["gpu_mem_mb_peak"] = gpu.peak_mem_mb

        # GPU utilization, in strict preference order (see kernel_util.py for
        # why NVML sampling is the least-trustworthy option for onnx/engine):
        #   1. CUPTI (exact, per-process). The only path for pt; for onnx/engine
        #      a >0 reading means CUPTI did capture their kernels this cell.
        #   2. onnx/engine, if CUPTI came up empty: the runtime's OWN profiler.
        #   3. NVML whole-device sampling -- last resort, flagged as such.
        # The winning method is recorded in gpu_util_method so a reader can see
        # exactly how each cell's number was obtained (and discount NVML ones).
        util_pct = None
        util_method = None
        cuda_busy_pct = timing.get("cuda_busy_pct")
        if cuda_busy_pct is not None and (fmt == "pt" or cuda_busy_pct > 0.0):
            util_pct = cuda_busy_pct
            util_method = "cupti"
        elif fmt in ("onnx", "engine") and is_cuda:
            # Size the native profiler's own pass to roughly match the timed
            # loop's duration (per_iter_s = batch_size / fps).
            native_iters = iterations
            fps = timing.get("fps") or 0.0
            if fps > 0 and min_duration_s > 0:
                native_iters = max(iterations, math.ceil(min_duration_s * fps / max(1, batch_size)))
            native = native_kernel_busy_pct(runnable, fmt, images, device, native_iters)
            if native is not None:
                util_pct = native
                util_method = "trt-iprofiler" if fmt == "engine" else "ort-profiling"

        if util_pct is not None:
            # Exact single reading -> mean == peak, exactly as the pt path reports.
            result["gpu_util_pct_mean"] = util_pct
            result["gpu_util_pct_peak"] = util_pct
            result["gpu_util_method"] = util_method
        else:
            result["gpu_util_pct_mean"] = gpu.mean_util_pct
            result["gpu_util_pct_peak"] = gpu.peak_util_pct
            result["gpu_util_method"] = "nvml-sampled"
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - fault isolation is the point
        result["status"] = "error"
        result["status_reason"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    return result


def benchmark_arch(arch: str, variants: List[VariantSpec], formats: List[str], **cell_kwargs):
    import pandas as pd

    columns: Dict[str, Dict[str, Any]] = {}
    counts = {"ok": 0, "error": 0, "skipped": 0}

    for variant in variants:
        for fmt in [f for f in formats if f in variant.formats]:
            column = f"{variant.name}__{fmt}"
            print(f"[benchmark] {arch} / {column} ...")
            cell = benchmark_cell(arch, variant, fmt, **cell_kwargs)
            columns[column] = cell
            counts[cell["status"]] = counts.get(cell["status"], 0) + 1
            if cell["status"] == "error":
                print(f"[benchmark]   -> error: {cell['status_reason']}")

    df = pd.DataFrame(columns).reindex(METRIC_ROW_ORDER)
    return df, counts


def run_sweep(
    archs: List[str],
    formats: List[str],
    *,
    output_dir: Path,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    iterations: int,
    warmup: int,
    image_size: int,
    precision: str,
    workspace_gb: float,
    gpu_poll_interval_s: float,
    force_rebuild: bool,
    variant_names: Optional[List[str]] = None,
    min_duration_s: float = 0.0,
    follow_image_size: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_kwargs = dict(
        output_dir=output_dir,
        num_classes=num_classes,
        device=device,
        batch_size=batch_size,
        iterations=iterations,
        warmup=warmup,
        image_size=image_size,
        precision=precision,
        workspace_gb=workspace_gb,
        gpu_poll_interval_s=gpu_poll_interval_s,
        force_rebuild=force_rebuild,
        min_duration_s=min_duration_s,
    )

    dataframes: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    for arch in archs:
        # rfdetr's resolution and yolox's canvas are architectural, so they
        # normally ignore --image-size; when asked to follow it, rebuild each at
        # the size (rfdetr snapped to the nearest valid resolution, yolox at the
        # size directly). retinanet/fasterrcnn/rtdetr follow via the static
        # profile in ARCHS_NEEDING_STATIC_FP16_PROFILE regardless of this flag.
        if arch == "rfdetr" and follow_image_size:
            variants = rfdetr_variant_specs(image_size)
        elif arch == "yolox" and follow_image_size:
            variants = yolox_variant_specs(image_size)
        else:
            variants = ARCH_VARIANTS[arch]
        if variant_names:
            wanted = set(variant_names)
            variants = [v for v in variants if v.name in wanted]
        if not variants:
            print(f"[benchmark] arch {arch!r}: no variants match --variants, skipping")
            summary[arch] = {"ok": 0, "error": 0, "skipped": 0}
            continue

        try:
            df, counts = benchmark_arch(arch, variants, formats, **cell_kwargs)
        except Exception as exc:  # noqa: BLE001 - one broken arch must not stop the rest
            traceback.print_exc()
            print(f"[benchmark] arch {arch!r} failed entirely: {exc}")
            summary[arch] = {"ok": 0, "error": 0, "skipped": 0, "arch_error": str(exc)}
            continue

        csv_path = output_dir / f"{arch}.csv"
        df.to_csv(csv_path)
        print(f"[benchmark] wrote {csv_path}")
        dataframes[arch] = df
        summary[arch] = counts

    return {"dataframes": dataframes, "summary": summary}
