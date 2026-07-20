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

import json
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
    from .gpu_monitor import GpuMonitor
    from .variants import ARCH_VARIANTS, NMS_STATUS, NO_TRT_ARCHS, VariantSpec
except ImportError:  # run as a flat script
    from gpu_monitor import GpuMonitor  # type: ignore
    from variants import ARCH_VARIANTS, NMS_STATUS, NO_TRT_ARCHS, VariantSpec  # type: ignore

METRIC_ROW_ORDER = [
    "status",
    "status_reason",
    "nms_status",
    "eval_seconds",
    "inference_seconds",
    "fps",
    "gpu_mem_mb_peak",
    "gpu_util_pct_mean",
    "gpu_util_pct_peak",
]


def _build_rtdetr_variant(repo_id: str, num_classes: int):
    """Config-only fetch (small ``config.json``) + fresh random-init build.

    Bypasses ``friendy_chachkalica.adapters.rtdetr.build_rtdetr``'s
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

    from friendy_chachkalica.adapters.rtdetr import RTDETRAdapter

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

    from friendy_chachkalica.onnx_export.registry import get_exporter

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


def export_engine_artifact(
    arch: str,
    variant: VariantSpec,
    adapter,
    onnx_path: Path,
    output_dir: Path,
    precision: str,
    workspace_gb: float,
    force_rebuild: bool,
) -> Optional[Path]:
    if arch in NO_TRT_ARCHS:
        return None

    engine_path = output_dir / arch / f"{variant.name}.engine"
    if engine_path.exists() and not force_rebuild:
        return engine_path

    from friendy_chachkalica.trt_export.cli import build_engine

    # adapter= is required since there's no .pt checkpoint file backing this
    # onnx_path; it's used internally only for retinanet/yolox's EfficientNMS
    # raw re-export and harmlessly ignored for passthrough archs.
    return build_engine(
        onnx_path,
        engine_path,
        precision=precision,
        adapter=adapter,
        workspace_gb=workspace_gb,
    )


def _synthetic_images(batch_size: int, image_size: int, device: torch.device) -> List[torch.Tensor]:
    return [torch.rand(3, image_size, image_size, device=device) for _ in range(batch_size)]


def time_predict(
    adapter,
    images: List[torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> Dict[str, float]:
    """Warmup + timed predict() loop, bracketed with torch.cuda.synchronize()
    so torch/onnxruntime-CUDA timings aren't optimistic the way
    inferlica/eval.py::run_eval's timer is (it stops before the
    .detach().cpu() call that would force a sync)."""
    from chachak.infer import predict_adapter

    is_cuda = str(device).startswith("cuda")
    iterations = max(1, iterations)

    for _ in range(max(0, warmup)):
        predict_adapter(adapter, images)

    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        predict_adapter(adapter, images)
    if is_cuda:
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start

    num_images = iterations * len(images)
    fps = num_images / inference_seconds if inference_seconds > 0 else 0.0
    return {
        "eval_seconds": round(inference_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "fps": round(fps, 3),
    }


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
    }

    if fmt == "engine" and arch in NO_TRT_ARCHS:
        result["status"] = "skipped"
        result["status_reason"] = "no TRT export path for this arch (data-dependent NMS/RPN ops)"
        return result

    try:
        pt_adapter = build_pt_adapter(arch, variant, num_classes)

        if fmt == "pt":
            runnable = pt_adapter.to(device)
        elif fmt == "onnx":
            onnx_path = export_onnx_artifact(arch, variant, pt_adapter, num_classes, output_dir, force_rebuild)
            from onnx_infer import load_onnx_adapter

            runnable, _ = load_onnx_adapter(onnx_path, device)
        elif fmt == "engine":
            onnx_path = export_onnx_artifact(arch, variant, pt_adapter, num_classes, output_dir, force_rebuild)
            engine_path = export_engine_artifact(
                arch, variant, pt_adapter, onnx_path, output_dir, precision, workspace_gb, force_rebuild
            )
            from trt_infer import load_trt_adapter

            runnable, _ = load_trt_adapter(engine_path, device)
        else:
            raise ValueError(f"unknown format {fmt!r}")

        images = _synthetic_images(batch_size, image_size, device)

        with GpuMonitor(interval_s=gpu_poll_interval_s) as mon:
            timing = time_predict(runnable, images, device, warmup, iterations)
        gpu = mon.result

        result.update(timing)
        result["gpu_mem_mb_peak"] = gpu.peak_mem_mb
        result["gpu_util_pct_mean"] = gpu.mean_util_pct
        result["gpu_util_pct_peak"] = gpu.peak_util_pct
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
    )

    dataframes: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    for arch in archs:
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
