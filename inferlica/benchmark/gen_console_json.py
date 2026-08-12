"""Generate the benchmark-console JSON files the Django admin page consumes.

Runs the inferlica benchmark sweep at each requested image size and writes one
``<image_size>.json`` envelope per size into OUTPUT_DIR -- the exact files
``chachkalica``'s admin benchmark console reads from ``data/benchmarks/``. Run
this, point OUTPUT_DIR at that directory (or copy the files there afterwards),
and the console's size switcher picks them up with no code change.

    python inferlica/benchmark/gen_console_json.py OUTPUT_DIR
    python inferlica/benchmark/gen_console_json.py chachkalica/data/benchmarks --sizes 320 640 960

To add or refresh ONE arch without re-timing the rest (the other archs' published
numbers then stay byte-identical rather than drifting with driver/thermal state):

    python inferlica/benchmark/gen_console_json.py chachkalica/data/benchmarks \
        --archs ecdet --merge --sizes 320 640 960

``--merge`` refuses a device mismatch, because latency and memory are not
comparable across GPUs.

rfdetr handling: its resolution is architectural, not a runtime input size. At
the reference size (640 by default) it runs at each variant's NATIVE resolution
(base 672, large 704); at every other size it follows the image size via the
nearest valid resolution per variant (multiples of 32, or 56 for base). Change
the reference with ``--reference-size`` (use ``-1`` to make every size follow).

Environment: needs ``torch`` + ``tensorrt`` + ``onnxruntime`` importable and a
CUDA GPU for the ``.engine`` / GPU-``.onnx`` rows. In this repo's trainer
container that means the isolated onnxruntime-gpu on ``PYTHONPATH`` and its CUDA
libs + the TensorRT plugin shim on ``LD_LIBRARY_PATH``:

    export PYTHONPATH=/tmp/ort_gpu_test
    export LD_LIBRARY_PATH=$(python3 -c 'import glob;print(":".join(glob.glob(
        "/usr/local/lib/python3.12/site-packages/nvidia/*/lib")))'):/work/shim

On a machine where onnxruntime-gpu and TensorRT are already on the default
loader path, no special env is needed. Without a CUDA onnxruntime the ONNX rows
degrade to CPU fp32 (reported honestly in the JSON); without the TensorRT
``libnvinfer_vc_plugin`` shim the EfficientNMS engine builds
(fasterrcnn/retinanet/yolox) fail with a build error in that cell.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .core import run_sweep
    from .variants import ARCH_VARIANTS, NMS_STATUS, rfdetr_variant_specs
except ImportError:  # run as a flat script
    from core import run_sweep  # type: ignore
    from variants import ARCH_VARIANTS, NMS_STATUS, rfdetr_variant_specs  # type: ignore

ALL_ARCHS = sorted(ARCH_VARIANTS)
FORMATS = ["pt", "onnx", "engine"]


def _num(value):
    """DataFrame cell -> float or None (NaN / missing -> None)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _text(value):
    """DataFrame cell -> str ('' for NaN / missing)."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _build_data(dataframes: dict) -> dict:
    """Turn run_sweep's per-arch DataFrames into the console's DATA object:
    ``{arch: {variants: [...], nms: str, cells: {"<variant>__<fmt>": {...}}}}``.
    """
    out = {}
    for arch, df in dataframes.items():
        columns = list(df.columns)
        variants = []
        for col in columns:
            variant = col.rsplit("__", 1)[0]
            if variant not in variants:
                variants.append(variant)

        cells = {}
        index = set(df.index)
        for col in columns:
            def cell(row):
                return df.at[row, col] if row in index else None

            cells[col] = {
                "status": _text(cell("status")),
                "reason": _text(cell("status_reason")) or None,
                "precision": _text(cell("precision")),
                "eval_s": _num(cell("eval_seconds")),
                "fps": _num(cell("fps")),
                "mem": _num(cell("gpu_mem_mb_peak")),
                "util_mean": _num(cell("gpu_util_pct_mean")),
                "util_peak": _num(cell("gpu_util_pct_peak")),
                "method": _text(cell("gpu_util_method")) or None,
            }
        out[arch] = {"variants": variants, "nms": NMS_STATUS[arch], "cells": cells}
    return out


def _rfdetr_note(image_size: int, follow: bool) -> str:
    """Human-readable summary of the rfdetr resolutions used at this size."""
    specs = rfdetr_variant_specs(image_size) if follow else ARCH_VARIANTS["rfdetr"]
    parts = ", ".join(f"{s.name} {s.build_kwargs.get('resolution')}" for s in specs)
    prefix = "snapped to image size" if follow else "native per-variant"
    return f"{prefix} ({parts})"


def _device_label(device) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"{device} · {torch.cuda.get_device_name(0)}"
    except Exception:  # noqa: BLE001 - label is cosmetic, never fail the run over it
        pass
    return str(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_dir", type=Path, help="Directory to write <image_size>.json into")
    parser.add_argument("--sizes", type=int, nargs="+", default=[320, 640, 960], help="Image sizes to sweep (default: 320 640 960)")
    parser.add_argument(
        "--reference-size", type=int, default=640,
        help="Size at which rfdetr runs at its NATIVE per-variant resolution; every other size makes "
        "rfdetr follow the image size. Use -1 to make every size follow (default: 640).",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda (default: auto)")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--gpu-poll-interval-ms", type=float, default=4.0)
    parser.add_argument("--gpu-util-min-duration-s", type=float, default=2.0)
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Rebuild cached .onnx/.engine artifacts (per-size scratch dir) even if present",
    )
    parser.add_argument(
        "--archs", nargs="+", default=None, choices=ALL_ARCHS,
        help="Limit the sweep to these archs (default: all). Pair with --merge to add or refresh "
        "one arch without re-timing — and so without churning — the others.",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Update only the swept archs inside an existing <size>.json instead of replacing the "
        "whole envelope. Refuses if the existing file was generated on a different device, since "
        "mixing numbers across GPUs would make the table lie (override with --force-merge).",
    )
    parser.add_argument(
        "--force-merge", action="store_true",
        help="Allow --merge across a device mismatch. The envelope's device label is then rewritten "
        "to the current device, which no longer describes the rows carried over.",
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=None,
        help="Where the per-size .onnx/.engine/.csv artifacts are cached (default: OUTPUT_DIR/_artifacts/<size>). "
        "Reused across runs so re-generating is fast unless --force-rebuild.",
    )
    args = parser.parse_args()

    from friendy_chachkalica.device import resolve_device

    device = resolve_device(args.device)
    device_label = _device_label(device)
    today = datetime.date.today().isoformat()
    archs = list(args.archs) if args.archs else list(ALL_ARCHS)
    if args.archs and not args.merge:
        print(f"[gen] WARNING: sweeping only {archs} without --merge — every other arch will be "
              f"DROPPED from the output files.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for size in args.sizes:
        follow = size != args.reference_size
        artifact_dir = (args.csv_dir / str(size)) if args.csv_dir else (args.output_dir / "_artifacts" / str(size))
        print(f"\n[gen] === image size {size} (rfdetr {'follows' if follow else 'native'}) ===")
        result = run_sweep(
            archs,
            FORMATS,
            output_dir=artifact_dir,
            num_classes=args.num_classes,
            device=device,
            batch_size=args.batch_size,
            iterations=args.iterations,
            warmup=args.warmup,
            image_size=size,
            precision=args.precision,
            workspace_gb=args.workspace_gb,
            gpu_poll_interval_s=args.gpu_poll_interval_ms / 1000.0,
            force_rebuild=args.force_rebuild,
            min_duration_s=args.gpu_util_min_duration_s,
            follow_image_size=follow,
        )
        data = _build_data(result["dataframes"])
        out_path = args.output_dir / f"{size}.json"
        envelope = {
            "image_size": size,
            "label": f"{size}×{size}",
            "device": device_label,
            "rfdetr_note": _rfdetr_note(size, follow),
            "generated": today,
            "data": data,
        }
        if args.merge and out_path.is_file():
            existing = json.loads(out_path.read_text())
            previous_device = existing.get("device")
            if previous_device != device_label and not args.force_merge:
                raise SystemExit(
                    f"[gen] refusing to merge into {out_path}: it was generated on "
                    f"{previous_device!r} but this run is on {device_label!r}. Latency and memory "
                    f"are not comparable across devices — re-sweep every arch, or pass "
                    f"--force-merge if you accept a mixed table."
                )
            merged = dict(existing)
            merged["data"] = {**existing.get("data", {}), **data}
            # rfdetr_note describes rfdetr's resolution handling for this size; keep the
            # existing one unless rfdetr was actually re-swept.
            if "rfdetr" not in data:
                merged["rfdetr_note"] = existing.get("rfdetr_note")
            merged["device"] = device_label
            merged["generated"] = today
            envelope = merged
            print(f"[gen] merged {sorted(data)} into {out_path} "
                  f"(kept {sorted(set(existing.get('data', {})) - set(data))})")
        out_path.write_text(json.dumps(envelope, separators=(",", ":")))
        written.append(out_path)
        print(f"[gen] wrote {out_path}  (summary: {result['summary']})")

    print("\n[gen] done. JSON files:")
    for path in written:
        print(f"  {path}")
    print(
        "\nDrop these into chachkalica/data/benchmarks/ (bind-mounted into the web "
        "container) and the admin benchmark console picks them up automatically."
    )


if __name__ == "__main__":
    main()
