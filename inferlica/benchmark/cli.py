"""Sweep every registered architecture x variant x export format, measuring
GPU usage and inference latency on synthetic random-init models -- no
checkpoints, no dataset. Writes one CSV per architecture into --output-dir:
columns are "<variant>__<format>", rows are metrics.

    python inferlica/benchmark/cli.py --output-dir bench_out/

See inferlica/commands.md for the full flag reference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .core import run_sweep
    from .variants import ARCH_VARIANTS
except ImportError:  # run as a flat script
    from core import run_sweep  # type: ignore
    from variants import ARCH_VARIANTS  # type: ignore

ALL_ARCHS = sorted(ARCH_VARIANTS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep every registered arch x variant x export format, "
        "measuring GPU usage and inference latency on synthetic inputs"
    )
    parser.add_argument(
        "--output-dir", "-o", default="bench_out",
        help="Directory for cached .onnx/.engine artifacts + <arch>.csv results (default: bench_out)",
    )
    parser.add_argument(
        "--archs", nargs="+", default=None, choices=ALL_ARCHS,
        help=f"Filter to these archs (default: all of {ALL_ARCHS})",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Filter to these variant names across whichever archs are selected (default: all)",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=["pt", "onnx", "engine"], default=["pt", "onnx", "engine"],
        help="Export formats to benchmark (default: all three)",
    )
    parser.add_argument("--num-classes", type=int, default=3, help="Synthetic num_classes (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1, help="Images per predict() call (default: 1)")
    parser.add_argument("--iterations", type=int, default=30, help="Timed predict() calls per cell (default: 30)")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed calls before timing starts (default: 5)")
    parser.add_argument("--image-size", type=int, default=640, help="Synthetic square image HxW (default: 640)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda (default: auto)")
    parser.add_argument(
        "--precision", choices=["fp16", "fp32"], default="fp16",
        help="Precision to attempt across every format: TensorRT engine build, PT via "
        "torch.autocast (skipped per-adapter if supports_amp=False), and ONNX via a graph-level "
        "fp16 cast run through onnxruntime's CUDA execution provider (falls back to fp32 for any "
        "of the three if that format's fp16 path isn't available; see the CSV's precision row)",
    )
    parser.add_argument("--workspace-gb", type=float, default=4.0, help="TensorRT builder workspace (default: 4.0)")
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Rebuild cached .onnx/.engine artifacts even if already present in --output-dir",
    )
    parser.add_argument(
        "--gpu-poll-interval-ms", type=float, default=4.0,
        help="GPU sampling interval in ms (default: 4). Most cells time out well under 100ms "
        "(TRT engines especially) -- at the old 50ms default a ~12ms cell got only 2 background-thread "
        "samples for its whole gpu_util_pct_mean/peak, so the number was closer to a coin flip on "
        "whichever instant it landed on than a real average. 4ms keeps NVML query overhead negligible "
        "relative to the GPU work itself while giving even a ~50ms cell a dozen-plus samples.",
    )
    parser.add_argument(
        "--gpu-util-min-duration-s", type=float, default=0.0,
        help="Grow --iterations (never shrink it) so each cell's timed loop runs at least this many "
        "seconds (default: 0, off). This GPU's NVML utilization counter only refreshes about once a "
        "second regardless of poll rate (measured: 1.28M queries/sec, 1 change) -- a cell shorter than "
        "that window gets a stale/lucky snapshot, not a real reading, no matter how it's sampled. "
        "2.0-3.0 reliably clears that window with margin; onnx/engine still rely on NVML sampling even "
        "with this set (it just gives them a fair shot at a real sample), pt gets an exact CUPTI-backed "
        "reading regardless via torch.profiler.",
    )
    parser.add_argument(
        "--follow-image-size", action="store_true",
        help="Make rfdetr, yolox AND ecdet follow --image-size too. All three have an architectural "
        "input size (rfdetr's per-variant resolution, yolox's and ecdet's fixed canvas) that otherwise "
        "ignores --image-size; with this flag rfdetr is rebuilt at the nearest valid resolution "
        "(multiples of 32, or 56 for base) and yolox/ecdet at the size directly (multiple of 32). "
        "fasterrcnn/rtdetr/retinanet follow via their static engine profile regardless. Leave off to "
        "keep native sizes.",
    )
    args = parser.parse_args()

    from friendy_chachkalica.device import resolve_device

    device = resolve_device(args.device)
    archs = args.archs or ALL_ARCHS

    result = run_sweep(
        archs,
        args.formats,
        output_dir=Path(args.output_dir),
        num_classes=args.num_classes,
        device=device,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup=args.warmup,
        image_size=args.image_size,
        precision=args.precision,
        workspace_gb=args.workspace_gb,
        gpu_poll_interval_s=args.gpu_poll_interval_ms / 1000.0,
        force_rebuild=args.force_rebuild,
        variant_names=args.variants,
        min_duration_s=args.gpu_util_min_duration_s,
        follow_image_size=args.follow_image_size,
    )

    print("\n=== benchmark summary ===")
    for arch, counts in result["summary"].items():
        print(f"{arch:>14}: {counts}")


if __name__ == "__main__":
    main()
