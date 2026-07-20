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
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16", help="TensorRT build precision")
    parser.add_argument("--workspace-gb", type=float, default=4.0, help="TensorRT builder workspace (default: 4.0)")
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Rebuild cached .onnx/.engine artifacts even if already present in --output-dir",
    )
    parser.add_argument(
        "--gpu-poll-interval-ms", type=float, default=50.0, help="GPU sampling interval in ms (default: 50)"
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
    )

    print("\n=== benchmark summary ===")
    for arch, counts in result["summary"].items():
        print(f"{arch:>14}: {counts}")


if __name__ == "__main__":
    main()
