"""Unified inference/eval CLI for .pt, .onnx, and .engine detectors.

    python inferlica/cli.py <dataset> <model> --threshold 0.25 [--nms 0.5]

See inferlica/commands.md for the full flag reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval import run_eval

SUMMARY_KEYS = [
    "map50",
    "map50_95",
    "precision",
    "recall",
    "f1",
    "num_images",
    "num_predictions",
    "num_targets",
    "eval_seconds",
    "inference_seconds",
    "fps",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference + detection metrics on a .pt/.onnx/.engine model")
    parser.add_argument("dataset", help="Dataset root: images/ (or flat) + labels/ + classes.txt")
    parser.add_argument("model", help="Model file: .pt, .pth, .onnx, or .engine/.plan/.trt")
    parser.add_argument("--threshold", type=float, required=True, help="Score threshold for metrics")
    parser.add_argument("--nms", type=float, default=None, help="Class-aware NMS IoU threshold (omit = no NMS)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda (default: auto)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1, help="Batches run before timing starts (default: 1)")
    parser.add_argument("--json", help="Write the full metrics dict (incl. per-class + confusion matrix) here")
    args = parser.parse_args()

    metrics = run_eval(
        dataset_path=args.dataset,
        model_path=args.model,
        threshold=args.threshold,
        nms=args.nms,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup=args.warmup,
    )

    print(f"\nmodel={metrics['model_path']}")
    print(f"dataset={metrics['dataset_path']}")
    print(f"threshold={metrics['threshold']} nms={metrics['nms']} device={metrics['device']}\n")
    for key in SUMMARY_KEYS:
        print(f"{key:>18}: {metrics[key]}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote full metrics: {json_path}")


if __name__ == "__main__":
    main()
