"""Confirm whether CUPTI (torch.profiler CUDA) actually captures the kernels
that onnxruntime and TensorRT launch -- the open question behind switching the
ONNX/ENGINE utilization metric off NVML sampling and onto an exact
per-process kernel-busy%.

Run this in the trainer container against artifacts an earlier sweep already
built (``bench_out/<arch>/<variant>.onnx`` / ``.engine``):

    python inferlica/benchmark/verify_cupti.py \
        --onnx  bench_out/rtdetr/r50vd.onnx \
        --engine bench_out/rtdetr/r50vd.engine \
        --image-size 640 --iters 200

For each artifact it runs two independent measurements over the same loop and
prints them side by side:

  * CUPTI  -- torch.profiler(ProfilerActivity.CUDA), summed
    self_device_time_total / wall. If this is > 0, CUPTI sees the runtime's
    kernels and Fix 1 (reuse the pt path for all formats) is valid.
  * NATIVE -- the runtime's own profiler (TensorRT IProfiler / onnxruntime
    enable_profiling) via kernel_util.py, the Fix 2 fallback.

The two should land close together. The verdict line states plainly whether
CUPTI captured that runtime, so the NVML path is only retired once this says
YES on real artifacts.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

try:
    from .kernel_util import native_kernel_busy_pct
except ImportError:  # run as a flat script
    from kernel_util import native_kernel_busy_pct  # type: ignore


def _synthetic_images(batch_size, image_size, device):
    return [torch.rand(3, image_size, image_size, device=device) for _ in range(batch_size)]


def _cupti_busy_pct(runnable, images, iterations):
    """Same measurement core.time_predict uses, standalone: returns
    (busy_pct, top_kernels) or (None, []) if CUPTI itself is unavailable."""
    from chachak.infer import predict_adapter
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):  # warmup
        predict_adapter(runnable, images)
    torch.cuda.synchronize()

    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            start = time.perf_counter()
            for _ in range(iterations):
                predict_adapter(runnable, images)
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001
        print(f"    CUPTI profiling raised: {type(exc).__name__}: {exc}")
        return None, []

    rows = prof.key_averages()
    busy_us = sum(e.self_device_time_total for e in rows)
    top = sorted(rows, key=lambda e: e.self_device_time_total, reverse=True)[:8]
    top_kernels = [(e.key, e.self_device_time_total) for e in top if e.self_device_time_total > 0]
    if wall_s <= 0:
        return None, top_kernels
    return round(min(busy_us / 1e6 / wall_s * 100.0, 100.0), 3), top_kernels


def _measure(fmt, path, device, image_size, batch_size, iters):
    print(f"\n=== {fmt.upper()}: {path} ===")
    if fmt == "engine":
        from trt_infer import load_trt_adapter

        runnable, _ = load_trt_adapter(path, device)
    else:
        from onnx_infer import load_onnx_adapter

        runnable, _ = load_onnx_adapter(path, device)
        providers = runnable._model.session.get_providers()
        print(f"    onnxruntime providers: {providers}")
        if "CUDAExecutionProvider" not in providers:
            print("    !! session is CPU-only -- GPU-util comparison is not meaningful here")

    images = _synthetic_images(batch_size, image_size, device)

    cupti_pct, top_kernels = _cupti_busy_pct(runnable, images, iters)
    native_pct = native_kernel_busy_pct(runnable, fmt, images, device, iters)

    print(f"    CUPTI  busy%: {cupti_pct if cupti_pct is not None else 'unavailable'}")
    print(f"    NATIVE busy%: {native_pct if native_pct is not None else 'unavailable'}")
    if top_kernels:
        print("    CUPTI top kernels (self device us):")
        for name, us in top_kernels:
            print(f"      {us/1000:9.3f} ms  {name}")

    captured = cupti_pct is not None and cupti_pct > 0.0
    verdict = "YES" if captured else "NO"
    print(f"    >>> CUPTI captures {fmt} kernels: {verdict}")
    if captured and native_pct:
        rel = abs(cupti_pct - native_pct) / native_pct * 100.0
        print(f"    >>> CUPTI vs NATIVE agreement: {rel:.1f}% apart")
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, default=None, help="path to a .onnx artifact to probe")
    parser.add_argument("--engine", type=Path, default=None, help="path to a .engine artifact to probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    if not args.onnx and not args.engine:
        parser.error("pass --onnx and/or --engine")
    if not torch.cuda.is_available():
        parser.error("no CUDA device available -- this check only makes sense on a GPU")

    device = torch.device(args.device)
    results = {}
    if args.engine:
        results["engine"] = _measure("engine", args.engine, device, args.image_size, args.batch_size, args.iters)
    if args.onnx:
        results["onnx"] = _measure("onnx", args.onnx, device, args.image_size, args.batch_size, args.iters)

    print("\n=== summary ===")
    for fmt, captured in results.items():
        print(f"  {fmt:7s}: CUPTI {'CAPTURES' if captured else 'DOES NOT capture'} its kernels")
    all_captured = all(results.values())
    print(
        "\nVERDICT: "
        + (
            "CUPTI captures every probed runtime -- safe to make CUPTI the primary "
            "util metric for all formats and retire NVML sampling."
            if all_captured
            else "at least one runtime is NOT captured by CUPTI -- keep the "
            "kernel_util.py native fallback (and NVML behind it) for that format."
        )
    )
    return 0 if all_captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
