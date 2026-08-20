# ECDet Benchmark Results

Measured 2026-08-13 on an NVIDIA GeForce RTX 4080 SUPER (16 GB), driver 580.173.02,
CUDA 13.0, inside the `chackalica_unified-trainer-1` trainer container.

**Method:** `inferlica/benchmark/cli.py`, all 4 registered ecdet variants
(`ecdet-s/m/l/x`, 10/18/31/49M params), synthetic random-init weights (no
trained checkpoint needed — timing/memory don't depend on weight values),
640×640 input, batch size 1, 30 timed iterations + 5 warmup per cell.

```
python inferlica/benchmark/cli.py --archs ecdet --formats pt     --precision fp32 --device cuda --output-dir bench_out
python inferlica/benchmark/cli.py --archs ecdet --formats engine --precision fp32 --device cuda --output-dir bench_out
```

fp32 was used for both formats deliberately — ecdet's TensorRT fp16 path is
untrusted (see commit `772f357`, "stop the TRT export form handing out an
untrusted fp16 engine"), so fp32 is the arch's only trusted export precision
today. The explicit `--precision fp32` above is now redundant: the sweep floors
any arch in `UNTRUSTED_FP16` to fp32 by default, in every format, so it only
publishes precisions that can actually ship. Pass `--allow-untrusted-fp16` to
measure ecdet's fp16 speed anyway.

## PyTorch (fp32, eager)

| variant | params | fps | latency | peak GPU mem (Δ) | GPU util |
|---|---|---|---|---|---|
| ecdet-s | 10M | 75.5 | 13.2 ms | 63.1 MB | 49.2% |
| ecdet-m | 18M | 58.1 | 17.2 ms | 79.2 MB | 56.7% |
| ecdet-l (default) | 31M | 51.4 | 19.5 ms | 78.3 MB | 59.9% |
| ecdet-x | 49M | 48.5 | 20.6 ms | 88.0 MB | 67.5% |

> Ran with two unrelated GPU processes already resident (~2.6 GB / ~13% util
> from an Xorg session + a live `processing_cctv` RTSP inference pipeline) —
> these are real-world "shared GPU" numbers, not an idle-machine peak.

## TensorRT (fp32 engine)

| variant | params | fps | latency | peak GPU mem (Δ) | GPU util |
|---|---|---|---|---|---|
| ecdet-s | 10M | 172.2 | 5.8 ms | 142 MB | 72.8% |
| ecdet-m | 18M | 128.6 | 7.8 ms | 210 MB | 82.9% |
| ecdet-l (default) | 31M | 89.1 | 11.2 ms | 318 MB | 87.8% |
| ecdet-x | 49M | 76.4 | 13.1 ms | 362 MB | 89.0% |

## TensorRT vs PyTorch speedup

| variant | fps speedup | latency reduction |
|---|---|---|
| ecdet-s | 2.28× | 13.2 → 5.8 ms |
| ecdet-m | 2.21× | 17.2 → 7.8 ms |
| ecdet-l (default) | 1.73× | 19.5 → 11.2 ms |
| ecdet-x | 1.58× | 20.6 → 13.1 ms |

Speedup shrinks as the model grows — small/medium variants are more
kernel-launch/overhead-bound in eager PyTorch, so TensorRT's fusion buys more;
the larger variants are already compute-bound, so there's less overhead left
to fuse away. GPU memory is higher under TensorRT than PyTorch at every
variant — expected, since the engine keeps its own workspace/activation
buffers separate from PyTorch's allocator, and the peak crept up with util
(89% for `ecdet-x` engine, vs 67.5% PyTorch) as each variant pushes the GPU
harder per inference call.

## Caveats

- Synthetic random-init weights, not a trained checkpoint — this measures
  pure architecture/compute cost, not accuracy-affecting behavior.
- Both runs shared the same background GPU contention (same two processes,
  ~2.6 GB / mid-teens % util, resident the whole time) — so the PT-vs-TRT
  comparison is apples-to-apples relative to each other, but neither set is
  an idle-machine peak number.
- fp16 not benchmarked for TensorRT — untrusted for this arch as of
  `772f357`.
