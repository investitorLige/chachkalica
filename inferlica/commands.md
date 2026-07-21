# inferlica

Unified CLI for running inference + detection metrics on `.pt`, `.onnx`, or `.engine` models.

```
python inferlica/cli.py <dataset> <model> --threshold 0.25 [flags]
```

- `dataset` — dataset root: `images/` (or images flat in the root) + `labels/*.txt` (YOLO format) + `classes.txt` (one name per line, 0-indexed)
- `model` — model file: `.pt`/`.pth` (this repo's training checkpoint), `.onnx`, or `.engine`/`.plan`/`.trt` (TensorRT); `.onnx`/`.engine` need a sibling `<name>.meta.json`

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--threshold` | *required* | score threshold used for precision/recall/F1/confusion matrix (mAP always uses the full curve) |
| `--nms` | none | class-aware NMS IoU threshold applied before operating-point metrics. Set this for raw/un-fused heads (e.g. YOLOX, RT-DETR exported without a fused NMS op). Leave unset for torchvision Faster R-CNN/RetinaNet or engines built with `EfficientNMS_TRT` already fused in — they're NMS'd already |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--batch-size` | `8` | images per inference batch |
| `--num-workers` | `4` | dataloader workers |
| `--warmup` | `1` | batches run once before timing starts (still scored for real afterwards) — absorbs onnxruntime/TensorRT first-call cost so `fps` isn't skewed |
| `--json` | none | write the full metrics dict (per-class breakdown + confusion matrix included) to this path |

## Output

Prints `map50`, `map50_95`, `precision`, `recall`, `f1`, `num_images`, `num_predictions`, `num_targets`, `eval_seconds` (wall time for the predict+metrics loop), `inference_seconds` (pure model time), `fps` (`num_images / inference_seconds`).

## Examples

```
python inferlica/cli.py data/val runs/best.pt      --threshold 0.25
python inferlica/cli.py data/val runs/best.onnx    --threshold 0.25 --nms 0.5
python inferlica/cli.py data/val runs/best.engine  --threshold 0.25 --nms 0.5 --device cuda --json out.json
```

# inferlica/export.py

Convert a checkpoint straight to a deployable engine: chains
`friendy_chachkalica.onnx_export` (`.pt` -> `.onnx` + `meta.json`) and
`friendy_chachkalica.trt_export` (`.onnx` -> `.engine` + `meta.json` +
provenance), writing every artifact into one output directory, then
smoke-tests both exported artifacts through `onnx_infer`/`trt_infer` on a
dummy image so a broken export fails here instead of downstream.

```
python inferlica/export.py <checkpoint.pt> --output-dir out/ [flags]
```

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `checkpoint` | *required* | path to a `.pt`/`.pth` checkpoint |
| `--output-dir` / `-o` | *required* | directory to write `<stem>.onnx` / `<stem>.engine` (+ sidecars) into |
| `--precision` | `fp16` | `fp16` \| `fp32` |
| `--min-hw` / `--opt-hw` / `--max-hw` | from `meta.json` | override the TensorRT optimization profile, `HxW` (e.g. `640x640`) |
| `--workspace-gb` | `4.0` | TensorRT builder workspace size |
| `--device` | `cuda` | device for the `trt_infer` smoke test |
| `--no-verify` | off | skip the `onnx_infer`/`trt_infer` smoke test |

## Output

`<output-dir>/<stem>.onnx`, `<stem>.meta.json`, `<stem>.engine`, `<stem>.engine.json` (build provenance), and — only for archs whose TRT engine needs a raw-output + `EfficientNMS_TRT` re-export (retinanet, yolox) — `<stem>.trt.onnx`.

## Example

```
python inferlica/export.py runs/foo/best.pt --output-dir out/foo --precision fp16
```

# inferlica/benchmark

Sweep every registered architecture (fasterrcnn, retinanet, rfdetr, rtdetr,
yolox) across every variant and every exportable format (`.pt`/`.onnx`/
`.engine`), measuring GPU memory/utilization and inference latency on
synthetic random-init models — no checkpoints, no dataset needed. Writes one
CSV per architecture: columns are `<variant>__<format>`, rows are metrics.

```
python inferlica/benchmark/cli.py --output-dir bench_out/
```

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--output-dir` / `-o` | `bench_out` | cached `.onnx`/`.engine` artifacts + `<arch>.csv` results |
| `--archs` | all 5 | filter to specific architectures |
| `--variants` | all | filter to specific variant names |
| `--formats` | `pt onnx engine` | which formats to benchmark |
| `--num-classes` | `3` | synthetic class count for every built adapter |
| `--batch-size` | `1` | images per `predict()` call |
| `--iterations` | `30` | timed `predict()` calls per cell |
| `--warmup` | `5` | untimed calls before timing starts |
| `--image-size` | `640` | synthetic square HxW |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--precision` | `fp16` | TensorRT build precision |
| `--workspace-gb` | `4.0` | TensorRT builder workspace |
| `--force-rebuild` | off | rebuild cached `.onnx`/`.engine` artifacts |
| `--gpu-poll-interval-ms` | `50` | GPU sampling interval |

## Output

`<output-dir>/<arch>/<variant>.onnx` / `.engine` (cached across re-runs — pass
`--force-rebuild` to regenerate), plus `<output-dir>/fasterrcnn.csv`,
`retinanet.csv`, `rfdetr.csv`, `rtdetr.csv`, `yolox.csv` — each with rows
`status`, `status_reason`, `nms_status`, `eval_seconds`, `inference_seconds`,
`fps`, `gpu_mem_mb_peak`, `gpu_util_pct_mean`, `gpu_util_pct_peak`.

NMS is never applied as a separate step by this benchmark — it's either
already baked into the arch's `.pt`/`.onnx` graph (torchvision NMS) or its
`.engine`'s fused `EfficientNMS_TRT` plugin (fasterrcnn, retinanet, yolox), or the
arch is NMS-free by design (rtdetr, rfdetr — DETR set-prediction). The
`nms_status` row just documents which case applies.

Known-expected: `.onnx` rows show ~0% GPU utilization in this environment
(onnxruntime-gpu is blocked here by a CUDA12/13 cudnn collision with torch —
CPU-only ORT is correct, not a bug).

## Example

```
python inferlica/benchmark/cli.py --archs retinanet yolox --formats pt onnx --output-dir bench_out
```
