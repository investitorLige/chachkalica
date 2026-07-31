# ppe_batch_detect_smoke_test

A self-contained object-detection pipeline, exported 2026-07-31T10:31:27Z.
Everything needed to run it is in this directory.

## Run it

```sh
pip install -r requirements.txt
python infer.py path/to/images --conf 0.4
```

`images` can be a single image or a directory (searched recursively). Results go
to `results/detections.json`; add `--save-images` to also get annotated copies.

```sh
python infer.py --help          # every option
```

## The one setting you should tune

`--conf` is the confidence threshold, defaulting to **0.4** —
the value this pipeline was exported with. Raise it for fewer, surer detections;
lower it to catch more and accept false positives.

Everything else in `pipeline.json` — tile geometry, person-crop expansion, the
NMS thresholds that merge overlapping regions — was tuned together with the
weights. Changing those changes what the pipeline detects, so they are not
command-line options.

## Classes

| id | name |
| --- | --- |
| 0 | helmet |
| 1 | vest |
| 2 | no_helmet |
| 3 | no_vest |

## What runs

**Pipeline: `batch_detect`**

The frame is cut into overlapping tiles, the model runs on each tile, and the per-tile detections are mapped back onto the full frame and merged with NMS. Tiling exists to keep small objects large enough for the model to see them.

| role | file | architecture | classes |
| --- | --- | --- | --- |
| model | `models/model.onnx` | yolox | helmet, vest, no_helmet, no_vest |

## Format

This bundle carries **onnx** artifacts; `infer.py` runs them by default.

The `.onnx` graphs run anywhere onnxruntime does, on CPU or GPU. Install
`onnxruntime-gpu` instead of `onnxruntime` and pass `--device cuda` for GPU.

## Output

`results/detections.json`:

```json
{
  "conf": 0.4,
  "classes": {"0": "helmet"},
  "images": [
    {
      "image": "frame_001.jpg",
      "width": 1920, "height": 1080,
      "detections": [
        {
          "class_id": 0, "class_name": "helmet", "confidence": 0.87,
          "box_xyxy": [812.4, 233.1, 902.0, 361.7],
          "box_xywhn": [0.446, 0.275, 0.047, 0.119]
        }
      ]
    }
  ]
}
```

`box_xyxy` is in that image's own pixels (top-left, bottom-right).
`box_xywhn` is centre-x, centre-y, width, height normalized to 0..1.
Detections are sorted by confidence, highest first.

## What's in here

```
infer.py          the entrypoint
pipeline.json     the pipeline: stages, geometry, thresholds, class space
models/           the exported weights + their preprocessing sidecars
runtime/          the inference code (vendored; no install needed)
requirements.txt
```

`pipeline.json` also records, under `provenance`, the checkpoints and request
this bundle was built from.
