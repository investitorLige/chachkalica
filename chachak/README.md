# chachak

Stackable inference/eval **pipelines** that wrap the trained detector from
`friendy_chachkalica` with custom pre/post-processing. Each pipeline turns a
directory of frames into Friendy-format predictions and scores them with
`metrics.evaluate_detection`, writing `predictions.pt` + `result.yaml` exactly
like `friendy_chachkalica/ml/eval_checkpoint.py`.

## The three pipelines

A **tile** and a **crop** are the same thing — a sub-region at an `(x1, y1)`
offset with its own size — so all three pipelines share one coordinate remap
(`boxes.remap_local_preds_to_frame`) and one merge/NMS step.

| pipeline | what it does |
|---|---|
| `batch_detect` | Tile each frame into overlapping sub-images (SAHI-style), run the model per tile, remap detections to full-frame coords, NMS-merge across seams. |
| `people_detect_first` | A detector checkpoint proposes person boxes → crop → run the model per crop → remap labels to the full frame → merge. |
| `batch_people` | Combines both: tile → run the detector per tile → lift person boxes to the full frame (+NMS) → crop each person from the **original** frame → run the model → remap → merge. |
| `chain` | Run several pipelines and merge their per-frame predictions. |

## Run

```bash
# from the repo root
python chachak/run.py chachak/configs/batch_detect.yaml
```

Point the config's `model_checkpoint` (and `detector.checkpoint` for the people
pipelines) at your trained `.pt` files and `images`/`labels` at a YOLO-format
dataset. See `configs/*.yaml` for every option. Output lands in `output_dir`:

- `predictions.pt` — list of `{image_path, label_path, orig_size, predictions}`
  records; `predictions` is an `(N, 6)` tensor
  `[x_center, y_center, width, height, confidence, class_id]`, normalized to the
  full frame.
- `result.yaml` — run metadata + metrics (`map50`, `map50_95`, `precision`, ...).

## Programmatic use

```python
from chachak import load_pipeline_config, run_pipeline
run_pipeline(load_pipeline_config("chachak/configs/batch_people.yaml"))
```

## Tests

Tests use stub model/detector adapters (a duck-typed `.predict`), so no
checkpoint or GPU is needed — but they do import the sibling `friendy_chachkalica`
stack (torch, torchvision, ...), so run them where those are installed. In this
repo that's the `trainer` container, with the source mounted:

```bash
docker compose run --rm -v "$PWD:/work" -w /work/chachak trainer \
    python -m unittest discover -s tests -p "test_*.py"
```

Coverage: `test_boxes.py` (tiling/crop/remap/merge geometry), `test_pipelines.py`
(all three pipelines + chain + the `run()` loop writing `predictions.pt`/metrics),
`test_config.py` (loader parsing + validation), `test_registry.py`
(`build_pipeline` + `Detector` person-class filtering).

## Adding a pipeline

1. Subclass `Pipeline` in `pipeline.py` and implement
   `process_batch(images, targets) -> list[(N, 6) tensor]` (full-frame
   normalized). Reuse `_tile_infer`, `self._crop_infer_remap`, and the helpers in
   `boxes.py`.
2. Register it in `registry.py::PIPELINE_REGISTRY` and add its name to
   `config.py::PIPELINE_NAMES`.

## Layout

- `_friendy.py` — puts sibling `friendy_chachkalica` on `sys.path` and re-exports
  the pieces reused (adapters, `formats`, dataloader, metrics, config dataclasses).
- `boxes.py` — tiling, cropping, coordinate remap, merge/NMS (pure, unit-tested).
- `infer.py` — checkpoint loading + threshold-tolerant, chunked adapter inference.
- `detector.py` — person-detector wrapper.
- `pipeline.py` — base + the three pipelines + `ChainedPipeline`.
- `config.py`, `registry.py`, `run.py` — config loader, builder, CLI.
