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

## Ship a pipeline to someone else

`friendy_chachkalica/ml/onnx_export` makes one *model* portable. `bundle_export`
makes the whole *pipeline* portable — the geometry that lives only in the config
here (tiling, person-crop expansion, merge NMS) plus every checkpoint the config
names:

```bash
# from the repo root
python -m chachak.bundle_export.cli chachak/configs/people_detect_first.yaml \
    -o dist/ppe --conf 0.4
```

The recipient gets a directory needing nothing from this repo — no architecture
packages, no `friendy_chachkalica` — and one command:

```bash
pip install -r requirements.txt
python infer.py path/to/images --conf 0.4     # --conf is the only routine knob
```

It writes `results/detections.json` (pixel + normalized boxes per image), and
annotated copies with `--save-images`.

| in the bundle | what it is |
| --- | --- |
| `pipeline.json` | Contract C: every non-path field of the request, the artifact paths, the confidence defaults, and the source provenance |
| `models/` | each checkpoint exported to `.onnx` + its `meta.json` (Contract B) |
| `runtime/` | `chachak` + `onnx_infer`, **copied verbatim at export time** so a bundle can't drift from the pipeline that produced it |
| `infer.py`, `README.md` | entrypoint, and docs generated from the manifest |

Notes:

- `--format engine` also compiles TensorRT engines and prefers them; `models/` still
  gets the ONNX files copied in (they only load on the GPU model + TensorRT version
  they were built for), but `pipeline.json` points `infer.py` at the `.engine`
  paths, so there is no automatic ONNX fallback at run time.
- Only two runtime files are generated rather than copied — `chachak/__init__.py`
  (drops `run.py`, whose batch eval needs the Friendy dataloader) and
  `chachak/_friendy.py` (re-exports the pure helpers from vendored copies and
  stubs the training-only names). Everything else is byte-identical, enforced by
  `tests/test_bundle_export.py`.
- A `score_threshold` under 0.01 is a metric-sweep setting, not a deployment one;
  the exporter warns if you ship one as the default. Pass `--conf`.

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
(`build_pipeline` + `Detector` person-class filtering),
`test_bundle_export.py` (the `pipeline.json` round trip, and that every vendored
runtime module imports with only stdlib + torch on the path).

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
- `bundle_export/` — export a whole pipeline as a self-contained runnable bundle
  (`manifest.py` is the `pipeline.json` contract, shared by exporter and bundle).
- `config.py`, `registry.py`, `run.py` — config loader, builder, CLI.
