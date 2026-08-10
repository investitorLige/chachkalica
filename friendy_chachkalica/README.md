# Friendy Chachkalica

A lightweight, config-driven **object-detection training & evaluation toolkit**.
It puts several detection architectures behind one common interface — the same
YOLO-format datasets, the same metrics, the same prediction format — so different
models can be trained and compared fairly from a single YAML experiment file.

## What it does

- **One config, many runs** — a single experiment YAML lists your train/val/test
  datasets and models; every model is trained once per train dataset (2 datasets ×
  3 models = 6 runs).
- **Pluggable architectures via adapters** — each model lives in `ml/adapters/` and is
  registered by name. Currently supported: `retinanet`, `fasterrcnn`, `rtdetr`, `rfdetr`,
  `yolox`, `ecdet`.
- **Shared everything else** — YOLO dataset loading, box/NMS postprocessing,
  detection metrics (P/R, mAP50, mAP50-95, per-class AP), and a normalized output
  format are model-agnostic, so comparisons are apples-to-apples.
- **Standalone checkpoint eval** — score any saved checkpoint against a dataset
  without retraining.
- **FastAPI service** — run training and evaluation as background jobs over HTTP
  (`/train`, `/eval`, `/runs/{id}`, `/evals/{id}`, `/health`).

## Prediction & label format

Labels stay in YOLO format:

```text
class_id x_center y_center width height
```

Model outputs are normalized to one internal prediction format:

```text
x_center y_center width height confidence class_id
```

## Install

Python deps are split so you only install the model stacks you need:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt          # core (torch, torchvision, fastapi…)
.venv/bin/python -m pip install -r requirements-rfdetr.txt   # RF-DETR extras
.venv/bin/python -m pip install -r requirements-yolox.txt    # YOLOX extras
.venv/bin/python -m pip install -r requirements-ecdet.txt    # ECDet extras
```

> Note: install the plain `rfdetr` package (Apache-2.0). Do **not** install
> `rfdetr[plus]` — its XLarge models are PML-1.0 licensed and not free for
> commercial use.

## Usage

### CLI — run a full experiment

```bash
.venv/bin/python run.py configs/experiment.yaml
```

Writes `last.pt` / `best.pt`, per-run history and results YAML, and raw test
predictions into the config's `output_dir`.

### CLI — evaluate a checkpoint

```bash
.venv/bin/python ml/eval_checkpoint.py --help
```

### Python API

```python
from train import train_from_config
results = train_from_config("configs/experiment.yaml")
```

### HTTP service

```bash
.venv/bin/uvicorn service:app --host 0.0.0.0 --port 8200
# or (reads .env — see "Running on another machine" first):
docker compose up -d
curl -s localhost:8200/health   # -> {"status": "ok", "active": 0}
```

## Running on another machine

A fresh clone does **not** run out of the box — some things are machine-specific
and are deliberately kept out of git. The trainer *code* is portable (dataset
paths resolve relative to the config file, model weights download on first use,
and it falls back to CPU when no GPU is visible), but you must set up the
following before `docker compose up` or a real run will work:

1. **Create `.env` from the template** (it is git-ignored, so it is never
   pushed):
   ```bash
   cp .env.example .env      # then edit for this machine
   ```
   At minimum set **`HOST_DATA_DIR`** to this machine's shared data directory
   (the `data/` dir of your Django / `chachkalica` checkout). Without it,
   `docker compose up` fails fast with `set HOST_DATA_DIR in .env`. See
   `.env.example` for every variable and what to change.

2. **Match the Django container user.** `FC_UID`/`FC_GID` in `.env` must match
   the user the `chachkalica` web/worker containers run as, so training outputs
   under `data/training/` have compatible ownership. Default is root (`0:0`).

3. **Bring your own datasets.** `data/` and `runs/` are git-ignored, so a clone
   has no images, labels, or checkpoints. Copy your datasets onto the machine
   (under `HOST_DATA_DIR`) and point the config at them.

4. **Fix dataset paths in the example configs.** `configs/experiment.yaml` and
   `configs/smoke_test.yaml` contain absolute paths from the original machine.
   Either edit them to your paths or make them relative to the config file
   (relative paths are resolved against the config's location).

5. **Enable the GPU (optional).** For real training on a GPU host, install the
   NVIDIA Container Toolkit and uncomment the `deploy:` block in
   `docker-compose.yml`. Without it the container runs on CPU — fine only for the
   synthetic smoke test.

The trainer talks to Django over HTTP on `FC_PORT` (8200) and shares files on
disk — keep `FC_PORT` equal to Django's `service_base_url` port, and keep
`TRAINER_DATA_DIR` (`/app/data`) equal to Django's `BASE_DIR/data` so the
absolute paths baked into the config YAMLs resolve inside the container.

## Experiment config

A minimal experiment file. Class IDs may differ between train and val/test as long
as the class *names* match; metrics are computed in the val/test class space, so
train-only classes are ignored.

```yaml
name: helmet-benchmark
output_dir: runs/helmet-benchmark

datasets:
  train:
    - name: field-v1
      images: datasets/field-v1/images/train
      labels: datasets/field-v1/labels/train
      classes: [helmet, head, vest]
  val:
    name: field-v1-val
    images: datasets/field-v1/images/val
    labels: datasets/field-v1/labels/val
    classes: [helmet, head, vest]

models:
  - name: retinanet
    num_classes: auto
    weights_backbone: DEFAULT
  - name: rfdetr
    num_classes: auto
    variant: base
  - name: yolox
    num_classes: auto
    variant: yolox-s
  - name: ecdet
    num_classes: auto
    variant: ecdet-l

training:
  epochs: 100
  batch_size: 4
  device: auto
  optimizer: { name: adamw, lr: 0.0001, weight_decay: 0.0001 }

evaluation:
  batch_size: 4
  score_threshold: 0.001
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

## Repository layout

| Path | Purpose |
|---|---|
| `run.py` | CLI entry point — runs a full experiment from a YAML config |
| `ml/train.py` / `ml/val.py` | Config-driven training and validation |
| `config.py` | Loads/validates the experiment YAML and expands it into runs |
| `data.py` | YOLO-format dataset loading, `Dataset` + collate |
| `metrics.py` | Model-agnostic detection metrics (mAP, per-class AP…) |
| `formats.py` | Shared label/prediction data structures + conversions |
| `export.py` | Prediction export helpers |
| `registry.py` | Maps model names to adapter builders |
| `ml/adapters/` | Architecture-specific code (retinanet, fasterrcnn, rtdetr, rfdetr, yolox, ecdet) |
| `ml/eval_checkpoint.py` | Standalone checkpoint evaluation |
| `preprocess/` | Batch cropping/tiling + coordinate re-mapping for train-time transforms |
| `service.py` | FastAPI service wrapping train/eval as background jobs |
| `configs/` | Example experiment configs |
| `vendor/` | Vendored model code (e.g. YOLOX) |
| `tests/` | Unit tests |

## Adding a model

1. Add `ml/adapters/<name>.py` with a `build_<name>(...)` builder that returns a model
   speaking the shared prediction/target format.
2. Register it in `registry.py`'s `MODEL_REGISTRY`.
3. Reference it by `name` in an experiment config.
