# Chachkalica Unified

Unified workspace for the Chachkalica annotation, training, inference, and evaluation tools.

| Path | What it is |
|---|---|
| [`chachkalica/`](./chachkalica) | Django annotator-fleet manager: admin UI, annotation webhook receiver, RQ worker, per-annotator Label Studio provisioning, COCO export, and a `training/` app that talks to the trainer over HTTP. |
| [`friendy_chachkalica/`](./friendy_chachkalica) | FastAPI + PyTorch object-detection training/evaluation service with config-driven runs for RetinaNet, RT-DETR, RF-DETR, and YOLOX. |
| [`chachak/`](./chachak) | Stackable inference/evaluation pipelines that run trained detectors over tiled frames, people-first crops, or chained pipeline configs. |
| [`onnx_infer/`](./onnx_infer) | Lightweight ONNX Runtime inference adapter/service code for running exported detector artifacts without the full training stack. |
| [`buildnode/`](./buildnode) | Slim, torch-free service deployed on **other** GPU machines so they can compile TensorRT engines and infer bundles on their own hardware. A TensorRT engine only loads on the GPU and TensorRT version that built it, so this is how a bundle gets built for the machine that will actually run it. |

Each main subproject keeps its own README, Dockerfile, and compose file where it can run independently. The root [`docker-compose.yml`](./docker-compose.yml) brings the Django fleet manager and trainer together on one shared network and one shared data mount.

## Quick Start

Prerequisites:

- Docker with the Compose plugin
- NVIDIA Container Toolkit if you want GPU-backed training
- Datasets under [`chachkalica/data/source/`](./chachkalica/data/source) for real annotation or training work

```bash
cp .env.example .env
docker compose up -d
```

The root compose stack starts:

| Service | Role | Host port |
|---|---|---|
| `web` | Django admin + webhook, served by Gunicorn | `9000` |
| `worker` | RQ background worker for fleet and training jobs | none |
| `trainer` | `friendy_chachkalica` FastAPI trainer | `8201` by default |
| `postgres` | Fleet state database | `5433` on host, `5432` in-network |
| `redis` | RQ queue | `6379` |

Useful endpoints:

- Django admin: <http://localhost:9000/admin/>
- Django health: <http://localhost:9000/health>
- Trainer health from the host: <http://localhost:8201/health>
- Trainer health inside the compose network: `http://trainer:8200/health`

On startup, `web` runs migrations and `collectstatic`. The trainer defaults to host port `8201` because the compose file reserves `8200` for the in-network service port; override it with `FC_HOST_PORT` in `.env` if needed.

## How The Stack Fits Together

- Django owns the annotation fleet, dataset metadata, admin workflows, and training job orchestration.
- The trainer exposes `/train`, `/eval`, `/pipeline`, run-status, and health endpoints for long-running ML jobs.
- Django calls the trainer at `http://trainer:8200`, injected through `TRAINING_SERVICE_URL` in the root compose file.
- Django writes training config YAMLs with absolute `/app/data/...` paths. The `web`, `worker`, and `trainer` containers all mount [`chachkalica/data/`](./chachkalica/data) at `/app/data`, so those paths resolve the same way in every container.
- `chachak` can run trained checkpoints through pipeline configs and is included in the trainer image so the trainer can execute pipeline jobs.
- `onnx_infer` is the lightweight runtime path for exported ONNX models and is intended to avoid depending on the full model architecture packages at inference time.

## Common Commands

Run the unified stack:

```bash
docker compose up -d
docker compose logs -f web trainer worker
```

Run only the fleet manager:

```bash
cd chachkalica
docker compose up -d
```

Run only the trainer:

```bash
cd friendy_chachkalica
docker compose up -d
```

Run a Chachak pipeline from the repo root:

```bash
python chachak/run.py chachak/configs/batch_detect.yaml
```

Run the Chachak unit tests in the trainer container:

```bash
docker compose run --rm -v "$PWD:/work" -w /work/chachak trainer \
    python -m unittest discover -s tests -p "test_*.py"
```

## Documentation

- [`chachkalica/README.md`](./chachkalica/README.md) - fleet manager overview, admin workflow, and Django service layout
- [`chachkalica/manual.md`](./chachkalica/manual.md) - operations manual and recovery runbook
- [`friendy_chachkalica/README.md`](./friendy_chachkalica/README.md) - trainer installation, config format, HTTP API, and model-adapter notes
- [`chachak/README.md`](./chachak/README.md) - pipeline types, config usage, and test commands
- [`onnx_infer/PLAN.md`](./onnx_infer/PLAN.md) - ONNX inference design and export contract
- [`buildnode/README.md`](./buildnode/README.md) - build-node internals, HTTP API, and why it carries no torch
- [`chachkalica/docs/build-nodes.md`](./chachkalica/docs/build-nodes.md) - registering GPU machines and building engines/bundles on them

## Data And Generated Files

The repository is meant to keep code, configs, and documentation in git. Runtime data is local:

- `chachkalica/data/source/` - source datasets and `classes.txt`
- `chachkalica/data/target/` - annotation outputs
- `chachkalica/data/training/` - generated training configs and artifacts
- trainer `runs/` directories - checkpoints, metrics, histories, and predictions

Keep large datasets, model weights, and generated run outputs out of git unless there is a specific reason to version a small fixture.

### Regenerate the person-detector graphs when the person model changes

`models/` is gitignored, and two of the files in it are **derived from the person
checkpoint** and baked into the build-node image:

- `models/people/detector.onnx` - standard graph, used in ONNX bundles
- `models/people/detector.trt.onnx` - EfficientNMS graph, the TensorRT compile input

**If the person detector checkpoint ever changes, these are stale and nothing will
tell you.** They are ordinary files; no build fails, no test goes red. Build nodes
keep compiling detector engines from the old graph and every bundle they produce
silently carries the previous person model. Regenerate them:

```bash
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$PWD/friendy_chachkalica:/app/friendy_chachkalica:ro" \
  -v "$PWD/buildnode:/app/buildnode:ro" \
  -v "$PWD/person_model_test:/app/person_model_test:ro" \
  -v "$PWD/models:/app/models" \
  -w /app chackalica_unified-trainer \
  python buildnode/scripts/make_person_graphs.py
```

Then **rebuild and redeploy the build-node image on every node** - the graphs are
baked in at image build time, so a running node keeps using the old ones until it
is replaced. Each node's compiled-detector cache is keyed by the graph's hash, so
it invalidates itself and recompiles on the next build; no volume to clear.

Two things this does *not* cover, and both need doing separately with the same
trigger: `models/people/best_ckpt.engine` (what the **local** trainer bundles) has
to be rebuilt on its own, and bundles already built against the old detector keep
it until you re-export them.

The meta sidecar the script copies is **hand-patched** (`bgr` / `byte` /
`pad_value: 114`) because the person model is a Megvii YOLOX, not a friendy-trained
one. The script preserves it verbatim; do not substitute a freshly exported meta,
which would say `rgb`/`unit` and silently mis-preprocess every frame. See
[`chachkalica/docs/build-nodes.md`](./chachkalica/docs/build-nodes.md).

## GPU Notes

The root compose file includes an NVIDIA GPU reservation for the trainer. On a machine without NVIDIA Container Toolkit support, remove or comment the `trainer.deploy.resources.reservations.devices` block before running the stack. Without GPU access, model configs that use `device: auto` fall back to CPU, which is useful for smoke tests but usually too slow for real training.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](./LICENSE).

Redistributions must preserve the license and applicable copyright, attribution, and NOTICE information. See [`NOTICE`](./NOTICE) for the project attribution notice.
