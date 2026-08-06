# Label Studio Module Docs

This module manages Label Studio projects and imports image tasks.

## Files

- `label-studio.py`: CLI and Label Studio API/project orchestration.
- `fleet.py`: per-annotator container fleet management.
- `ml_backends/sam/`: interactive SAM + text-prompt Grounding-SAM ML backend.
- `configs/instance/`: Label Studio instance connection/runtime configs.
- `configs/project/`: project configs.

## Python

Use the module-local virtualenv:

```bash
.venv/bin/python
```

Do not rely on system Python/pip for this module.

## Quick Local Smoke Test

```bash
.venv/bin/python -m py_compile label-studio.py fleet.py

.venv/bin/python label-studio.py start

printf 'Mock PPE Dataset\n' | .venv/bin/python label-studio.py \
  --project-config configs/project/mock.yaml \
  create-project

docker rm -f label-studio
```

The cleanup command removes only the container. It does not delete the Docker volume.

## More

- [Operations Manual](../manual.md) — full flow + recovery runbook
- [Configuration](configuration.md)
- [Commands](commands.md)
- [Annotator Fleet](annotator-fleet.md)
- [Camera Live Preview](camera-live-preview.md) — RTSP cameras + MJPEG preview via Redis
- [Camera Live Inference](camera-live-inference.md) — running a model on a camera's live frames
- [Pipeline Metadata](pipeline-metadata.md) — a model carries the pipeline it was trained through, in every format; every action prefills from it
- [Infer Bundles](infer-bundles.md) — drop a self-contained bundle in `data/bundles` and run it from any inference form; what "Sync bundle" checks
- [Build Nodes](build-nodes.md) — register other GPU machines and compile TensorRT engines/bundles on them, so a bundle is valid on the GPU it will actually run on
