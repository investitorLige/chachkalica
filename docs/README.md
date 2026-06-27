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
