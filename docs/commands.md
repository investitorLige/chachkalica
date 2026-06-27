# Commands

All commands use the module-local virtualenv:

```bash
.venv/bin/python
```


## Start Local Label Studio

Requires `label_studio.mode: local`.

```bash
.venv/bin/python label-studio.py start
```

## Check Instance

```bash
.venv/bin/python label-studio.py check
```

In local mode, this reports Docker/container status and API readiness.

In remote mode, this skips Docker status and reports API readiness.

## Create Project

```bash
.venv/bin/python label-studio.py \
  --project-config configs/project/mock.yaml \
  create-project
```

If a project with the same title exists, the script asks for exact title confirmation before deleting/recreating it.

## Start Interactive SAM Backend

This is the backend to connect in Label Studio `Project Settings -> Model`.

### Docker Compose

Start Label Studio on `8080` and the SAM backend on `9090`:

```bash
docker compose -f ml_backends/sam/docker-compose.yml up --build
```

The first SAM backend startup downloads the checkpoint into the `sam-models`
Docker volume if it is not already present. The default is `vit_b`, which is
safer for CPU-only Linux machines than the larger `vit_h` checkpoint. On CPU,
the backend refuses to load SAM unless enough RAM is available. The default
minimum is 6 GB for `vit_b`; override it with `SAM_MIN_AVAILABLE_RAM_GB` if
you need a different threshold.

In Label Studio, connect the model with:

```text
Backend URL: http://sam-backend:9090
```

This URL works because both containers are on the same compose network. From your host browser, the backend is also available at:

```text
http://localhost:9090
```

but Label Studio’s Model settings should use the compose service URL above.

If your checkpoint has a different path or model type:

```bash
SAM_MODEL_TYPE=vit_l \
SAM_CHECKPOINT=/models/sam_vit_l_0b3195.pth \
docker compose -f ml_backends/sam/docker-compose.yml up --build
```

Supported automatic downloads are `vit_h`, `vit_l`, and `vit_b`. For a custom
checkpoint URL:

```bash
SAM_MODEL_TYPE=vit_b \
SAM_CHECKPOINT=/models/custom-sam.pth \
SAM_CHECKPOINT_URL=https://example.com/custom-sam.pth \
docker compose -f ml_backends/sam/docker-compose.yml up --build
```

For mask/RLE output, use a `BrushLabels` control and start with:

```bash
SAM_FROM_NAME=brush SAM_OUTPUT=brush docker compose -f ml_backends/sam/docker-compose.yml up --build
```

### Local Python

Install the optional backend dependencies:

```bash
.venv/bin/python -m pip install label-studio-ml segment-anything opencv-python pillow torch label-studio-converter
```

Start the backend:

```bash
SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth \
SAM_MIN_AVAILABLE_RAM_GB=6 \
SAM_DATA_ROOT=label_data \
SAM_FROM_NAME=segmentation \
SAM_TO_NAME=image \
SAM_LABEL=Object \
SAM_OUTPUT=polygon \
LABEL_STUDIO_URL=http://localhost:8080 \
LABEL_STUDIO_API_TOKEN=<token> \
.venv/bin/label-studio-ml start ml_backends/sam --port 9090
```

Then in Label Studio, connect:

```text
Backend URL: http://host.docker.internal:9090
```

Use `http://localhost:9090` only when Label Studio is not running in Docker.
Enable interactive preannotations in the model settings.

For mask/RLE output, set `SAM_OUTPUT=brush` and use a `BrushLabels` control named by `SAM_FROM_NAME`.

## Per-Annotator Container Fleet

`fleet.py` runs one isolated Label Studio container per annotator. Each gets its
own port, data volume, and admin user/token, but all containers share the same
`data/source` (source images) and `data/target` (labels out) mounts. Fleet state
— ports, passwords, and API tokens — is written to `configs/fleet.local.yaml`,
which is gitignored because it holds secrets.

### Add an Annotator

```bash
.venv/bin/python fleet.py add alice
```

Picks the next free port (from `8081`), generates a password and API token,
starts the container with the `data/source` and `data/target` mounts, waits for
first-boot migrations, enables legacy `Token` auth (disabled by default in
Label Studio >= 1.23), and confirms the token works. Re-running for an existing
annotator is a no-op. Override the login email with `--email`:

```bash
.venv/bin/python fleet.py add alice --email alice@themercury.ai
```

### Declare Annotators by Hand

You can also pre-declare annotators by editing the `annotators:` section of
`configs/fleet.local.yaml` directly, then provisioning them. `add` reuses any
values you set (`email`, `password`, `port`, `container_name`, `volume_name`,
`token`) and generates only the ones you leave out. This is the supported way to
pin a specific email/password/port instead of accepting the auto-generated ones:

```yaml
annotators:
  charlie:
    email: charlie@themercury.ai
    password: my-chosen-password
    port: 8083
```

Provision a single hand-written entry:

```bash
.venv/bin/python fleet.py add charlie
```

Or provision **every** annotator declared in the config at once (idempotent —
already-running containers are left alone):

```bash
.venv/bin/python fleet.py add
```

Before touching Docker, `add` validates the config and refuses to run if two
annotators share a `port`, `container_name`, or `volume_name`, so a bad hand-edit
fails loudly instead of corrupting the fleet. Other than the records under
`annotators:`, the top-level keys (`base_port`, `image_name`, `source_dir`,
`target_dir`) are also safe to edit by hand.

### List the Fleet

```bash
.venv/bin/python fleet.py list
```

Shows each annotator's URL, container name, and whether it is running.

### Remove an Annotator

```bash
.venv/bin/python fleet.py remove alice
```

Stops and removes the container, deletes its data volume, and drops it from the
fleet state. Pass `--keep-volume` to preserve that annotator's annotations:

```bash
.venv/bin/python fleet.py remove alice --keep-volume
```

### Setup a Dataset

Create the same dataset project in one or all annotator containers. The dataset
lives at `data/source/<dataset>/` and must contain a `classes.txt` plus the
images; every annotator gets a project titled `"<dataset> — <username>"` that
reads from the shared source mount.

For a single annotator:

```bash
.venv/bin/python fleet.py setup-dataset dataset1 --annotator alice
```

For the whole fleet:

```bash
.venv/bin/python fleet.py setup-dataset dataset1 --all
```

It builds the labeling config from `data/source/<dataset>/classes.txt`, creates
the project, attaches a local-files storage pointing at `source/<dataset>`, and
imports one task per image. It is idempotent: an annotator that already has the
project is left untouched (reported as `exists`), so re-running after adding a
new annotator only fills in the gaps. Containers that are not running are
skipped with a warning.

## Annotation Export Pipeline (`coco_sync/`)

Annotations flow out to `data/target/<dataset>/<username>/` as a custom
per-image `.txt` (header `W H`, then one line per object: `class_id cx cy w h`
plus optional normalized polygon points). These are compiled into a per-annotator
`data/target/<dataset>/<username>.coco.json`.

There are two paths into that layout:

### Live: the webhook receiver

`coco_sync/` is a standalone service with its own compose file. Each annotation
submit/update/delete in Label Studio POSTs to it, and it writes/overwrites/deletes
exactly one per-image `.txt` (no COCO is touched live).

```bash
docker compose -f coco_sync/docker-compose.yml up --build
```

It listens on `:9000`, mounts `data/source` (read-only, for `classes.txt`),
`data/target` (read-write), and `configs/` (read-only, for per-annotator tokens).
`fleet.py setup-dataset` auto-registers the webhook on each project it creates,
encoding the annotator/dataset/project in the URL. For containers to reach the
receiver, they are started with `--add-host=host.docker.internal:host-gateway`
(handled by `fleet.py add`); containers created before this change must be
recreated to use live webhooks.

### Batch: `fleet.py sync`

Reconciles `target/` against the authoritative Label Studio state — rebuilds every
per-image `.txt` and the `<username>.coco.json`, pruning files for annotations that
no longer exist. Use it for backfill, after editing already-running containers, or
to recover from missed webhook deliveries.

```bash
# whole fleet, every dataset
.venv/bin/python fleet.py sync

# one annotator, or one dataset
.venv/bin/python fleet.py sync --annotator alice
.venv/bin/python fleet.py sync dataset1 --all
```

Stopped containers are skipped with a warning.

## Cleanup

Remove only the local Label Studio container:

```bash
docker rm -f label-studio
```

Do not delete the Docker volume unless intentionally resetting Label Studio data.
