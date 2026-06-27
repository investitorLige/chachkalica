# Annotator Fleet — Status & Roadmap

A summary of the per-annotator Label Studio setup: what is built and verified,
and what still needs to be done.

## Goal

Each annotator works in their own isolated Label Studio instance. All instances
read the **same source** images and write to the **same target** location, with
per-annotator separation:

```
data/
  source/
    <dataset>/                 # source images (one copy, shared by all annotators)
      *.jpg
      classes.txt
      labels/                  # OPTIONAL pre-existing labels (see below)
        <image>.txt
  target/
    <dataset>/
      labels/
        <username>/            # that annotator's labels for the dataset
```

- A dataset may ship an optional `source/<dataset>/labels/` folder. When present
  (detected on add and stored as `Dataset.has_labels`), its per-image `.txt`
  files are imported as **predictions** (editable pre-annotations) when projects
  are created. Both standard YOLO (`class_id cx cy w h …`, no header) and this
  tool's own `width height`-header format are auto-detected; filenames may be
  either `<image>.txt` or `<image>.<ext>.txt`. Predictions don't sync back to
  `target/` until an annotator reviews and submits them.
- One container **per annotator** (not per project) — gives true isolation and
  sidesteps Label Studio Community-edition role limits.
- Within a container, an annotator has **one project per dataset**.
- `source` and `target` are local mounts today; planned to become S3 buckets.

## Done

### Data layout & config
- Created `data/source` and `data/target`.
- Repointed config: `configs/instance/local.yaml` `data_root: ./data`;
  `configs/project/default.yaml` uses `dataset_dir: source`,
  `classes_file: source/classes.txt`, `target_dir: target`.

### Fleet management — `fleet.py` (verified working)
- `add <username>`, `list`, `remove <username>` (see `docs/commands.md`).
- Each `add`: assigns the next free port (from 8081), generates a password and
  API token, starts a container with both `data/source` and `data/target`
  mounts and a private data volume, waits for first-boot migrations, and
  bootstraps the API token.
- Fleet state (ports, passwords, tokens) is stored in
  `configs/fleet.local.yaml` (gitignored).
- **Verified live:** `alice` (http://localhost:8081) and `bob`
  (http://localhost:8082) are provisioned and running; their tokens
  authenticate against `/api/projects` (200); mounts are correct; re-running
  `add` is idempotent. These two test containers are intentionally left running.

### Key finding — legacy token auth
Label Studio >= 1.23 **disables legacy `Token <token>` auth by default**, which
this entire codebase relies on. `fleet.py` re-enables it during bootstrap via
`POST /api/jwt/settings` (`legacy_api_tokens_enabled: true`). The existing
`label-studio.py` commands (`setup`, `create-project`, …) hit the same wall
against any fresh non-fleet container and would need the same toggle.

### Export helper (pull-based, may be superseded)
`label-studio.py` gained an `export_project` function and an `export-project`
command that pulls a project's annotations via the export API (JSON / COCO /
zip). This was the original export approach; see the webhook plan below for the
new direction.

## Done (continued)

### `setup-dataset` — create per-annotator dataset projects
`fleet.py setup-dataset <D> --annotator <username>` (or `--all`) creates, in
each selected annotator's container, a project titled `"<D> — <username>"`:
- Builds the label config from `data/source/<D>/classes.txt`.
- Attaches a source local-files storage pointing at `source/<D>` and imports
  one task per image.
- Reuses `create_dataset_project` in `label-studio.py` (which chains
  `build_label_config` → `create_project` → `create_local_files_storage` →
  `make_tasks_from_data_root` → `import_tasks`).
- **Idempotent:** a non-interactive `on_conflict="skip"` path was threaded
  through `create_project`/`create_dataset_project`, so an annotator that
  already has the project is left untouched. This resolved the original caveat
  that `create_project` prompted on a title collision (the default `"prompt"`
  behavior is preserved for the standalone `create-project` command).
- Stopped containers are skipped with a warning; the dataset is validated once
  up front before any container is touched.
- **Verified live:** projects created for `alice`/`bob`, 3 tasks each, storage
  path `/label-studio/data/local/source/dataset1`, and images serve through
  Label Studio's local-files endpoint (HTTP 200).

### Export on submit — webhook + sync (`coco_sync/`)
Built. Annotations land in `target/<dataset>/<username>/` as a custom per-image
`.txt` (header `W H`; one line per object `class_id cx cy w h` + optional
normalized polygon points), compiled to `target/<dataset>/<username>.coco.json`.

- **`coco_sync/`** is a standalone service with its own `docker-compose.yml`
  (port `9000`). It owns all format/validation/write logic and exposes a single
  `POST /hook` endpoint.
- `fleet.py setup-dataset` **auto-registers** a per-project webhook on
  `ANNOTATION_CREATED` / `ANNOTATION_UPDATED` / `ANNOTATIONS_DELETED`, encoding
  annotator/dataset/project in the URL so the receiver needs no project lookup.
- Each webhook is a **pure single-file op**: write/overwrite/delete one image's
  `.txt`. No COCO is touched live (keeps writes cheap and the file always valid
  per-image). Box edits and removals ride along for free, since each submit
  carries the full surviving set; deletes are handled via the delete event.
- **`fleet.py sync`** reconciles `target/` from authoritative LS state and
  (re)builds the `<username>.coco.json`. It supersedes the pull-based
  `export-project` for the normal flow; that command stays as a manual fallback.
- Only **bbox + polygon** are exported (brush was removed project-wide; SAM now
  emits polygons via `SAM_OUTPUT=polygon`).

## To do

### Consensus (later)
Once each annotator's labels collect under `target/<dataset>/<username>/` (with
their `<username>.coco.json`), a `build-consensus --dataset D` step can merge
them (IoU agreement / majority vote) into a single consensus label set. The
layout is already consensus-ready by construction.

## Open decisions
- Annotator roster source — currently usernames are passed ad hoc to
  `fleet.py add`; may want a `configs/annotators.yaml` roster to drive both
  fleet provisioning and `setup-dataset`.
- S3 migration for `source` and `target` (currently local mounts).
