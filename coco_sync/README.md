# coco_sync

Exports Label Studio annotations from the per-annotator container fleet into
`data/target/<dataset>/<username>/` and compiles them to COCO.

## What it produces

Per annotated image, one custom `.txt` (`<image_filename>.txt`):

```
<width> <height>                               # image pixel size
<class_id> <cx> <cy> <w> <h> [<px0> <py0> ...] # one line per object
```

Coordinates are normalized to `[0, 1]`. The 5-token prefix is always a YOLO-style
bounding box; any tokens after it are normalized polygon points. `class_id` is the
0-based line index in the dataset's `classes.txt`.

`fleet.py sync` compiles all of an annotator's `.txt` files into a single
`data/target/<dataset>/<username>.coco.json` (COCO `category_id` = `class_id + 1`).

## Two write paths

| Path | Trigger | Scope |
|------|---------|-------|
| Webhook (`app.py`) | Label Studio annotation create/update/delete | one image's `.txt` |
| `fleet.py sync` | manual | full rebuild + COCO assembly + prune |

The webhook never touches COCO — it is a cheap single-file op. COCO is assembled
only on `sync`. `sync` is also the authoritative reconciler (recovers from missed
webhook deliveries, bulk imports, and deletions).

## Modules

- `labels.py` — read `classes.txt` → class names / index map
- `txt_format.py` — LS result ↔ objects ↔ `.txt`; image-filename + latest-annotation helpers
- `validate.py` — clamp/drop bad geometry; structural COCO checks
- `writer.py` — target paths + atomic, per-path-locked writes/deletes
- `coco.py` — assemble a COCO dict from a directory of `.txt` files
- `ls_client.py` — fetch a task from a container (token/port from fleet state)
- `app.py` — Flask `POST /hook` receiver

## Run the receiver

```bash
docker compose -f coco_sync/docker-compose.yml up --build
```

Listens on `:9000`. Mounts `../data/source` (ro), `../data/target` (rw), and
`../configs` (ro, for per-annotator tokens). `fleet.py setup-dataset` registers
the webhook on each project automatically.
