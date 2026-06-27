"""Label Studio annotation export pipeline.

Two entry points share one set of pure-logic modules:

  - The webhook API (`app.py`, run via docker-compose) writes one custom
    per-image `.txt` to `target/<dataset>/<username>/` on every annotation
    submit/update/delete. This is a cheap single-file write — no COCO touched.

  - `fleet.py sync` pulls the authoritative state from each Label Studio
    container, reconciles those same `.txt` files, then assembles them into a
    single `target/<dataset>/<username>.coco.json`.

The `.txt` format is the live working format; COCO is the compiled artifact.

Per-image `.txt` layout (`<image_filename>.txt`, e.g. `img01.jpg.txt`):

    <width> <height>                          # image pixel size (header)
    <class_id> <cx> <cy> <w> <h> [<px0> <py0> ...]   # one line per object

Coordinates are normalized to [0, 1]. The 5-token prefix is always a bounding
box (YOLO-style centre/size); any tokens after it are normalized polygon
points. A line with only the 5 prefix tokens is a box-only object.
"""
