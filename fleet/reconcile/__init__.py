"""Pure annotation-format logic shared by the webhook view and the sync service.

Moved verbatim from the standalone `coco_sync` package: these modules have no
Flask, Django, or YAML dependency — they convert between Label Studio results,
the per-image `.txt` working format, and the assembled COCO document.

Per-image `.txt` layout (`<image_filename>.txt`, e.g. `img01.jpg.txt`):

    <width> <height>                                 # image pixel size (header)
    <class_id> <cx> <cy> <w> <h> [<px0> <py0> ...]   # one line per object

Coordinates are normalized to [0, 1]; the 5-token prefix is a YOLO-style box,
any trailing tokens are normalized polygon points.
"""
