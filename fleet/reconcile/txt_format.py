"""Convert between Label Studio annotation results and the per-image `.txt`.

Label Studio gives geometry as percentages (0-100) of the image; we store
fractions (0-1). The conversion in both directions lives here so the webhook
(LS result -> txt) and sync/COCO build (txt -> objects) cannot drift apart.

An "object" is the in-memory shape both sides speak:

    {"class_id": int, "bbox": (cx, cy, w, h), "polygon": [x0, y0, ...] | None}

all coordinates normalized to [0, 1]. `bbox` is always present (computed from
the polygon's extent when the source region is a polygon); `polygon` is None
for plain bounding-box regions.
"""

# Label Studio result `type` values we export. Anything else (e.g. the smart
# `keypointlabels` SAM trigger) is ignored.
_RECT = "rectanglelabels"
_POLY = "polygonlabels"


def _fmt(value: float) -> str:
    """Compact fixed-point formatting: 6 decimals, trailing zeros trimmed."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _is_axis_aligned_rectangle(polygon: list[float], *, tol: float = 1e-4) -> bool:
    """True if a flat ``[x0, y0, x1, y1, ...]`` polygon is a 4-point box.

    Some YOLO exporters emit bounding boxes in segmentation style — four corner
    points instead of ``cx cy w h``. Such a polygon has exactly four points with
    only two distinct x values and two distinct y values (the box edges), so it
    should round-trip as a rectangle, not a free-form polygon. Genuine polygon
    annotations effectively never satisfy this, so the check is safe.
    """
    if len(polygon) != 8:
        return False

    def distinct(values: list[float]) -> list[float]:
        out: list[float] = []
        for v in values:
            if not any(abs(v - u) <= tol for u in out):
                out.append(v)
        return out

    return len(distinct(polygon[0::2])) == 2 and len(distinct(polygon[1::2])) == 2


def _label_index(value: dict, key: str, name_to_index: dict[str, int]) -> int | None:
    labels = value.get(key) or []
    if not labels:
        return None
    return name_to_index.get(labels[0])


def result_item_to_object(item: dict, name_to_index: dict[str, int]) -> dict | None:
    """Convert a single LS result region into an object, or None to skip it."""
    value = item.get("value") or {}
    region_type = item.get("type")

    if region_type == _RECT:
        class_id = _label_index(value, _RECT, name_to_index)
        if class_id is None:
            return None
        x = value.get("x", 0.0) / 100.0
        y = value.get("y", 0.0) / 100.0
        w = value.get("width", 0.0) / 100.0
        h = value.get("height", 0.0) / 100.0
        # Rotation is dropped: the export formats are axis-aligned.
        return {"class_id": class_id, "bbox": (x + w / 2, y + h / 2, w, h), "polygon": None}

    if region_type == _POLY:
        class_id = _label_index(value, _POLY, name_to_index)
        if class_id is None:
            return None
        points = value.get("points") or []
        polygon = [coord / 100.0 for point in points for coord in point[:2]]
        if len(polygon) < 6:  # fewer than 3 points is not a polygon
            return None
        xs = polygon[0::2]
        ys = polygon[1::2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox = ((min_x + max_x) / 2, (min_y + max_y) / 2, max_x - min_x, max_y - min_y)
        return {"class_id": class_id, "bbox": bbox, "polygon": polygon}

    return None


def index_to_name(names: list[str]) -> dict[int, str]:
    return {index: name for index, name in enumerate(names)}


# Control `from_name`s must match what `lsapi.build_label_config` emits from
# LABEL_TOOLS: bbox -> "bbox" (RectangleLabels), polygon -> "segmentation"
# (PolygonLabels). A prediction whose from_name isn't a control in the project's
# label config simply won't render, so these are kept in lockstep with that map.
_RECT_CONTROL = "bbox"
_POLY_CONTROL = "segmentation"


def object_to_result_item(
    obj: dict, index_to_name: dict[int, str], *, width: int, height: int
) -> dict | None:
    """Convert an object into one LS result region — inverse of result_item_to_object.

    Returns None when the object's class index has no name (skip it). Geometry
    goes from fractions (0-1) back to LS percentages (0-100). original_width/
    height are attached only when known (the app's txt format carries them; bare
    YOLO does not — percentage regions still render without them).
    """
    name = index_to_name.get(obj["class_id"])
    if name is None:
        return None

    polygon = obj.get("polygon")
    if polygon:
        points = [[polygon[i] * 100.0, polygon[i + 1] * 100.0] for i in range(0, len(polygon) - 1, 2)]
        item = {
            "from_name": _POLY_CONTROL,
            "to_name": "image",
            "type": _POLY,
            "value": {"points": points, "polygonlabels": [name]},
        }
    else:
        cx, cy, w, h = obj["bbox"]
        item = {
            "from_name": _RECT_CONTROL,
            "to_name": "image",
            "type": _RECT,
            "value": {
                "x": (cx - w / 2) * 100.0,
                "y": (cy - h / 2) * 100.0,
                "width": w * 100.0,
                "height": h * 100.0,
                "rotation": 0,
                "rectanglelabels": [name],
            },
        }
    if width and height:
        item["original_width"] = int(width)
        item["original_height"] = int(height)
    return item


def results_for_label_text(text: str, names: list[str]) -> list[dict]:
    """Parse an on-disk label `.txt` and convert it to a list of LS result regions."""
    width, height, objects = parse_label_text(text)
    mapping = index_to_name(names)
    results = []
    for obj in objects:
        item = object_to_result_item(obj, mapping, width=width, height=height)
        if item is not None:
            results.append(item)
    return results


def result_dimensions(result: list[dict]) -> tuple[int, int]:
    """Pull the image pixel size from the first region that carries it."""
    for item in result:
        w = item.get("original_width")
        h = item.get("original_height")
        if w and h:
            return int(w), int(h)
    return 0, 0


def result_to_image(result: list[dict], name_to_index: dict[str, int]) -> tuple[int, int, list[dict]]:
    """Convert a full annotation `result` list into (width, height, objects)."""
    objects = []
    for item in result:
        obj = result_item_to_object(item, name_to_index)
        if obj is not None:
            objects.append(obj)
    width, height = result_dimensions(result)
    return width, height, objects


def objects_to_text(width: int, height: int, objects: list[dict]) -> str:
    """Serialize (width, height, objects) into the per-image `.txt` content."""
    lines = [f"{int(width)} {int(height)}"]
    for obj in objects:
        cx, cy, w, h = obj["bbox"]
        parts = [str(obj["class_id"]), _fmt(cx), _fmt(cy), _fmt(w), _fmt(h)]
        if obj.get("polygon"):
            parts.extend(_fmt(coord) for coord in obj["polygon"])
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def parse_text(text: str) -> tuple[int, int, list[dict]]:
    """Parse per-image `.txt` content back into (width, height, objects)."""
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        return 0, 0, []

    header = rows[0].split()
    width = int(float(header[0]))
    height = int(float(header[1])) if len(header) > 1 else 0

    objects: list[dict] = []
    for row in rows[1:]:
        tokens = row.split()
        if len(tokens) < 5:
            continue
        class_id = int(tokens[0])
        cx, cy, w, h = (float(t) for t in tokens[1:5])
        rest = [float(t) for t in tokens[5:]]
        polygon = rest if len(rest) >= 6 else None
        if polygon and _is_axis_aligned_rectangle(polygon):
            polygon = None
        objects.append({"class_id": class_id, "bbox": (cx, cy, w, h), "polygon": polygon})
    return width, height, objects


def parse_yolo_text(text: str) -> tuple[int, int, list[dict]]:
    """Parse a header-less YOLO label file into (0, 0, objects).

    Each line is ``class_id cx cy w h`` (a box) or ``class_id x0 y0 x1 y1 …``
    with an even number of >=6 coordinates (a polygon). All coords are already
    normalized to [0, 1]. YOLO carries no image size, so width/height are 0.
    """
    objects: list[dict] = []
    for raw in text.splitlines():
        tokens = raw.split()
        if len(tokens) < 5:
            continue
        class_id = int(float(tokens[0]))
        coords = [float(t) for t in tokens[1:]]
        if len(coords) >= 6 and len(coords) % 2 == 0:
            polygon = coords
            xs, ys = polygon[0::2], polygon[1::2]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox = ((min_x + max_x) / 2, (min_y + max_y) / 2, max_x - min_x, max_y - min_y)
            # A 4-corner axis-aligned polygon is a bounding box stored in
            # segmentation style — keep it a box rather than a polygon.
            if _is_axis_aligned_rectangle(polygon):
                polygon = None
            objects.append({"class_id": class_id, "bbox": bbox, "polygon": polygon})
        else:
            cx, cy, w, h = coords[:4]
            objects.append({"class_id": class_id, "bbox": (cx, cy, w, h), "polygon": None})
    return 0, 0, objects


def parse_label_text(text: str) -> tuple[int, int, list[dict]]:
    """Parse a label `.txt`, auto-detecting the format.

    A first non-empty line of exactly two integer tokens is the app's own
    ``width height`` header (delegate to parse_text); anything else is treated as
    a header-less standard YOLO file.
    """
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    header = first.split()
    if len(header) == 2 and all(_is_int_token(tok) for tok in header):
        return parse_text(text)
    return parse_yolo_text(text)


def _is_int_token(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def image_filename_from_value(image_value: str) -> str | None:
    """Derive the source image's basename from a task's `data.image` value.

    Handles the local-files form `/data/local-files/?d=source/ds/img01.jpg`
    and plain URLs/paths. Returns e.g. `img01.jpg`.
    """
    from urllib.parse import parse_qs, unquote, urlparse

    if not image_value:
        return None
    parsed = urlparse(image_value)
    query = parse_qs(parsed.query)
    path = unquote(query["d"][0]) if "d" in query else unquote(parsed.path)
    path = path.rstrip("/")
    if not path:
        return None
    return path.rsplit("/", 1)[-1]


def latest_annotation(task: dict) -> dict | None:
    """Return a task's most recent non-cancelled annotation, or None.

    One container == one annotator, so a task normally has a single annotation;
    if several exist we take the highest id (the latest) to match the webhook,
    which always writes the annotation that was just submitted.
    """
    annotations = [
        ann
        for ann in (task.get("annotations") or [])
        if not ann.get("was_cancelled")
    ]
    if not annotations:
        return None
    return max(annotations, key=lambda ann: ann.get("id", 0))
