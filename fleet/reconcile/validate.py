"""Validation for parsed objects and assembled COCO documents.

`clean_objects` is forgiving — it drops or clamps bad geometry and reports it,
so one malformed region never blocks a whole annotation from being written.
`validate_coco` is a structural sanity check run once after assembly.
"""

import math


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def clean_objects(objects: list[dict], num_classes: int) -> tuple[list[dict], list[str]]:
    """Return (valid objects, warnings). Coordinates are clamped to [0, 1]."""
    clean: list[dict] = []
    warnings: list[str] = []

    for index, obj in enumerate(objects):
        class_id = obj.get("class_id")
        if not isinstance(class_id, int) or not (0 <= class_id < num_classes):
            warnings.append(f"object {index}: class_id {class_id} out of range")
            continue

        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4 or not _finite(*bbox):
            warnings.append(f"object {index}: invalid bbox {bbox!r}")
            continue
        cx, cy, w, h = (min(1.0, max(0.0, float(v))) for v in bbox)
        if w <= 0 or h <= 0:
            warnings.append(f"object {index}: non-positive bbox size")
            continue

        polygon = obj.get("polygon")
        if polygon is not None:
            if len(polygon) < 6 or len(polygon) % 2 != 0 or not _finite(*polygon):
                warnings.append(f"object {index}: invalid polygon, dropping points")
                polygon = None
            else:
                polygon = [min(1.0, max(0.0, float(v))) for v in polygon]

        clean.append({"class_id": class_id, "bbox": (cx, cy, w, h), "polygon": polygon})

    return clean, warnings


def validate_coco(doc: dict) -> list[str]:
    """Structural checks on an assembled COCO dict. Returns a list of errors."""
    errors: list[str] = []
    image_ids = {img["id"] for img in doc.get("images", [])}
    category_ids = {cat["id"] for cat in doc.get("categories", [])}

    seen_ann_ids: set[int] = set()
    for ann in doc.get("annotations", []):
        ann_id = ann.get("id")
        if ann_id in seen_ann_ids:
            errors.append(f"duplicate annotation id {ann_id}")
        seen_ann_ids.add(ann_id)
        if ann.get("image_id") not in image_ids:
            errors.append(f"annotation {ann_id}: unknown image_id {ann.get('image_id')}")
        if ann.get("category_id") not in category_ids:
            errors.append(f"annotation {ann_id}: unknown category_id {ann.get('category_id')}")

    return errors
