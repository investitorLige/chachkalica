"""Split a labeled dataset's regions into a box-only dataset and a polygon-only one.

A source dataset on disk is ``data/source/<name>/`` = images (flat or under
``images/``) + a ``classes.txt`` + a ``labels/`` folder of YOLO ``.txt`` files
(see ``datasets.detect_labels``). Every annotation line carries a ``bbox``
*and*, when the region is a polygon, a ``polygon`` (see
``txt_format`` module docstring). ``bbox`` is meant to always be the tight
extent of ``polygon`` when both are present, but a dataset imported from an
external tool may have recorded them independently, so they can disagree.

``check_shape_consistency`` reports how often that happens; ``split_by_shape``
recomputes every polygon region's bbox from its own polygon (so the two can
never disagree in the output) and writes the result as two brand-new sibling
datasets: one with every region reduced to a plain box, one with only the
regions that carry a real polygon. Nothing under the source dataset itself is
modified. Mirrors ``merge.py``'s "new Dataset row + new on-disk folder"
pattern.
"""

import re
import shutil
from pathlib import Path

from fleet.models import Dataset
from fleet.reconcile import writer
from fleet.reconcile.txt_format import objects_to_text, parse_label_text
from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services.data_quality_solve import _bbox_from_polygon
from fleet.services.paths import source_root

# A bare directory name: letters/digits/_/-/. with no path separators or `..`.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Slack when comparing a stored bbox to its polygon's own tight extent.
_BBOX_TOL = 1e-3


def _find_label_file(labels_dir: Path, image_filename: str) -> Path | None:
    """Locate an image's YOLO label file, mirroring ``load_predictions_for_image``.

    Tries ``<image_filename>.txt`` (the app's own convention, e.g. ``img.jpg.txt``)
    then ``<stem>.txt`` (standard YOLO, e.g. ``img.txt``). Returns None if neither.
    """
    for candidate in (
        labels_dir / f"{image_filename}.txt",
        labels_dir / f"{Path(image_filename).stem}.txt",
    ):
        if candidate.exists():
            return candidate
    return None


def _bbox_close(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return all(abs(x - y) <= _BBOX_TOL for x, y in zip(a, b))


def _dataset_images(dataset_dir: Path) -> list[Path]:
    image_dir = lsapi.image_source_dir(dataset_dir)
    return [
        p for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in lsapi.IMAGE_EXTENSIONS
    ]


def check_shape_consistency(dataset: Dataset) -> dict:
    """Count bbox/polygon regions and how often a stored bbox isn't the
    polygon's own tight extent. Reads disk only — no DB writes.
    """
    dataset_dir = source_root() / dataset.name
    labels_dir = datasets_svc.labels_source_dir(dataset)

    total_regions = bbox_regions = polygon_regions = mismatched_bbox_regions = 0
    for img in _dataset_images(dataset_dir):
        label_file = _find_label_file(labels_dir, img.name) if labels_dir.is_dir() else None
        if label_file is None:
            continue
        _w, _h, objects = parse_label_text(label_file.read_text(encoding="utf-8"))
        for obj in objects:
            total_regions += 1
            polygon = obj.get("polygon")
            if not polygon:
                bbox_regions += 1
                continue
            polygon_regions += 1
            tight = _bbox_from_polygon(polygon)
            if tight is not None and not _bbox_close(obj["bbox"], tight):
                mismatched_bbox_regions += 1

    return {
        "dataset": dataset.name,
        "total_regions": total_regions,
        "bbox_regions": bbox_regions,
        "polygon_regions": polygon_regions,
        "mismatched_bbox_regions": mismatched_bbox_regions,
    }


def _tight_bbox(obj: dict) -> tuple[float, float, float, float]:
    """The object's bbox, recomputed from its polygon's own extent when it has one."""
    polygon = obj.get("polygon")
    if not polygon:
        return obj["bbox"]
    return _bbox_from_polygon(polygon) or obj["bbox"]


def _write_classes_file(dest_dir: Path, names: list[str], tool: str) -> None:
    writer.write_atomic(dest_dir / "classes.txt", f"# tools: {tool}\n" + "\n".join(names) + "\n")


def _validate_new_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise RuntimeError(
            f"Invalid dataset name {name!r}: use letters, digits, '.', '_', '-' "
            "and no path separators."
        )
    if Dataset.objects.filter(name=name).exists():
        raise RuntimeError(f"A dataset named {name!r} already exists.")
    return name


def split_by_shape(dataset: Dataset, boxes_name: str, polygons_name: str) -> dict:
    """Build two new sibling datasets from one. Returns a summary dict.

    Skips (without writing anything) when the source has no polygon regions —
    there is nothing to separate from a purely box-labeled dataset.
    """
    if dataset.storage_type != Dataset.LOCAL:
        raise RuntimeError(f"Split supports local-storage datasets only; {dataset.name!r} is cloud.")

    boxes_name = _validate_new_name(boxes_name)
    polygons_name = _validate_new_name(polygons_name)
    if boxes_name == polygons_name:
        raise RuntimeError("Boxes and polygons dataset names must differ.")

    src = source_root()
    dataset_dir = src / dataset.name
    classes_file = dataset_dir / "classes.txt"
    lsapi.require_path(classes_file, kind="Classes file")
    names, _tools = lsapi.parse_classes_file(classes_file)

    consistency = check_shape_consistency(dataset)
    if consistency["polygon_regions"] == 0:
        return {**consistency, "skipped": True, "reason": "no polygon regions found"}

    boxes_dir = src / boxes_name
    polygons_dir = src / polygons_name
    if boxes_dir.exists():
        raise RuntimeError(f"Source directory already exists: {boxes_dir}")
    if polygons_dir.exists():
        raise RuntimeError(f"Source directory already exists: {polygons_dir}")

    boxes_dir.mkdir(parents=True)
    try:
        polygons_dir.mkdir(parents=True)
        try:
            boxes_images = boxes_dir / "images"
            boxes_labels = boxes_dir / "labels"
            polygons_images = polygons_dir / "images"
            polygons_labels = polygons_dir / "labels"
            for path in (boxes_images, boxes_labels, polygons_images, polygons_labels):
                path.mkdir()

            labels_dir = datasets_svc.labels_source_dir(dataset)
            images = 0
            box_labels = 0
            polygon_labels = 0
            for img in _dataset_images(dataset_dir):
                shutil.copy2(img, boxes_images / img.name)
                shutil.copy2(img, polygons_images / img.name)
                images += 1

                label_file = _find_label_file(labels_dir, img.name) if labels_dir.is_dir() else None
                if label_file is None:
                    continue
                width, height, objects = parse_label_text(label_file.read_text(encoding="utf-8"))
                if not objects:
                    continue

                # Written via objects_to_text (always header'd) rather than bare YOLO rows:
                # a header-less polygon row is read back as one flat coordinate list with no
                # bbox/polygon split (see txt_format.parse_yolo_text), which would silently
                # corrupt the very geometry this split is meant to make trustworthy.
                box_objects = [
                    {"class_id": obj["class_id"], "bbox": _tight_bbox(obj), "polygon": None}
                    for obj in objects
                ]
                writer.write_atomic(
                    boxes_labels / f"{img.name}.txt", objects_to_text(width, height, box_objects)
                )
                box_labels += 1

                polygon_objects = [
                    {"class_id": obj["class_id"], "bbox": _tight_bbox(obj), "polygon": obj["polygon"]}
                    for obj in objects
                    if obj.get("polygon")
                ]
                if polygon_objects:
                    writer.write_atomic(
                        polygons_labels / f"{img.name}.txt",
                        objects_to_text(width, height, polygon_objects),
                    )
                    polygon_labels += 1

            _write_classes_file(boxes_dir, names, "bbox")
            _write_classes_file(polygons_dir, names, "polygon")

            boxes_row = Dataset.objects.create(name=boxes_name, storage_type=Dataset.LOCAL)
            datasets_svc.detect_labels(boxes_row)
            polygons_row = Dataset.objects.create(name=polygons_name, storage_type=Dataset.LOCAL)
            datasets_svc.detect_labels(polygons_row)
        except BaseException:
            shutil.rmtree(polygons_dir, ignore_errors=True)
            raise
    except BaseException:
        shutil.rmtree(boxes_dir, ignore_errors=True)
        raise

    return {
        **consistency,
        "skipped": False,
        "boxes_name": boxes_name,
        "polygons_name": polygons_name,
        "boxes_dataset_id": boxes_row.id,
        "polygons_dataset_id": polygons_row.id,
        "images": images,
        "box_labels": box_labels,
        "polygon_labels": polygon_labels,
    }
