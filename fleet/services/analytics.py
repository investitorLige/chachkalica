"""Read-only class-distribution analytics for a labeled dataset.

A labeled dataset on disk is ``source/<name>/`` = images (flat or under
``images/``) + a ``classes.txt`` + a ``labels/`` folder of YOLO ``.txt`` files
(see ``datasets.detect_labels``). This walks those label files and reports two
per-class distributions, reading disk only — no Label Studio API, no DB writes —
so the admin can render the result synchronously:

  * **labels** — each class's share of *all* annotation regions.
  * **images** — each class's share of images that contain it, where an image
    with several same-class regions counts once.

Both share one colour per class so a single legend reads against both donuts.
Alongside the distributions it reports object-size buckets, per-image density,
per-class average box size and a set of data-quality flags.

Box areas are computed from the normalized ``w*h`` (a fraction of the image),
so they need no pixel dimensions — YOLO files carry none. Buckets and the
out-of-bounds test are therefore relative to the image, not absolute pixels.
"""

import statistics
from pathlib import Path

from fleet.models import Dataset
from fleet.reconcile.txt_format import parse_label_text
from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services.paths import source_root

# Cycled across classes; picked to stay distinct on a white admin background.
_PALETTE = [
    "#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2",
    "#db2777", "#65a30d", "#ea580c", "#0d9488", "#4f46e5", "#9333ea",
]

# Object-size buckets by box area as a fraction of the image (w*h).
_SMALL_MAX = 0.01   # < 1% of the frame
_MEDIUM_MAX = 0.10  # 1%–10%; larger is "large"
# An image with this many labels or more is flagged as "crowded".
_CROWDED_MIN = 10
# Slack on the [0, 1] box-extent check so float rounding doesn't false-positive.
_BOUNDS_TOL = 1e-3


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


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


def _conic_gradient(rows: list[dict], pct_key: str) -> str:
    """Build a CSS ``conic-gradient`` value from per-class slices.

    Rows are taken in order; zero-share classes are skipped and the final slice
    is closed at exactly 100% so float rounding never leaves a seam in the ring.
    Returns a neutral full ring when there is nothing to show.
    """
    slices = [r for r in rows if r[pct_key] > 0]
    if not slices:
        return "#e5e7eb 0 100%"
    stops: list[str] = []
    cursor = 0.0
    for i, row in enumerate(slices):
        start = cursor
        cursor = 100.0 if i == len(slices) - 1 else cursor + row[pct_key]
        stops.append(f"{row['color']} {start:.3f}% {cursor:.3f}%")
    return ", ".join(stops)


def analyze_dataset(dataset: Dataset) -> dict:
    """Compute class-distribution stats for a labeled dataset (reads disk only).

    Raises ``FileNotFoundError``/``RuntimeError`` (via ``parse_classes_file``)
    when the dataset directory or its ``classes.txt`` is missing or empty.
    """
    src = source_root()
    dataset_dir = src / dataset.name
    classes_file = dataset_dir / "classes.txt"
    lsapi.require_path(classes_file, kind="Classes file")
    names, _tools = lsapi.parse_classes_file(classes_file)

    image_dir = lsapi.image_source_dir(dataset_dir)
    labels_dir = datasets_svc.labels_source_dir(dataset)

    images = [
        p for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in lsapi.IMAGE_EXTENSIONS
    ]

    region_counts = [0] * len(names)  # annotation regions per class
    image_counts = [0] * len(names)   # images containing the class at least once
    area_sums = [0.0] * len(names)    # summed box area per class (for the average)
    labeled_images = 0
    total_regions = 0
    bbox_regions = 0
    polygon_regions = 0

    # Object-size buckets and per-image density.
    size_small = size_medium = size_large = 0
    per_image_counts: list[int] = []
    crowded_images = 0

    # Data-quality tallies.
    images_without_label_file = 0
    empty_label_files = 0
    invalid_class_regions = 0
    out_of_bounds_boxes = 0
    zero_area_boxes = 0

    for img in images:
        label_file = _find_label_file(labels_dir, img.name) if labels_dir.is_dir() else None
        if label_file is None:
            images_without_label_file += 1
            continue
        _w, _h, objects = parse_label_text(label_file.read_text(encoding="utf-8"))
        valid = [o for o in objects if 0 <= o["class_id"] < len(names)]
        invalid_class_regions += len(objects) - len(valid)
        if not valid:
            empty_label_files += 1
            continue

        labeled_images += 1
        per_image_counts.append(len(valid))
        if len(valid) >= _CROWDED_MIN:
            crowded_images += 1

        present: set[int] = set()
        for obj in valid:
            cid = obj["class_id"]
            cx, cy, w, h = obj["bbox"]
            area = max(w, 0.0) * max(h, 0.0)

            region_counts[cid] += 1
            area_sums[cid] += area
            total_regions += 1
            present.add(cid)
            if obj.get("polygon"):
                polygon_regions += 1
            else:
                bbox_regions += 1

            if area < _SMALL_MAX:
                size_small += 1
            elif area < _MEDIUM_MAX:
                size_medium += 1
            else:
                size_large += 1

            if w <= 0 or h <= 0:
                zero_area_boxes += 1
            if (cx - w / 2 < -_BOUNDS_TOL or cy - h / 2 < -_BOUNDS_TOL
                    or cx + w / 2 > 1 + _BOUNDS_TOL or cy + h / 2 > 1 + _BOUNDS_TOL):
                out_of_bounds_boxes += 1
        for cid in present:
            image_counts[cid] += 1

    orphan_count, orphan_examples = _orphan_label_files(labels_dir, images)

    image_count = len(images)
    present_total = sum(image_counts)
    color_map = {names[i]: _PALETTE[i % len(_PALETTE)] for i in range(len(names))}

    rows = [
        {
            "name": names[i],
            "color": color_map[names[i]],
            "region_count": region_counts[i],
            "region_pct": _pct(region_counts[i], total_regions),
            "image_count": image_counts[i],
            "image_pct": _pct(image_counts[i], present_total),
            # Average box area as a percentage of the image.
            "avg_size_pct": round(100 * area_sums[i] / region_counts[i], 2) if region_counts[i] else 0,
        }
        for i in range(len(names))
    ]
    # Legend/table: most-annotated class first.
    rows.sort(key=lambda r: (-r["region_count"], -r["image_count"], r["name"]))

    summary = {
        "image_count": image_count,
        "labeled_images": labeled_images,
        "unlabeled_images": image_count - labeled_images,
        "labeled_pct": _pct(labeled_images, image_count),
        "total_regions": total_regions,
        "class_count": len(names),
        "unused_classes": [r["name"] for r in rows if r["region_count"] == 0],
        "avg_regions_per_image": round(total_regions / image_count, 2) if image_count else 0,
        "avg_regions_per_labeled_image": (
            round(total_regions / labeled_images, 2) if labeled_images else 0
        ),
        "median_regions_per_labeled_image": (
            round(statistics.median(per_image_counts), 1) if per_image_counts else 0
        ),
        "max_regions_per_image": max(per_image_counts) if per_image_counts else 0,
        "crowded_images": crowded_images,
        "crowded_min": _CROWDED_MIN,
        "empty_images": images_without_label_file + empty_label_files,
        "bbox_regions": bbox_regions,
        "polygon_regions": polygon_regions,
    }

    size_dist = [
        {"label": "Small (<1%)", "count": size_small, "pct": _pct(size_small, total_regions), "color": "#f59e0b"},
        {"label": "Medium (1–10%)", "count": size_medium, "pct": _pct(size_medium, total_regions), "color": "#22c55e"},
        {"label": "Large (≥10%)", "count": size_large, "pct": _pct(size_large, total_regions), "color": "#2563eb"},
    ]

    quality = {
        "orphan_label_files": orphan_count,
        "orphan_examples": orphan_examples,
        "empty_label_files": empty_label_files,
        "invalid_class_regions": invalid_class_regions,
        "out_of_bounds_boxes": out_of_bounds_boxes,
        "zero_area_boxes": zero_area_boxes,
    }
    # `empty_label_files` is excluded: an empty .txt is the YOLO convention for a
    # background/negative image, usually intentional (shown in the summary card),
    # not a defect. issue_total counts genuine errors only.
    quality["issue_total"] = (
        orphan_count + invalid_class_regions + out_of_bounds_boxes + zero_area_boxes
    )

    return {
        "dataset": dataset,
        "summary": summary,
        "rows": rows,
        "size_dist": size_dist,
        "quality": quality,
        "label_gradient": _conic_gradient(
            sorted(rows, key=lambda r: (-r["region_count"], r["name"])), "region_pct"
        ),
        "image_gradient": _conic_gradient(
            sorted(rows, key=lambda r: (-r["image_count"], r["name"])), "image_pct"
        ),
    }


def _orphan_label_files(labels_dir: Path, images: list[Path]) -> tuple[int, list[str]]:
    """Count label ``.txt`` files that match no image, mirroring the lookup convention.

    A label file is ``<image_filename>.txt`` or ``<stem>.txt``, so stripping the
    ``.txt`` yields either a full image name (``img.jpg``) or a stem (``img``). A
    file matching neither any image name nor any image stem is an orphan — labels
    with no image to attach to. Returns the total and up to five example names.
    """
    if not labels_dir.is_dir():
        return 0, []
    image_names = {p.name for p in images}
    image_stems = {p.stem for p in images}
    orphans: list[str] = []
    for path in sorted(labels_dir.iterdir()):
        if path.suffix.lower() != ".txt":
            continue
        base = path.name[: -len(".txt")]
        if base not in image_names and base not in image_stems:
            orphans.append(path.name)
    return len(orphans), orphans[:5]
