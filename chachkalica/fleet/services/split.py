"""Split one dataset into two new sibling datasets by percentage, then delete the source.

A source dataset on disk is ``data/source/<name>/`` = images (flat or under
``images/``) + a ``classes.txt``, and *optionally* a ``labels/`` folder of YOLO
``.txt`` files (see ``datasets.detect_labels``). Splitting partitions the
dataset's own sorted image order (or a shuffled copy of it) at ``left_percent``:
the first share goes to a new ``left_name`` dataset, the rest to ``right_name``.
Each image's label file (if any) travels with whichever side its image lands
on; ``classes.txt`` is copied verbatim to both sides since the class space is
unchanged (unlike ``merge.py``, there's no intersection/remap to do).

Both new datasets use the canonical YOLO layout: images under ``images/`` and,
when the source is labeled, its label files under ``labels/``. Once both are
built successfully, the source dataset's directory is removed from disk and
its ``Dataset`` row deleted — so no duplicate copy of the data lingers. Mirrors
``merge.py``'s "new Dataset row(s) + new on-disk folder(s)" pattern.
"""

import random
import re
import shutil
from pathlib import Path

from fleet.models import Dataset
from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services.paths import source_root

# A bare directory name: letters/digits/_/-/. with no path separators or `..`.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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


def _dataset_images(dataset_dir: Path) -> list[Path]:
    image_dir = lsapi.image_source_dir(dataset_dir)
    return [
        p for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in lsapi.IMAGE_EXTENSIONS
    ]


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


def _populate_side(
    dest_dir: Path, images: list[Path], labels_dir: Path | None, classes_file: Path
) -> dict:
    images_dir = dest_dir / "images"
    labels_out = dest_dir / "labels"
    images_dir.mkdir()
    labels_out.mkdir()

    image_count = 0
    label_count = 0
    for img in images:
        shutil.copy2(img, images_dir / img.name)
        image_count += 1
        label_file = _find_label_file(labels_dir, img.name) if labels_dir is not None else None
        if label_file is None:
            continue
        shutil.copy2(label_file, labels_out / label_file.name)
        label_count += 1

    shutil.copy2(classes_file, dest_dir / "classes.txt")
    return {"images": image_count, "labels": label_count}


def dataset_image_count(dataset: Dataset) -> int:
    """Number of images in a dataset's source dir (for the split preview)."""
    dataset_dir = source_root() / dataset.name
    lsapi.require_path(dataset_dir, kind="Dataset directory")
    return len(_dataset_images(dataset_dir))


def split_by_percentage(
    dataset: Dataset,
    left_name: str,
    right_name: str,
    left_percent: int,
    *,
    shuffle: bool = False,
) -> dict:
    """Build two new sibling datasets from one, then delete the source.

    ``left_percent`` (1-99) of the dataset's images go to ``left_name``, the
    rest to ``right_name``, taken in the dataset's own sorted order unless
    ``shuffle`` asks for a random split instead. Returns a summary dict.
    """
    if dataset.storage_type != Dataset.LOCAL:
        raise RuntimeError(f"Split supports local-storage datasets only; {dataset.name!r} is cloud.")
    if not 1 <= left_percent <= 99:
        raise RuntimeError("Split percentage must be between 1 and 99.")

    left_name = _validate_new_name(left_name)
    right_name = _validate_new_name(right_name)
    if left_name == right_name:
        raise RuntimeError("The two split dataset names must differ.")

    src = source_root()
    dataset_dir = src / dataset.name
    classes_file = dataset_dir / "classes.txt"
    lsapi.require_path(classes_file, kind="Classes file")

    images = _dataset_images(dataset_dir)
    if not images:
        raise RuntimeError(f"{dataset.name!r} has no images to split.")
    if shuffle:
        images = images.copy()
        random.shuffle(images)

    split_at = min(len(images), max(0, round(len(images) * left_percent / 100)))
    left_images, right_images = images[:split_at], images[split_at:]

    left_dir = src / left_name
    right_dir = src / right_name
    if left_dir.exists():
        raise RuntimeError(f"Source directory already exists: {left_dir}")
    if right_dir.exists():
        raise RuntimeError(f"Source directory already exists: {right_dir}")

    labeled = datasets_svc.detect_labels(dataset, persist=False)
    labels_dir = datasets_svc.labels_source_dir(dataset) if labeled else None

    left_dir.mkdir(parents=True)
    try:
        right_dir.mkdir(parents=True)
        try:
            left_counts = _populate_side(left_dir, left_images, labels_dir, classes_file)
            right_counts = _populate_side(right_dir, right_images, labels_dir, classes_file)

            left_row = Dataset.objects.create(name=left_name, storage_type=Dataset.LOCAL)
            datasets_svc.detect_labels(left_row)
            right_row = Dataset.objects.create(name=right_name, storage_type=Dataset.LOCAL)
            datasets_svc.detect_labels(right_row)
        except BaseException:
            shutil.rmtree(right_dir, ignore_errors=True)
            raise
    except BaseException:
        shutil.rmtree(left_dir, ignore_errors=True)
        raise

    # Both new datasets are built — safe to remove the source now, so no
    # duplicate copy of the images/labels is left lying around on disk.
    shutil.rmtree(dataset_dir, ignore_errors=True)
    source_name = dataset.name
    dataset.delete()

    return {
        "source": source_name,
        "left_name": left_name,
        "right_name": right_name,
        "left_dataset_id": left_row.id,
        "right_dataset_id": right_row.id,
        "left_images": left_counts["images"],
        "left_labels": left_counts["labels"],
        "right_images": right_counts["images"],
        "right_labels": right_counts["labels"],
    }
