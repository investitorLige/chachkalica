"""Resolve a YOLO-style dataset root into images dir + labels dir + class map.

Convention (matches ``chachkalica/fleet/services/lsapi.py`` and
``chachkalica/coco_sync/labels.py``): a dataset root has an ``images/``
subdirectory (or images sitting flat in the root), a ``labels/`` subdirectory
of YOLO ``.txt`` files, and a ``classes.txt`` with one class name per line
(0-indexed, blank/``#`` lines skipped).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def resolve_dataset(root: str | Path) -> Tuple[Path, Path, Dict[int, str]]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    images_subdir = root / "images"
    if images_subdir.is_dir() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in images_subdir.iterdir()):
        images_dir = images_subdir
    else:
        images_dir = root

    labels_dir = root / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels/ directory under dataset root: {root}")

    classes_path = root / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Missing classes.txt under dataset root: {root}")
    classes = _load_classes(classes_path)

    return images_dir, labels_dir, classes


def _load_classes(classes_path: Path) -> Dict[int, str]:
    names = []
    for raw_line in classes_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    if not names:
        raise ValueError(f"{classes_path} has no class names")
    return {index: name for index, name in enumerate(names)}


def build_dataloader(images_dir: Path, labels_dir: Path, classes: Dict[int, str], batch_size: int, num_workers: int):
    from torch.utils.data import DataLoader

    from friendy_chachkalica.data import YoloDetectionDataset, detection_collate_fn

    dataset = YoloDetectionDataset(images_dir, labels_dir, classes, require_labels=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )
