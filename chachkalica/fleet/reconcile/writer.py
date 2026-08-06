"""Filesystem layout and atomic writes for the export target.

Layout under the target root:

    <dataset>/<username>/<image_stem>.txt       # one per annotated image
    <dataset>/<username>.coco.json              # assembled by `fleet.py sync`

The per-image `.txt` uses the image stem (e.g. `img01.txt` for
`img01.jpg`). This is the conventional YOLO layout and prevents duplicate
labels for one image under two different filename conventions.

Writes are atomic (temp file + os.replace) and serialized per path, so the
threaded webhook server never exposes a half-written file or races itself on
the same image.
"""

import os
import tempfile
import threading
from collections import defaultdict
from pathlib import Path

_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


def _lock_for(path: Path) -> threading.Lock:
    return _locks[str(path)]


def labels_dir(target_root: Path, dataset: str, username: str) -> Path:
    return Path(target_root) / dataset / username


def label_filename(image_filename: str) -> str:
    """Return the canonical YOLO label filename for an image filename."""
    return f"{Path(image_filename).stem}.txt"


def label_path(target_root: Path, dataset: str, username: str, image_filename: str) -> Path:
    return labels_dir(target_root, dataset, username) / label_filename(image_filename)


def legacy_label_path(target_root: Path, dataset: str, username: str, image_filename: str) -> Path:
    """Return the old extension-preserving path, for cleanup only."""
    return labels_dir(target_root, dataset, username) / f"{image_filename}.txt"


def coco_path(target_root: Path, dataset: str, username: str) -> Path:
    return Path(target_root) / dataset / f"{username}.coco.json"


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_for(path):
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            os.chmod(fd, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise


def delete(path: Path) -> bool:
    """Remove a label file if present. Returns True if a file was deleted."""
    with _lock_for(path):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
