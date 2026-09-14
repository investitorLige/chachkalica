"""Per-image measurements a model eval can later be sliced by.

Tag analytics can cut an eval by anything that labels its images. Annotators
supply some of that; this supplies the part nobody should have to answer by
hand — how bright the frame is, how much tonal range it has. One pass over a
dataset's images, cached beside them, reused by every eval of that dataset
forever after.

Two things make this cheap enough to be worth caching rather than recomputing:

* it depends **only on the images**, not on any label set, so one file serves
  every annotator's labels and every eval; and
* the numbers are raw floats, bucketed into low/medium/high at read time by
  :mod:`training.services.tag_analytics`, so a partial pass is still usable and
  a changed bucketing rule needs no recompute.

The file lands in ``source/<dataset>/.measures/`` — a dotted subdirectory of the
dataset root, which is inert to every existing walker here (they all filter on
image or ``.txt`` extensions) and, being outside ``images/``, can never be
mistaken for content by something that assumes everything in there is a frame.
"""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from fleet.models import Dataset
from fleet.reconcile import writer
from fleet.services import lsapi
from fleet.services.paths import source_root

MEASURES_DIRNAME = ".measures"
MEASURES_FILENAME = "image_measures.json"
VERSION = 1

#: Bumped when a measure's definition changes, so a stale value is recomputed
#: rather than silently compared against one computed a different way.
ALGO_VERSION = 1

#: The loader walks ``rglob`` over a wider extension set than this app's own
#: flat listing does (``lsapi.list_dataset_images`` uses ``iterdir``, and the
#: trainer's ``data.IMAGE_EXTENSIONS`` adds bmp/tif). Measuring fewer images
#: than the eval scored would silently leave them unbucketed, so take the union
#: and recurse.
IMAGE_EXTENSIONS = lsapi.IMAGE_EXTENSIONS | {".bmp", ".tif", ".tiff"}

#: Pixels sampled per image before the histogram. A stride is an unbiased sample
#: of the pixel distribution, so mean and percentiles survive it; a *resize*
#: would not — an area-average shrinks the tails, by an amount that depends on
#: the source resolution, which would quietly make "contrast" partly a proxy for
#: image size on a mixed-resolution dataset.
SAMPLE_PIXELS = 1_000_000


def measures_dir(dataset_name: str) -> Path:
    return source_root() / dataset_name / MEASURES_DIRNAME


def measures_path(dataset_name: str) -> Path:
    return measures_dir(dataset_name) / MEASURES_FILENAME


def dataset_images(dataset_name: str) -> tuple[Path, list[Path]]:
    """``(images_dir, sorted image paths)`` — recursive, like the trainer's loader."""
    directory = source_root() / dataset_name
    images_dir = lsapi.image_source_dir(directory)
    images = sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images_dir, images


def signature(images: list[Path]) -> str:
    """A digest of the image set, so a changed dataset is noticed.

    Name, size and mtime of every file. Not content hashes: this runs over tens
    of thousands of images and the point is to detect "the dataset moved on",
    not to be a tamper seal. An in-place edit preserving both size and mtime
    would slip through; nothing here pretends otherwise.
    """
    digest = hashlib.sha1()
    for path in images:
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def measure_image(path: Path) -> dict | None:
    """Brightness and contrast for one image, or None if it cannot be read.

    Decoded straight to grey — the same call ``overlap._dhash`` makes — which
    gives Rec.601 luma without a separate colour conversion.

    ``contrast`` is the 5th-to-95th percentile luma spread rather than the
    standard deviation. The question the tag exists to answer is "does this
    model do worse when the tone curve is compressed" — fog, glare, night IR, a
    badly exposed camera — and the percentile spread measures exactly that
    usable dynamic range, robust to one blown highlight or a letterbox bar.
    RMS contrast instead conflates tone compression with *scene complexity*: a
    busy, well-exposed street has a high standard deviation purely for having
    many objects in it, which would make the tag partly a second crowding tag.
    Both fall out of the same histogram, so the RMS figure is kept alongside.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]
    stride = max(1, int(math.sqrt(image.size / SAMPLE_PIXELS)))
    sample = image[::stride, ::stride]

    histogram = cv2.calcHist([sample], [0], None, [256], [0, 256]).ravel()
    total = float(histogram.sum())
    if total <= 0:
        return None
    levels = np.arange(256, dtype=np.float64)
    mean = float((histogram * levels).sum() / total)
    variance = float((histogram * (levels - mean) ** 2).sum() / total)
    cumulative = np.cumsum(histogram) / total
    low = float(np.searchsorted(cumulative, 0.05))
    high = float(np.searchsorted(cumulative, 0.95))

    return {
        "brightness": round(mean, 3),
        "contrast": round(high - low, 3),
        "contrast_rms": round(math.sqrt(variance), 3),
        "width": int(width),
        "height": int(height),
    }


def measure_dataset(dataset: Dataset, *, run=None) -> dict:
    """Measure every image in ``dataset`` and write the sidecar.

    ``run`` is an optional row whose ``images_processed``/``images_failed``
    counters are updated as it goes, so a long pass can be watched rather than
    only awaited.
    """
    images_dir, images = dataset_images(dataset.name)
    if not images:
        raise FileNotFoundError(f"No images found under {images_dir}.")

    if run is not None:
        run.images_total = len(images)
        run.save(update_fields=["images_total"])

    entries: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    seen_basenames: dict[str, int] = {}
    for position, path in enumerate(images, start=1):
        relative = path.relative_to(images_dir).as_posix()
        seen_basenames[path.name] = seen_basenames.get(path.name, 0) + 1
        measured = measure_image(path)
        if measured is None:
            skipped[relative] = "could not be decoded"
        else:
            entries[relative] = measured
        if run is not None and (position % 200 == 0 or position == len(images)):
            run.images_processed = position
            run.images_failed = len(skipped)
            run.save(update_fields=["images_processed", "images_failed"])

    document = {
        "version": VERSION,
        "algo_version": ALGO_VERSION,
        "dataset": dataset.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images_dir": str(images_dir),
        "signature": signature(images),
        "image_count": len(images),
        "measures": [
            {"name": "brightness", "unit": "mean luma, 0-255"},
            {"name": "contrast", "unit": "5th-95th percentile luma spread, 0-255"},
        ],
        # A consumer only has basenames to join on (the match table stores
        # nothing else), so a name under two subdirectories cannot be attributed
        # to either. Named here, refused there -- the same posture the per-box
        # row check takes rather than guessing.
        "basename_collisions": sorted(
            name for name, count in seen_basenames.items() if count > 1),
        "images": entries,
        "skipped": skipped,
    }

    path = measures_path(dataset.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer.write_atomic(path, json.dumps(document))
    if run is not None:
        run.signature = document["signature"]
        run.save(update_fields=["signature"])
    return {
        "path": str(path),
        "images": len(images),
        "measured": len(entries),
        "skipped": len(skipped),
        "collisions": len(document["basename_collisions"]),
    }


def load(dataset_name: str) -> dict | None:
    """The dataset's measures sidecar, or None when it has never been built."""
    path = measures_path(dataset_name)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("version") != VERSION:
        return None
    return document


def freshness(dataset_name: str, document: dict | None) -> tuple[str, str]:
    """``(state, detail)`` for the sidecar: ok / stale / missing.

    Stale rather than wrong: the numbers still describe the images they were
    taken from, so the honest move is to say they are behind and offer a
    recompute, not to throw them away.
    """
    if document is None:
        return "missing", (
            "Brightness and contrast have not been measured for this dataset yet. "
            "Run 'Measure image statistics' on it — no GPU, no re-inference."
        )
    _images_dir, images = dataset_images(dataset_name)
    if signature(images) != document.get("signature"):
        return "stale", (
            f"The images have changed since these measurements were taken "
            f"({len(images)} now, {document.get('image_count')} then). Re-run "
            "'Measure image statistics'."
        )
    if document.get("algo_version") != ALGO_VERSION:
        return "stale", (
            "These measurements were taken by an older definition of the "
            "measures. Re-run 'Measure image statistics'."
        )
    return "ok", (
        f"{len(document.get('images') or {})} image(s) measured on "
        f"{document.get('generated_at', 'an unknown date')}"
    )
