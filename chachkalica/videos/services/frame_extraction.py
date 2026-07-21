"""Sample frames from a video into a brand-new dataset.

Ports the sampling loop from the sibling ``video_dataset_tools/frame_extractor.py``
(not vendored as a dependency, same reasoning as ``videos.services.downloader``).
Builds the dataset on disk and registers it exactly like
``fleet.services.merge.merge_datasets`` does for merges: images under
``images/``, a ``classes.txt`` at the dataset root, then a ``Dataset`` row —
so the result shows up in the Dataset tab as soon as the job finishes.
"""

import re
import shutil
from pathlib import Path

from django.conf import settings

from fleet.models import Dataset
from fleet.reconcile import writer
from fleet.services import datasets as datasets_svc
from fleet.services.paths import source_root

# Same bare-directory-name rule as fleet.services.merge._NAME_RE.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Fixed confidence for the "only with people" filter — not exposed in the form.
_PEOPLE_SCORE_THRESHOLD = 0.5


def unique_dataset_name(base: str) -> str:
    """A dataset name derived from ``base`` that collides with no ``Dataset`` row
    or existing source directory — used to prefill the "Extract frames…" form."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", (base or "").strip()) or "video"
    if not _NAME_RE.match(base):
        base = "video"
    src = source_root()
    candidate = base
    n = 2
    while Dataset.objects.filter(name=candidate).exists() or (src / candidate).exists():
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def _extract_frames(
    video_path: Path, images_dir: Path, every_n_frames: int,
    max_frames: int | None, image_ext: str,
) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    frame_idx = 0
    stem = video_path.stem
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % every_n_frames == 0:
                out_path = images_dir / f"{stem}_frame_{saved:03d}{image_ext}"
                cv2.imwrite(str(out_path), frame)
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
            frame_idx += 1
    finally:
        capture.release()
    return saved


def _difference_hash(gray, hash_size: int = 8) -> int:
    """dHash of a grayscale image — ports ``similar_frame_filter.difference_hash``."""
    import cv2

    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def _dedupe_similar_frames(images_dir: Path, threshold: int) -> int:
    """Drop frames whose dHash is within ``threshold`` of the previously kept frame.

    Ports ``video_dataset_tools/similar_frame_filter.py``'s single-previous-frame
    window (``window=1``) — frames are visited in name order, which matches
    extraction order (``_frame_000``, ``_frame_001``, ...).
    """
    import cv2

    dropped = 0
    kept_hash = None
    for path in sorted(images_dir.iterdir()):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        digest = _difference_hash(gray)
        if kept_hash is not None and (digest ^ kept_hash).bit_count() <= threshold:
            path.unlink()
            dropped += 1
            continue
        kept_hash = digest
    return dropped


def _resolve_checkpoint(checkpoint: str) -> Path:
    p = Path(checkpoint)
    return p if p.is_absolute() else Path(settings.BASE_DIR) / p


def _has_person(image_path: Path, checkpoint: Path) -> bool:
    from training.services import runner

    response = runner.predict_image({
        "model_checkpoint": str(checkpoint),
        "image_path": str(image_path),
        "pipeline": "raw",
        "score_threshold": _PEOPLE_SCORE_THRESHOLD,
    })
    return any(box.get("class_name") == "person" for box in response.get("boxes", []))


def _filter_people_frames(images_dir: Path) -> int:
    """Drop frames with no person detected, via the trainer's people-detector engine."""
    from training.models import DEFAULT_PERSON_DETECTOR_CHECKPOINT

    checkpoint = _resolve_checkpoint(DEFAULT_PERSON_DETECTOR_CHECKPOINT)
    dropped = 0
    for path in sorted(images_dir.iterdir()):
        if not _has_person(path, checkpoint):
            path.unlink()
            dropped += 1
    return dropped


def extract_to_dataset(
    video, dataset_name: str, every_n_frames: int, max_frames_per_video: int | None,
    image_ext: str, classes: list[str],
    only_with_people: bool = False, similarity_threshold: int | None = None,
) -> dict:
    """Build ``<source_root>/<dataset_name>/{images/,classes.txt}`` and register it."""
    dataset_name = (dataset_name or "").strip()
    if not _NAME_RE.match(dataset_name):
        raise RuntimeError(
            f"Invalid dataset name {dataset_name!r}: use letters, digits, '.', '_', '-' "
            "and no path separators."
        )
    if Dataset.objects.filter(name=dataset_name).exists():
        raise RuntimeError(f"A dataset named {dataset_name!r} already exists.")
    if not classes:
        raise RuntimeError("Provide at least one class name.")

    src = source_root()
    new_dir = src / dataset_name
    if new_dir.exists():
        raise RuntimeError(f"Source directory already exists: {new_dir}")

    new_dir.mkdir(parents=True)
    try:
        images_dir = new_dir / "images"
        images_dir.mkdir()

        count = _extract_frames(
            video.path(), images_dir, every_n_frames, max_frames_per_video, image_ext,
        )
        if count == 0:
            raise RuntimeError(
                "No frames extracted — check the video file and the frame interval."
            )

        dropped_similar = None
        if similarity_threshold is not None:
            dropped_similar = _dedupe_similar_frames(images_dir, similarity_threshold)

        dropped_no_person = None
        if only_with_people:
            dropped_no_person = _filter_people_frames(images_dir)

        remaining = sum(1 for _ in images_dir.iterdir())
        if remaining == 0:
            raise RuntimeError(
                "No frames left after filtering — relax the similarity/people filters."
            )

        writer.write_atomic(new_dir / "classes.txt", "\n".join(classes) + "\n")

        dataset_row = Dataset.objects.create(name=dataset_name, storage_type=Dataset.LOCAL)
        datasets_svc.detect_labels(dataset_row)  # no labels/ yet -> has_labels stays False
    except BaseException:
        shutil.rmtree(new_dir, ignore_errors=True)
        raise

    return {
        "dataset_id": dataset_row.id, "dataset_name": dataset_name, "frames": remaining,
        "dropped_similar": dropped_similar, "dropped_no_person": dropped_no_person,
    }
