"""On-disk discovery, browser uploads, and probing for the VLM video library.

The listing half mirrors :mod:`videos.services.videos`, pointed at
``vlm_videos_root()``. The upload and probe halves are new: nothing in this
project accepted a browser file upload before, and nothing needed a video's own
frame rate until "sample at N fps" had to become a frame stride.
"""

import logging
import re
from pathlib import Path

from fleet.services.paths import vlm_videos_root

logger = logging.getLogger(__name__)

# Same set videos.services.videos offers for import: downloads normalize to
# .mp4, but a file already in the folder may be any of these containers.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def list_vlm_video_files() -> list[str]:
    """Sorted file names under the VLM videos root that look like videos.

    A missing or unreadable root yields an empty list rather than raising, so
    the Add page still renders on a fresh install.
    """
    try:
        root = vlm_videos_root()
        return sorted(
            p.name for p in root.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    except (FileNotFoundError, NotADirectoryError, OSError):
        return []


def _safe_stem(name: str) -> str:
    """A file stem safe to write to disk, in the spirit of ``downloader.sanitize_title``."""
    stem = re.sub(r'[<>:"/\\|?*]', "", Path(name).stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or "video"


def unique_filename(preferred: str) -> str:
    """A file name under the VLM root that no existing file uses."""
    root = vlm_videos_root()
    suffix = Path(preferred).suffix.lower() or ".mp4"
    stem = _safe_stem(preferred)
    candidate = f"{stem}{suffix}"
    n = 2
    while (root / candidate).exists():
        candidate = f"{stem} ({n}){suffix}"
        n += 1
    return candidate


def save_upload(uploaded_file) -> str:
    """Stream an uploaded video to the VLM root; return the file name written.

    Streamed in chunks rather than read whole: Django has already spooled
    anything past ``FILE_UPLOAD_MAX_MEMORY_SIZE`` to a temp file, and a video is
    exactly the case that setting exists for.
    """
    root = vlm_videos_root()
    root.mkdir(parents=True, exist_ok=True)
    filename = unique_filename(uploaded_file.name)
    destination = root / filename
    try:
        with open(destination, "wb") as fh:
            for chunk in uploaded_file.chunks():
                fh.write(chunk)
    except Exception:
        # Never leave a half-written file behind to be "imported" later.
        destination.unlink(missing_ok=True)
        raise
    return filename


def probe(path: Path) -> dict:
    """Duration, fps and frame size for a video, or empty values if unreadable.

    ``source_fps`` is the one that matters: it is what turns the operator's
    requested frames-per-second into a frame stride. Probing never raises — a
    video we cannot read here should still produce a row the operator can see
    and delete, and the run itself falls back to a nominal fps.
    """
    empty = {"duration_seconds": None, "source_fps": None, "width": None, "height": None}
    try:
        import cv2
    except ImportError:  # pragma: no cover - opencv is a hard dependency of the image
        logger.warning("opencv unavailable; skipping probe of %s", path)
        return empty

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return empty
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    except Exception:  # noqa: BLE001 - a probe must never break the add flow
        logger.exception("probe failed for %s", path)
        return empty
    finally:
        cap.release()

    return {
        "duration_seconds": (frames / fps) if fps > 0 and frames > 0 else None,
        "source_fps": fps if fps > 0 else None,
        "width": width or None,
        "height": height or None,
    }


def apply_probe(video) -> None:
    """Fill a :class:`~vlm.models.VlmVideo`'s geometry fields from its file."""
    if not video.exists():
        return
    values = probe(video.path())
    for field, value in values.items():
        setattr(video, field, value)
    video.save(update_fields=[*values.keys(), "updated_at"])
