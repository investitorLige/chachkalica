"""On-disk video discovery — the analog of fleet's source-folder listing."""

from pathlib import Path

from fleet.services.paths import videos_root

# Files offered for import. Downloads always normalize to .mp4, but existing
# files in the folder may be any of these container formats.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def list_video_files() -> list[str]:
    """Sorted file names under the videos root that look like videos.

    Mirrors :meth:`fleet.admin.DatasetAdminForm._source_dirs`: a missing or
    unreadable root yields an empty list rather than raising.
    """
    try:
        root = videos_root()
        return sorted(
            p.name for p in root.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    except (FileNotFoundError, NotADirectoryError, OSError):
        return []
