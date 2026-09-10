"""On-disk video discovery for the studio's own library.

The twin of :mod:`videos.services.videos`, differing only in which root it
scans — that module hardcodes the detection videos root, and re-deriving the
twenty lines here is shorter than parameterizing it.
"""

from marketing_studio.services.paths import marketing_videos_root

# Files offered for import. Downloads always normalize to .mp4, but existing
# files in the folder may be any of these container formats.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def list_video_files() -> list[str]:
    """Sorted file names under the studio's video root that look like videos.

    A missing or unreadable root yields an empty list rather than raising — an
    operator who has not pointed ``marketing_videos_dir`` anywhere yet should
    get an empty dropdown, not a 500.
    """
    try:
        root = marketing_videos_root()
        return sorted(
            p.name for p in root.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    except (FileNotFoundError, NotADirectoryError, OSError):
        return []
