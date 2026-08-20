"""Download a video from a URL to an MP4 on disk.

Vendored from ``video_dataset_tools/youtube_downloader.py`` — the sibling repo is
not copied into the Docker image, so the ~30 lines we actually need live here.
Unlike the original ``download_videos`` (which returns ``None`` and swallows
``DownloadError``), :func:`download` returns the saved path and raises on failure
so the background job can record it on the row.

Requires ``yt-dlp`` (+ ``ffmpeg`` on PATH for the mp4 merge/convert). A JS
runtime (deno) lets ``yt-dlp-ejs`` solve YouTube's player sig/n challenge, and
``bgutil-ytdlp-pot-provider`` (talking to the ``yt-pot-server`` companion
container) mints the separate PO Token YouTube now requires on the actual
videoplayback fetch — the JS challenge alone no longer unlocks a download.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

JS_RUNTIMES = ["deno", "node", "nodejs", "bun", "qjs", "quickjs"]


def sanitize_title(title: str) -> str:
    title = re.sub(r"https?://\S+", "", title)
    title = re.sub(r"www\.\S+", "", title)
    title = re.sub(r'[<>:"/\\|?*]', "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title or "video"


def build_format(quality: int | None) -> str:
    if quality:
        return f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]/best"
    return "bestvideo+bestaudio/best"


def _detect_js_runtime() -> tuple[str, str] | None:
    """Return (runtime_name, path) for the first JS runtime found, or None."""
    for name in JS_RUNTIMES:
        path = shutil.which(name)
        if path:
            canonical = "node" if name in ("node", "nodejs") else name
            canonical = "quickjs" if name in ("qjs", "quickjs") else canonical
            return canonical, path
    return None


def download(url: str, output_dir: str | Path, quality: int | None = None) -> Path:
    """Download ``url`` into ``output_dir`` as an mp4; return the saved path.

    Raises ``RuntimeError`` on any download/convert failure or if the expected
    output file is missing afterwards.
    """
    import yt_dlp  # imported lazily so the module loads even where yt-dlp isn't installed

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts: dict = {
        "format": build_format(quality),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
        "quiet": True,
        "no_warnings": True,
    }
    # A JS runtime lets yt-dlp solve YouTube's player challenge and reach the
    # >360p streams (the exact problem this was tested against). We do NOT pin
    # ``player_client`` — with current yt-dlp the built-in default picks a client
    # that serves HD, whereas forcing web/tv clients trips "format not available"
    # / DRM errors on many videos.
    runtime = _detect_js_runtime()
    if runtime:
        name, path = runtime
        ydl_opts["js_runtimes"] = {name: {"path": path}}
        ydl_opts["remote_components"] = ["ejs:github"]

    # Points bgutil-ytdlp-pot-provider (a yt-dlp plugin, active once installed —
    # no opt-in flag needed) at its companion BotGuard-solving server. Without a
    # PO Token, YouTube 403s the videoplayback fetch even after the JS challenge
    # above is solved correctly.
    pot_server_url = os.environ.get("YT_POT_SERVER_URL", "http://127.0.0.1:4416")
    ydl_opts["extractor_args"] = {"youtubepot-bgutilhttp": {"base_url": [pot_server_url]}}
    # This network's TLS-inspecting proxy MITMs the fetch the POT server makes to
    # www.google.com (see yt_pot_server/Dockerfile); yt-dlp forwards this flag to
    # the server as `disable_tls_verification`, which is what actually silences
    # that failure — NODE_TLS_REJECT_UNAUTHORIZED on the server alone does not,
    # since bgutil's own HTTPS agent hardcodes rejectUnauthorized from this flag.
    ydl_opts["nocheckcertificate"] = True

    try:
        with yt_dlp.YoutubeDL(dict(ydl_opts)) as ydl:
            info = ydl.extract_info(url, download=False)
            clean_title = sanitize_title(info.get("title", "video"))

        per_video_opts = dict(ydl_opts)
        per_video_opts["outtmpl"] = str(output_dir / f"{clean_title}.%(ext)s")
        with yt_dlp.YoutubeDL(per_video_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(str(exc)) from exc

    result = output_dir / f"{clean_title}.mp4"
    if not result.is_file():
        # Merge/convert should always yield the mp4; fall back to any file with
        # the sanitized stem before giving up.
        matches = sorted(output_dir.glob(f"{glob_escape(clean_title)}.*"))
        if not matches:
            raise RuntimeError(f"download produced no file for {url!r}")
        result = matches[0]
    return result


def glob_escape(text: str) -> str:
    """Escape glob metacharacters in a literal filename stem."""
    return re.sub(r"([\[\]*?])", r"[\1]", text)
