"""Range-aware file streaming for the admin's HTML5 video players.

Django's ``FileResponse`` does not implement HTTP Range on its own, so a player
pointed straight at it cannot seek. This serves a ``Range`` request as a bounded
``206`` instead.

Extracted because three admin pages now need it — the raw video player, the
annotated inference-output player, and the VLM live view — and the logic had
already been copy-pasted once.
"""

import re
from pathlib import Path

from django.http import FileResponse, StreamingHttpResponse

CHUNK_SIZE = 8192

_RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


def file_chunks(path, start: int, length: int, chunk_size: int = CHUNK_SIZE):
    with open(path, "rb") as fh:
        fh.seek(start)
        remaining = length
        while remaining > 0:
            data = fh.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def serve(request, path: Path, content_type: str = "video/mp4"):
    """Stream ``path`` to ``request``, honouring HTTP Range so seeking works.

    Callers check that the file exists and raise their own ``Http404`` first —
    the message is more useful when it names the row that was missing.
    """
    size = path.stat().st_size

    match = _RANGE_RE.match(request.headers.get("Range", "").strip())
    if not match:
        resp = FileResponse(open(path, "rb"), content_type=content_type)
        resp["Accept-Ranges"] = "bytes"
        return resp

    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else size - 1
    end = min(end, size - 1)
    start = min(start, end)
    length = end - start + 1

    resp = StreamingHttpResponse(
        file_chunks(path, start, length), status=206, content_type=content_type
    )
    resp["Content-Length"] = str(length)
    resp["Content-Range"] = f"bytes {start}-{end}/{size}"
    resp["Accept-Ranges"] = "bytes"
    return resp
