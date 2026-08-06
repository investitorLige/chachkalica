"""Environment-driven settings for a build node.

Everything is overridable so the same image serves a bare-metal GPU box, a
container with a small disk, and the profile-gated service in the repo's own
docker-compose. Defaults assume the container layout the Dockerfile sets up.
"""

import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# Where a build's uploads, intermediates and finished tarball live.
WORK_DIR = Path(os.environ.get("BUILDNODE_WORK_DIR", "/var/lib/buildnode/builds"))

# How long a FINISHED build is kept before its scratch dir is reaped. Long enough
# that a caller whose download was interrupted can retry, short enough that a
# forgotten build doesn't hold ~200 MB forever.
BUILD_TTL = _int("BUILDNODE_BUILD_TTL", 24 * 60 * 60)

# Lines of build log retained per build and returned by the status endpoint.
LOG_LINES = _int("BUILDNODE_LOG_LINES", 400)

# Reject an upload larger than this outright. A model ONNX is tens to a few
# hundred MB; anything far past that is a mistake or an attempt to fill the disk.
MAX_UPLOAD_MB = _int("BUILDNODE_MAX_UPLOAD_MB", 2048)

PORT = _int("BUILDNODE_PORT", 8300)
