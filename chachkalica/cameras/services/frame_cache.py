"""Redis-backed cache of each camera's most recent frame.

Written continuously by the ``stream_cameras`` management command, read by
the admin's MJPEG view — this is what decouples "one RTSP connection per
camera" from "however many admin tabs happen to be watching it". Reuses the
same Redis connection django-rq already holds (``REDIS_URL`` in settings),
rather than opening a second one.
"""

import django_rq

_KEY_PREFIX = "camera:"
_KEY_SUFFIX = ":frame"

# A bit longer than the worker's push interval (see FRAME_PUSH_INTERVAL_SECONDS
# in stream_cameras.py), so a dead/stalled worker's last frame expires instead
# of looking falsely live forever.
FRAME_TTL_SECONDS = 8


def _key(camera_id) -> str:
    return f"{_KEY_PREFIX}{camera_id}{_KEY_SUFFIX}"


def _redis():
    return django_rq.get_connection("default")


def set_frame(camera_id, jpeg_bytes: bytes) -> None:
    _redis().setex(_key(camera_id), FRAME_TTL_SECONDS, jpeg_bytes)


def get_frame(camera_id):
    return _redis().get(_key(camera_id))
