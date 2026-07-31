"""Redis-backed cache of each camera's most recent frame.

Written continuously by the ``stream_cameras`` management command, read by
the admin's MJPEG view — this is what decouples "one RTSP connection per
camera" from "however many admin tabs happen to be watching it". Reuses the
same Redis connection django-rq already holds (``REDIS_URL`` in settings),
rather than opening a second one.

Three frames are cached per camera, and which one a reader wants depends on
whether it is a model or a person:

- ``:frame`` — the *raw* frame ``stream_cameras`` pushes straight off the RTSP
  stream, at full sensor resolution. This is the one the model gets.
- ``:preview`` — the same frame downscaled for viewing (see
  ``cameras.services.preview`` for why the producer, not the viewer, does it).
- ``:annotated`` — preview-sized, with detection boxes burnt in, pushed by the
  ``infer_cameras`` worker (which reads ``:frame`` from here rather than
  opening its own RTSP session — see ``docs/camera-live-inference.md``). The
  slowest of the three, and simply absent when live inference is off.

A fourth key, ``:viewed``, isn't a frame at all — it's a heartbeat the admin's
annotated MJPEG stream refreshes every poll, so ``infer_cameras`` can tell
"someone has the Live view open" from "nobody's watching" and only run the
model in the former case.
"""

import json

import django_rq

_KEY_PREFIX = "camera:"
_KEY_SUFFIX = ":frame"
_PREVIEW_SUFFIX = ":preview"
_ANNOTATED_SUFFIX = ":annotated"
_BOXES_SUFFIX = ":boxes"
_VIEWED_SUFFIX = ":viewed"

# A bit longer than the worker's push interval (see FRAME_PUSH_INTERVAL_SECONDS
# in stream_cameras.py), so a dead/stalled worker's last frame expires instead
# of looking falsely live forever.
FRAME_TTL_SECONDS = 8

# Annotated frames arrive at CameraInference.target_fps (default 2/s), far
# slower than the raw push rate, and a single slow inference call can stretch
# the gap further — so they need a much more forgiving TTL than the raw frame
# to avoid a viewer seeing gaps between perfectly healthy detections.
ANNOTATED_TTL_SECONDS = 30

# The MJPEG generator refreshes this every _MJPEG_POLL_SECONDS (0.1s) while a
# viewer is connected, so it's always fresh well before it could expire; a TTL
# a few multiples of infer_cameras' RECONCILE_INTERVAL_SECONDS (15s) means a
# closed tab reads as "not viewed" within one or two reconcile ticks, not
# instantly (which would flap a worker on a momentary network hiccup) and not
# for minutes after (which would burn GPU time on an empty tab).
VIEWED_TTL_SECONDS = 30


def _key(camera_id) -> str:
    return f"{_KEY_PREFIX}{camera_id}{_KEY_SUFFIX}"


def _preview_key(camera_id) -> str:
    return f"{_KEY_PREFIX}{camera_id}{_PREVIEW_SUFFIX}"


def _annotated_key(camera_id) -> str:
    return f"{_KEY_PREFIX}{camera_id}{_ANNOTATED_SUFFIX}"


def _boxes_key(camera_id) -> str:
    return f"{_KEY_PREFIX}{camera_id}{_BOXES_SUFFIX}"


def _viewed_key(camera_id) -> str:
    return f"{_KEY_PREFIX}{camera_id}{_VIEWED_SUFFIX}"


def _redis():
    return django_rq.get_connection("default")


def set_frame(camera_id, jpeg_bytes: bytes) -> None:
    _redis().setex(_key(camera_id), FRAME_TTL_SECONDS, jpeg_bytes)


def get_frame(camera_id):
    return _redis().get(_key(camera_id))


def set_preview_frame(camera_id, jpeg_bytes: bytes) -> None:
    # Same TTL as the raw frame it was made from: they're pushed together, so a
    # preview outliving its source would only ever mislead a viewer.
    _redis().setex(_preview_key(camera_id), FRAME_TTL_SECONDS, jpeg_bytes)


def get_preview_frame(camera_id):
    """The downscaled raw frame, for viewers rather than models.

    Falls back to the full-resolution frame so the live view still shows
    something during the seconds between a ``camera-workers`` restart and its
    first preview push. That fallback streams 4K to the browser, which is the
    entire cost this key exists to avoid — it's a stopgap for a cold cache, not
    a mode to sit in.
    """
    return _redis().get(_preview_key(camera_id)) or _redis().get(_key(camera_id))


def set_annotated_frame(camera_id, jpeg_bytes: bytes) -> None:
    _redis().setex(_annotated_key(camera_id), ANNOTATED_TTL_SECONDS, jpeg_bytes)


def get_annotated_frame(camera_id):
    return _redis().get(_annotated_key(camera_id))


def set_boxes(camera_id, boxes: list) -> None:
    """Cache the most recent detections as JSON, for the admin's live readout."""
    _redis().setex(_boxes_key(camera_id), ANNOTATED_TTL_SECONDS, json.dumps(boxes))


def get_boxes(camera_id) -> list:
    raw = _redis().get(_boxes_key(camera_id))
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return []


def mark_viewed(camera_id) -> None:
    """Record that the annotated live stream has a viewer right now."""
    _redis().setex(_viewed_key(camera_id), VIEWED_TTL_SECONDS, b"1")


def recently_viewed(camera_id) -> bool:
    """Whether some annotated MJPEG stream is (recently) connected for this camera."""
    return bool(_redis().exists(_viewed_key(camera_id)))


def clear_viewed(camera_id) -> None:
    """Forget that this camera was being viewed — mainly for tests.

    Production code lets the heartbeat's TTL do this instead, so a brief
    disconnect doesn't flap the worker off and on.
    """
    _redis().delete(_viewed_key(camera_id))


def clear_annotated(camera_id) -> None:
    """Drop a camera's annotated frame + boxes.

    Called when live inference is disabled, so the admin stops showing a frozen
    last detection as though it were current.
    """
    _redis().delete(_annotated_key(camera_id), _boxes_key(camera_id))
