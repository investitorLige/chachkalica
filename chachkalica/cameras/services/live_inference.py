"""Run a model over a camera's live frames and cache the annotated result.

Split into **two independent rates**, because the model is far slower than the
camera and tying them together would make the annotated stream judder along at
the model's pace:

- :func:`predict_frame` — the slow half. Runs the model at
  ``CameraInference.target_fps`` (default 2/s) and publishes just the *boxes*.
- :func:`composite_frame` — the cheap half. Draws the most recent boxes onto
  **every** raw frame ``stream_cameras`` pushes (~6-7/s), so the annotated view
  is as smooth as the raw one.

The worker runs them as two threads (see
``cameras/management/commands/infer_cameras.py``) precisely so the compositor
keeps producing frames while the inference thread is blocked on its HTTP call.
The trade is that between inferences the boxes are up to one inference interval
stale — the same trade the recorded-video path makes with ``frame_stride``.

Neither loop touches RTSP. ``stream_cameras`` already holds the one connection
per camera and keeps its latest frame in Redis; both read that key instead. Two
things fall out of that indirection for free:

- **No second RTSP session.** Most IP cameras cap concurrent sessions at 2-4,
  so opening another one just to run a model is the wrong trade.
- **Frames can never queue up.** There is only ever *one* latest frame in
  Redis, so a slow model drops intermediate frames rather than falling
  progressively further behind reality — which is what a queue would do.

Inference itself goes out to the trainer's ``POST /predict_image`` (the same
endpoint the interactive preview and ``videos.services.inference`` use), which
keeps one model warm in GPU memory. That endpoint takes a *file path*, not
image bytes, so each frame is written to a per-camera temp file on the shared
data mount first — the raw JPEG bytes are written through unchanged rather than
decoded and re-encoded, since the model only needs the pixels.

Two consequences of reusing that endpoint are worth knowing about, both
accepted at this scale (see ``docs/camera-live-inference.md``):

- The trainer's model cache holds exactly **one** entry, so a camera running
  model A and an admin previewing model B will evict each other, and each
  eviction costs a full model rebuild. One camera on one model is fine; two
  cameras on *different* models would thrash badly.
- ``_predict_lock`` serializes the GPU call with the interactive preview and
  contends with any active training run, so latency is bursty. ``target_fps``
  is an upper bound, never a guarantee.
"""

import hashlib
import logging
import time

from fleet.services.paths import videos_root

logger = logging.getLogger(__name__)

# No new frame in Redis (camera offline, or stream_cameras not running) — wait
# this long before looking again. Short enough that recovery is quick, long
# enough not to spin on Redis.
IDLE_POLL_SECONDS = 0.5

# Cap on how long a single failing inference call may retry-loop at full speed.
ERROR_BACKOFF_SECONDS = 5.0

# How often the compositor looks for a new raw frame to draw on. Deliberately
# faster than stream_cameras' push interval (0.15s) so no pushed frame is
# systematically missed; a frame it has already drawn is skipped by digest, so
# polling faster costs a Redis GET and a hash, not a re-encode.
COMPOSITE_POLL_SECONDS = 0.1


def frames_dir():
    """Directory for the per-camera temp frames handed to the trainer.

    Must live on the data tree that web/worker/trainer all bind-mount at the
    same path (see the SHARED FILESYSTEM note in docker-compose.yml), because
    ``/predict_image`` receives a path string and opens it itself. Sits beside
    ``inferred/`` under the videos root, which is already used for exactly this
    reason by ``videos.services.inference``.
    """
    d = videos_root() / "camera_frames"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_predict_payload(config) -> dict:
    """The ``/predict_image`` payload for ``config``, minus the per-frame path.

    Mirrors :func:`videos.services.inference.build_predict_payload` — resolves
    the model (trained checkpoint, exported artifact, or the model inside a
    bundle) and the pipeline knobs once, so the per-frame work is just swapping
    ``image_path``. Raises
    ``RuntimeError`` for an unusable model or an inconsistent pipeline config.
    """
    from pathlib import Path

    from training.services import config_gen
    from videos.services.frame_extraction import _resolve_checkpoint

    try:
        checkpoint = config.model_checkpoint()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if not Path(checkpoint).exists():
        raise RuntimeError(f"Model artifact not found: {checkpoint}")

    detector_checkpoint = (config.detector_checkpoint or "").strip()
    if detector_checkpoint:
        detector_checkpoint = str(_resolve_checkpoint(detector_checkpoint))

    try:
        return config_gen.build_predict_request(
            checkpoint,
            config.pipeline,
            image_path="",  # filled in per frame
            detector_checkpoint=detector_checkpoint,
            detector_expand_ratio=config.detector_expand_ratio,
            detector_min_box_size=config.detector_min_box_size,
            tile_size_px=config.tile_size_px,
            tile_width_pct=config.tile_width_pct,
            tile_height_pct=config.tile_height_pct,
            overlap=config.overlap,
            merge_nms_iou=config.merge_nms_iou,
            chain=config.chain,
            score_threshold=config.score_threshold,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def predict_frame(camera_id, jpeg_bytes: bytes, payload: dict, frame_path) -> dict:
    """Run one cached JPEG through the model; return its boxes and latency.

    Does *not* draw anything — that's :func:`composite_frame`, which runs on its
    own schedule (see the module docstring on the two loops). Returns
    ``{"boxes": [...], "latency_ms": int}``. Raises whatever ``predict_image``
    raises (``RuntimeError`` for a trainer-side failure), leaving the caller to
    decide whether that's fatal or worth retrying.
    """
    from cameras.services import frame_cache
    from training.services import runner

    # Written through as-is: the cached frame is already a JPEG, and the model
    # only needs the pixels — decoding just to re-encode would cost CPU and a
    # generation of quality for nothing.
    frame_path.write_bytes(jpeg_bytes)

    started = time.monotonic()
    response = runner.predict_image({**payload, "image_path": str(frame_path)})
    latency_ms = int((time.monotonic() - started) * 1000)
    boxes = response.get("boxes", [])

    frame_cache.set_boxes(camera_id, boxes)
    return {"boxes": boxes, "latency_ms": latency_ms}


def composite_frame(camera_id, jpeg_bytes: bytes, boxes: list) -> bytes:
    """Draw ``boxes`` onto a raw camera JPEG and cache it as the annotated frame.

    The result is preview-sized, not full resolution (see
    ``cameras.services.preview``): a browser is the only thing that ever reads
    the annotated key, and the model reads the raw one.

    Called for *every* raw frame, not just the inferred ones, so the annotated
    stream stays as smooth as the raw one while the boxes themselves refresh
    only as fast as the model manages. Same trade the recorded-video path makes
    with ``frame_stride`` (see ``videos.services.inference``): between
    inferences the most recent boxes are reused, so they can lag the picture by
    up to one inference interval.

    An empty ``boxes`` is normal and meaningful — before the first inference
    lands, this publishes clean frames so the annotated view shows live video
    immediately rather than nothing at all.
    """
    import cv2
    import numpy as np

    from cameras.services import frame_cache, preview
    from videos.services.inference import draw_boxes

    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Cached frame could not be decoded as an image.")
    # Downscale *before* drawing. The annotated frame's only consumer is a
    # browser, so unlike the raw frame there is no reason for a full-resolution
    # copy to exist at all — and drawing second keeps the labels crisp, because
    # draw_boxes sizes its lines and text to the frame it is handed rather than
    # having them shrunk along with the picture afterwards.
    frame = preview.resize(frame)
    draw_boxes(frame, boxes)

    annotated = preview.encode(frame)
    frame_cache.set_annotated_frame(camera_id, annotated)
    return annotated


def frame_digest(jpeg_bytes: bytes) -> str:
    """Cheap identity for a cached frame, so the same one is never inferred twice.

    ``stream_cameras`` pushes at ~6-7fps and this loop usually runs slower, but
    a camera that stalls (or a push interval that happens to line up) would
    otherwise burn GPU re-running an identical image.
    """
    return hashlib.blake2b(jpeg_bytes, digest_size=16).hexdigest()
