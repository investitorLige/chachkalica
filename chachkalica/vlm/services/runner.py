"""Walk a video at the requested frame rate, asking the VLM about each sample.

Split from :mod:`vlm.jobs` the same way ``videos.services.inference`` is split
from ``videos.jobs``: the job owns the status transitions, this owns the loop.

Three things here are less obvious than they look:

* **Frames go to disk, not over the wire.** Each sampled frame is written to one
  reused scratch path under ``data/`` and the *path* is what the backend gets —
  the shared-mount convention the camera live-inference path already follows.
  One path per run is safe because the loop is strictly sequential.
* **The timestamp has a fallback.** ``cv2.CAP_PROP_POS_MSEC`` returns 0 for some
  codecs and containers, which would stamp every alert at t=0. When it looks
  unusable we compute the position from the frame index instead.
* **Nobody watching means pause, not stop.** The run holds its place while the
  tab is closed and picks up where it left off when the page is reopened.
"""

import logging
import time
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from vlm.models import VlmAlert, VlmRun
from vlm.services import backend, heartbeat

logger = logging.getLogger(__name__)

#: Assumed frame rate when a video does not report a usable one.
NOMINAL_FPS = 25.0

#: How long to wait between checks while paused for lack of a viewer.
IDLE_POLL_SECONDS = 2.0

#: A run gets this long to have its live page opened before "nobody is watching"
#: starts pausing it — the redirect, the first paint, and the first poll all have
#: to happen after the worker has already picked the job up.
VIEWER_GRACE_SECONDS = 60.0

#: How long to stay paused before giving up entirely. Pausing indefinitely would
#: hold one of the three "default" rq workers for the job's whole 6-hour budget
#: on the strength of a tab someone closed and forgot. The alerts produced so far
#: are already in the database, so stopping loses nothing.
ABANDONED_AFTER_SECONDS = 600.0


def frames_dir() -> Path:
    """Scratch directory for the frame currently being inferred."""
    return Path(settings.BASE_DIR) / "data" / "vlm_frames"


def _timestamp_seconds(cap, frame_index: int, source_fps: float) -> float:
    """Where in the video this frame sits, in seconds."""
    import cv2

    try:
        millis = cap.get(cv2.CAP_PROP_POS_MSEC)
    except Exception:  # noqa: BLE001
        millis = 0.0
    if millis and millis > 0:
        return millis / 1000.0
    return frame_index / source_fps if source_fps > 0 else 0.0


def _wait_for_viewer(run: VlmRun, started_at: float) -> bool:
    """Block while nobody is watching.

    Returns False when the run should stop — either it was cancelled, or the
    page has been gone long enough to call it abandoned.
    """
    if time.monotonic() - started_at < VIEWER_GRACE_SECONDS:
        return True

    paused_at = None
    while not heartbeat.recently_viewed(run.pk):
        if paused_at is None:
            paused_at = time.monotonic()
            logger.info("run %s paused — no viewer", run.pk)
        elif time.monotonic() - paused_at > ABANDONED_AFTER_SECONDS:
            logger.info("run %s abandoned after %.0fs unwatched", run.pk,
                        ABANDONED_AFTER_SECONDS)
            return False

        time.sleep(IDLE_POLL_SECONDS)
        run.refresh_from_db(fields=["status"])
        if run.status == VlmRun.CANCEL_REQUESTED:
            return False

    if paused_at is not None:
        logger.info("run %s resumed", run.pk)
    return True


def run_vlm_on_video(run: VlmRun) -> dict:
    """Sample ``run.video`` at ``run.fps`` and write a :class:`VlmAlert` per frame.

    Returns a summary dict; the caller records the terminal status.
    """
    import cv2

    video = run.video
    if not video.exists():
        raise RuntimeError(f"{video.name}: no file on disk at {video.path()}")

    cap = cv2.VideoCapture(str(video.path()))
    if not cap.isOpened():
        raise RuntimeError(f"{video.name}: could not open the video file")

    scratch_dir = frames_dir()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    frame_path = scratch_dir / f"run_{run.pk}.jpg"

    cancelled = False
    seq = 0
    frame_index = 0
    started_at = time.monotonic()

    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if source_fps <= 0:
            source_fps = video.source_fps or NOMINAL_FPS
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

        stride = max(1, round(source_fps / run.fps)) if run.fps > 0 else 1
        run.frame_stride = stride
        run.frames_total = int(frame_count // stride) if frame_count > 0 else None
        run.save(update_fields=["frame_stride", "frames_total"])

        config = run.model_config_snapshot or {}
        prompt = run.prompt_snapshot

        while True:
            sampling = frame_index % stride == 0
            # Read the position BEFORE decoding: afterwards cv2 reports where the
            # *next* frame starts, which would stamp every alert one frame late.
            timestamp = _timestamp_seconds(cap, frame_index, source_fps) if sampling else 0.0

            ok, frame = cap.read()
            if not ok:
                break

            if sampling:
                cv2.imwrite(str(frame_path), frame)

                call_started = time.monotonic()
                result = backend.infer(str(frame_path), config, prompt)
                latency_ms = result.get("latency_ms")
                if latency_ms is None:
                    latency_ms = int((time.monotonic() - call_started) * 1000)

                seq += 1
                VlmAlert.objects.create(
                    run=run,
                    seq=seq,
                    frame_index=frame_index,
                    video_timestamp_seconds=timestamp,
                    text=(result.get("text") or "").strip(),
                    latency_ms=latency_ms,
                )
                run.frames_processed = seq
                run.alerts_count = seq
                run.save(update_fields=["frames_processed", "alerts_count"])

                # Cancellation is cooperative and checked once per VLM call — a
                # call already in flight cannot be interrupted, which is why the
                # client bounds it with a timeout.
                run.refresh_from_db(fields=["status"])
                if run.status == VlmRun.CANCEL_REQUESTED:
                    cancelled = True
                    break
                if not _wait_for_viewer(run, started_at):
                    cancelled = True
                    break

            frame_index += 1
    finally:
        cap.release()
        frame_path.unlink(missing_ok=True)

    return {
        "cancelled": cancelled,
        "frames_total": run.frames_total,
        "frames_processed": seq,
        "alerts": seq,
        "finished_at": timezone.now(),
    }
