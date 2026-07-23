"""Long-running worker: one process per camera, holding the RTSP connection
and continuously refreshing its latest frame in Redis.

Run as its own docker-compose service (``camera-workers``), the same way
``rqworker`` is run for the fleet's background jobs — see docker-compose.yml.
Every ``RECONCILE_INTERVAL_SECONDS`` it re-reads the ``Camera`` table and
starts/stops per-camera workers to match, so adding, editing, or deleting a
camera in the admin takes effect without restarting the worker.

Each camera gets its own *process*, not a thread: cv2/FFmpeg's OPEN/READ
timeout properties don't reliably bound a hang against an unreachable host,
so a worker can get stuck inside a blocking call that no amount of
cooperative signalling will interrupt. A process can be killed outright,
which guarantees a camera we've decided to stop actually releases its RTSP
session instead of leaking a connection nobody is tracking anymore.
"""

import logging
import multiprocessing
import signal
import time

import cv2
from django.core.management.base import BaseCommand
from django.utils import timezone

from cameras.models import Camera
from cameras.services import frame_cache

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL_SECONDS = 15
FRAME_PUSH_INTERVAL_SECONDS = 0.15  # caps push rate to ~6-7fps regardless of camera fps
RECONNECT_BACKOFF_SECONDS = 5
# Doubles on each consecutive failed attempt (bad credentials, camera down, ...),
# capped here. A flat 5s retry forever hammers the camera with the same failed
# login repeatedly, which is exactly the pattern that trips (or re-arms) an
# NVR's brute-force lockout — this backs off so a bad camera stops being
# hammered instead of getting retried a few hundred times an hour.
MAX_RECONNECT_BACKOFF_SECONDS = 300
STATUS_WRITE_INTERVAL_SECONDS = 30
OPEN_READ_TIMEOUT_MS = 5000
JOIN_TIMEOUT_SECONDS = 5
TERMINATE_GRACE_SECONDS = 2


class CameraWorker(multiprocessing.Process):
    """Owns one camera's RTSP connection for the life of the process."""

    def __init__(self, camera_id, rtsp_url, stop_event):
        super().__init__(daemon=True, name=f"camera-{camera_id}")
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self._stop_event = stop_event
        self._last_status_write = 0.0
        self._last_status = None

    def run(self):
        # Forked from the parent, so: don't react to the parent's shutdown
        # signals (a plain kill of this one process must not be mistaken by
        # the child for "the whole command got SIGTERM"), and don't reuse
        # the DB connection inherited from the parent's fork.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        from django.db import connections
        connections.close_all()

        consecutive_failures = 0
        while not self._stop_event.is_set():
            capture = cv2.VideoCapture()
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_READ_TIMEOUT_MS)
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, OPEN_READ_TIMEOUT_MS)
            connected = False
            try:
                if not capture.open(self.rtsp_url, cv2.CAP_FFMPEG):
                    self._record_status(Camera.OFFLINE, "Could not open stream.")
                else:
                    connected = self._read_loop(capture)
            except Exception as exc:  # noqa: BLE001 - a bad camera must not kill the worker
                logger.exception("camera %s: worker error", self.camera_id)
                self._record_status(Camera.OFFLINE, str(exc))
            finally:
                capture.release()

            consecutive_failures = 0 if connected else consecutive_failures + 1
            backoff = min(
                RECONNECT_BACKOFF_SECONDS * (2 ** consecutive_failures),
                MAX_RECONNECT_BACKOFF_SECONDS,
            )
            self._stop_event.wait(backoff)

    def _read_loop(self, capture):
        """Read frames until the stream drops or we're told to stop.

        Returns whether at least one frame was successfully read — used by
        the caller to distinguish "camera reachable, briefly flaked" from
        "never authenticated/connected at all" for backoff purposes.
        """
        last_push = 0.0
        read_any_frame = False
        while not self._stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                self._record_status(Camera.OFFLINE, "Stream opened but no frame could be read.")
                return read_any_frame
            read_any_frame = True
            now = time.monotonic()
            if now - last_push >= FRAME_PUSH_INTERVAL_SECONDS:
                encoded, buf = cv2.imencode(".jpg", frame)
                if encoded:
                    frame_cache.set_frame(self.camera_id, buf.tobytes())
                    self._record_status(Camera.ONLINE, "")
                last_push = now
        return read_any_frame

    def _record_status(self, status, error):
        now = time.monotonic()
        transitioned = status != self._last_status
        if not transitioned and (now - self._last_status_write) < STATUS_WRITE_INTERVAL_SECONDS:
            return
        self._last_status = status
        self._last_status_write = now
        Camera.objects.filter(pk=self.camera_id).update(
            status=status, last_error=error, last_checked_at=timezone.now(),
        )


def _stop_worker(worker, stop_event):
    """Signal a worker to stop and guarantee it actually does.

    Cooperative shutdown (set the event, join) is tried first, but a worker
    stuck inside a blocking cv2/FFmpeg call won't notice the event in time —
    in that case the process is killed outright, so it's never silently
    forgotten while still holding the camera's RTSP session.
    """
    stop_event.set()
    worker.join(JOIN_TIMEOUT_SECONDS)
    if worker.is_alive():
        logger.warning("camera worker %s did not stop in time, terminating", worker.name)
        worker.terminate()
        worker.join(TERMINATE_GRACE_SECONDS)
        if worker.is_alive():
            worker.kill()
            worker.join()


class Command(BaseCommand):
    help = "Continuously pull frames from every registered camera into Redis."

    def handle(self, *args, **options):
        workers = {}  # camera_id -> (CameraWorker, multiprocessing.Event, rtsp_url)
        shutdown = multiprocessing.Event()
        signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
        signal.signal(signal.SIGINT, lambda *_: shutdown.set())

        self.stdout.write("stream_cameras: starting")
        while not shutdown.is_set():
            try:
                self._reconcile(workers)
            except Exception:  # noqa: BLE001 - a transient DB/Redis blip must not kill the command
                logger.exception("stream_cameras: reconcile failed, will retry next cycle")
                from django.db import connections
                connections.close_all()  # drop a broken connection so the next cycle reconnects
            shutdown.wait(RECONCILE_INTERVAL_SECONDS)

        self.stdout.write("stream_cameras: shutting down")
        for worker, stop_event, _ in workers.values():
            _stop_worker(worker, stop_event)

    def _reconcile(self, workers):
        current = {c.id: c.rtsp_url for c in Camera.objects.all()}

        for camera_id in list(workers):
            worker, stop_event, rtsp_url = workers[camera_id]
            if camera_id not in current or current[camera_id] != rtsp_url:
                _stop_worker(worker, stop_event)
                del workers[camera_id]
                self.stdout.write(f"stream_cameras: stopped worker for camera {camera_id}")

        for camera_id, rtsp_url in current.items():
            if camera_id in workers:
                continue
            stop_event = multiprocessing.Event()
            worker = CameraWorker(camera_id, rtsp_url, stop_event)
            worker.start()
            workers[camera_id] = (worker, stop_event, rtsp_url)
            self.stdout.write(f"stream_cameras: started worker for camera {camera_id}")
