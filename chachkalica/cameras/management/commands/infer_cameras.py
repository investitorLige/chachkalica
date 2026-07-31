"""Long-running worker: run each camera's configured model over its live frames.

The inference-side sibling of ``stream_cameras``, run as its own
docker-compose service (``camera-inference``). Every
``RECONCILE_INTERVAL_SECONDS`` it re-reads the ``CameraInference`` table and
starts/stops per-camera workers to match, so enabling, retargeting, or
disabling live inference in the admin takes effect without restarting the
command.

Unlike ``stream_cameras``, no worker here ever opens an RTSP connection — each
one reads the latest frame ``stream_cameras`` already cached in Redis (see
``cameras.services.live_inference`` for why). A worker is still a separate
*process* rather than a thread, for the same reason as there: the GPU-bound
HTTP call it makes can be slow or hang, and a process can be killed outright.
"""

import logging
import multiprocessing
import signal
import threading
import time

from django.core.management.base import BaseCommand
from django.db import connections
from django.utils import timezone

from cameras.models import CameraInference
from cameras.services import frame_cache, live_inference

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL_SECONDS = 15
STATUS_WRITE_INTERVAL_SECONDS = 30
JOIN_TIMEOUT_SECONDS = 10
TERMINATE_GRACE_SECONDS = 2

# A model that fails to build (missing checkpoint, bad pipeline config) will
# keep failing. Report it and stop rather than hammering the trainer forever;
# the reconcile loop restarts the worker as soon as the config is edited.
FATAL_ERROR_SLEEP_SECONDS = 60

COMPOSITOR_JOIN_SECONDS = 5


class _BoxState:
    """The newest boxes, handed from the inference thread to the compositor.

    ``version`` increments on every set, so the compositor can tell "same boxes
    as last time" from "same boxes, but a fresh inference" without comparing box
    lists — and can skip re-encoding a frame it has already drawn.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._boxes = []
        self._version = 0

    def set(self, boxes):
        with self._lock:
            self._boxes = boxes
            self._version += 1

    def get(self):
        with self._lock:
            return self._boxes, self._version


class InferenceWorker(multiprocessing.Process):
    """Runs one camera's model over its cached frames for the life of the process.

    Two threads inside: the inference loop (slow, ``target_fps``) and the
    compositor loop (cheap, every raw frame). They must be concurrent — the
    inference loop spends most of its time blocked on the trainer's HTTP call,
    and the compositor has to keep publishing annotated frames throughout, or
    the annotated stream would only advance twice a second.
    """

    def __init__(self, config_id, camera_id, interval, stop_event):
        super().__init__(daemon=True, name=f"camera-inference-{camera_id}")
        self.config_id = config_id
        self.camera_id = camera_id
        self.interval = interval
        self._stop_event = stop_event
        self._last_status_write = 0.0
        self._last_status = None

    def run(self):
        # Forked from the parent, so: don't react to the parent's shutdown
        # signals, and don't reuse the inherited DB connection. Same reasoning
        # as CameraWorker.run in stream_cameras.py.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        connections.close_all()

        config = CameraInference.objects.filter(pk=self.config_id).first()
        if config is None:  # deleted between reconcile and start
            return

        try:
            payload = live_inference.build_predict_payload(config)
        except Exception as exc:  # noqa: BLE001 - report and idle, don't crash-loop
            logger.exception("camera %s: could not build model", self.camera_id)
            self._record_status(CameraInference.ERROR, str(exc), force=True)
            self._stop_event.wait(FATAL_ERROR_SLEEP_SECONDS)
            return

        frame_path = live_inference.frames_dir() / f"camera_{self.camera_id}.jpg"
        state = _BoxState()
        compositor = threading.Thread(
            target=self._composite_loop, args=(state,),
            name=f"compositor-{self.camera_id}", daemon=True)
        compositor.start()
        try:
            self._infer_loop(payload, frame_path, state)
        finally:
            # Both loops watch the same stop event, so setting it here also ends
            # the compositor if we're leaving for our own reasons.
            self._stop_event.set()
            compositor.join(COMPOSITOR_JOIN_SECONDS)
            frame_path.unlink(missing_ok=True)

    def _infer_loop(self, payload, frame_path, state):
        """The slow half: run the model at ``target_fps`` and publish boxes."""
        last_digest = None
        while not self._stop_event.is_set():
            cycle_started = time.monotonic()

            jpeg_bytes = frame_cache.get_frame(self.camera_id)
            if jpeg_bytes is None:
                # stream_cameras has nothing cached — camera offline, or its
                # worker isn't running. Nothing to infer; don't treat it as our
                # error, the Camera row already carries the real reason.
                self._record_status(
                    CameraInference.IDLE, "Waiting for a live frame from the camera.")
                self._stop_event.wait(live_inference.IDLE_POLL_SECONDS)
                continue

            digest = live_inference.frame_digest(jpeg_bytes)
            if digest == last_digest:
                # Same frame we already ran — a stalled stream. Skip the GPU.
                self._stop_event.wait(live_inference.IDLE_POLL_SECONDS)
                continue

            try:
                result = live_inference.predict_frame(
                    self.camera_id, jpeg_bytes, payload, frame_path)
            except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the worker
                logger.exception("camera %s: inference failed", self.camera_id)
                self._record_status(CameraInference.ERROR, str(exc))
                self._stop_event.wait(live_inference.ERROR_BACKOFF_SECONDS)
                continue

            last_digest = digest
            state.set(result["boxes"])
            self._record_metrics(result)

            # target_fps is a ceiling: sleep only for whatever's left of the
            # budget, which is usually nothing at all because the model is
            # slower than the interval.
            remaining = self.interval - (time.monotonic() - cycle_started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _composite_loop(self, state):
        """The cheap half: draw the newest boxes onto every raw frame.

        This is what makes the annotated stream as smooth as the raw one. It runs
        regardless of whether inference has produced anything yet — with no boxes
        it publishes clean frames, so the annotated view shows live video from
        the start instead of staying blank until the first detection.
        """
        last_key = None
        while not self._stop_event.is_set():
            jpeg_bytes = frame_cache.get_frame(self.camera_id)
            if jpeg_bytes is None:
                self._stop_event.wait(live_inference.IDLE_POLL_SECONDS)
                continue

            boxes, version = state.get()
            # Skip work when neither the picture nor the boxes have moved on —
            # re-encoding a 4K frame to identical output is the one cost worth
            # avoiding here.
            key = (live_inference.frame_digest(jpeg_bytes), version)
            if key == last_key:
                self._stop_event.wait(live_inference.COMPOSITE_POLL_SECONDS)
                continue

            try:
                live_inference.composite_frame(self.camera_id, jpeg_bytes, boxes)
            except Exception:  # noqa: BLE001 - a bad frame must not kill the compositor
                logger.exception("camera %s: compositing failed", self.camera_id)
                self._stop_event.wait(live_inference.ERROR_BACKOFF_SECONDS)
                continue

            last_key = key
            self._stop_event.wait(live_inference.COMPOSITE_POLL_SECONDS)

    def _record_metrics(self, result):
        """Write a successful inference's numbers back, throttled like the status."""
        now = time.monotonic()
        transitioned = self._last_status != CameraInference.RUNNING
        if not transitioned and (now - self._last_status_write) < STATUS_WRITE_INTERVAL_SECONDS:
            return
        self._last_status = CameraInference.RUNNING
        self._last_status_write = now
        CameraInference.objects.filter(pk=self.config_id).update(
            status=CameraInference.RUNNING,
            last_error="",
            last_inference_at=timezone.now(),
            last_latency_ms=result["latency_ms"],
            last_box_count=len(result["boxes"]),
        )

    def _record_status(self, status, error, force=False):
        now = time.monotonic()
        transitioned = status != self._last_status
        if not (force or transitioned) and (
                now - self._last_status_write) < STATUS_WRITE_INTERVAL_SECONDS:
            return
        self._last_status = status
        self._last_status_write = now
        CameraInference.objects.filter(pk=self.config_id).update(
            status=status, last_error=error)


def _stop_worker(worker, stop_event):
    """Signal a worker to stop and guarantee it actually does.

    Same escalation as ``stream_cameras._stop_worker``: cooperative first, then
    terminate, then kill. The join timeout is longer here because a worker may
    legitimately be blocked on a slow ``/predict_image`` call.
    """
    stop_event.set()
    worker.join(JOIN_TIMEOUT_SECONDS)
    if worker.is_alive():
        logger.warning("inference worker %s did not stop in time, terminating", worker.name)
        worker.terminate()
        worker.join(TERMINATE_GRACE_SECONDS)
        if worker.is_alive():
            worker.kill()
            worker.join()


class Command(BaseCommand):
    help = "Continuously run each camera's configured model over its live frames."

    def handle(self, *args, **options):
        workers = {}  # config_id -> (worker, stop_event, fingerprint, camera_id)
        shutdown = multiprocessing.Event()
        signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
        signal.signal(signal.SIGINT, lambda *_: shutdown.set())

        self.stdout.write("infer_cameras: starting")
        while not shutdown.is_set():
            try:
                self._reconcile(workers)
            except Exception:  # noqa: BLE001 - a transient DB/Redis blip must not kill the command
                logger.exception("infer_cameras: reconcile failed, will retry next cycle")
                connections.close_all()  # drop a broken connection so the next cycle reconnects
            shutdown.wait(RECONCILE_INTERVAL_SECONDS)

        self.stdout.write("infer_cameras: shutting down")
        for worker, stop_event, _, _ in workers.values():
            _stop_worker(worker, stop_event)

    def _reconcile(self, workers):
        # Only cameras someone actually has the annotated Live view open for —
        # see frame_cache.mark_viewed/recently_viewed. A config that's enabled
        # but unwatched just drops out of `current`, which the loop below
        # already treats exactly like disabled/deleted: it stops the worker
        # and reports IDLE. Restarting it once viewing resumes costs at most
        # one reconcile tick plus the model's own warm-up.
        current = {
            c.id: c for c in
            CameraInference.objects.filter(enabled=True).select_related("camera")
            if frame_cache.recently_viewed(c.camera_id)
        }

        for config_id in list(workers):
            worker, stop_event, fingerprint, camera_id = workers[config_id]
            config = current.get(config_id)
            if config is None or config.worker_fingerprint() != fingerprint:
                _stop_worker(worker, stop_event)
                del workers[config_id]
                # The stale annotated frame would otherwise sit in Redis for its
                # whole TTL, looking to a viewer like a current detection.
                frame_cache.clear_annotated(camera_id)
                if config is None:
                    CameraInference.objects.filter(pk=config_id).update(
                        status=CameraInference.IDLE, last_error="")
                self.stdout.write(f"infer_cameras: stopped worker for camera {camera_id}")

        for config_id, config in current.items():
            if config_id in workers:
                continue
            stop_event = multiprocessing.Event()
            worker = InferenceWorker(
                config_id, config.camera_id, config.interval_seconds(), stop_event)
            # Resolve the label (a lazy trained_model fetch) BEFORE forking, then
            # hand the child no connection to inherit: fork duplicates our socket,
            # and the child's own close_all() terminates the server-side backend
            # on the far end of it — which breaks *our* connection too, with a
            # bewildering "server closed the connection unexpectedly" on whatever
            # query we happen to run next. We reconnect lazily.
            label = f"{config.model_label()} / {config.pipeline}"
            fingerprint = config.worker_fingerprint()
            connections.close_all()
            worker.start()
            workers[config_id] = (worker, stop_event, fingerprint, config.camera_id)
            self.stdout.write(
                f"infer_cameras: started worker for camera {config.camera_id} ({label})")
