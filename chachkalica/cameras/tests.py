from unittest import mock

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.urls import reverse

from cameras.admin import mask_rtsp_url
from cameras.models import Camera, CameraInference, validate_rtsp_url
from cameras.services import frame_cache

RTSP_URL = "rtsp://admin:AxProVideo2024@10.10.10.24:554/Streaming/Channels/101"

# CameraAdmin carries the CameraInference inline (see CameraInferenceInline), so
# every POST to the camera add/change form must include the inline's management
# form or Django rejects the whole submission. Prefix is the OneToOne's accessor
# name ("inference"); TOTAL_FORMS 0 means "no inference config submitted".
NO_INFERENCE_INLINE = {
    "inference-TOTAL_FORMS": "0",
    "inference-INITIAL_FORMS": "0",
    "inference-MIN_NUM_FORMS": "0",
    "inference-MAX_NUM_FORMS": "1",
}


class ValidatorTests(TestCase):
    def test_accepts_rtsp_scheme(self):
        validate_rtsp_url(RTSP_URL)  # no raise

    def test_rejects_non_rtsp_scheme(self):
        with self.assertRaises(ValidationError):
            validate_rtsp_url("https://example.com/stream")


class MaskUrlTests(TestCase):
    def test_masks_password(self):
        self.assertEqual(
            mask_rtsp_url(RTSP_URL),
            "rtsp://admin:••••@10.10.10.24:554/Streaming/Channels/101",
        )

    def test_leaves_credential_free_url_untouched(self):
        url = "rtsp://10.10.10.24:554/Streaming/Channels/101"
        self.assertEqual(mask_rtsp_url(url), url)


class SaveModelTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

    def test_successful_test_marks_online(self):
        with mock.patch("cameras.admin.rtsp.test_connection", return_value=(True, "")):
            self.client.post(reverse("admin:cameras_camera_add"), {
                "name": "Front door", "rtsp_url": RTSP_URL, **NO_INFERENCE_INLINE,
            })
        camera = Camera.objects.get(name="Front door")
        self.assertEqual(camera.status, Camera.ONLINE)
        self.assertEqual(camera.last_error, "")
        self.assertIsNotNone(camera.last_checked_at)

    def test_failed_test_marks_offline_but_still_saves(self):
        with mock.patch("cameras.admin.rtsp.test_connection",
                        return_value=(False, "Could not open stream.")):
            self.client.post(reverse("admin:cameras_camera_add"), {
                "name": "Back yard", "rtsp_url": RTSP_URL, **NO_INFERENCE_INLINE,
            })
        camera = Camera.objects.get(name="Back yard")
        self.assertEqual(camera.status, Camera.OFFLINE)
        self.assertIn("Could not open stream", camera.last_error)


class AdminRenderTests(TestCase):
    def setUp(self):
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

    def test_index_shows_cameras_section(self):
        resp = self.client.get(reverse("admin:index"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Cameras")

    def test_list_shows_masked_url(self):
        Camera.objects.create(name="Lobby", rtsp_url=RTSP_URL)
        resp = self.client.get(reverse("admin:cameras_camera_changelist"))
        self.assertContains(resp, "••••")
        self.assertNotContains(resp, "AxProVideo2024")


class LiveStreamTests(TestCase):
    def setUp(self):
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")
        self.camera = Camera.objects.create(name="Lobby", rtsp_url=RTSP_URL)

    def test_mjpeg_requires_login(self):
        self.client.logout()
        resp = self.client.get(reverse("admin:cameras_camera_mjpeg"), {"camera": self.camera.pk})
        self.assertEqual(resp.status_code, 302)

    def test_mjpeg_unknown_camera_404s(self):
        resp = self.client.get(reverse("admin:cameras_camera_mjpeg"), {"camera": 999999})
        self.assertEqual(resp.status_code, 404)

    def test_mjpeg_content_type(self):
        resp = self.client.get(reverse("admin:cameras_camera_mjpeg"), {"camera": self.camera.pk})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "multipart/x-mixed-replace; boundary=frame")

    def test_live_view_renders_camera(self):
        resp = self.client.get(reverse("admin:cameras_camera_live"), {"camera": self.camera.pk})
        self.assertContains(resp, "Lobby")
        self.assertContains(resp, f"?camera={self.camera.pk}")

    def test_live_view_unknown_camera_404s(self):
        resp = self.client.get(reverse("admin:cameras_camera_live"), {"camera": 999999})
        self.assertEqual(resp.status_code, 404)

    def test_change_page_shows_preview_and_link(self):
        resp = self.client.get(reverse("admin:cameras_camera_change", args=[self.camera.pk]))
        self.assertContains(resp, "Open full live view")
        self.assertContains(resp, f"/admin/cameras/camera/mjpeg/?camera={self.camera.pk}")

    def test_action_requires_single_selection(self):
        Camera.objects.create(name="Back yard", rtsp_url=RTSP_URL.replace("101", "102"))
        resp = self.client.post(reverse("admin:cameras_camera_changelist"), {
            "action": "view_live_stream",
            "_selected_action": [str(c.pk) for c in Camera.objects.all()],
        }, follow=True)
        self.assertContains(resp, "Select exactly one camera")

    def test_action_redirects_to_live_view(self):
        resp = self.client.post(reverse("admin:cameras_camera_changelist"), {
            "action": "view_live_stream",
            "_selected_action": [str(self.camera.pk)],
        })
        self.assertRedirects(
            resp, reverse("admin:cameras_camera_live") + f"?camera={self.camera.pk}",
        )


def _jpeg_bytes(color=(0, 0, 0), size=(32, 32)):
    """A real, decodable JPEG — the live-inference path runs cv2.imdecode on it."""
    import cv2
    import numpy as np

    frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


class LiveInferenceModelTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(name="Lobby", rtsp_url=RTSP_URL)

    def _config(self, **kwargs):
        return CameraInference(camera=self.camera, **kwargs)

    def test_interval_from_target_fps(self):
        self.assertAlmostEqual(self._config(target_fps=4).interval_seconds(), 0.25)

    def test_zero_fps_means_no_throttle(self):
        # Guards against a ZeroDivisionError in the worker's sleep budget.
        self.assertEqual(self._config(target_fps=0).interval_seconds(), 0.0)

    def test_trained_source_without_model_raises(self):
        config = self._config(model_source=CameraInference.TRAINED)
        with self.assertRaises(ValueError):
            config.model_checkpoint()

    def test_exported_source_without_artifact_raises(self):
        config = self._config(model_source=CameraInference.EXPORTED, artifact_path="")
        with self.assertRaises(ValueError):
            config.model_checkpoint()

    def test_absolute_trained_checkpoint_passes_through(self):
        from training.models import TrainedModel

        model = TrainedModel.objects.create(
            name="ppe-v1", arch="rtdetr", checkpoint_path="/abs/best.pt")
        config = self._config(model_source=CameraInference.TRAINED, trained_model=model)
        self.assertEqual(config.model_checkpoint(), "/abs/best.pt")

    def test_relative_trained_checkpoint_resolves_under_base_dir(self):
        from django.conf import settings
        from training.models import TrainedModel

        model = TrainedModel.objects.create(
            name="ppe-v2", arch="rtdetr", checkpoint_path="runs/best.pt")
        config = self._config(model_source=CameraInference.TRAINED, trained_model=model)
        self.assertEqual(
            config.model_checkpoint(), str(settings.BASE_DIR / "runs/best.pt"))

    def test_fingerprint_changes_with_model(self):
        from training.models import TrainedModel

        a = TrainedModel.objects.create(name="a", arch="rtdetr", checkpoint_path="/a.pt")
        b = TrainedModel.objects.create(name="b", arch="rtdetr", checkpoint_path="/b.pt")
        first = self._config(trained_model=a).worker_fingerprint()
        second = self._config(trained_model=b).worker_fingerprint()
        self.assertNotEqual(first, second)

    def test_fingerprint_ignores_worker_written_status(self):
        # Status writeback must not make the reconcile loop restart the worker
        # it just heard from — that would be an endless restart cycle.
        config = self._config(status=CameraInference.RUNNING, last_box_count=3)
        plain = self._config()
        self.assertEqual(config.worker_fingerprint(), plain.worker_fingerprint())


class LiveInferenceServiceTests(TestCase):
    """The per-frame path, with the trainer's HTTP call mocked out."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Lobby", rtsp_url=RTSP_URL)

    BOXES = [{"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4,
              "confidence": 0.9, "class_id": 0, "class_name": "helmet"}]

    def test_predict_frame_hands_the_trainer_a_path_to_the_original_bytes(self):
        import tempfile
        from pathlib import Path

        from cameras.services import live_inference

        raw = _jpeg_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            frame_path = Path(tmp) / "camera_1.jpg"
            with mock.patch("training.services.runner.predict_image",
                            return_value={"boxes": self.BOXES}) as predict, \
                 mock.patch("cameras.services.frame_cache.set_boxes") as set_boxes:
                result = live_inference.predict_frame(
                    self.camera.pk, raw, {"pipeline": "raw"}, frame_path)

            self.assertEqual(predict.call_args[0][0]["image_path"], str(frame_path))
            # Written through, not re-encoded — the model gets the camera's bytes.
            self.assertEqual(frame_path.read_bytes(), raw)

        self.assertEqual(result["boxes"], self.BOXES)
        self.assertIsInstance(result["latency_ms"], int)
        set_boxes.assert_called_once_with(self.camera.pk, self.BOXES)

    def test_predict_frame_does_not_draw_or_publish_annotated(self):
        """Drawing belongs to the compositor — the two run at different rates."""
        import tempfile
        from pathlib import Path

        from cameras.services import live_inference

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("training.services.runner.predict_image",
                        return_value={"boxes": self.BOXES}), \
             mock.patch("cameras.services.frame_cache.set_boxes"), \
             mock.patch("cameras.services.frame_cache.set_annotated_frame") as set_frame:
            live_inference.predict_frame(
                self.camera.pk, _jpeg_bytes(), {}, Path(tmp) / "f.jpg")
        set_frame.assert_not_called()

    def test_composite_frame_burns_boxes_in_and_caches_the_result(self):
        from cameras.services import live_inference

        raw = _jpeg_bytes()
        with mock.patch("cameras.services.frame_cache.set_annotated_frame") as set_frame:
            annotated = live_inference.composite_frame(self.camera.pk, raw, self.BOXES)

        self.assertTrue(annotated.startswith(b"\xff\xd8"))  # a real JPEG
        self.assertNotEqual(annotated, raw)  # boxes were drawn
        set_frame.assert_called_once_with(self.camera.pk, annotated)

    def test_composite_frame_publishes_clean_frames_with_no_boxes(self):
        """Before the first inference lands, the annotated view must still show
        live video rather than nothing at all."""
        from cameras.services import live_inference

        with mock.patch("cameras.services.frame_cache.set_annotated_frame") as set_frame:
            annotated = live_inference.composite_frame(self.camera.pk, _jpeg_bytes(), [])
        self.assertTrue(annotated.startswith(b"\xff\xd8"))
        set_frame.assert_called_once()

    def test_undecodable_frame_raises(self):
        from cameras.services import live_inference

        with self.assertRaises(RuntimeError):
            live_inference.composite_frame(self.camera.pk, b"not a jpeg", [])

    def test_frame_digest_distinguishes_frames(self):
        from cameras.services import live_inference

        black, white = _jpeg_bytes((0, 0, 0)), _jpeg_bytes((255, 255, 255))
        self.assertEqual(
            live_inference.frame_digest(black), live_inference.frame_digest(black))
        self.assertNotEqual(
            live_inference.frame_digest(black), live_inference.frame_digest(white))

    def test_build_payload_rejects_missing_checkpoint(self):
        from cameras.services import live_inference
        from training.models import TrainedModel

        model = TrainedModel.objects.create(
            name="gone", arch="rtdetr", checkpoint_path="/nope/best.pt")
        config = CameraInference(
            camera=self.camera, model_source=CameraInference.TRAINED, trained_model=model)
        with self.assertRaisesMessage(RuntimeError, "Model artifact not found"):
            live_inference.build_predict_payload(config)


class BoxStateTests(TestCase):
    def test_version_increments_on_every_set(self):
        from cameras.management.commands.infer_cameras import _BoxState

        state = _BoxState()
        self.assertEqual(state.get(), ([], 0))
        state.set([{"class_name": "helmet"}])
        boxes, version = state.get()
        self.assertEqual(version, 1)
        self.assertEqual(boxes[0]["class_name"], "helmet")

    def test_identical_boxes_still_bump_the_version(self):
        # The compositor keys off the version, not box equality — a fresh
        # inference that happened to find the same thing is still fresh.
        from cameras.management.commands.infer_cameras import _BoxState

        state = _BoxState()
        state.set([])
        state.set([])
        self.assertEqual(state.get()[1], 2)


class CompositeLoopTests(TestCase):
    """The cheap loop — the one that makes the annotated stream smooth."""

    def setUp(self):
        self.camera = Camera.objects.create(name="Lobby", rtsp_url=RTSP_URL)

    def _run(self, frames, on_frame=None):
        """Drive ``_composite_loop`` over a scripted list of raw frames.

        Returns the ``boxes`` argument of every ``composite_frame`` call. The
        loop stops once the script runs out.
        """
        import multiprocessing

        from cameras.management.commands.infer_cameras import (
            InferenceWorker, _BoxState,
        )

        worker = InferenceWorker(1, self.camera.pk, 0.5, multiprocessing.Event())
        state = _BoxState()
        remaining = list(frames)
        composited = []

        def get_frame(_camera_id):
            if not remaining:
                worker._stop_event.set()
                return None
            if on_frame is not None:
                on_frame(state, len(frames) - len(remaining))
            return remaining.pop(0)

        with mock.patch("cameras.services.frame_cache.get_frame", side_effect=get_frame), \
             mock.patch("cameras.services.live_inference.composite_frame",
                        side_effect=lambda cid, jpeg, boxes: composited.append(boxes)), \
             mock.patch("cameras.services.live_inference.COMPOSITE_POLL_SECONDS", 0), \
             mock.patch("cameras.services.live_inference.IDLE_POLL_SECONDS", 0):
            worker._composite_loop(state)
        return composited

    def test_every_new_frame_is_composited_even_without_new_boxes(self):
        """The whole point: annotated output keeps up with the camera, not the
        model. Three distinct frames and zero inferences -> three annotated
        frames."""
        frames = [_jpeg_bytes((0, 0, 0)), _jpeg_bytes((80, 80, 80)),
                  _jpeg_bytes((255, 255, 255))]
        self.assertEqual(len(self._run(frames)), 3)

    def test_unchanged_frame_is_not_recomposited(self):
        # Re-encoding a 4K frame to byte-identical output is the one cost worth
        # skipping, so a stalled stream must not spin the CPU.
        same = _jpeg_bytes()
        self.assertEqual(len(self._run([same, same, same])), 1)

    def test_new_boxes_recomposite_even_on_an_unchanged_frame(self):
        same = _jpeg_bytes()

        def bump(state, index):
            if index == 1:  # fresh inference arrives before the second read
                state.set([{"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1,
                            "confidence": 0.8, "class_id": 0, "class_name": "vest"}])

        composited = self._run([same, same], on_frame=bump)
        self.assertEqual(len(composited), 2)
        self.assertEqual(composited[0], [])  # nothing detected yet
        self.assertEqual(composited[1][0]["class_name"], "vest")

    def test_missing_raw_frame_composites_nothing(self):
        self.assertEqual(self._run([]), [])


class LiveInferenceWorkerTests(TestCase):
    """The reconcile loop: which cameras get a worker, and when they're stopped."""

    def setUp(self):
        from cameras.management.commands.infer_cameras import Command

        self.command = Command()
        self.camera = Camera.objects.create(name="Lobby", rtsp_url=RTSP_URL)

    def _enabled_config(self, enabled=True, **kwargs):
        from training.models import TrainedModel

        model = TrainedModel.objects.create(
            name=f"m{CameraInference.objects.count()}", arch="rtdetr",
            checkpoint_path="/best.pt")
        return CameraInference.objects.create(
            camera=self.camera, enabled=enabled, trained_model=model, **kwargs)

    def _reconcile(self, workers, viewed=True):
        """Run one reconcile with process start/stop stubbed out.

        ``connections.close_all()`` is stubbed too: in production it drops the
        parent's Postgres socket before forking so the child can't kill it, but
        inside a ``TestCase`` it would tear down the transaction wrapping the
        test. Irrelevant to what these tests assert (which workers start/stop).

        ``viewed`` marks (or, if ``False``, explicitly clears) the shared
        camera's viewed heartbeat right before reconciling — real Redis, not
        mocked, same as production. Cleared rather than just "not marked
        again" so a viewer-gating test doesn't depend on a 30s TTL actually
        elapsing; every other test wants "someone's watching" held constant so
        it can isolate the enabled/fingerprint dimension.
        """
        if viewed:
            frame_cache.mark_viewed(self.camera.pk)
        else:
            frame_cache.clear_viewed(self.camera.pk)
        with mock.patch(
            "cameras.management.commands.infer_cameras.InferenceWorker"
        ) as worker_cls, mock.patch(
            "cameras.management.commands.infer_cameras._stop_worker"
        ) as stop, mock.patch(
            "cameras.management.commands.infer_cameras.frame_cache.clear_annotated"
        ) as clear, mock.patch(
            "cameras.management.commands.infer_cameras.connections.close_all"
        ):
            self.command._reconcile(workers)
        return worker_cls, stop, clear

    def test_starts_worker_for_enabled_config(self):
        config = self._enabled_config()
        workers = {}
        worker_cls, _, _ = self._reconcile(workers)
        self.assertIn(config.pk, workers)
        worker_cls.assert_called_once()

    def test_skips_disabled_config(self):
        self._enabled_config(enabled=False)
        workers = {}
        self._reconcile(workers)
        self.assertEqual(workers, {})

    def test_disabling_stops_worker_and_clears_annotated_frame(self):
        config = self._enabled_config()
        workers = {}
        self._reconcile(workers)

        CameraInference.objects.filter(pk=config.pk).update(enabled=False)
        _, stop, clear = self._reconcile(workers)
        self.assertEqual(workers, {})
        stop.assert_called_once()
        clear.assert_called_once_with(self.camera.pk)
        config.refresh_from_db()
        self.assertEqual(config.status, CameraInference.IDLE)

    def test_editing_the_model_restarts_the_worker(self):
        config = self._enabled_config()
        workers = {}
        self._reconcile(workers)
        first_fingerprint = workers[config.pk][2]

        CameraInference.objects.filter(pk=config.pk).update(score_threshold=0.9)
        _, stop, _ = self._reconcile(workers)
        stop.assert_called_once()
        self.assertIn(config.pk, workers)  # restarted, not just stopped
        self.assertNotEqual(workers[config.pk][2], first_fingerprint)

    def test_unchanged_config_leaves_worker_alone(self):
        config = self._enabled_config()
        workers = {}
        self._reconcile(workers)
        worker = workers[config.pk][0]

        worker_cls, stop, _ = self._reconcile(workers)
        stop.assert_not_called()
        worker_cls.assert_not_called()
        self.assertIs(workers[config.pk][0], worker)

    def test_skips_enabled_config_nobody_is_viewing(self):
        self._enabled_config()
        workers = {}
        self._reconcile(workers, viewed=False)
        self.assertEqual(workers, {})

    def test_worker_stops_once_the_viewer_leaves(self):
        config = self._enabled_config()
        workers = {}
        self._reconcile(workers)  # viewed=True: starts the worker

        _, stop, clear = self._reconcile(workers, viewed=False)
        self.assertEqual(workers, {})
        stop.assert_called_once()
        clear.assert_called_once_with(self.camera.pk)
        config.refresh_from_db()
        self.assertEqual(config.status, CameraInference.IDLE)

    def test_worker_restarts_once_viewing_resumes(self):
        config = self._enabled_config()
        workers = {}
        self._reconcile(workers)
        self._reconcile(workers, viewed=False)
        self.assertEqual(workers, {})

        worker_cls, _, _ = self._reconcile(workers)  # viewed=True again
        self.assertIn(config.pk, workers)
        worker_cls.assert_called_once()


class AnnotatedStreamTests(TestCase):
    def setUp(self):
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")
        self.camera = Camera.objects.create(name="Lobby", rtsp_url=RTSP_URL)

    # The mocked getters must return real bytes, not None: _mjpeg_frames only
    # yields when a frame exists, so a None-returning getter would spin forever
    # and next() would never return.
    def test_annotated_mjpeg_reads_the_annotated_key(self):
        frame = _jpeg_bytes()
        with mock.patch("cameras.admin.frame_cache.get_annotated_frame",
                        return_value=frame) as annotated, \
             mock.patch("cameras.admin.frame_cache.get_frame", return_value=frame) as raw:
            resp = self.client.get(reverse("admin:cameras_camera_mjpeg"),
                                   {"camera": self.camera.pk, "annotated": "1"})
            self.assertEqual(resp.status_code, 200)
            # StreamingHttpResponse is lazy — pull one chunk out of the generator.
            chunk = next(resp.streaming_content)
        self.assertIn(frame, chunk)
        self.assertTrue(annotated.called)
        self.assertFalse(raw.called)

    def test_raw_mjpeg_reads_the_preview_key(self):
        """The un-annotated stream serves the downscaled copy, not the 4K one the
        model runs on."""
        frame = _jpeg_bytes()
        with mock.patch("cameras.admin.frame_cache.get_annotated_frame",
                        return_value=frame) as annotated, \
             mock.patch("cameras.admin.frame_cache.get_preview_frame",
                        return_value=frame) as preview_frame, \
             mock.patch("cameras.admin.frame_cache.get_frame", return_value=frame) as raw:
            resp = self.client.get(reverse("admin:cameras_camera_mjpeg"),
                                   {"camera": self.camera.pk})
            next(resp.streaming_content)
        self.assertTrue(preview_frame.called)
        self.assertFalse(raw.called)
        self.assertFalse(annotated.called)

    def test_live_view_warns_when_inference_is_off(self):
        resp = self.client.get(reverse("admin:cameras_camera_live"),
                               {"camera": self.camera.pk, "annotated": "1"})
        self.assertContains(resp, "Live inference is not enabled")

    def test_live_view_shows_model_when_enabled(self):
        from training.models import TrainedModel

        model = TrainedModel.objects.create(
            name="ppe-v1", arch="rtdetr", checkpoint_path="/best.pt")
        CameraInference.objects.create(
            camera=self.camera, enabled=True, trained_model=model,
            status=CameraInference.RUNNING, last_latency_ms=140)
        with mock.patch("cameras.admin.frame_cache.get_boxes", return_value=[]):
            resp = self.client.get(reverse("admin:cameras_camera_live"),
                                   {"camera": self.camera.pk, "annotated": "1"})
        self.assertContains(resp, "ppe-v1")
        self.assertContains(resp, "140 ms/inference")
        self.assertContains(resp, "Show raw stream")
        # The readout must not imply the video is limited to target_fps — only
        # the boxes are.
        self.assertContains(resp, "boxes refresh at up to 2.0 fps")

    def test_change_page_prefers_annotated_preview_when_enabled(self):
        from training.models import TrainedModel

        model = TrainedModel.objects.create(
            name="ppe-v1", arch="rtdetr", checkpoint_path="/best.pt")
        CameraInference.objects.create(
            camera=self.camera, enabled=True, trained_model=model)
        resp = self.client.get(reverse("admin:cameras_camera_change", args=[self.camera.pk]))
        self.assertContains(resp, f"?camera={self.camera.pk}&amp;annotated=1")

    def test_change_page_uses_raw_preview_when_disabled(self):
        resp = self.client.get(reverse("admin:cameras_camera_change", args=[self.camera.pk]))
        self.assertNotContains(resp, "annotated=1")

    def test_changelist_shows_inference_column(self):
        resp = self.client.get(reverse("admin:cameras_camera_changelist"))
        self.assertContains(resp, "Live inference")  # capfirst'd column header


class PreviewSizingTests(TestCase):
    """Frames are downscaled once by the producer, not once per viewer."""

    def test_resize_shrinks_a_wide_frame_and_keeps_the_aspect_ratio(self):
        import numpy as np

        from cameras.services import preview

        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        small = preview.resize(frame)
        self.assertEqual(small.shape[1], preview.PREVIEW_MAX_WIDTH)
        self.assertEqual(small.shape[0], round(2160 * preview.PREVIEW_MAX_WIDTH / 3840))

    def test_resize_leaves_an_already_small_frame_untouched(self):
        """Returned as the *same object*, which is how callers know to skip a
        pointless second JPEG encode."""
        import numpy as np

        from cameras.services import preview

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIs(preview.resize(frame), frame)

    def test_preview_getter_falls_back_to_the_full_frame(self):
        from cameras.services import frame_cache

        full = _jpeg_bytes()
        with mock.patch("cameras.services.frame_cache._redis") as redis:
            redis.return_value.get.side_effect = [None, full]  # no preview yet
            self.assertEqual(frame_cache.get_preview_frame(1), full)

    def test_preview_getter_prefers_the_preview_key(self):
        from cameras.services import frame_cache

        small = _jpeg_bytes()
        with mock.patch("cameras.services.frame_cache._redis") as redis:
            redis.return_value.get.return_value = small
            self.assertEqual(frame_cache.get_preview_frame(1), small)
            self.assertEqual(redis.return_value.get.call_count, 1)  # no fallback GET

    def test_stream_worker_publishes_full_and_preview_frames(self):
        import multiprocessing

        import cv2
        import numpy as np

        from cameras.management.commands.stream_cameras import CameraWorker
        from cameras.services import preview

        camera = Camera.objects.create(name="4K", rtsp_url=RTSP_URL)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        capture = mock.Mock()
        # One good frame, then a failed read so _read_loop returns.
        capture.read.side_effect = [(True, frame), (False, None)]

        worker = CameraWorker(camera.pk, RTSP_URL, multiprocessing.Event())
        with mock.patch("cameras.services.frame_cache.set_frame") as set_full, \
             mock.patch("cameras.services.frame_cache.set_preview_frame") as set_preview:
            worker._read_loop(capture)

        full_bytes = set_full.call_args[0][1]
        preview_bytes = set_preview.call_args[0][1]
        self.assertEqual(
            cv2.imdecode(np.frombuffer(full_bytes, np.uint8), cv2.IMREAD_COLOR).shape[1],
            1920)
        self.assertEqual(
            cv2.imdecode(np.frombuffer(preview_bytes, np.uint8), cv2.IMREAD_COLOR).shape[1],
            preview.PREVIEW_MAX_WIDTH)
        self.assertLess(len(preview_bytes), len(full_bytes))

    def test_annotated_frame_is_preview_sized(self):
        import cv2
        import numpy as np

        from cameras.services import live_inference, preview

        camera = Camera.objects.create(name="4K", rtsp_url=RTSP_URL)
        raw = _jpeg_bytes(size=(3840, 2160))
        with mock.patch("cameras.services.frame_cache.set_annotated_frame"):
            annotated = live_inference.composite_frame(camera.pk, raw, [])

        decoded = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[1], preview.PREVIEW_MAX_WIDTH)


class MjpegDedupTests(TestCase):
    """The stream sends each frame once instead of whatever Redis holds at poll time."""

    def test_unchanged_frame_is_not_resent(self):
        from cameras.admin import _mjpeg_frames

        first, second = _jpeg_bytes((0, 0, 0)), _jpeg_bytes((255, 255, 255))
        with mock.patch("cameras.admin.frame_cache.get_preview_frame",
                        side_effect=[first, first, first, second]):
            stream = _mjpeg_frames(1)
            self.assertIn(first, next(stream))
            # The two repeats in between are skipped, not sent again.
            self.assertIn(second, next(stream))

    def test_keepalive_resends_a_stalled_frame(self):
        from cameras.admin import _mjpeg_frames

        frame = _jpeg_bytes()
        with mock.patch("cameras.admin._MJPEG_KEEPALIVE_SECONDS", 0), \
             mock.patch("cameras.admin.frame_cache.get_preview_frame", return_value=frame):
            stream = _mjpeg_frames(1)
            self.assertIn(frame, next(stream))
            self.assertIn(frame, next(stream))  # same bytes, sent again


class InferenceFormTests(TestCase):
    def setUp(self):
        from training.models import TrainedModel

        self.camera = Camera.objects.create(name="Lobby", rtsp_url=RTSP_URL)
        self.model = TrainedModel.objects.create(
            name="ppe-v1", arch="rtdetr", checkpoint_path="/best.pt")

    def _form(self, **overrides):
        from cameras.admin import CameraInferenceForm

        data = {
            "camera": self.camera.pk,
            "enabled": "on",
            "target_fps": "2",
            "model_source": CameraInference.TRAINED,
            "trained_model": self.model.pk,
            "artifact_path": "",
            "score_threshold": "0.5",
            "pipeline": CameraInference.RAW,
            "detector_checkpoint": "",
            "chain": "[]",
        }
        data.update(overrides)
        return CameraInferenceForm(data=data)

    def test_valid_config_passes_the_payload_probe(self):
        with mock.patch("cameras.services.live_inference.build_predict_payload",
                        return_value={}) as build:
            self.assertTrue(self._form().is_valid(), self._form().errors)
        self.assertTrue(build.called)

    def test_payload_error_surfaces_as_a_form_error(self):
        with mock.patch("cameras.services.live_inference.build_predict_payload",
                        side_effect=RuntimeError("Model artifact not found: /best.pt")):
            form = self._form()
            self.assertFalse(form.is_valid())
        self.assertIn("Model artifact not found", str(form.errors))

    def test_disabled_config_skips_the_probe(self):
        # An operator must be able to save a half-finished config without the
        # checkpoint existing yet; only *enabling* it demands a working model.
        with mock.patch("cameras.services.live_inference.build_predict_payload") as build:
            form = self._form(enabled="")
            self.assertTrue(form.is_valid(), form.errors)
        self.assertFalse(build.called)

    def test_trained_source_requires_a_model(self):
        form = self._form(trained_model="")
        self.assertFalse(form.is_valid())
        self.assertIn("trained_model", form.errors)

    def test_exported_source_requires_an_artifact(self):
        form = self._form(model_source=CameraInference.EXPORTED, trained_model="")
        self.assertFalse(form.is_valid())
        self.assertIn("artifact_path", form.errors)

    def test_rejects_non_positive_fps(self):
        form = self._form(target_fps="0")
        self.assertFalse(form.is_valid())
        self.assertIn("target_fps", form.errors)

    def test_artifact_choices_come_from_the_export_root(self):
        from cameras.admin import CameraInferenceForm

        artifacts = [{"relpath": "ppe/best.onnx", "kind": "ONNX", "size_mb": 12.3}]
        with mock.patch("training.services.exports.list_artifacts", return_value=artifacts):
            form = CameraInferenceForm()
        labels = dict(form.fields["artifact_path"].choices)
        self.assertIn("ppe/best.onnx", labels)
        self.assertIn("ONNX", labels["ppe/best.onnx"])

    def test_inline_post_creates_the_config(self):
        """The whole round trip: change page POST → a CameraInference row."""
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

        data = {
            "name": self.camera.name, "rtsp_url": self.camera.rtsp_url,
            "inference-TOTAL_FORMS": "1", "inference-INITIAL_FORMS": "0",
            "inference-MIN_NUM_FORMS": "0", "inference-MAX_NUM_FORMS": "1",
            "inference-0-enabled": "on",
            "inference-0-target_fps": "3",
            "inference-0-model_source": CameraInference.TRAINED,
            "inference-0-trained_model": str(self.model.pk),
            "inference-0-artifact_path": "",
            "inference-0-score_threshold": "0.4",
            "inference-0-pipeline": CameraInference.RAW,
            "inference-0-detector_checkpoint": "",
            "inference-0-chain": "[]",
        }
        with mock.patch("cameras.admin.rtsp.test_connection", return_value=(True, "")), \
             mock.patch("cameras.services.live_inference.build_predict_payload",
                        return_value={}):
            resp = self.client.post(
                reverse("admin:cameras_camera_change", args=[self.camera.pk]), data)

        self.assertEqual(resp.status_code, 302, getattr(resp, "context", None))
        config = CameraInference.objects.get(camera=self.camera)
        self.assertTrue(config.enabled)
        self.assertEqual(config.target_fps, 3.0)
        self.assertEqual(config.trained_model_id, self.model.pk)
        self.assertEqual(config.score_threshold, 0.4)
        # Worker-owned fields are excluded from the form, so they keep defaults.
        self.assertEqual(config.status, CameraInference.IDLE)

    def test_missing_saved_artifact_stays_selectable(self):
        from cameras.admin import CameraInferenceForm

        instance = CameraInference(camera=self.camera, artifact_path="old/gone.engine")
        with mock.patch("training.services.exports.list_artifacts", return_value=[]):
            form = CameraInferenceForm(instance=instance)
        labels = dict(form.fields["artifact_path"].choices)
        self.assertIn("old/gone.engine", labels)
        self.assertIn("missing", labels["old/gone.engine"])
