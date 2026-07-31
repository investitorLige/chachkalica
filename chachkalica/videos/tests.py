import json
import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME

from fleet.models import FleetSettings
from training.models import TrainedModel, TrainingSettings
from training.services import exports, pipeline_meta
from videos import jobs
from videos.admin import VideoAddForm
from videos.models import InferenceJob, Video
from videos.services import inference
from videos.services.videos import list_video_files


class VideosBase(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fs = FleetSettings.load()
        fs.videos_dir = str(self.root)  # absolute -> used verbatim by videos_root()
        fs.save()

    def tearDown(self):
        self.tmp.cleanup()

    def _touch(self, name: str):
        (self.root / name).write_bytes(b"fake-mp4")


class ScannerTests(VideosBase):
    def test_lists_only_video_extensions_sorted(self):
        for name in ["b.mp4", "a.mov", "notes.txt", ".hidden.mp4", "c.webm"]:
            self._touch(name)
        self.assertEqual(list_video_files(), ["a.mov", "b.mp4", "c.webm"])

    def test_missing_root_is_empty(self):
        self.tmp.cleanup()  # remove the dir
        self.assertEqual(list_video_files(), [])


class AddFormTests(VideosBase):
    def test_requires_exactly_one_source(self):
        self._touch("clip.mp4")
        # neither
        self.assertFalse(VideoAddForm(data={"name": ""}).is_valid())
        # both
        both = VideoAddForm(data={"import_file": "clip.mp4",
                                  "source_url": "https://youtu.be/x"})
        self.assertFalse(both.is_valid())

    def test_import_defaults_name_to_stem(self):
        self._touch("clip.mp4")
        form = VideoAddForm(data={"import_file": "clip.mp4"})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["name"], "clip")

    def test_registered_files_excluded_from_choices(self):
        self._touch("clip.mp4")
        self._touch("other.mp4")
        Video.objects.create(name="clip", filename="clip.mp4", status=Video.READY)
        values = [v for v, _ in VideoAddForm().fields["import_file"].choices]
        self.assertIn("other.mp4", values)
        self.assertNotIn("clip.mp4", values)


class DownloadJobTests(VideosBase):
    def test_success_sets_filename_name_ready(self):
        video = Video.objects.create(
            name="https://youtu.be/x", source_url="https://youtu.be/x",
            status=Video.DOWNLOADING,
        )
        produced = self.root / "My Clip.mp4"
        produced.write_bytes(b"data")
        with mock.patch("videos.jobs.downloader.download", return_value=produced) as dl:
            jobs.download_video(video.id)
        dl.assert_called_once()
        video.refresh_from_db()
        self.assertEqual(video.status, Video.READY)
        self.assertEqual(video.filename, "My Clip.mp4")
        self.assertEqual(video.name, "My Clip")

    def test_failure_records_error(self):
        video = Video.objects.create(
            name="https://youtu.be/bad", source_url="https://youtu.be/bad",
            status=Video.DOWNLOADING,
        )
        with mock.patch("videos.jobs.downloader.download",
                        side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                jobs.download_video(video.id)
        video.refresh_from_db()
        self.assertEqual(video.status, Video.ERROR)
        self.assertIn("boom", video.last_error)

    def test_unique_name_dedupes(self):
        Video.objects.create(name="My Clip", filename="a.mp4", status=Video.READY)
        self.assertEqual(jobs._unique_name("My Clip"), "My Clip (2)")


class AdminRenderTests(VideosBase):
    def setUp(self):
        super().setUp()
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

    def test_index_shows_videos_section(self):
        resp = self.client.get(reverse("admin:index"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Videos")

    def test_add_page_has_both_sources(self):
        (self.root / "clip.mp4").write_bytes(b"x")
        resp = self.client.get(reverse("admin:videos_video_add"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "import_file")
        self.assertContains(resp, "source_url")
        self.assertContains(resp, "id_quality")

    def test_change_page_embeds_player(self):
        (self.root / "clip.mp4").write_bytes(b"x")
        v = Video.objects.create(name="clip", filename="clip.mp4", status=Video.READY)
        resp = self.client.get(reverse("admin:videos_video_change", args=[v.pk]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "<video")

    def test_play_page_renders(self):
        (self.root / "clip.mp4").write_bytes(b"x")
        v = Video.objects.create(name="clip", filename="clip.mp4", status=Video.READY)
        url = reverse("admin:videos_video_play") + f"?video={v.pk}"
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "<video")


class ExportedArtifactTests(VideosBase):
    """Scanning + resolving the export output directory (no DB rows behind it)."""

    def setUp(self):
        super().setUp()
        self.exports = self.root / "exports"
        (self.exports / "nested").mkdir(parents=True)
        ts = TrainingSettings.load()
        ts.exports_root = str(self.exports)  # absolute -> used verbatim
        ts.save()

    def test_lists_onnx_and_engine_recursively_ignoring_others(self):
        (self.exports / "m-best.onnx").write_bytes(b"x")
        (self.exports / "m-best.meta.json").write_text("{}")
        (self.exports / "nested" / "m-last.engine").write_bytes(b"x")
        found = {a["relpath"]: a["kind"] for a in exports.list_artifacts()}
        self.assertEqual(found, {"m-best.onnx": "ONNX",
                                 "nested/m-last.engine": "TensorRT"})

    def test_missing_root_is_empty(self):
        ts = TrainingSettings.load()
        ts.exports_root = str(self.root / "nope")
        ts.save()
        self.assertEqual(exports.list_artifacts(), [])

    def test_resolve_returns_absolute_path(self):
        (self.exports / "m-best.onnx").write_bytes(b"x")
        self.assertEqual(exports.resolve("m-best.onnx"),
                         (self.exports / "m-best.onnx").resolve())

    def test_resolve_rejects_escape_wrong_suffix_and_missing(self):
        (self.root / "secret.onnx").write_bytes(b"x")
        (self.exports / "notes.txt").write_text("x")
        for value in ["", "../secret.onnx", "notes.txt", "gone.onnx"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                exports.resolve(value)


class InferencePayloadTests(VideosBase):
    """``build_predict_payload`` — what each job shape sends to /predict_image."""

    def setUp(self):
        super().setUp()
        self.exports = self.root / "exports"
        self.exports.mkdir()
        ts = TrainingSettings.load()
        ts.exports_root = str(self.exports)
        ts.default_device = "cpu"
        ts.save()
        self.video = Video.objects.create(
            name="clip", filename="clip.mp4", status=Video.READY)
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"x")
        self.tm = TrainedModel.objects.create(
            name="m", arch="yolox", checkpoint_path=str(self.checkpoint))

    def _job(self, **kwargs):
        return InferenceJob(video=self.video, score_threshold=0.4, **kwargs)

    def test_trained_raw_job_sends_checkpoint_and_no_pipeline_knobs(self):
        payload = inference.build_predict_payload(
            self._job(trained_model=self.tm, pipeline=InferenceJob.RAW))
        self.assertEqual(payload["model_checkpoint"], str(self.checkpoint))
        self.assertEqual(payload["pipeline"], "raw")
        self.assertEqual(payload["score_threshold"], 0.4)
        for key in ["detector_checkpoint", "tile_size_px", "tile_width_pct", "chain"]:
            self.assertNotIn(key, payload)

    def test_exported_artifact_job_sends_resolved_artifact(self):
        artifact = self.exports / "m-best.engine"
        artifact.write_bytes(b"x")
        payload = inference.build_predict_payload(self._job(
            model_source=InferenceJob.EXPORTED, artifact_path="m-best.engine"))
        self.assertEqual(payload["model_checkpoint"], str(artifact.resolve()))

    def test_bundle_job_sends_the_bundled_model_and_detector(self):
        """The end of the road: a bundle-sourced job runs the bundle's own files.

        Neither service needed changing for this — both build their payload from
        ``model_checkpoint()`` plus the row's own fields, and a bundle fills both
        in. This asserts that stays true.
        """
        from training.tests_bundles import MANIFEST

        root = self.root / "bundles"
        bundle = root / "ppe-bundle"
        (bundle / "models").mkdir(parents=True)
        (bundle / "pipeline.json").write_text(json.dumps(MANIFEST))
        (bundle / "models/model.engine").write_bytes(b"x")
        (bundle / "models/detector.engine").write_bytes(b"x")
        ts = TrainingSettings.load()
        ts.bundles_root = str(root)
        ts.save()

        job = self._job(model_source=InferenceJob.BUNDLE, bundle_path="ppe-bundle")
        job.sync_bundle()

        payload = inference.build_predict_payload(job)

        self.assertEqual(payload["model_checkpoint"], str(bundle / "models/model.engine"))
        self.assertEqual(payload["detector_checkpoint"],
                         str(bundle / "models/detector.engine"))
        self.assertEqual(payload["pipeline"], "people_detect_first")
        self.assertEqual(payload["detector_expand_ratio"], 0.1)
        self.assertEqual(payload["detector_min_box_size"], 224.0)
        self.assertEqual(payload["merge_nms_iou"], 0.3)

    def test_pipeline_knobs_are_forwarded(self):
        detector = self.root / "person.engine"
        detector.write_bytes(b"x")
        payload = inference.build_predict_payload(self._job(
            trained_model=self.tm,
            pipeline="people_detect_first",
            detector_checkpoint=str(detector),
            detector_expand_ratio=0.15,
            detector_min_box_size=224.0,
            tile_size_px=640,
            overlap=0.3,
        ))
        self.assertEqual(payload["detector_checkpoint"], str(detector))
        self.assertEqual(payload["detector_expand_ratio"], 0.15)
        self.assertEqual(payload["detector_min_box_size"], 224.0)
        self.assertEqual(payload["tile_size_px"], 640)
        self.assertEqual(payload["overlap"], 0.3)

    def test_person_pipeline_without_detector_is_rejected(self):
        with self.assertRaises(RuntimeError):
            inference.build_predict_payload(self._job(
                trained_model=self.tm, pipeline="people_detect_first"))

    def test_merge_nms_iou_is_forwarded_only_when_set(self):
        # chachak parses merge_nms_iou with an unconditional float(), so the
        # payload must omit the key rather than send null.
        blank = inference.build_predict_payload(self._job(
            trained_model=self.tm, pipeline="batch_detect"))
        self.assertNotIn("merge_nms_iou", blank)

        payload = inference.build_predict_payload(self._job(
            trained_model=self.tm, pipeline="batch_detect", merge_nms_iou=0.55))
        self.assertEqual(payload["merge_nms_iou"], 0.55)

    def test_chain_pipeline_requires_members(self):
        with self.assertRaises(RuntimeError):
            inference.build_predict_payload(self._job(
                trained_model=self.tm, pipeline="chain", chain=[]))

    def test_missing_artifact_is_rejected(self):
        with self.assertRaises(RuntimeError):
            inference.build_predict_payload(self._job(
                model_source=InferenceJob.EXPORTED, artifact_path="absent.onnx"))


class RunInferenceActionTests(VideosBase):
    """The two-step "Run model inference…" action."""

    def setUp(self):
        super().setUp()
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

        self.exports = self.root / "exports"
        self.exports.mkdir()
        (self.exports / "m-best.onnx").write_bytes(b"x")
        ts = TrainingSettings.load()
        ts.exports_root = str(self.exports)
        ts.save()

        (self.root / "clip.mp4").write_bytes(b"x")
        self.video = Video.objects.create(
            name="clip", filename="clip.mp4", status=Video.READY)
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"x")
        self.tm = TrainedModel.objects.create(
            name="m", arch="yolox", checkpoint_path=str(self.checkpoint))
        self.url = reverse("admin:videos_video_changelist")

    def _post(self, **extra):
        data = {
            "action": "run_inference",
            ACTION_CHECKBOX_NAME: [str(self.video.pk)],
            **extra,
        }
        return self.client.post(self.url, data, follow=True)

    def test_step_one_asks_for_model_type(self):
        resp = self._post()
        self.assertContains(resp, 'name="model_source"')
        self.assertContains(resp, "exported artifact")
        self.assertNotContains(resp, 'name="pipeline"')

    def test_step_two_trained_renders_model_and_pipeline_selects(self):
        resp = self._post(configure="1", model_source=InferenceJob.TRAINED)
        self.assertContains(resp, 'name="trained_model"')
        self.assertContains(resp, 'name="pipeline"')
        self.assertContains(resp, "people_detect_first")
        self.assertNotContains(resp, 'name="artifact_path"')

    def test_step_two_exported_renders_artifact_select(self):
        resp = self._post(configure="1", model_source=InferenceJob.EXPORTED)
        self.assertContains(resp, 'name="artifact_path"')
        self.assertContains(resp, "m-best.onnx")
        self.assertNotContains(resp, 'name="trained_model"')

    @mock.patch("videos.admin._queue")
    def test_apply_trained_with_pipeline_creates_job(self, queue):
        detector = self.root / "person.engine"
        detector.write_bytes(b"x")
        self._post(
            apply="1", model_source=InferenceJob.TRAINED,
            trained_model=str(self.tm.pk), pipeline="people_detect_first",
            detector_checkpoint=str(detector), detector_expand_ratio="0.1",
            score_threshold="0.6", frame_stride="5",
        )
        job = InferenceJob.objects.get()
        self.assertEqual(job.model_source, InferenceJob.TRAINED)
        self.assertEqual(job.trained_model, self.tm)
        self.assertEqual(job.pipeline, "people_detect_first")
        self.assertEqual(job.detector_expand_ratio, 0.1)
        self.assertEqual(job.frame_stride, 5)
        self.assertTrue(job.output_filename)
        queue.return_value.enqueue.assert_called_once()

    @mock.patch("videos.admin._queue")
    def test_apply_exported_creates_job_without_trained_model(self, queue):
        self._post(
            apply="1", model_source=InferenceJob.EXPORTED,
            artifact_path="m-best.onnx", pipeline="raw",
            score_threshold="0.5", frame_stride="1",
        )
        job = InferenceJob.objects.get()
        self.assertEqual(job.model_source, InferenceJob.EXPORTED)
        self.assertIsNone(job.trained_model)
        self.assertEqual(job.artifact_path, "m-best.onnx")
        self.assertEqual(job.model_label(), "m-best.onnx")
        queue.return_value.enqueue.assert_called_once()

    @mock.patch("videos.admin._queue")
    def test_apply_rejects_out_of_root_artifact(self, queue):
        (self.root / "secret.onnx").write_bytes(b"x")
        resp = self._post(
            apply="1", model_source=InferenceJob.EXPORTED,
            artifact_path="../secret.onnx", pipeline="raw",
            score_threshold="0.5", frame_stride="1",
        )
        self.assertFalse(InferenceJob.objects.exists())
        queue.return_value.enqueue.assert_not_called()
        self.assertContains(resp, "outside the export output directory")

    @mock.patch("videos.admin._queue")
    def test_apply_rejects_person_pipeline_without_detector(self, queue):
        resp = self._post(
            apply="1", model_source=InferenceJob.TRAINED,
            trained_model=str(self.tm.pk), pipeline="batch_people",
            score_threshold="0.5", frame_stride="1",
        )
        self.assertFalse(InferenceJob.objects.exists())
        queue.return_value.enqueue.assert_not_called()
        self.assertContains(resp, "requires a detector checkpoint")
        # Re-renders step 2 rather than bouncing back to the model-type page.
        self.assertContains(resp, 'name="pipeline"')

    # ------------------------------------------------------------- bundles
    def _bundle(self, name="ppe-bundle"):
        """A complete bundle under a bundle root pointed at by TrainingSettings."""
        from training.tests_bundles import MANIFEST

        root = self.root / "bundles"
        bundle = root / name
        (bundle / "models").mkdir(parents=True)
        (bundle / "pipeline.json").write_text(json.dumps(MANIFEST))
        (bundle / "models/model.engine").write_bytes(b"x")
        (bundle / "models/detector.engine").write_bytes(b"x")
        (bundle / "runtime").mkdir()
        (bundle / "infer.py").write_text("")
        ts = TrainingSettings.load()
        ts.bundles_root = str(root)
        ts.save()
        return bundle

    def test_step_one_offers_bundles(self):
        self._bundle()
        resp = self._post()
        self.assertContains(resp, "infer bundle")
        self.assertContains(resp, "1 found")

    def test_step_two_bundle_renders_the_bundle_select(self):
        self._bundle()
        resp = self._post(configure="1", model_source=InferenceJob.BUNDLE)
        self.assertContains(resp, 'name="bundle_path"')
        self.assertContains(resp, "ppe-bundle")
        self.assertContains(resp, "Sync bundle")
        # Nothing preselected: the geometry arrives via an explicit sync, and a
        # preselected bundle with filled-in fields would imply that happened.
        self.assertContains(resp, 'value="">———')

    @mock.patch("videos.admin._queue")
    def test_apply_bundle_takes_its_geometry_from_the_manifest(self, queue):
        bundle = self._bundle()
        self._post(
            apply="1", model_source=InferenceJob.BUNDLE, bundle_path="ppe-bundle",
            # What a stale page or a hand-edited POST might send for the locked
            # fields — the bundle's own values must win.
            pipeline="raw", detector_expand_ratio="9", merge_nms_iou="0.99",
            detector_checkpoint="/wrong/person.pt",
            score_threshold="0.5", frame_stride="2",
        )
        job = InferenceJob.objects.get()
        self.assertEqual(job.model_source, InferenceJob.BUNDLE)
        self.assertEqual(job.bundle_path, "ppe-bundle")
        self.assertEqual(job.pipeline, "people_detect_first")
        self.assertEqual(job.detector_expand_ratio, 0.1)
        self.assertEqual(job.detector_min_box_size, 224.0)
        self.assertEqual(job.merge_nms_iou, 0.3)
        self.assertEqual(job.detector_checkpoint,
                         str(bundle / "models/detector.engine"))
        # The one knob the operator keeps.
        self.assertEqual(job.score_threshold, 0.5)
        self.assertEqual(job.frame_stride, 2)
        self.assertEqual(job.model_checkpoint(), str(bundle / "models/model.engine"))
        queue.return_value.enqueue.assert_called_once()

    @mock.patch("videos.admin._queue")
    def test_apply_rejects_a_bundle_missing_its_model(self, queue):
        bundle = self._bundle()
        (bundle / "models/model.engine").unlink()
        resp = self._post(
            apply="1", model_source=InferenceJob.BUNDLE, bundle_path="ppe-bundle",
            pipeline="raw", score_threshold="0.5", frame_stride="1",
        )
        self.assertFalse(InferenceJob.objects.exists())
        queue.return_value.enqueue.assert_not_called()
        self.assertContains(resp, "is not usable")

    @mock.patch("videos.admin._queue")
    def test_apply_rejects_an_out_of_root_bundle(self, queue):
        self._bundle()
        resp = self._post(
            apply="1", model_source=InferenceJob.BUNDLE, bundle_path="../../etc",
            pipeline="raw", score_threshold="0.5", frame_stride="1",
        )
        self.assertFalse(InferenceJob.objects.exists())
        self.assertContains(resp, "outside the bundle directory")

    @mock.patch("videos.admin._queue")
    def test_apply_bundle_without_a_selection_asks_for_one(self, queue):
        self._bundle()
        resp = self._post(
            apply="1", model_source=InferenceJob.BUNDLE, bundle_path="",
            pipeline="raw", score_threshold="0.5", frame_stride="1",
        )
        self.assertFalse(InferenceJob.objects.exists())
        self.assertContains(resp, "Choose a bundle")

    @mock.patch("videos.admin._queue")
    def test_apply_drops_knobs_the_pipeline_cannot_use(self, queue):
        """Rows hidden by CSS still POST — the saved job must not claim them."""
        self._post(
            apply="1", model_source=InferenceJob.TRAINED,
            trained_model=str(self.tm.pk), pipeline="raw",
            detector_checkpoint="/ckpts/person.pt", detector_expand_ratio="0.2",
            tile_size_px="640", overlap="0.3", chain="batch_detect",
            score_threshold="0.5", frame_stride="1",
        )
        job = InferenceJob.objects.get()
        self.assertEqual(job.detector_checkpoint, "")
        self.assertIsNone(job.detector_expand_ratio)
        self.assertIsNone(job.tile_size_px)
        self.assertIsNone(job.overlap)
        self.assertEqual(job.chain, [])

    @mock.patch("videos.admin._queue")
    def test_apply_saves_merge_nms_iou(self, queue):
        self._post(
            apply="1", model_source=InferenceJob.TRAINED,
            trained_model=str(self.tm.pk), pipeline="batch_detect",
            merge_nms_iou="0.55", score_threshold="0.5", frame_stride="1",
        )
        self.assertEqual(InferenceJob.objects.get().merge_nms_iou, 0.55)

    @mock.patch("videos.admin._queue")
    def test_apply_drops_merge_nms_iou_for_raw(self, queue):
        """"raw" runs the model once on the whole frame — nothing to merge."""
        self._post(
            apply="1", model_source=InferenceJob.TRAINED,
            trained_model=str(self.tm.pk), pipeline="raw",
            merge_nms_iou="0.55", score_threshold="0.5", frame_stride="1",
        )
        self.assertIsNone(InferenceJob.objects.get().merge_nms_iou)

    @mock.patch("videos.admin._queue")
    def test_apply_rejects_bad_merge_nms_iou(self, queue):
        self._post(
            apply="1", model_source=InferenceJob.TRAINED,
            trained_model=str(self.tm.pk), pipeline="batch_detect",
            merge_nms_iou="1.5", score_threshold="0.5", frame_stride="1",
        )
        self.assertFalse(InferenceJob.objects.exists())
        queue.return_value.enqueue.assert_not_called()

    @mock.patch("videos.admin._queue")
    def test_apply_rejects_bad_overlap(self, queue):
        self._post(
            apply="1", model_source=InferenceJob.TRAINED,
            trained_model=str(self.tm.pk), pipeline="batch_detect",
            overlap="1.5", score_threshold="0.5", frame_stride="1",
        )
        self.assertFalse(InferenceJob.objects.exists())
        queue.return_value.enqueue.assert_not_called()


class RunInferencePrefillTests(VideosBase):
    """Step 2 arrives configured from the model's pipeline metadata.

    The point of the metadata convention: running a model the way it was trained
    should take zero selections, so the form is rendered server-side from the
    preselected model's frozen record rather than waiting for a change event.
    """

    def setUp(self):
        super().setUp()
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

        self.exports = self.root / "exports"
        self.exports.mkdir()
        ts = TrainingSettings.load()
        ts.exports_root = str(self.exports)
        ts.save()

        (self.root / "clip.mp4").write_bytes(b"x")
        self.video = Video.objects.create(
            name="clip", filename="clip.mp4", status=Video.READY)
        self.url = reverse("admin:videos_video_changelist")

        self.detector = self.root / "person.pt"
        self.detector.write_bytes(b"x")
        # Real files on disk: the submit path pre-flights the model artifact, and a
        # missing one would mask the validation error a test is actually after.
        for name in ("a.pt", "b.pt", "c.pt", "d.pt"):
            (self.root / name).write_bytes(b"x")
        self.tiled = TrainedModel.objects.create(
            name="tiled", arch="yolox", checkpoint_path=str(self.root / "a.pt"),
            pipeline_metadata=pipeline_meta.normalize({
                "pipeline": "batch_detect",
                "tile_size_px": 640,
                "overlap": 0.25,
                "score_threshold": 0.42,
            }),
        )

    def _step_two(self, **extra):
        return self.client.post(self.url, {
            "action": "run_inference",
            ACTION_CHECKBOX_NAME: [str(self.video.pk)],
            "configure": "1",
            "model_source": InferenceJob.TRAINED,
            **extra,
        }, follow=True)

    def test_newest_model_is_preselected_with_its_pipeline_filled_in(self):
        resp = self._step_two()
        html = resp.content.decode()

        self.assertIn(f'value="{self.tiled.pk}"\n                  selected', html)
        self.assertIn('<option value="batch_detect" selected>', html)
        self.assertIn('id="tile_size_px" min="1" step="1"\n           value="640"', html)
        self.assertIn('value="0.25"', html)
        # The recorded operating point, not the form's generic 0.5 fallback.
        self.assertIn('value="0.42"', html)

    def test_newest_model_wins_when_several_are_catalogued(self):
        newer = TrainedModel.objects.create(
            name="cropped", arch="yolox", checkpoint_path=str(self.root / "b.pt"),
            pipeline_metadata=pipeline_meta.normalize({
                "pipeline": "people_detect_first",
                "detector_checkpoint": str(self.detector),
                "detector_expand_ratio": 0.12,
            }),
        )
        html = self._step_two().content.decode()

        self.assertIn(f'value="{newer.pk}"\n                  selected', html)
        self.assertIn('<option value="people_detect_first" selected>', html)
        self.assertIn(f'value="{self.detector}"', html)
        self.assertIn('value="0.12"', html)

    def test_the_defaults_maps_are_rendered_as_json_script(self):
        html = self._step_two().content.decode()
        self.assertIn('<script id="trained-defaults" type="application/json">', html)
        self.assertIn('JSON.parse', html)

    def test_a_script_tag_in_the_metadata_cannot_break_out(self):
        # The maps are keyed by model pk / artifact filename and carry operator-set
        # paths, so their values reach a <script> block; json.dumps escapes JSON
        # syntax but not HTML, which json_script does.
        TrainedModel.objects.create(
            name="sneaky", arch="yolox", checkpoint_path=str(self.root / "d.pt"),
            pipeline_metadata=pipeline_meta.normalize({
                "pipeline": "people_detect_first",
                "detector_checkpoint": '</script><script>alert(1)</script>',
            }),
        )
        html = self._step_two().content.decode()

        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("\\u003Cscript\\u003Ealert(1)", html)

    def test_a_model_with_no_pipeline_on_record_renders_raw_and_blank(self):
        TrainedModel.objects.create(
            name="plain", arch="yolox", checkpoint_path=str(self.root / "c.pt"))
        html = self._step_two().content.decode()

        self.assertIn('<option value="raw" selected>', html)
        # No geometry to prefill, and the generic threshold fallback applies.
        self.assertIn('id="tile_size_px" min="1" step="1"\n           value=""', html)
        self.assertIn('value="0.5"', html)

    def test_explicit_selection_overrides_the_preselected_model(self):
        older_wins = self._step_two(trained_model=str(self.tiled.pk))
        TrainedModel.objects.create(
            name="newer", arch="yolox", checkpoint_path=str(self.root / "d.pt"))

        html = older_wins.content.decode()
        self.assertIn('<option value="batch_detect" selected>', html)

    def test_rerender_after_a_warning_keeps_what_the_operator_typed(self):
        # batch_people with no detector fails validation; the re-rendered form must
        # show the operator's own values, not reset to the model's metadata.
        with mock.patch("videos.admin._queue"):
            resp = self.client.post(self.url, {
                "action": "run_inference",
                ACTION_CHECKBOX_NAME: [str(self.video.pk)],
                "apply": "1",
                "model_source": InferenceJob.TRAINED,
                "trained_model": str(self.tiled.pk),
                "pipeline": "batch_people",
                "tile_size_px": "1024",
                "score_threshold": "0.77",
                "frame_stride": "9",
            }, follow=True)

        html = resp.content.decode()
        self.assertContains(resp, "requires a detector checkpoint")
        self.assertIn('<option value="batch_people" selected>', html)
        self.assertIn('value="1024"', html)
        self.assertIn('value="0.77"', html)
        self.assertIn('value="9"', html)
        # Not the model's recorded values.
        self.assertNotIn('value="640"', html)

    def test_merge_nms_iou_prefills_from_the_metadata(self):
        merged = TrainedModel.objects.create(
            name="merged", arch="yolox", checkpoint_path=str(self.root / "e.pt"),
            pipeline_metadata=pipeline_meta.normalize({
                "pipeline": "batch_detect",
                "merge_nms_iou": 0.55,
            }),
        )
        html = self._step_two().content.decode()

        self.assertIn(f'value="{merged.pk}"\n                  selected', html)
        self.assertIn('id="merge_nms_iou"', html)
        self.assertIn('min="0" max="1" step="0.05" value="0.55"', html)

    def test_exported_artifact_prefills_from_its_sidecar(self):
        artifact = self.exports / "tiled-best.onnx"
        artifact.write_bytes(b"x")
        exports.export_pipeline_sidecar(self.tiled, artifact)

        html = self.client.post(self.url, {
            "action": "run_inference",
            ACTION_CHECKBOX_NAME: [str(self.video.pk)],
            "configure": "1",
            "model_source": InferenceJob.EXPORTED,
        }, follow=True).content.decode()

        self.assertIn('value="tiled-best.onnx"\n                  selected', html)
        self.assertIn('<option value="batch_detect" selected>', html)
        self.assertIn('value="640"', html)


class StreamViewTests(VideosBase):
    def setUp(self):
        super().setUp()
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")
        (self.root / "clip.mp4").write_bytes(b"0123456789")
        self.video = Video.objects.create(
            name="clip", filename="clip.mp4", status=Video.READY,
        )

    def test_full_response_advertises_ranges(self):
        url = reverse("admin:videos_video_stream") + f"?video={self.video.pk}"
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Accept-Ranges"], "bytes")

    def test_range_request_returns_206_slice(self):
        url = reverse("admin:videos_video_stream") + f"?video={self.video.pk}"
        resp = self.client.get(url, headers={"range": "bytes=2-5"})
        self.assertEqual(resp.status_code, 206)
        self.assertEqual(resp["Content-Range"], "bytes 2-5/10")
        self.assertEqual(resp["Content-Length"], "4")
        self.assertEqual(b"".join(resp.streaming_content), b"2345")


class DrawBoxesTests(TestCase):
    """``inference.draw_boxes`` — shared with the live camera path, so it has to
    hold up on 4K frames as well as the ~1080p video it was written for."""

    @staticmethod
    def _frame(width, height):
        import numpy as np

        return np.zeros((height, width, 3), dtype=np.uint8)

    @staticmethod
    def _box(cx, cy, w, h, name="helmet"):
        return {"cx": cx, "cy": cy, "w": w, "h": h,
                "confidence": 0.9, "class_id": 0, "class_name": name}

    def _drawn_rows(self, frame):
        """Row indices that got drawn on, i.e. where any annotation landed.

        Checked across all channels, not just green: different class labels
        are now colored differently, so the drawn color isn't fixed.
        """
        return set(frame.any(axis=2).nonzero()[0].tolist())

    def test_draws_within_frame(self):
        frame = self._frame(1920, 1080)
        inference.draw_boxes(frame, [self._box(0.5, 0.5, 0.2, 0.2)])
        self.assertTrue(self._drawn_rows(frame))

    def test_box_extending_past_top_edge_still_labels_inside_the_frame(self):
        # The regression this guards: an unclamped label y went negative and the
        # text silently vanished. Every drawn row must be inside the frame.
        frame = self._frame(3840, 2160)
        inference.draw_boxes(frame, [self._box(0.04, 0.04, 0.05, 0.2)])
        rows = self._drawn_rows(frame)
        self.assertTrue(rows)
        self.assertTrue(min(rows) >= 0 and max(rows) < 2160)

    def test_box_fully_off_frame_does_not_raise(self):
        frame = self._frame(1920, 1080)
        inference.draw_boxes(frame, [self._box(1.5, 1.5, 0.2, 0.2)])  # no raise

    def test_different_labels_get_different_colors(self):
        frame = self._frame(1920, 1080)
        boxes = [self._box(0.3, 0.5, 0.1, 0.1, name="helmet"),
                 self._box(0.7, 0.5, 0.1, 0.1, name="vest")]
        colors = {tuple(c) for c in
                  (inference._color_for_class(b["class_name"]) for b in boxes)}
        self.assertEqual(len(colors), 2)
        inference.draw_boxes(frame, boxes)
        left_colors = {tuple(px) for px in frame[:, :960].reshape(-1, 3)
                       if px.any()}
        right_colors = {tuple(px) for px in frame[:, 960:].reshape(-1, 3)
                        if px.any()}
        self.assertTrue(left_colors)
        self.assertTrue(right_colors)
        self.assertNotEqual(left_colors, right_colors)

    def test_class_color_is_stable_across_calls(self):
        self.assertEqual(inference._color_for_class("helmet"),
                          inference._color_for_class("helmet"))

    def _bottom_edge_thickness(self, width, height):
        """Measured stroke width of a box's bottom edge, in pixels.

        Sampled down the middle of the box, where no label text can interfere.
        """
        frame = self._frame(width, height)
        inference.draw_boxes(frame, [self._box(0.5, 0.5, 0.2, 0.2)])
        column = frame[:, width // 2, :]
        near_bottom = column[int(0.55 * height):]  # y2 is at 0.6 * height
        return int(near_bottom.any(axis=1).sum())

    def test_annotations_scale_up_with_resolution(self):
        """A 4K frame must get thicker lines than 1080p, or the boxes are
        hairlines on the showroom camera (which is 3840x2160).

        Asserted on measured stroke width rather than total drawn pixels: the
        drawn *fraction* of the frame stays flat when strokes scale linearly and
        area grows quadratically, so it cannot tell the two cases apart.
        """
        hd = self._bottom_edge_thickness(1920, 1080)
        uhd = self._bottom_edge_thickness(3840, 2160)
        # Compared, not pinned to exact pixel counts — cv2 rasterizes a stroke
        # slightly wider than the requested thickness, which is its business.
        self.assertGreater(uhd, hd)

    def test_sub_1080p_frames_are_not_scaled_down(self):
        # scale is floored at 1.0 — a 1px box and sub-pixel text would be worse
        # than keeping the 1080p sizes on a small frame.
        self.assertEqual(self._bottom_edge_thickness(640, 480),
                         self._bottom_edge_thickness(1920, 1080))
