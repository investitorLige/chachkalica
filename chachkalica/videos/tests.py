import json
import shutil
import subprocess
import tempfile
import unittest
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
from videos.models import InferenceJob, RenderPreset, Video
from videos.services import inference, render_style
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


class CleanupSignalTests(VideosBase):
    """Deleting a Video/InferenceJob row should also remove its file on disk
    (see ``videos.signals``), the same convention ``training.signals`` uses."""

    def test_deleting_video_removes_its_file(self):
        self._touch("clip.mp4")
        video = Video.objects.create(name="clip", filename="clip.mp4", status=Video.READY)
        path = video.path()
        self.assertTrue(path.exists())
        video.delete()
        self.assertFalse(path.exists())

    def test_deleting_video_with_no_file_is_safe(self):
        # e.g. a row still downloading, or one whose download failed.
        video = Video.objects.create(name="pending", filename="", status=Video.DOWNLOADING)
        video.delete()  # must not raise

    def test_deleting_inference_job_removes_annotated_video(self):
        self._touch("clip.mp4")
        video = Video.objects.create(name="clip", filename="clip.mp4", status=Video.READY)
        out = inference.output_dir() / "clip_inferred.mp4"
        out.write_bytes(b"fake-annotated-mp4")
        job = InferenceJob.objects.create(video=video, output_filename=out.name)
        job.delete()
        self.assertFalse(out.exists())

    def test_deleting_inference_job_with_no_output_is_safe(self):
        self._touch("clip.mp4")
        video = Video.objects.create(name="clip", filename="clip.mp4", status=Video.READY)
        job = InferenceJob.objects.create(video=video)  # never finished, output_filename blank
        job.delete()  # must not raise (was IsADirectoryError before this used a signal)

    def test_deleting_trained_model_cascades_and_cleans_up_inference_job_output(self):
        """A path that never goes through InferenceJobAdmin at all."""
        self._touch("clip.mp4")
        video = Video.objects.create(name="clip", filename="clip.mp4", status=Video.READY)
        model = TrainedModel.objects.create(name="m1", arch="yolox", checkpoint_path="/tmp/best.pt")
        out = inference.output_dir() / "clip_inferred.mp4"
        out.write_bytes(b"fake-annotated-mp4")
        job = InferenceJob.objects.create(
            video=video, model_source=InferenceJob.TRAINED, trained_model=model,
            output_filename=out.name,
        )
        model.delete()
        self.assertFalse(InferenceJob.objects.filter(pk=job.pk).exists())
        self.assertFalse(out.exists())


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


class RenderStyleFieldTests(TestCase):
    """``render_style``'s two entry points: :func:`normalize` (never rejects, used
    on the render side) and :func:`parse_form` (rejects loudly, used at the form).
    The split is the whole design, so both halves are pinned here."""

    def test_defaults_are_complete_and_in_range(self):
        style = render_style.defaults()
        self.assertEqual(style, render_style.normalize(style))
        self.assertIn("box_style", style)
        self.assertIn("crf", style)

    def test_normalize_fills_in_missing_and_clamps_nonsense(self):
        style = render_style.normalize({
            "thickness": 9999, "box_opacity": -4, "palette": "not-a-palette",
            "smoothing": "0.5", "watermark_text": "  hi  ", "fixed_color": "nope",
        })
        self.assertEqual(style["thickness"], 40)          # clamped to the maximum
        self.assertEqual(style["box_opacity"], 0.0)       # clamped to the minimum
        self.assertEqual(style["palette"], "vivid")       # unknown choice -> default
        self.assertEqual(style["smoothing"], 0.5)         # numeric string coerced
        self.assertEqual(style["watermark_text"], "hi")
        self.assertEqual(style["fixed_color"], "#22c55e")  # bad colour -> default

    def test_normalize_survives_a_non_dict(self):
        # A hand-edited or older row must still render rather than 500 the worker.
        self.assertEqual(render_style.normalize(None), render_style.defaults())
        self.assertEqual(render_style.normalize("{}"), render_style.defaults())

    def test_parse_form_reads_the_style_prefixed_fields(self):
        style, error = render_style.parse_form({
            "style_box_style": "corners", "style_thickness": "6",
            "style_palette": "neon", "style_watermark_text": "Mercury",
            "style_class_filter": "helmet, vest",
        })
        self.assertIsNone(error)
        self.assertEqual(style["box_style"], "corners")
        self.assertEqual(style["thickness"], 6)
        self.assertEqual(style["palette"], "neon")
        self.assertEqual(style["class_filter"], ["helmet", "vest"])

    def test_parse_form_blank_means_default_not_zero(self):
        style, error = render_style.parse_form({"style_thickness": "", "style_crf": ""})
        self.assertIsNone(error)
        self.assertEqual(style["thickness"], render_style.defaults()["thickness"])
        self.assertEqual(style["crf"], render_style.defaults()["crf"])

    def test_a_bad_field_does_not_discard_the_good_ones(self):
        # The form is re-rendered from what comes back, so one out-of-range knob
        # must not reset the other twenty-nine.
        style, error = render_style.parse_form({
            "style_thickness": "9000", "style_palette": "neon",
            "style_watermark_text": "Keep me",
        })
        self.assertIn("Thickness", error)
        self.assertEqual(style["palette"], "neon")
        self.assertEqual(style["watermark_text"], "Keep me")
        self.assertEqual(style["thickness"], render_style.defaults()["thickness"])

    def test_parse_form_rejects_out_of_range_and_names_the_field(self):
        _style, error = render_style.parse_form({"style_thickness": "500"})
        self.assertIn("Thickness", error)
        _style, error = render_style.parse_form({"style_crf": "nope"})
        self.assertIn("CRF", error)
        _style, error = render_style.parse_form({"style_box_style": "sparkles"})
        self.assertIn("sparkles", error)
        _style, error = render_style.parse_form({"style_fixed_color": "reddish"})
        self.assertIn("hex", error)

    def test_checkboxes_need_their_companion_marker_to_read_as_unticked(self):
        # An absent checkbox in a form that *did* render it means "off"; an absent
        # checkbox in a POST that never had the field must not silently turn a
        # default-on knob off.
        style, _e = render_style.parse_form({"style_has_shadow": "1"})
        self.assertFalse(style["shadow"])
        style, _e = render_style.parse_form({})
        self.assertTrue(style["shadow"])
        style, _e = render_style.parse_form({"style_has_shadow": "1", "style_shadow": "on"})
        self.assertTrue(style["shadow"])

    def test_per_class_colors_round_trip(self):
        style, error = render_style.parse_form(
            {"style_class_colors": "helmet=#FF0000, vest=#3b82f6"})
        self.assertIsNone(error)
        self.assertEqual(style["class_colors"], {"helmet": "#ff0000", "vest": "#3b82f6"})
        self.assertEqual(render_style.format_class_colors(style["class_colors"]),
                         "helmet=#ff0000, vest=#3b82f6")

    def test_per_class_colors_reject_junk(self):
        _style, error = render_style.parse_form({"style_class_colors": "helmet"})
        self.assertIn("helmet", error)
        _style, error = render_style.parse_form({"style_class_colors": "helmet=puce"})
        self.assertIn("puce", error)

    def test_force_class_round_trips_and_is_off_by_default(self):
        self.assertEqual(render_style.defaults()["force_class"], "")
        style, error = render_style.parse_form({"style_force_class": " helmet "})
        self.assertIsNone(error)
        self.assertEqual(style["force_class"], "helmet")
        self.assertEqual(
            render_style.form_values(style)["style_force_class"], "helmet")
        # Not validated against any class space: it is only ever drawn, and the
        # dropdown that offers it is filled from whichever model is selected.
        self.assertEqual(
            render_style.normalize({"force_class": "violation"})["force_class"],
            "violation")

    def test_encode_options_are_read_back_from_a_stored_style(self):
        self.assertEqual(render_style.encode_options({"crf": 30, "output_height": 720}),
                         {"crf": 30, "output_height": 720})
        self.assertEqual(render_style.encode_options({})["output_height"], 0)

    def test_class_colors_are_stable_and_frame_independent(self):
        style = render_style.defaults()
        first = render_style._class_color("helmet", style)
        self.assertEqual(first, render_style._class_color("helmet", style))
        # Not affected by what else has been drawn — the preview and the video it
        # previews must agree even when they contain different classes.
        render_style._class_color("vest", style)
        self.assertEqual(first, render_style._class_color("helmet", style))

    def test_a_per_class_override_wins_over_the_palette(self):
        style = render_style.normalize({"class_colors": {"helmet": "#ff0000"}})
        self.assertEqual(render_style._class_color("helmet", style), (0, 0, 255))


class MarketingRendererTests(TestCase):
    """The renderer itself. Assertions are about *what changed on the frame*
    rather than exact pixel values — the point of each knob is that it visibly
    does something and never draws outside the frame."""

    @staticmethod
    def _frame(width=1280, height=720):
        import numpy as np

        return np.zeros((height, width, 3), dtype=np.uint8)

    @staticmethod
    def _box(cx=0.5, cy=0.5, w=0.2, h=0.2, name="helmet", confidence=0.9):
        return {"cx": cx, "cy": cy, "w": w, "h": h, "confidence": confidence,
                "class_id": 0, "class_name": name}

    def _render(self, style, boxes, **frame_kwargs):
        frame = self._frame(**frame_kwargs)
        drawn = render_style.MarketingRenderer(style, still=True).draw(frame, boxes)
        return frame, drawn

    def test_draws_something_within_the_frame(self):
        frame, drawn = self._render({}, [self._box()])
        self.assertEqual(drawn, 1)
        self.assertTrue(frame.any())

    def test_every_outline_style_draws(self):
        for box_style, _label in render_style.BOX_STYLE_CHOICES:
            with self.subTest(box_style=box_style):
                frame, _n = self._render(
                    {"box_style": box_style, "show_label": False,
                     "fill_opacity": 0 if box_style == "none" else 0.2},
                    [self._box()])
                self.assertEqual(frame.any(), box_style != "none")

    def test_every_label_style_and_position_draws_inside_the_frame(self):
        for styled, _l in render_style.LABEL_STYLE_CHOICES:
            for position, _p in render_style.LABEL_POSITION_CHOICES:
                with self.subTest(style=styled, position=position):
                    # A box hard against the top-left corner: the placement that
                    # used to put a label at a negative y and lose it entirely.
                    frame, _n = self._render(
                        {"label_style": styled, "label_position": position,
                         "box_style": "none", "fill_opacity": 0},
                        [self._box(cx=0.06, cy=0.04, w=0.1, h=0.06)])
                    self.assertTrue(frame.any(), "label vanished")

    def test_a_box_off_the_frame_is_skipped_not_drawn(self):
        frame, drawn = self._render({}, [self._box(cx=1.8, cy=1.8)])
        self.assertEqual(drawn, 0)
        self.assertFalse(frame.any())

    def test_malformed_boxes_do_not_raise(self):
        frame, drawn = self._render({}, [{"nope": 1}, {"cx": 0.5, "cy": 0.5}])
        self.assertEqual(drawn, 0)

    def test_class_filter_hides_other_classes(self):
        boxes = [self._box(cx=0.25, name="helmet"), self._box(cx=0.75, name="vest")]
        frame, drawn = self._render({"class_filter": ["helmet"]}, boxes)
        self.assertEqual(drawn, 1)
        self.assertTrue(frame[:, :640].any())
        self.assertFalse(frame[:, 640:].any())

    def test_force_class_draws_every_box_as_that_class(self):
        """Name and colour both, since both come off ``class_name`` — asserted as
        "identical to the frame that class would have drawn anyway"."""
        import numpy as np

        relabelled, drawn = self._render(
            {"force_class": "helmet"}, [self._box(name="vest")])
        as_helmet, _n = self._render({}, [self._box(name="helmet")])
        as_vest, _n = self._render({}, [self._box(name="vest")])
        self.assertEqual(drawn, 1)
        self.assertTrue(np.array_equal(relabelled, as_helmet))
        self.assertFalse(np.array_equal(relabelled, as_vest))

    def test_force_class_leaves_confidence_and_geometry_alone(self):
        import numpy as np

        box = self._box(cx=0.3, cy=0.4, w=0.15, h=0.25, name="vest", confidence=0.41)
        relabelled, _n = self._render({"force_class": "helmet"}, [box])
        expected, _n = self._render(
            {}, [{**box, "class_name": "helmet"}])
        self.assertTrue(np.array_equal(relabelled, expected))

    def test_force_class_does_not_change_which_boxes_are_drawn(self):
        """It is applied after the "what to show" knobs, so those still select on
        what the model actually returned."""
        boxes = [self._box(cx=0.25, name="helmet"), self._box(cx=0.75, name="vest")]
        frame, drawn = self._render(
            {"class_filter": ["helmet"], "force_class": "vest"}, boxes)
        self.assertEqual(drawn, 1)
        self.assertTrue(frame[:, :640].any())
        self.assertFalse(frame[:, 640:].any(), "the filter went by the relabel")

    def test_min_box_area_drops_specks(self):
        # 0.02 x 0.02 of the frame is 0.04% of its area.
        boxes = [self._box(w=0.02, h=0.02)]
        self.assertEqual(self._render({"min_box_area": 0.1}, boxes)[1], 0)
        self.assertEqual(self._render({"min_box_area": 0.01}, boxes)[1], 1)

    def test_max_boxes_keeps_the_most_confident(self):
        boxes = [self._box(cx=0.25, confidence=0.4, name="a"),
                 self._box(cx=0.75, confidence=0.95, name="b")]
        frame, drawn = self._render({"max_boxes": 1}, boxes)
        self.assertEqual(drawn, 1)
        self.assertTrue(frame[:, 640:].any())
        self.assertFalse(frame[:, :640].any())

    def test_spotlight_darkens_outside_the_box_and_leaves_it_alone(self):
        import numpy as np

        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
        render_style.MarketingRenderer(
            {"dim_background": 0.5, "box_style": "none", "fill_opacity": 0,
             "show_label": False}, still=True).draw(frame, [self._box()])
        self.assertEqual(int(frame[10, 10, 0]), 100)     # corner, dimmed
        self.assertEqual(int(frame[360, 640, 0]), 200)   # box centre, untouched

    def test_the_counter_and_watermark_land_in_the_corners_they_are_told_to(self):
        frame, _n = self._render(
            {"counter": "top_left", "watermark_text": "mercury",
             "watermark_position": "bottom_right", "box_style": "none",
             "fill_opacity": 0, "show_label": False},
            [self._box()])
        self.assertTrue(frame[:120, :400].any(), "no counter top-left")
        self.assertTrue(frame[600:, 900:].any(), "no watermark bottom-right")

    def test_sizes_scale_with_the_frame_in_both_directions(self):
        """Unlike ``draw_boxes``, which floors at 1080p: a marketing render is
        watched at its own size, so a 720p cut gets proportionally thinner lines
        and a 4K one thicker ones."""
        def stroke_width(height):
            width = round(height * 16 / 9)
            frame, _n = self._render(
                {"box_style": "solid", "corner_radius": 0, "thickness": 4,
                 "show_label": False, "fill_opacity": 0, "shadow": False},
                [self._box()], width=width, height=height)
            column = frame[:, width // 2, :]
            return int(column[int(0.55 * height):].any(axis=1).sum())

        self.assertGreater(stroke_width(2160), stroke_width(1080))
        self.assertGreater(stroke_width(1080), stroke_width(540))

    def test_still_mode_drops_the_temporal_treatments(self):
        # A single frame has no history, so a preview must not show every box
        # part-way through its fade-in.
        renderer = render_style.MarketingRenderer({"fade_frames": 8, "smoothing": 0.9},
                                                  still=True)
        self.assertEqual(renderer.style["fade_frames"], 0)
        self.assertEqual(renderer.style["smoothing"], 0.0)


class BoxTrackerTests(TestCase):
    """The easing/fading tracker — the part of the renderer with state, and the
    only part that can draw something the current frame's model output does not
    contain."""

    @staticmethod
    def _box(cx, name="helmet"):
        return {"cx": cx, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.9,
                "class_name": name}

    def test_a_new_box_fades_in_and_a_lost_one_fades_out(self):
        tracker = render_style._Tracker({"smoothing": 0.0, "fade_frames": 3})
        alphas = [tracker.update([self._box(0.5)])[0][1] for _ in range(4)]
        self.assertEqual(alphas, sorted(alphas))
        self.assertAlmostEqual(alphas[-1], 1.0)

        fading = []
        for _ in range(4):
            tracked = tracker.update([])
            fading.append(tracked[0][1] if tracked else None)
        self.assertEqual(fading[:3], sorted(fading[:3], reverse=True))
        self.assertIsNone(fading[3], "a lost box must not linger past fade_frames")

    def test_fade_frames_zero_draws_exactly_what_the_model_returned(self):
        tracker = render_style._Tracker({"smoothing": 0.0, "fade_frames": 0})
        self.assertEqual(tracker.update([self._box(0.5)])[0][1], 1.0)
        self.assertEqual(tracker.update([]), [])

    def test_smoothing_eases_toward_the_new_position_and_gets_there(self):
        tracker = render_style._Tracker({"smoothing": 0.5, "fade_frames": 0})
        tracker.update([self._box(0.30)])
        positions = [tracker.update([self._box(0.36)])[0][0]["cx"] for _ in range(8)]
        self.assertTrue(all(a < b for a, b in zip(positions, positions[1:])))
        self.assertLess(positions[0], 0.36)
        self.assertAlmostEqual(positions[-1], 0.36, places=2)

    def test_a_box_that_jumps_further_than_its_own_width_is_still_the_same_box(self):
        """Overlap alone loses a fast subject — or any subject once frame_stride
        puts several frames between updates — and a lost track means a box that
        fades out under its own replacement."""
        tracker = render_style._Tracker({"smoothing": 0.0, "fade_frames": 0})
        tracker.update([self._box(0.30)])            # box is 0.2 wide
        tracked = tracker.update([self._box(0.42)])  # moved 0.12: no overlap at all
        self.assertEqual(len(tracked), 1, "the jump started a second track")

    def test_a_box_on_the_other_side_of_the_frame_is_a_different_box(self):
        tracker = render_style._Tracker({"smoothing": 0.0, "fade_frames": 2})
        tracker.update([self._box(0.15)])
        tracked = tracker.update([self._box(0.85)])
        self.assertEqual(len(tracked), 2, "unrelated boxes were matched")

    def test_boxes_of_different_classes_are_never_matched_to_each_other(self):
        tracker = render_style._Tracker({"smoothing": 0.9, "fade_frames": 0})
        tracker.update([self._box(0.5, name="helmet")])
        tracked = tracker.update([self._box(0.5, name="vest")])
        # The vest is a new track at its true position, not the helmet eased.
        vest = [b for b, _a in tracked if b["class_name"] == "vest"][0]
        self.assertEqual(vest["cx"], 0.5)

    def test_a_relabel_does_not_let_two_classes_share_a_track(self):
        """Every box on the frame reads as one class under ``force_class``; the
        tracker still has to match on what the model said, or a helmet's track
        would ease onto the vest that replaced it."""
        renderer = render_style.MarketingRenderer({"force_class": "subject"})
        helmet = renderer._select([self._box(0.30, name="helmet")])
        vest = renderer._select([self._box(0.36, name="vest")])
        self.assertEqual([b["class_name"] for b in helmet + vest],
                         ["subject", "subject"])

        tracker = render_style._Tracker({"smoothing": 0.9, "fade_frames": 0})
        tracker.update(helmet)
        tracked = tracker.update(vest)
        self.assertEqual(len(tracked), 1)
        # Drawn where the vest is, not eased 10% of the way there from the helmet.
        self.assertAlmostEqual(tracked[0][0]["cx"], 0.36)

    def test_two_boxes_of_one_class_keep_their_own_identities(self):
        tracker = render_style._Tracker({"smoothing": 0.5, "fade_frames": 0})
        tracker.update([self._box(0.2), self._box(0.8)])
        tracked = tracker.update([self._box(0.25), self._box(0.85)])
        eased = sorted(b["cx"] for b, _a in tracked)
        self.assertEqual(len(eased), 2)
        # Each matched its own neighbour rather than crossing over.
        self.assertLess(eased[0], 0.5)
        self.assertGreater(eased[1], 0.5)


class RunInferenceMarketingActionTests(VideosBase):
    """"Run model inference for marketing…" — the same wizard as the plain action
    plus a Look section, which lands on the job as ``render_style``."""

    def setUp(self):
        super().setUp()
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

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
            "action": "run_inference_marketing",
            ACTION_CHECKBOX_NAME: [str(self.video.pk)],
            **extra,
        }
        return self.client.post(self.url, data, follow=True)

    def _apply(self, **style):
        return self._post(
            apply="1", model_source=InferenceJob.TRAINED,
            trained_model=str(self.tm.pk), pipeline="raw",
            score_threshold="0.5", frame_stride="1", **style)

    def test_the_action_is_offered_on_the_changelist(self):
        resp = self.client.get(self.url)
        self.assertContains(resp, "run_inference_marketing")
        self.assertContains(resp, "Run model inference for marketing")

    def test_step_one_asks_for_the_model_type_like_the_plain_action(self):
        resp = self._post()
        self.assertContains(resp, 'name="model_source"')
        self.assertNotContains(resp, 'name="style_box_style"')

    def test_step_two_renders_the_look_section_and_the_preview(self):
        resp = self._post(configure="1", model_source=InferenceJob.TRAINED)
        self.assertContains(resp, 'name="style_box_style"')
        self.assertContains(resp, 'name="style_palette"')
        self.assertContains(resp, 'name="style_watermark_text"')
        self.assertContains(resp, 'name="style_smoothing"')
        self.assertContains(resp, 'name="style_crf"')
        # …on top of the shared model/pipeline half, not instead of it.
        self.assertContains(resp, 'name="trained_model"')
        self.assertContains(resp, 'name="pipeline"')
        self.assertContains(resp, reverse("admin:videos_video_marketing_preview"))

    def test_step_two_renders_for_every_model_source(self):
        """The Look section is bolted onto the shared model/pipeline include, so
        each of the three sources has to still render through it."""
        for source, marker in ((InferenceJob.TRAINED, 'name="trained_model"'),
                               (InferenceJob.EXPORTED, 'name="artifact_path"'),
                               (InferenceJob.BUNDLE, 'name="bundle_path"')):
            with self.subTest(source=source):
                resp = self._post(configure="1", model_source=source)
                self.assertContains(resp, marker)
                self.assertContains(resp, 'name="style_box_style"')

    @mock.patch("videos.admin._queue")
    def test_apply_stores_the_style_on_the_job(self, queue):
        self._apply(style_box_style="corners", style_palette="neon",
                    style_thickness="7", style_watermark_text="Mercury",
                    style_has_shadow="1")
        job = InferenceJob.objects.get()
        self.assertEqual(job.pipeline, "raw")
        self.assertEqual(job.render_style["box_style"], "corners")
        self.assertEqual(job.render_style["palette"], "neon")
        self.assertEqual(job.render_style["thickness"], 7)
        self.assertEqual(job.render_style["watermark_text"], "Mercury")
        self.assertFalse(job.render_style["shadow"])  # rendered, left unticked
        self.assertTrue(job.is_marketing())
        queue.return_value.enqueue.assert_called_once()

    @mock.patch("videos.admin._queue")
    def test_a_bad_style_field_is_reported_and_nothing_is_queued(self, queue):
        resp = self._apply(style_thickness="9000")
        self.assertContains(resp, "Thickness")
        self.assertFalse(InferenceJob.objects.exists())
        queue.return_value.enqueue.assert_not_called()

    @mock.patch("videos.admin._queue")
    def test_a_rejected_submit_rerenders_with_what_was_typed(self, queue):
        resp = self._apply(style_thickness="9000", style_watermark_text="Keep me")
        self.assertContains(resp, "Keep me")

    @mock.patch("videos.admin._queue")
    def test_the_plain_action_still_stores_no_style(self, queue):
        self.client.post(self.url, {
            "action": "run_inference",
            ACTION_CHECKBOX_NAME: [str(self.video.pk)],
            "apply": "1", "model_source": InferenceJob.TRAINED,
            "trained_model": str(self.tm.pk), "pipeline": "raw",
            "score_threshold": "0.5", "frame_stride": "1",
            # Style fields posted by a stale/pasted form must not leak into the
            # plain action's job — the two actions differ in exactly this field.
            "style_box_style": "corners",
        }, follow=True)
        job = InferenceJob.objects.get()
        self.assertEqual(job.render_style, {})
        self.assertFalse(job.is_marketing())

    def test_the_relabel_dropdown_offers_the_selected_models_classes(self):
        self.tm.classes = ["helmet", "vest"]
        self.tm.save()
        resp = self._post(configure="1", model_source=InferenceJob.TRAINED)
        self.assertEqual(resp.context["class_choices"], ["helmet", "vest"])
        # …and every model's classes reach the page, for the dropdown to refill
        # itself from when a different model is picked.
        self.assertEqual(resp.context["classes_by_model"],
                         {str(self.tm.pk): ["helmet", "vest"]})
        self.assertContains(resp, 'name="style_force_class"')

    @mock.patch("videos.admin._queue")
    def test_a_relabel_the_model_has_no_class_for_stays_in_the_dropdown(self, queue):
        """Carried over from the last render, or loaded from a preset: it is only
        ever drawn, so it must not vanish the moment the form re-renders."""
        self.tm.classes = ["helmet"]
        self.tm.save()
        resp = self._apply(style_force_class="violation")
        resp = self._post(configure="1", model_source=InferenceJob.TRAINED)
        self.assertEqual(resp.context["class_choices"], ["helmet", "violation"])

    @mock.patch("videos.admin._queue")
    def test_apply_stores_the_relabel_on_the_job(self, queue):
        self._apply(style_force_class="helmet")
        self.assertEqual(InferenceJob.objects.get().render_style["force_class"],
                         "helmet")

    @mock.patch("videos.admin._queue")
    def test_the_form_arrives_carrying_the_last_marketing_look(self, queue):
        self._apply(style_palette="sunset", style_watermark_text="House style")
        resp = self._post(configure="1", model_source=InferenceJob.TRAINED)
        self.assertEqual(resp.context["style_values"]["style_palette"], "sunset")
        self.assertContains(resp, "House style")


class MarketingPreviewViewTests(VideosBase):
    """The preview endpoint. The model is mocked out — what is under test is the
    plumbing (auth, validation, the caching, the data URI), not the detector."""

    def setUp(self):
        super().setUp()
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")
        self.video = Video.objects.create(
            name="clip", filename="clip.mp4", status=Video.READY)
        (self.root / "clip.mp4").write_bytes(b"x")
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"x")
        self.tm = TrainedModel.objects.create(
            name="m", arch="yolox", checkpoint_path=str(self.checkpoint))
        self.url = reverse("admin:videos_video_marketing_preview")

    def _payload(self, **extra):
        return {
            "video": str(self.video.pk), "model_source": InferenceJob.TRAINED,
            "trained_model": str(self.tm.pk), "pipeline": "raw",
            "score_threshold": "0.5", "frame_stride": "1", "position": "0.5",
            **extra,
        }

    def test_get_is_refused(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    def test_an_anonymous_visitor_is_bounced_to_the_login(self):
        self.client.logout()
        resp = self.client.post(self.url, self._payload())
        self.assertEqual(resp.status_code, 302)
        self.assertIn("login", resp["Location"])

    def test_an_unknown_video_is_a_404_not_a_crash(self):
        resp = self.client.post(self.url, self._payload(video="99999"))
        self.assertEqual(resp.status_code, 404)

    def test_a_bad_style_is_reported_as_json_not_a_500(self):
        resp = self.client.post(self.url, self._payload(style_thickness="9000"))
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Thickness", resp.json()["error"])

    @mock.patch("videos.services.inference.preview_frame")
    def test_a_rendered_frame_comes_back_as_a_data_uri(self, preview):
        preview.return_value = {
            "jpeg": b"\xff\xd8jpeg", "width": 640, "height": 360, "boxes": 2,
            "detections": 3, "frame_index": 12, "frames_total": 100, "cached": True,
        }
        resp = self.client.post(self.url, self._payload(style_box_style="corners"))
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["image"].startswith("data:image/jpeg;base64,"))
        self.assertEqual((body["boxes"], body["detections"], body["cached"]),
                         (2, 3, True))
        # The style the operator is looking at is the style that reaches the
        # renderer — that is the entire promise of the preview.
        job = preview.call_args.args[0]
        self.assertEqual(job.render_style["box_style"], "corners")
        self.assertIsNone(job.pk, "the preview must not save a job row")

    @mock.patch("videos.services.inference.preview_frame",
                side_effect=RuntimeError("trainer is down"))
    def test_a_failing_render_is_reported_in_the_panel(self, _preview):
        resp = self.client.post(self.url, self._payload())
        self.assertEqual(resp.status_code, 500)
        self.assertIn("trainer is down", resp.json()["error"])


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg to build a test clip")
class MarketingRenderEndToEndTests(VideosBase):
    """The two halves that only a real video file can exercise: the preview's
    frame-grab + box cache, and the render loop's styled output.

    The model is mocked (the trainer is a separate container and holds the GPU);
    everything on this side of that call is real — decode, draw, encode.
    """

    BOXES = [{"cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.4, "confidence": 0.9,
              "class_id": 0, "class_name": "helmet"}]

    def setUp(self):
        super().setUp()
        self.clip = self.root / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc=size=640x360:rate=10:duration=1",
             "-pix_fmt", "yuv420p", str(self.clip)],
            check=True, capture_output=True,
        )
        self.video = Video.objects.create(
            name="clip", filename="clip.mp4", status=Video.READY)
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"x")
        self.tm = TrainedModel.objects.create(
            name="m", arch="yolox", checkpoint_path=str(self.checkpoint))

    def _job(self, **kwargs):
        kwargs.setdefault("output_filename", inference.unique_output_filename("clip"))
        return InferenceJob(
            video=self.video, model_source=InferenceJob.TRAINED,
            trained_model=self.tm, pipeline="raw", **kwargs)

    @mock.patch("training.services.runner.predict_image")
    def test_a_styled_job_renders_at_the_requested_delivery_height(self, predict):
        import cv2

        predict.return_value = {"boxes": self.BOXES}
        job = self._job(render_style={"output_height": 180, "box_style": "corners",
                                      "counter": "top_left", "crf": 30})
        result = inference.run_inference_on_video(job)

        self.assertEqual(result["frames_total"], 10)
        output = inference.output_dir() / job.output_filename
        self.assertTrue(output.is_file())
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 180)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 320)
        finally:
            capture.release()

    @mock.patch("training.services.runner.predict_image")
    def test_an_unstyled_job_keeps_the_source_size(self, predict):
        import cv2

        predict.return_value = {"boxes": self.BOXES}
        job = self._job()
        inference.run_inference_on_video(job)
        capture = cv2.VideoCapture(str(inference.output_dir() / job.output_filename))
        try:
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 360)
        finally:
            capture.release()

    @mock.patch("training.services.runner.predict_image")
    def test_the_renderer_runs_on_every_frame_not_only_inferred_ones(self, predict):
        """With a stride the model is asked once and the easing carries the box
        across the frames in between — which only happens if drawing is outside
        the ``if inferred`` branch."""
        predict.return_value = {"boxes": self.BOXES}
        job = self._job(frame_stride=5, render_style={"smoothing": 0.6})
        result = inference.run_inference_on_video(job)
        self.assertEqual(result["frames_total"], 10)
        self.assertEqual(result["frames_processed"], 2)

    @mock.patch("training.services.runner.predict_image")
    def test_preview_renders_a_frame_and_reuses_the_detections(self, predict):
        predict.return_value = {"boxes": self.BOXES}
        job = self._job(render_style={"box_style": "rounded"})

        first = inference.preview_frame(job, 0.5)
        self.assertTrue(first["jpeg"].startswith(b"\xff\xd8"))
        self.assertEqual((first["boxes"], first["detections"]), (1, 1))
        self.assertFalse(first["cached"])
        self.assertEqual(predict.call_count, 1)

        # A style change must not cost another inference on the same frame…
        job.render_style = {"box_style": "corners", "palette": "neon"}
        second = inference.preview_frame(job, 0.5)
        self.assertTrue(second["cached"])
        self.assertEqual(predict.call_count, 1)
        self.assertNotEqual(first["jpeg"], second["jpeg"], "style change not drawn")

        # …but asking for it explicitly must.
        inference.preview_frame(job, 0.5, use_cache=False)
        self.assertEqual(predict.call_count, 2)

    @mock.patch("training.services.runner.predict_image")
    def test_preview_positions_land_on_different_frames(self, predict):
        predict.return_value = {"boxes": []}
        job = self._job()
        self.assertEqual(inference.preview_frame(job, 0.0)["frame_index"], 0)
        self.assertGreater(inference.preview_frame(job, 1.0)["frame_index"], 0)

    @mock.patch("training.services.runner.predict_image")
    def test_preview_caps_a_large_frame_at_1080p(self, predict):
        predict.return_value = {"boxes": []}
        job = self._job()
        with mock.patch.object(inference, "_PREVIEW_MAX_HEIGHT", 180):
            result = inference.preview_frame(job, 0.5)
        self.assertEqual((result["width"], result["height"]), (320, 180))


class InferredVideosListTests(VideosBase):
    """The Inferred videos changelist — the page an operator lands on straight
    after queuing, so both kinds of row have to render there."""

    def setUp(self):
        super().setUp()
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")
        self.video = Video.objects.create(
            name="clip", filename="clip.mp4", status=Video.READY)

    def test_both_a_plain_and_a_marketing_row_render(self):
        InferenceJob.objects.create(video=self.video, pipeline="raw")
        InferenceJob.objects.create(
            video=self.video, pipeline="raw",
            render_style=render_style.normalize({"box_style": "corners",
                                                 "palette": "neon"}))
        resp = self.client.get(reverse("admin:videos_inferencejob_changelist"))
        self.assertContains(resp, "plain")
        self.assertContains(resp, "corners · neon")


class OutlineStyleTests(TestCase):
    """The outline vocabulary. Every entry in ``BOX_STYLE_CHOICES`` has to draw
    something and stay inside the frame — including the ones that fan well past
    their nominal line width, which is where the drawing ROI can clip them."""

    @staticmethod
    def _box(**over):
        return {"cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.4, "confidence": 0.9,
                "class_id": 0, "class_name": "helmet", **over}

    def _draw(self, style, box=None, size=(1280, 720)):
        import numpy as np

        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        renderer = render_style.MarketingRenderer(
            {"show_label": False, "fill_opacity": 0, **style}, still=True)
        renderer.draw(frame, [box or self._box()])
        return frame

    def test_every_outline_draws_and_none_draws_nothing(self):
        for value, _label in render_style.BOX_STYLE_CHOICES:
            with self.subTest(box_style=value):
                frame = self._draw({"box_style": value})
                self.assertEqual(bool(frame.any()), value != "none")

    def test_every_outline_survives_a_box_clipped_by_the_frame_edge(self):
        for value, _label in render_style.BOX_STYLE_CHOICES:
            with self.subTest(box_style=value):
                # No raise, and nothing drawn outside the array.
                self._draw({"box_style": value},
                           box=self._box(cx=0.02, cy=0.02, w=0.2, h=0.2))

    def test_a_glow_reaches_further_out_than_its_line(self):
        """The regression this guards: the drawing ROI was sized from the line
        thickness, which cropped the bloom into a square."""
        import numpy as np

        narrow = self._draw({"box_style": "glow", "glow_spread": 1, "thickness": 2})
        wide = self._draw({"box_style": "glow", "glow_spread": 20, "thickness": 2})
        self.assertGreater(int(np.count_nonzero(wide)), int(np.count_nonzero(narrow)))

    def test_the_sketch_wobble_is_the_same_wobble_every_frame(self):
        # A jitter reseeded per frame boils like video noise; it has to be fixed
        # per class instead.
        first = self._draw({"box_style": "sketch"})
        second = self._draw({"box_style": "sketch"})
        self.assertTrue((first == second).all())

    def test_the_sketch_wobble_differs_between_classes(self):
        a = self._draw({"box_style": "sketch"}, box=self._box(class_name="helmet"))
        b = self._draw({"box_style": "sketch"}, box=self._box(class_name="vest"))
        self.assertFalse((a == b).all())

    def test_underline_draws_only_along_the_bottom(self):
        frame = self._draw({"box_style": "underline"})
        rows = frame.any(axis=2).nonzero()[0]
        # The box spans 0.3–0.7 of a 720-high frame; the bar and its two short
        # lifts must all sit near the bottom edge of it.
        self.assertGreater(int(rows.min()), int(0.45 * 720))


class OutlineAnimationTests(TestCase):
    """``animation`` is the only thing on the frame that changes when nothing
    else did, so it needs the frame counter to actually advance."""

    BOX = {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4, "confidence": 0.9,
           "class_name": "helmet"}

    @staticmethod
    def _blank():
        import numpy as np

        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def _frames(self, style, count=4):
        # fade/smoothing off: they also change the frame from one call to the
        # next, and would let these assertions pass without any animation at all.
        renderer = render_style.MarketingRenderer(
            {"show_label": False, "fill_opacity": 0, "shadow": False,
             "fade_frames": 0, "smoothing": 0, **style})
        shots = []
        for _ in range(count):
            frame = self._blank()
            renderer.draw(frame, [self.BOX])
            shots.append(frame)
        return shots

    def test_marching_dashes_move_between_frames(self):
        shots = self._frames({"box_style": "dashed", "animation": "march",
                              "animation_period": 4})
        self.assertFalse((shots[0] == shots[1]).all())

    def test_marching_returns_to_where_it_started_after_one_period(self):
        shots = self._frames({"box_style": "dashed", "animation": "march",
                              "animation_period": 4}, count=5)
        self.assertTrue((shots[0] == shots[4]).all())

    def test_pulse_changes_the_line_weight(self):
        import numpy as np

        shots = self._frames({"box_style": "solid", "animation": "pulse",
                              "animation_period": 8, "thickness": 6})
        weights = {int(np.count_nonzero(shot)) for shot in shots}
        self.assertGreater(len(weights), 1)

    def test_a_still_never_animates(self):
        # A preview must show the outline at rest rather than at whatever phase
        # the counter happened to reach.
        renderer = render_style.MarketingRenderer(
            {"box_style": "dashed", "animation": "march"}, still=True)
        self.assertEqual(renderer.style["animation"], "none")
        first, second = self._blank(), self._blank()
        renderer.draw(first, [self.BOX])
        renderer.draw(second, [self.BOX])
        self.assertTrue((first == second).all())


class RenderPresetTests(VideosBase):
    """Saving a look by name, and offering it back on the next clip."""

    def setUp(self):
        super().setUp()
        User = get_user_model()
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")
        (self.root / "clip.mp4").write_bytes(b"x")
        self.video = Video.objects.create(
            name="clip", filename="clip.mp4", status=Video.READY)
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"x")
        self.tm = TrainedModel.objects.create(
            name="m", arch="yolox", checkpoint_path=str(self.checkpoint))
        self.url = reverse("admin:videos_video_render_preset_save")

    def _save(self, **extra):
        return self.client.post(self.url, {
            "preset_name": "Deck look", "model_source": InferenceJob.TRAINED,
            "trained_model": str(self.tm.pk), "pipeline": "raw",
            "style_box_style": "glow", "style_palette": "neon", **extra,
        })

    def test_get_is_refused(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    def test_an_anonymous_visitor_is_bounced_to_the_login(self):
        self.client.logout()
        resp = self._save()
        self.assertEqual(resp.status_code, 302)
        self.assertIn("login", resp["Location"])

    def test_saving_stores_the_whole_look(self):
        resp = self._save(style_thickness="9", style_watermark_text="Mercury")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["created"])

        preset = RenderPreset.objects.get()
        self.assertEqual(preset.name, "Deck look")
        self.assertEqual(preset.style["box_style"], "glow")
        self.assertEqual(preset.style["palette"], "neon")
        self.assertEqual(preset.style["thickness"], 9)
        self.assertEqual(preset.style["watermark_text"], "Mercury")
        # …and the look only. A preset that carried the pipeline would quietly
        # override the model's own recorded metadata.
        self.assertNotIn("pipeline", preset.style)
        self.assertNotIn("trained_model", preset.style)

    def test_saving_the_same_name_again_updates_rather_than_duplicates(self):
        self._save(style_thickness="4")
        resp = self._save(style_thickness="11")
        self.assertFalse(resp.json()["created"])
        self.assertEqual(RenderPreset.objects.count(), 1)
        self.assertEqual(RenderPreset.objects.get().style["thickness"], 11)

    def test_a_nameless_preset_is_refused(self):
        resp = self._save(preset_name="   ")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Name", resp.json()["error"])
        self.assertFalse(RenderPreset.objects.exists())

    def test_a_bad_style_is_refused_rather_than_stored(self):
        resp = self._save(style_thickness="9000")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Thickness", resp.json()["error"])
        self.assertFalse(RenderPreset.objects.exists())

    def test_the_reply_is_in_the_forms_own_shape(self):
        # The page applies it by assigning each value to the input of that name,
        # so the keys have to be the form's, not the style dict's.
        values = self._save().json()["values"]
        self.assertEqual(values["style_box_style"], "glow")
        self.assertIn("style_class_colors", values)

    def test_the_marketing_form_offers_every_saved_preset(self):
        RenderPreset.objects.create(
            name="House", style=render_style.normalize({"box_style": "chamfer"}))
        resp = self.client.post(reverse("admin:videos_video_changelist"), {
            "action": "run_inference_marketing",
            ACTION_CHECKBOX_NAME: [str(self.video.pk)],
            "configure": "1", "model_source": InferenceJob.TRAINED,
        }, follow=True)
        self.assertContains(resp, "House")
        self.assertContains(resp, 'id="mk-preset-data"')
        self.assertEqual(resp.context["presets"][0]["values"]["style_box_style"],
                         "chamfer")

    def test_a_preset_saved_before_a_knob_existed_still_loads(self):
        # Presets are normalized on the way out, so an older or hand-edited row
        # fills in whatever it is missing rather than breaking the form.
        RenderPreset.objects.create(name="Ancient", style={"box_style": "dotted"})
        preset = RenderPreset.objects.get(name="Ancient")
        values = render_style.form_values(render_style.normalize(preset.style))
        self.assertEqual(values["style_box_style"], "dotted")
        self.assertEqual(values["style_crf"], render_style.defaults()["crf"])
        self.assertEqual(preset.summary(), "dotted · vivid")
