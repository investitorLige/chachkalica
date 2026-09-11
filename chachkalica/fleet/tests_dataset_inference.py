"""Tests for the Datasets tab's "Run model inference…" action and its runs tab.

The trainer is always mocked: what is under test is the *measurement* — which
clock a number came from, what is excluded from it, and what the form records —
not whether a GPU is fast. ``BASE_DIR`` is overridden per test class so a
dataset written into a tempdir still counts as inside the shared data mount (the
one thing the action refuses outright, since images reach the trainer as paths).
"""

import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from fleet import jobs
from fleet.models import (
    Dataset, DatasetInferenceResult, DatasetInferenceRun, FleetSettings,
)
from fleet.services import dataset_inference
from training.models import TrainedModel, TrainingSettings
from training.services import inference_form, pipeline_meta


def _box(cx=0.5, cy=0.5, name="helmet", conf=0.9):
    return {"cx": cx, "cy": cy, "w": 0.2, "h": 0.2,
            "confidence": conf, "class_id": 0, "class_name": name}


class _RunTestCase(TestCase):
    """A dataset of real (tiny) JPEGs under a tempdir that passes as the data mount."""

    IMAGE_COUNT = 4

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.src = self.base / "data" / "source"
        self.src.mkdir(parents=True)

        self.enter_context = override_settings(BASE_DIR=str(self.base))
        self.enter_context.enable()
        self.addCleanup(self.enter_context.disable)

        fs = FleetSettings.load()
        fs.source_dir = str(self.src)  # absolute -> used verbatim by source_root()
        fs.save()

        self.dataset_dir = self.src / "ds"
        self.dataset_dir.mkdir()
        (self.dataset_dir / "classes.txt").write_text("helmet\nvest\n", encoding="utf-8")
        for index in range(self.IMAGE_COUNT):
            cv2.imwrite(str(self.dataset_dir / f"img{index}.jpg"),
                        np.full((32, 48, 3), 100 + index, dtype=np.uint8))
        self.dataset = Dataset.objects.create(name="ds")

        self.checkpoint = self.base / "model.pt"
        self.checkpoint.write_bytes(b"not-really-a-checkpoint")
        self.model = TrainedModel.objects.create(
            name="m", arch="yolox", checkpoint_path=str(self.checkpoint))

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **kwargs):
        defaults = {
            "dataset": self.dataset,
            "dataset_name_snapshot": "ds",
            "model_source": DatasetInferenceRun.TRAINED,
            "trained_model": self.model,
            "model_label_snapshot": "m",
            "warmup": 0,
            "image_limit": 0,
        }
        return DatasetInferenceRun.objects.create(**{**defaults, **kwargs})

    def _login_admin(self):
        User = get_user_model()
        User.objects.create_superuser(
            username="admin", email="admin@example.com", password="pw")
        self.client.login(username="admin", password="pw")


class SelectImagesTests(_RunTestCase):
    IMAGE_COUNT = 10

    def test_no_limit_takes_every_image_in_order(self):
        picked = dataset_inference.select_images(self.dataset_dir, 0)
        self.assertEqual([p.name for p in picked],
                         [f"img{i}.jpg" for i in range(10)])

    def test_limit_spans_the_dataset_rather_than_its_first_files(self):
        picked = dataset_inference.select_images(self.dataset_dir, 4)
        # Evenly spaced, not img0..img3 — a run capped at 4 of 10 images must
        # still see the whole dataset.
        self.assertEqual([p.name for p in picked],
                         ["img0.jpg", "img2.jpg", "img5.jpg", "img7.jpg"])

    def test_limit_above_the_dataset_size_is_every_image(self):
        self.assertEqual(len(dataset_inference.select_images(self.dataset_dir, 99)), 10)


class TrainerVisibilityTests(_RunTestCase):
    def test_dataset_under_the_data_mount_is_visible(self):
        self.assertTrue(dataset_inference.visible_to_trainer(self.dataset_dir))

    def test_dataset_outside_the_data_mount_is_not(self):
        outside = self.base / "elsewhere"
        outside.mkdir()
        self.assertFalse(dataset_inference.visible_to_trainer(outside))


class SummarizeTests(_RunTestCase):
    def _result(self, seq, **kwargs):
        defaults = {"image_filename": f"img{seq}.jpg", "detections": 1}
        return DatasetInferenceResult.objects.create(
            run=self.run, seq=seq, **{**defaults, **kwargs})

    def setUp(self):
        super().setUp()
        self.run = self._run()

    def test_fps_comes_from_the_model_clock_when_the_trainer_reports_one(self):
        self._result(1, model_ms=10.0, request_ms=30.0)
        self._result(2, model_ms=20.0, request_ms=40.0)

        timings = dataset_inference.summarize(self.run)

        self.assertEqual(timings["timing_source"], "model")
        # 1000 / mean(10, 20) — the model's own time, not the round trip's.
        self.assertAlmostEqual(timings["fps"], 1000 / 15.0)
        self.assertAlmostEqual(timings["request_fps"], 1000 / 35.0)
        self.run.refresh_from_db()
        self.assertAlmostEqual(self.run.fps, 1000 / 15.0)
        self.assertAlmostEqual(self.run.p50_ms, 15.0)

    def test_falls_back_to_the_round_trip_and_says_so(self):
        self._result(1, model_ms=None, request_ms=50.0)

        timings = dataset_inference.summarize(self.run)

        self.assertEqual(timings["timing_source"], "request")
        self.assertAlmostEqual(timings["fps"], 20.0)
        self.assertEqual(timings["model_ms"], {})

    def test_failed_images_are_counted_but_never_timed(self):
        self._result(1, model_ms=10.0, request_ms=10.0)
        self._result(2, model_ms=None, request_ms=180000.0, detections=0,
                     error="trainer /predict_image returned HTTP 500")
        self.run.errors_count = 1
        self.run.save(update_fields=["errors_count"])

        timings = dataset_inference.summarize(self.run)

        # The timeout must not land in the distribution: it is a gap in the
        # measurement, not a slow frame.
        self.assertEqual(timings["images_timed"], 1)
        self.assertEqual(timings["images_failed"], 1)
        self.assertAlmostEqual(timings["model_ms"]["max"], 10.0)

    def test_stage_shares_sum_to_one(self):
        self._result(1, model_ms=10.0, request_ms=12.0,
                     stages={"load_ms": 6.0, "infer_ms": 3.0, "format_ms": 1.0})

        timings = dataset_inference.summarize(self.run)

        self.assertAlmostEqual(sum(timings["stage_share"].values()), 1.0)
        self.assertAlmostEqual(timings["stage_share"]["load_ms"], 0.6)

    def test_detections_are_totalled_onto_the_run(self):
        self._result(1, model_ms=1.0, detections=3)
        self._result(2, model_ms=1.0, detections=2)

        dataset_inference.summarize(self.run)

        self.run.refresh_from_db()
        self.assertEqual(self.run.detections_total, 5)


class RunLoopTests(_RunTestCase):
    def _reply(self, boxes=(), total_ms=12.0):
        return {"boxes": list(boxes), "classes": {},
                "timings": {"total_ms": total_ms, "load_ms": 4.0, "infer_ms": 8.0}}

    def test_every_image_gets_a_row_with_both_clocks(self):
        run = self._run()
        with mock.patch("training.services.runner.predict_image",
                        return_value=self._reply([_box()])) as predict:
            result = dataset_inference.run_model_on_dataset(run)

        self.assertEqual(predict.call_count, self.IMAGE_COUNT)
        self.assertEqual(result["images_processed"], self.IMAGE_COUNT)
        rows = list(run.results.order_by("seq"))
        self.assertEqual([r.image_filename for r in rows],
                         [f"img{i}.jpg" for i in range(self.IMAGE_COUNT)])
        self.assertEqual([r.detections for r in rows], [1] * self.IMAGE_COUNT)
        self.assertEqual([r.model_ms for r in rows], [12.0] * self.IMAGE_COUNT)
        self.assertTrue(all(r.request_ms is not None for r in rows))
        # The stage split is stored per image, with the total left out of it.
        self.assertEqual(rows[0].stages, {"load_ms": 4.0, "infer_ms": 8.0})
        run.refresh_from_db()
        self.assertAlmostEqual(run.fps, 1000 / 12.0)

    def test_stored_boxes_are_the_models_own_output(self):
        run = self._run()
        with mock.patch("training.services.runner.predict_image",
                        return_value=self._reply([_box(name="vest")])):
            dataset_inference.run_model_on_dataset(run)

        self.assertEqual(run.results.first().boxes[0]["class_name"], "vest")

    def test_warmup_calls_run_but_are_not_measured(self):
        run = self._run(warmup=2)
        with mock.patch("training.services.runner.predict_image",
                        return_value=self._reply()) as predict:
            dataset_inference.run_model_on_dataset(run)

        # Two extra calls on the first image, and no rows for them: a model load
        # is not a frame time.
        self.assertEqual(predict.call_count, self.IMAGE_COUNT + 2)
        self.assertEqual(run.results.count(), self.IMAGE_COUNT)

    def test_the_trainer_is_asked_for_a_stage_split(self):
        run = self._run()
        with mock.patch("training.services.runner.predict_image",
                        return_value=self._reply()) as predict:
            dataset_inference.run_model_on_dataset(run)

        self.assertTrue(predict.call_args.args[0]["stage_timings"])

    def test_a_trainer_without_timings_falls_back_to_the_round_trip(self):
        run = self._run()
        with mock.patch("training.services.runner.predict_image",
                        return_value={"boxes": [], "classes": {}}):
            dataset_inference.run_model_on_dataset(run)

        run.refresh_from_db()
        self.assertEqual(run.timings["timing_source"], "request")
        self.assertIsNone(run.results.first().model_ms)
        self.assertIsNotNone(run.results.first().request_ms)

    def test_one_bad_image_is_recorded_and_the_run_carries_on(self):
        run = self._run()
        replies = [RuntimeError("boom"), self._reply(), self._reply(), self._reply()]
        with mock.patch("training.services.runner.predict_image",
                        side_effect=replies):
            result = dataset_inference.run_model_on_dataset(run)

        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["images_processed"], self.IMAGE_COUNT)
        self.assertEqual(run.results.get(seq=1).error, "boom")

    def test_a_wedged_trainer_aborts_rather_than_writing_a_row_per_image(self):
        run = self._run()
        with mock.patch("training.services.runner.predict_image",
                        side_effect=RuntimeError("trainer down")), \
             mock.patch.object(dataset_inference, "MAX_CONSECUTIVE_ERRORS", 2):
            with self.assertRaises(RuntimeError):
                dataset_inference.run_model_on_dataset(run)

        self.assertEqual(run.results.count(), 2)

    def test_stop_ends_the_run_after_the_current_image(self):
        run = self._run()

        def stop_after_first(*args, **kwargs):
            DatasetInferenceRun.objects.filter(pk=run.pk).update(
                status=DatasetInferenceRun.CANCEL_REQUESTED)
            return self._reply()

        with mock.patch("training.services.runner.predict_image",
                        side_effect=stop_after_first):
            result = dataset_inference.run_model_on_dataset(run)

        self.assertTrue(result["cancelled"])
        self.assertEqual(run.results.count(), 1)
        # Cancelled or not, what was measured is summarized — those images are
        # already paid for.
        run.refresh_from_db()
        self.assertIsNotNone(run.fps)

    def test_an_unreadable_model_fails_the_run_before_any_image(self):
        self.checkpoint.unlink()
        run = self._run()
        with mock.patch("training.services.runner.predict_image") as predict:
            with self.assertRaises(RuntimeError):
                dataset_inference.run_model_on_dataset(run)
        predict.assert_not_called()

    def test_job_records_cancelled_rather_than_error(self):
        run = self._run()

        def stop_immediately(*args, **kwargs):
            DatasetInferenceRun.objects.filter(pk=run.pk).update(
                status=DatasetInferenceRun.CANCEL_REQUESTED)
            return self._reply()

        with mock.patch("training.services.runner.predict_image",
                        side_effect=stop_immediately):
            jobs.run_dataset_inference(run.pk)

        run.refresh_from_db()
        self.assertEqual(run.status, DatasetInferenceRun.CANCELLED)
        self.assertTrue(run.finished_at)

    def test_job_records_the_error_and_reraises(self):
        run = self._run()
        with mock.patch("training.services.runner.predict_image",
                        side_effect=RuntimeError("trainer down")), \
             mock.patch.object(dataset_inference, "MAX_CONSECUTIVE_ERRORS", 1):
            with self.assertRaises(RuntimeError):
                jobs.run_dataset_inference(run.pk)

        run.refresh_from_db()
        self.assertEqual(run.status, DatasetInferenceRun.ERROR)
        self.assertIn("trainer down", run.last_error)


class ActionTests(_RunTestCase):
    def setUp(self):
        super().setUp()
        self._login_admin()
        self.url = reverse("admin:fleet_dataset_changelist")
        self.tiled = TrainedModel.objects.create(
            name="tiled", arch="yolox", checkpoint_path=str(self.checkpoint),
            pipeline_metadata=pipeline_meta.normalize({
                "pipeline": "batch_detect",
                "tile_size_px": 640,
                "overlap": 0.25,
                "merge_nms_iou": 0.5,
                "score_threshold": 0.3,
            }),
        )

    def _post(self, **extra):
        return self.client.post(self.url, {
            "action": "run_model_inference",
            ACTION_CHECKBOX_NAME: [str(self.dataset.pk)],
            **extra,
        }, follow=True)

    def test_step_one_offers_the_three_model_sources(self):
        response = self._post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["model_source_choices"],
                         inference_form.MODEL_SOURCE_CHOICES)

    def test_step_two_arrives_prefilled_from_the_models_own_pipeline(self):
        """The standing convention: an action never asks for geometry the model
        already records. Prefilled server-side, not by a JS change event."""
        response = self._post(model_source=DatasetInferenceRun.TRAINED,
                              configure="1", trained_model=str(self.tiled.pk))

        values = response.context["values"]
        self.assertEqual(values["pipeline"], "batch_detect")
        self.assertEqual(values["tile_size_px"], 640)
        self.assertEqual(values["overlap"], 0.25)
        self.assertEqual(values["score_threshold"], 0.3)

    def test_submitting_creates_a_queued_run_and_opens_its_report(self):
        with mock.patch("fleet.admin._queue") as queue:
            response = self._post(
                model_source=DatasetInferenceRun.TRAINED, apply="1",
                trained_model=str(self.model.pk), pipeline="raw",
                score_threshold="0.4", image_limit="2", warmup="5")

        run = DatasetInferenceRun.objects.get()
        self.assertEqual(run.status, DatasetInferenceRun.QUEUED)
        self.assertEqual(run.dataset_name_snapshot, "ds")
        self.assertEqual(run.model_label_snapshot, "m")
        self.assertEqual(run.score_threshold, 0.4)
        self.assertEqual(run.image_limit, 2)
        self.assertEqual(run.warmup, 5)
        queue.return_value.enqueue.assert_called_once()
        self.assertEqual(queue.return_value.enqueue.call_args.args[1], run.pk)
        self.assertEqual(response.redirect_chain[-1][0].split("?")[0],
                         reverse("admin:fleet_datasetinferencerun_report"))

    def test_a_raw_run_records_no_tiling_knobs_even_if_the_form_posts_them(self):
        """The form hides irrelevant rows with CSS, which still submits them."""
        with mock.patch("fleet.admin._queue"):
            self._post(model_source=DatasetInferenceRun.TRAINED, apply="1",
                       trained_model=str(self.model.pk), pipeline="raw",
                       score_threshold="0.5", tile_size_px="640", overlap="0.2",
                       detector_checkpoint="/x/d.pt", merge_nms_iou="0.6")

        run = DatasetInferenceRun.objects.get()
        self.assertIsNone(run.tile_size_px)
        self.assertIsNone(run.overlap)
        self.assertIsNone(run.merge_nms_iou)
        self.assertEqual(run.detector_checkpoint, "")

    def test_a_person_pipeline_without_a_detector_is_refused_at_the_form(self):
        with mock.patch("fleet.admin._queue") as queue:
            response = self._post(
                model_source=DatasetInferenceRun.TRAINED, apply="1",
                trained_model=str(self.model.pk), pipeline="people_detect_first",
                score_threshold="0.5")

        queue.return_value.enqueue.assert_not_called()
        self.assertFalse(DatasetInferenceRun.objects.exists())
        self.assertContains(response, "requires a detector checkpoint")

    def test_a_dataset_the_trainer_cannot_see_is_refused(self):
        outside = Path(self.tmp.name) / "outside"
        (outside / "other").mkdir(parents=True)
        fs = FleetSettings.load()
        fs.source_dir = str(outside)
        fs.save()
        Dataset.objects.create(name="other")

        with mock.patch("fleet.admin._queue") as queue:
            response = self.client.post(self.url, {
                "action": "run_model_inference",
                ACTION_CHECKBOX_NAME: [str(Dataset.objects.get(name="other").pk)],
                "model_source": DatasetInferenceRun.TRAINED,
                "apply": "1",
                "trained_model": str(self.model.pk),
                "pipeline": "raw",
                "score_threshold": "0.5",
            }, follow=True)

        queue.return_value.enqueue.assert_not_called()
        self.assertContains(response, "only tree the trainer can open")

    def test_step_two_renders_for_an_exported_artifact(self):
        """The exported branch of the shared partial, with a real artifact on disk."""
        exports_root = self.base / "data" / "exports"
        exports_root.mkdir(parents=True)
        (exports_root / "m-best.onnx").write_bytes(b"onnx")
        ts = TrainingSettings.load()
        ts.exports_root = str(exports_root)
        ts.save()

        response = self._post(model_source=DatasetInferenceRun.EXPORTED, configure="1")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["is_exported"])
        self.assertEqual([a["relpath"] for a in response.context["artifacts"]],
                         ["m-best.onnx"])
        self.assertContains(response, "m-best.onnx")

    def test_step_two_renders_for_a_bundle_with_nothing_preselected(self):
        """A bundle's geometry arrives via the explicit Sync bundle press, so the
        page must not land with one chosen and its fields filled in."""
        response = self._post(model_source=DatasetInferenceRun.BUNDLE, configure="1")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["is_bundle"])
        self.assertEqual(response.context["selected_model"], "")
        self.assertEqual(response.context["values"]["pipeline"], "raw")

    def test_two_datasets_at_once_are_refused(self):
        Dataset.objects.create(name="second")
        response = self.client.post(self.url, {
            "action": "run_model_inference",
            ACTION_CHECKBOX_NAME: [str(d.pk) for d in Dataset.objects.all()],
        }, follow=True)
        self.assertContains(response, "exactly one dataset")


class ReportTests(_RunTestCase):
    def setUp(self):
        super().setUp()
        self._login_admin()
        self.run = self._run(status=DatasetInferenceRun.OK)
        self.first = DatasetInferenceResult.objects.create(
            run=self.run, seq=1, image_filename="img0.jpg", detections=1,
            boxes=[_box()], model_ms=8.0, request_ms=11.0,
            stages={"load_ms": 3.0, "infer_ms": 5.0})
        DatasetInferenceResult.objects.create(
            run=self.run, seq=2, image_filename="img1.jpg", detections=0,
            model_ms=12.0, request_ms=15.0)
        dataset_inference.summarize(self.run)

    def test_report_renders_the_headline_numbers_and_the_geometry(self):
        response = self.client.get(
            reverse("admin:fleet_datasetinferencerun_report") + f"?run={self.run.pk}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["timings"]["timing_source"], "model")
        self.assertEqual(len(response.context["rows"]), 2)
        self.assertIn(("score threshold", 0.5), response.context["geometry"])
        # Stage rows are ordered biggest share first.
        self.assertEqual([s["name"] for s in response.context["stage_rows"]],
                         ["infer", "load"])

    def test_report_can_show_only_images_with_detections(self):
        response = self.client.get(
            reverse("admin:fleet_datasetinferencerun_report")
            + f"?run={self.run.pk}&only=detections")

        self.assertEqual([r.seq for r in response.context["rows"]], [1])
        # A filtered slice must not be live-appended to.
        self.assertFalse(response.context["live_append"])

    def test_progress_returns_only_rows_after_the_cursor(self):
        response = self.client.get(
            reverse("admin:fleet_datasetinferencerun_progress")
            + f"?run={self.run.pk}&after=1")

        payload = response.json()
        self.assertEqual([r["seq"] for r in payload["results"]], [2])
        self.assertTrue(payload["run"]["terminal"])
        self.assertEqual(payload["run"]["timing_source"], "model")

    def test_cancel_marks_a_live_run_for_stopping(self):
        run = self._run(status=DatasetInferenceRun.RUNNING)
        self.client.post(reverse("admin:fleet_datasetinferencerun_cancel"),
                         {"run": str(run.pk)})

        run.refresh_from_db()
        self.assertEqual(run.status, DatasetInferenceRun.CANCEL_REQUESTED)

    def test_cancel_leaves_a_finished_run_alone(self):
        self.client.post(reverse("admin:fleet_datasetinferencerun_cancel"),
                         {"run": str(self.run.pk)})

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, DatasetInferenceRun.OK)

    def test_image_view_serves_a_thumbnail_and_an_annotated_copy(self):
        url = reverse("admin:fleet_datasetinferencerun_image")

        plain = self.client.get(f"{url}?result={self.first.pk}")
        drawn = self.client.get(f"{url}?result={self.first.pk}&boxes=1")

        self.assertEqual(plain.status_code, 200)
        self.assertEqual(drawn.status_code, 200)
        self.assertEqual(drawn["Content-Type"], "image/jpeg")
        # Different bytes: one is the plain frame, the other has boxes burned in.
        self.assertNotEqual(plain.getvalue(), drawn.getvalue())

    def test_image_view_refuses_a_filename_that_escapes_its_dataset(self):
        escaped = DatasetInferenceResult.objects.create(
            run=self.run, seq=3, image_filename="../../model.pt")
        response = self.client.get(
            reverse("admin:fleet_datasetinferencerun_image") + f"?result={escaped.pk}")
        # Basename-only resolution means this addresses a file that isn't there,
        # never one outside the dataset.
        self.assertEqual(response.status_code, 404)

    def test_changelist_marks_a_round_trip_fps_as_such(self):
        DatasetInferenceResult.objects.filter(run=self.run).update(model_ms=None)
        dataset_inference.summarize(self.run)

        response = self.client.get(
            reverse("admin:fleet_datasetinferencerun_changelist"))

        self.assertContains(response, "round trip")


class PipelineFieldParsingTests(TestCase):
    """The shared parse layer, which both this action and the video one use."""

    def test_chain_keeps_every_knob(self):
        values, error = inference_form.parse_pipeline_fields({
            "pipeline": "chain", "chain": "batch_detect, people_detect_first",
            "detector_checkpoint": "/d.pt", "tile_size_px": "640",
            "merge_nms_iou": "0.5", "score_threshold": "0.5",
        })
        self.assertIsNone(error)
        self.assertEqual(values["chain"], ["batch_detect", "people_detect_first"])
        self.assertEqual(values["tile_size_px"], 640)
        self.assertEqual(values["detector_checkpoint"], "/d.pt")

    def test_a_non_chain_pipeline_drops_a_posted_chain(self):
        values, _ = inference_form.parse_pipeline_fields({
            "pipeline": "batch_detect", "chain": "people_detect_first",
            "score_threshold": "0.5",
        })
        self.assertEqual(values["chain"], [])

    def test_every_key_of_the_metadata_vocabulary_is_returned(self):
        """A knob added to pipeline_meta.FIELDS must flow through here, not be
        hand-listed at each call site."""
        values, _ = inference_form.parse_pipeline_fields({"pipeline": "raw"})
        self.assertEqual(set(values), set(pipeline_meta.FIELDS))

    def test_out_of_range_values_are_reported_not_clamped(self):
        _values, error = inference_form.parse_pipeline_fields({
            "pipeline": "raw", "score_threshold": "3"})
        self.assertIn("between 0 and 1", error)
