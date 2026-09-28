"""Tests for the Datasets tab's "Evaluate a model on this dataset…" action.

What is under test is the routing, not the scoring: which *kind* of eval row a
posted form produces, what it records about the model it scored, and what the
request handed to the trainer then names — because that is where an eval of an
exported artifact or a bundle differs from the Models tab's eval of a
checkpoint. The trainer itself is never called; the queue is mocked.

``BASE_DIR`` is overridden per test so a dataset written into a tempdir still
counts as inside the shared data mount, the one thing this action refuses
outright (images reach the trainer as paths, not copies).
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from eval_pipelines.models import PipelineEvalRun
from fleet.models import Dataset, FleetSettings
from training.models import EvalRun, TrainedModel, TrainingSettings
from training.services import config_gen, eval_analytics, inference_form


class DatasetEvalActionTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.src = self.base / "data" / "source"
        self.src.mkdir(parents=True)

        self.settings_override = override_settings(BASE_DIR=str(self.base))
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

        fs = FleetSettings.load()
        fs.source_dir = str(self.src)
        fs.save()

        self.dataset_dir = self.src / "ds"
        self.dataset_dir.mkdir()
        (self.dataset_dir / "classes.txt").write_text("helmet\nvest\n", encoding="utf-8")
        (self.dataset_dir / "labels").mkdir()
        cv2.imwrite(str(self.dataset_dir / "img0.jpg"),
                    np.full((32, 48, 3), 120, dtype=np.uint8))
        self.dataset = Dataset.objects.create(name="ds")

        self.checkpoint = self.base / "model.pt"
        self.checkpoint.write_bytes(b"not-really-a-checkpoint")
        self.model = TrainedModel.objects.create(
            name="m", arch="yolox", checkpoint_path=str(self.checkpoint))

        ts = TrainingSettings.load()
        self.exports_root = self.base / "data" / "exports"
        self.exports_root.mkdir(parents=True)
        self.bundles_root = self.base / "data" / "bundles"
        self.bundles_root.mkdir(parents=True)
        ts.exports_root = str(self.exports_root)
        ts.bundles_root = str(self.bundles_root)
        ts.configs_root = str(self.base / "data" / "configs")
        ts.runs_root = str(self.base / "data" / "runs")
        ts.save()

        User = get_user_model()
        User.objects.create_superuser(
            username="admin", email="admin@example.com", password="pw")
        self.client.login(username="admin", password="pw")
        self.url = reverse("admin:fleet_dataset_changelist")

    # ------------------------------------------------------------------ helpers
    def _post(self, **extra):
        return self.client.post(self.url, {
            "action": "evaluate_on_dataset",
            ACTION_CHECKBOX_NAME: [str(self.dataset.pk)],
            **extra,
        }, follow=True)

    def _artifact(self, name="m-best.onnx", classes=("helmet", "vest")):
        """An exported artifact with the ``.meta.json`` its loader reads."""
        (self.exports_root / name).write_bytes(b"onnx")
        if classes is not None:
            (self.exports_root / name).with_suffix(".meta.json").write_text(json.dumps({
                "arch": "yolox",
                "num_classes": len(classes),
                "class_map": {str(i): n for i, n in enumerate(classes)},
            }))
        return name

    def _bundle(self, name="ppe-bundle"):
        from training.tests_bundles import MANIFEST

        bundle = self.bundles_root / name
        (bundle / "models").mkdir(parents=True)
        (bundle / "pipeline.json").write_text(json.dumps(MANIFEST))
        (bundle / "models/model.engine").write_bytes(b"x")
        (bundle / "models/detector.engine").write_bytes(b"x")
        (bundle / "runtime").mkdir()
        (bundle / "infer.py").write_text("")
        return name

    def _request_yaml(self, eval_obj) -> dict:
        return yaml.safe_load(Path(eval_obj.request_yaml_path).read_text())

    # ------------------------------------------------------------------- step 1
    def test_step_one_offers_the_three_model_sources(self):
        response = self._post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["model_source_choices"],
                         inference_form.MODEL_SOURCE_CHOICES)

    def test_step_two_arrives_prefilled_and_asks_for_the_labels(self):
        response = self._post(model_source=EvalRun.TRAINED, configure="1",
                              trained_model=str(self.model.pk))
        self.assertContains(response, 'name="label_source"')
        self.assertContains(response, 'name="map_score_threshold"')

    # -------------------------------------------------------- the three sources
    def test_a_trained_model_on_raw_makes_a_base_eval(self):
        with mock.patch("fleet.admin._queue") as queue:
            response = self._post(
                model_source=EvalRun.TRAINED, apply="1",
                trained_model=str(self.model.pk), pipeline="raw",
                score_threshold="0.4", map_score_threshold="0.002")

        eval_run = EvalRun.objects.get()
        self.assertEqual(eval_run.status, EvalRun.QUEUED)
        self.assertEqual(eval_run.model_source, EvalRun.TRAINED)
        self.assertEqual(eval_run.trained_model, self.model)
        self.assertEqual(eval_run.model_label_snapshot, "m")
        self.assertEqual(eval_run.score_threshold, 0.4)
        self.assertEqual(eval_run.map_score_threshold, 0.002)
        self.assertFalse(PipelineEvalRun.objects.exists())
        queue.return_value.enqueue.assert_called_once()
        # The request keeps naming the checkpoint exactly as recorded — an
        # artifact is what gets resolved, a catalogued model never was.
        self.assertEqual(self._request_yaml(eval_run)["checkpoint_path"],
                         str(self.checkpoint))
        self.assertEqual(response.redirect_chain[-1][0].split("?")[0],
                         reverse("admin:eval_pipelines_combinedeval_changelist"))

    def _second_dataset(self, classes="helmet\nvest\n"):
        other = self.src / "ds-night"
        other.mkdir()
        (other / "classes.txt").write_text(classes, encoding="utf-8")
        (other / "labels").mkdir()
        cv2.imwrite(str(other / "img0.jpg"), np.full((32, 48, 3), 30, dtype=np.uint8))
        return Dataset.objects.create(name="ds-night")

    def test_several_datasets_are_several_grouped_evals(self):
        night = self._second_dataset()
        with mock.patch("fleet.admin._queue") as queue:
            self._post(**{
                ACTION_CHECKBOX_NAME: [str(self.dataset.pk), str(night.pk)],
                "model_source": EvalRun.TRAINED, "apply": "1",
                "trained_model": str(self.model.pk), "pipeline": "raw",
            })

        rows = list(EvalRun.objects.order_by("dataset__name"))
        self.assertEqual([r.dataset.name for r in rows], ["ds", "ds-night"])
        self.assertIsNotNone(rows[0].test_group)
        self.assertEqual(rows[0].test_group, rows[1].test_group)
        self.assertTrue(all(r.status == EvalRun.QUEUED for r in rows))
        self.assertEqual(queue.return_value.enqueue.call_count, 2)

    def test_a_class_map_applies_to_each_set_by_class_name(self):
        night = self._second_dataset(classes="helmet\nperson\n")
        with mock.patch("fleet.admin._queue"):
            self._post(**{
                ACTION_CHECKBOX_NAME: [str(self.dataset.pk), str(night.pk)],
                "model_source": EvalRun.TRAINED, "apply": "1",
                "trained_model": str(self.model.pk), "pipeline": "raw",
                "class_map__helmet": "hardhat", "class_map__vest": "",
            })

        maps = {r.dataset.name: r.class_map for r in EvalRun.objects.all()}
        self.assertEqual(maps["ds"], {"helmet": "hardhat", "vest": None})
        self.assertEqual(maps["ds-night"], {"helmet": "hardhat"})

    def test_one_set_with_no_labels_refuses_the_whole_request(self):
        night = self._second_dataset()
        (self.src / "ds-night" / "labels").rmdir()
        with mock.patch("fleet.admin._queue") as queue:
            response = self._post(**{
                ACTION_CHECKBOX_NAME: [str(self.dataset.pk), str(night.pk)],
                "model_source": EvalRun.TRAINED, "apply": "1",
                "trained_model": str(self.model.pk), "pipeline": "raw",
            })

        self.assertFalse(EvalRun.objects.exists())
        queue.return_value.enqueue.assert_not_called()
        self.assertContains(response, "ds-night")

    def test_an_exported_artifact_is_evaluated_as_itself(self):
        relpath = self._artifact()
        with mock.patch("fleet.admin._queue"):
            self._post(model_source=EvalRun.EXPORTED, apply="1",
                       artifact_path=relpath, pipeline="raw", score_threshold="0.5")

        eval_run = EvalRun.objects.get()
        self.assertEqual(eval_run.model_source, EvalRun.EXPORTED)
        self.assertIsNone(eval_run.trained_model)
        self.assertEqual(eval_run.artifact_path, relpath)
        self.assertEqual(eval_run.model_label_snapshot, relpath)
        # What the trainer is told to load: the artifact, resolved under the
        # export root — not the checkpoint it was exported from.
        self.assertEqual(self._request_yaml(eval_run)["checkpoint_path"],
                         str(self.exports_root / relpath))

    def test_a_pipeline_choice_makes_a_pipeline_eval_of_the_artifact(self):
        relpath = self._artifact()
        with mock.patch("fleet.admin._queue"):
            self._post(model_source=EvalRun.EXPORTED, apply="1",
                       artifact_path=relpath, pipeline="batch_detect",
                       score_threshold="0.5", tile_size_px="640", overlap="0.2")

        pe = PipelineEvalRun.objects.get()
        self.assertFalse(EvalRun.objects.exists())
        self.assertEqual(pe.model_source, PipelineEvalRun.EXPORTED)
        self.assertEqual(pe.tile_size_px, 640)
        self.assertEqual(self._request_yaml(pe)["model_checkpoint"],
                         str(self.exports_root / relpath))

    def test_a_bundle_is_evaluated_through_its_manifests_pipeline(self):
        """Even when the form posts raw: a bundle's geometry is the bundle's.

        The page locks those fields after "Sync bundle", but nothing forces an
        operator to press it, and a stale page can carry an older geometry.
        """
        relpath = self._bundle()
        with mock.patch("fleet.admin._queue"):
            self._post(model_source=EvalRun.BUNDLE, apply="1",
                       bundle_path=relpath, pipeline="raw", score_threshold="0.5")

        pe = PipelineEvalRun.objects.get()
        self.assertFalse(EvalRun.objects.exists())
        self.assertEqual(pe.pipeline, "people_detect_first")
        self.assertEqual(pe.bundle_path, relpath)
        self.assertEqual(pe.merge_nms_iou, 0.3)
        self.assertEqual(
            pe.detector_checkpoint,
            str(self.bundles_root / relpath / "models/detector.engine"))
        # The model the trainer loads is the one inside the bundle.
        self.assertEqual(self._request_yaml(pe)["model_checkpoint"],
                         str(self.bundles_root / relpath / "models/model.engine"))

    # ------------------------------------------------------------- refusals
    def test_annotator_labels_without_an_annotator_are_refused(self):
        with mock.patch("fleet.admin._queue") as queue:
            response = self._post(
                model_source=EvalRun.TRAINED, apply="1",
                trained_model=str(self.model.pk), pipeline="raw",
                score_threshold="0.5", label_source=EvalRun.ANNOTATOR)

        queue.return_value.enqueue.assert_not_called()
        self.assertFalse(EvalRun.objects.exists())
        self.assertContains(response, "Pick an annotator")

    def test_labels_that_are_not_there_yet_are_refused_before_queueing(self):
        """An annotator who never synced is the ordinary case, and "eval #12
        errored" half an hour later is a worse way to find out."""
        from fleet.models import Annotator

        annotator = Annotator.objects.create(username="bob", status=Annotator.ACTIVE)
        with mock.patch("fleet.admin._queue") as queue:
            response = self._post(
                model_source=EvalRun.TRAINED, apply="1",
                trained_model=str(self.model.pk), pipeline="raw",
                score_threshold="0.5", label_source=EvalRun.ANNOTATOR,
                annotator=str(annotator.pk))

        queue.return_value.enqueue.assert_not_called()
        self.assertFalse(EvalRun.objects.exists())
        self.assertContains(response, "nothing to score against")

    def test_a_dataset_the_trainer_cannot_see_is_refused(self):
        outside = self.base / "outside"
        (outside / "other").mkdir(parents=True)
        fs = FleetSettings.load()
        fs.source_dir = str(outside)
        fs.save()
        other = Dataset.objects.create(name="other")

        with mock.patch("fleet.admin._queue") as queue:
            response = self.client.post(self.url, {
                "action": "evaluate_on_dataset",
                ACTION_CHECKBOX_NAME: [str(other.pk)],
                "model_source": EvalRun.TRAINED, "apply": "1",
                "trained_model": str(self.model.pk),
                "pipeline": "raw", "score_threshold": "0.5",
            }, follow=True)

        queue.return_value.enqueue.assert_not_called()
        self.assertContains(response, "only tree the trainer can open")


class ArtifactEvalDownstreamTests(TestCase):
    """What the services that re-read a finished eval do with an artifact one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.settings_override = override_settings(BASE_DIR=str(self.base))
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

        self.exports_root = self.base / "exports"
        self.exports_root.mkdir(parents=True)
        ts = TrainingSettings.load()
        ts.exports_root = str(self.exports_root)
        ts.save()

        self.dataset = Dataset.objects.create(name="ds")

    def _artifact_eval(self, classes=("helmet", "vest")):
        (self.exports_root / "m.onnx").write_bytes(b"onnx")
        if classes is not None:
            (self.exports_root / "m.meta.json").write_text(json.dumps({
                "arch": "yolox", "num_classes": len(classes),
                "class_map": {str(i): n for i, n in enumerate(classes)},
            }))
        return EvalRun.objects.create(
            dataset=self.dataset, model_source=EvalRun.EXPORTED,
            artifact_path="m.onnx", model_label_snapshot="m.onnx",
            output_dir=str(self.base / "out"),
        )

    def test_prediction_space_comes_from_the_artifacts_meta(self):
        """No checkpoint exists to read the class order back out of, so the
        sidecar's class_map is sent instead — and it must be sent, not guessed."""
        eval_run = self._artifact_eval()
        self.assertEqual(config_gen.prediction_space(eval_run, ["helmet", "vest"]),
                         {"prediction_classes": ["helmet", "vest"]})

    def test_an_artifact_with_no_class_map_is_refused_rather_than_assumed(self):
        eval_run = self._artifact_eval(classes=None)
        with self.assertRaises(ValueError):
            config_gen.prediction_space(eval_run, ["helmet", "vest"])

    def test_a_trained_eval_still_sends_its_checkpoint(self):
        model = TrainedModel.objects.create(
            name="m", arch="yolox", checkpoint_path="runs/x/best.pt")
        eval_run = EvalRun.objects.create(dataset=self.dataset, trained_model=model)
        self.assertEqual(config_gen.prediction_space(eval_run, ["helmet"]),
                         {"checkpoint_path": "runs/x/best.pt"})

    def test_the_comparison_page_names_an_artifact_eval_by_its_artifact(self):
        """The compare page joins nothing back to the catalogue: an artifact eval
        has no row there, and used to raise rather than render."""
        eval_run = self._artifact_eval()
        eval_run.metrics = {"map50": 0.5}
        eval_run.save()
        column = eval_analytics.compare([eval_run])["columns"][0]
        self.assertEqual(column["model"], "m.onnx")
        self.assertEqual(column["arch"], "exported artifact (ONNX / TensorRT)")
