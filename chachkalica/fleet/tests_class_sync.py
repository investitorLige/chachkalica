"""Tests for the "Check classes" half of the dataset-eval form.

An eval matches predictions to ground truth **by class name** and drops both
sides of anything the two spaces don't share, so a model and a dataset with
different taxonomies score 0.0 for everything while looking like a model that
detects nothing. This is the surface that catches that before the run: it reads
both class lists, says how they line up, and produces the ``class_map`` that
translates one into the other.

Two properties matter most and are pinned hardest:

* **the map is opt-in** — a form submitted without touching the block queues
  exactly the eval it queued before this existed, with an empty ``class_map``
  and a request YAML that never mentions one;
* **a posted map reaches the trainer** — stored on the row and emitted in the
  request, for both kinds of eval row.

Nothing here runs a model; the queue is mocked, as in ``tests_dataset_eval``.
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
from training.services import class_sync


class ClassSyncServiceTests(TestCase):
    """The pure half: how two class lists are compared and a map proposed."""

    def test_suggest_maps_a_shared_name_to_itself(self):
        self.assertEqual(
            class_sync.suggest(["helmet", "vest"], ["helmet", "vest"]),
            {"helmet": "helmet", "vest": "vest"},
        )

    def test_suggest_collapses_everything_onto_a_single_class_model(self):
        """The case the feature exists for: a coarse model, a fine test set."""
        self.assertEqual(
            class_sync.suggest(["handgun", "rifle", "knife"], ["gun"]),
            {"handgun": "gun", "rifle": "gun", "knife": "gun"},
        )

    def test_suggest_drops_what_a_multi_class_model_does_not_name(self):
        """No guessing between two named taxonomies — an unmatched class is dropped."""
        self.assertEqual(
            class_sync.suggest(["helmet", "gloves"], ["helmet", "vest"]),
            {"helmet": "helmet", "gloves": None},
        )

    def test_suggest_normalizes_a_case_only_difference_to_the_models_spelling(self):
        """The eval matches names exactly, so "Helmet" and "helmet" need mapping."""
        self.assertEqual(
            class_sync.suggest(["Helmet"], ["helmet"]), {"Helmet": "helmet"})

    def test_parse_posted_is_empty_when_the_block_was_never_used(self):
        self.assertEqual(class_sync.parse_posted({}, ["helmet", "vest"]), ({}, None))

    def test_parse_posted_reads_one_field_per_class(self):
        posted = {"class_map__handgun": "gun", "class_map__rifle": "gun",
                  "class_map__bat": ""}
        class_map, error = class_sync.parse_posted(posted, ["handgun", "rifle", "bat"])
        self.assertIsNone(error)
        self.assertEqual(class_map, {"handgun": "gun", "rifle": "gun", "bat": None})

    def test_parse_posted_records_a_no_op_map_as_no_map(self):
        """Every class mapped to itself is the default; storing it would be noise."""
        posted = {"class_map__helmet": "helmet", "class_map__vest": "vest"}
        self.assertEqual(class_sync.parse_posted(posted, ["helmet", "vest"]), ({}, None))

    def test_parse_posted_refuses_a_map_that_drops_everything(self):
        posted = {"class_map__helmet": "", "class_map__vest": ""}
        class_map, error = class_sync.parse_posted(posted, ["helmet", "vest"])
        self.assertEqual(class_map, {})
        self.assertIn("drops every class", error)

    def test_describe_renders_a_map_for_a_human(self):
        self.assertEqual(
            class_sync.describe({"handgun": "gun", "bat": None}),
            "bat → (dropped), handgun → gun",
        )
        self.assertEqual(class_sync.describe({}), "")


@override_settings(ALLOWED_HOSTS=["*"])
class ClassSyncEndpointTests(TestCase):
    """The JSON endpoint behind the button."""

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

        dataset_dir = self.src / "ds"
        dataset_dir.mkdir()
        (dataset_dir / "classes.txt").write_text(
            "handgun\nrifle\nknife\nbat\n", encoding="utf-8")
        self.dataset = Dataset.objects.create(name="ds")

        ts = TrainingSettings.load()
        self.exports_root = self.base / "data" / "exports"
        self.exports_root.mkdir(parents=True)
        ts.exports_root = str(self.exports_root)
        ts.save()

        User = get_user_model()
        User.objects.create_superuser("admin", "admin@example.com", "pw")
        self.client.login(username="admin", password="pw")
        self.url = reverse("class-sync")

    def _artifact(self, name, classes):
        (self.exports_root / name).write_bytes(b"onnx")
        (self.exports_root / name).with_suffix(".meta.json").write_text(json.dumps({
            "arch": "rtdetr",
            "num_classes": len(classes),
            "class_map": {str(i): n for i, n in enumerate(classes)},
        }))
        return name

    def test_disjoint_spaces_are_reported_as_not_ok_with_a_full_map_proposed(self):
        """The real case: a one-class 'gun' engine against a four-class test set."""
        artifact = self._artifact("gun.onnx", ["gun"])
        response = self.client.post(self.url, {
            "dataset": str(self.dataset.pk),
            "model_source": EvalRun.EXPORTED,
            "artifact_path": artifact,
        })
        body = response.json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["model_classes"], ["gun"])
        self.assertEqual(body["dataset_classes"], ["handgun", "rifle", "knife", "bat"])
        self.assertEqual(body["matched"], [])
        self.assertEqual(body["suggested"],
                         {"handgun": "gun", "rifle": "gun", "knife": "gun", "bat": "gun"})
        self.assertTrue(any(c["status"] == "fail" and "No shared names" in c["label"]
                            for c in body["checks"]))

    def test_a_matching_taxonomy_is_reported_as_needing_nothing(self):
        artifact = self._artifact("full.onnx", ["handgun", "rifle", "knife", "bat"])
        body = self.client.post(self.url, {
            "dataset": str(self.dataset.pk),
            "model_source": EvalRun.EXPORTED,
            "artifact_path": artifact,
        }).json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["unmatched"], [])
        self.assertTrue(any("no map needed" in c["detail"] for c in body["checks"]))

    def test_a_partial_overlap_names_what_would_be_dropped(self):
        artifact = self._artifact("part.onnx", ["handgun", "rifle"])
        body = self.client.post(self.url, {
            "dataset": str(self.dataset.pk),
            "model_source": EvalRun.EXPORTED,
            "artifact_path": artifact,
        }).json()
        self.assertTrue(body["ok"])          # it would still score something
        self.assertEqual(body["matched"], ["handgun", "rifle"])
        self.assertEqual(body["unmatched"], ["knife", "bat"])

    def test_a_trained_models_classes_come_off_its_row(self):
        model = TrainedModel.objects.create(
            name="m", arch="rtdetr", checkpoint_path="/x.pt", classes=["gun"])
        body = self.client.post(self.url, {
            "dataset": str(self.dataset.pk),
            "model_source": EvalRun.TRAINED,
            "trained_model": str(model.pk),
        }).json()
        self.assertEqual(body["model_classes"], ["gun"])

    def test_a_model_with_no_recorded_classes_says_so_rather_than_guessing(self):
        (self.exports_root / "bare.onnx").write_bytes(b"onnx")   # no .meta.json
        body = self.client.post(self.url, {
            "dataset": str(self.dataset.pk),
            "model_source": EvalRun.EXPORTED,
            "artifact_path": "bare.onnx",
        }).json()
        self.assertTrue(body["unknown"])
        self.assertFalse(body["ok"])
        self.assertEqual(body["suggested"], {})

    def test_the_dataset_is_addressed_by_pk_not_by_name(self):
        """Its name is a path segment under the source root; a posted one would be too."""
        response = self.client.post(self.url, {
            "dataset": "../../etc",
            "model_source": EvalRun.EXPORTED,
            "artifact_path": "x.onnx",
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("No such dataset", response.json()["error"])

    def test_get_is_refused(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)


class ClassMapReachesTheTrainerTests(TestCase):
    """The form → row → request YAML path, for both kinds of eval row."""

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

        dataset_dir = self.src / "ds"
        dataset_dir.mkdir()
        (dataset_dir / "classes.txt").write_text(
            "handgun\nrifle\nknife\nbat\n", encoding="utf-8")
        (dataset_dir / "labels").mkdir()
        cv2.imwrite(str(dataset_dir / "img0.jpg"),
                    np.full((32, 48, 3), 120, dtype=np.uint8))
        self.dataset = Dataset.objects.create(name="ds")

        self.checkpoint = self.base / "model.pt"
        self.checkpoint.write_bytes(b"x")
        self.model = TrainedModel.objects.create(
            name="m", arch="rtdetr", checkpoint_path=str(self.checkpoint),
            classes=["gun"])

        ts = TrainingSettings.load()
        ts.exports_root = str(self.base / "data" / "exports")
        ts.bundles_root = str(self.base / "data" / "bundles")
        ts.configs_root = str(self.base / "data" / "configs")
        ts.runs_root = str(self.base / "data" / "runs")
        ts.save()

        User = get_user_model()
        User.objects.create_superuser("admin", "admin@example.com", "pw")
        self.client.login(username="admin", password="pw")
        self.url = reverse("admin:fleet_dataset_changelist")

    def _post(self, **extra):
        return self.client.post(self.url, {
            "action": "evaluate_on_dataset",
            ACTION_CHECKBOX_NAME: [str(self.dataset.pk)],
            **extra,
        }, follow=True)

    def _request_yaml(self, eval_obj) -> dict:
        return yaml.safe_load(Path(eval_obj.request_yaml_path).read_text())

    COLLAPSE = {
        "class_map__handgun": "gun",
        "class_map__rifle": "gun",
        "class_map__knife": "gun",
        "class_map__bat": "",
    }

    def test_the_form_offers_the_block(self):
        response = self._post(model_source=EvalRun.TRAINED, configure="1",
                              trained_model=str(self.model.pk))
        self.assertContains(response, "data-class-sync")
        self.assertContains(response, "Check classes")
        self.assertContains(response, reverse("class-sync"))

    def test_no_map_posted_leaves_the_eval_exactly_as_it_was(self):
        with mock.patch("fleet.admin._queue"):
            self._post(model_source=EvalRun.TRAINED, apply="1",
                       trained_model=str(self.model.pk), pipeline="raw",
                       score_threshold="0.4")
        eval_run = EvalRun.objects.get()
        self.assertEqual(eval_run.class_map, {})
        # Absent from the request entirely, so an ordinary eval's YAML is unchanged.
        self.assertNotIn("class_map", self._request_yaml(eval_run))

    def test_a_posted_map_is_stored_and_sent_on_a_base_eval(self):
        with mock.patch("fleet.admin._queue"):
            self._post(model_source=EvalRun.TRAINED, apply="1",
                       trained_model=str(self.model.pk), pipeline="raw",
                       score_threshold="0.4", **self.COLLAPSE)
        eval_run = EvalRun.objects.get()
        expected = {"handgun": "gun", "rifle": "gun", "knife": "gun", "bat": None}
        self.assertEqual(eval_run.class_map, expected)
        self.assertEqual(self._request_yaml(eval_run)["class_map"], expected)

    def test_a_posted_map_is_stored_and_sent_on_a_pipeline_eval(self):
        with mock.patch("fleet.admin._queue"):
            self._post(model_source=EvalRun.TRAINED, apply="1",
                       trained_model=str(self.model.pk), pipeline="batch_detect",
                       score_threshold="0.4", **self.COLLAPSE)
        pipeline_run = PipelineEvalRun.objects.get()
        expected = {"handgun": "gun", "rifle": "gun", "knife": "gun", "bat": None}
        self.assertEqual(pipeline_run.class_map, expected)
        self.assertEqual(self._request_yaml(pipeline_run)["class_map"], expected)

    def test_a_map_that_drops_everything_is_refused_before_queueing(self):
        with mock.patch("fleet.admin._queue") as queue:
            response = self._post(
                model_source=EvalRun.TRAINED, apply="1",
                trained_model=str(self.model.pk), pipeline="raw",
                score_threshold="0.4",
                **{f"class_map__{n}": "" for n in
                   ["handgun", "rifle", "knife", "bat"]})
        self.assertFalse(EvalRun.objects.exists())
        queue.return_value.enqueue.assert_not_called()
        self.assertContains(response, "drops every class")
