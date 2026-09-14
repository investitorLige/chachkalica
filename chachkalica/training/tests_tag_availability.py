"""The "Sliceable by" panel both evaluate forms show.

The point of the panel is that the label source chosen on the form is what
decides which tag answers a later Tag analytics page can find. These check that
it says so accurately — including for the cases that used to be invisible: a tag
defined but never synced, and a dataset with no tags at all.
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.admin.sites import AdminSite
from django.test import RequestFactory, TestCase

from fleet.models import AnnotationTag, Dataset, FleetSettings
from fleetsite.admin_views import dataset_tags_view
from training.models import EvalRun
from training.services import tag_availability


def _dataset_on_disk(source: Path, name: str) -> Dataset:
    directory = source / name
    (directory / "labels").mkdir(parents=True)
    (directory / "classes.txt").write_text("helmet\nperson\n", encoding="utf-8")
    return Dataset.objects.create(name=name)


class TagAvailabilitySetup(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.source = root / "source"
        self.source.mkdir()
        fs = FleetSettings.load()
        fs.source_dir = str(self.source)
        fs.target_dir = str(root / "target")
        fs.save()
        self.dataset = _dataset_on_disk(self.source, "ds1")

    def tearDown(self):
        self._tmp.cleanup()

    def _tag(self, name, scope=AnnotationTag.FRAME, choices=("sun", "rain")):
        return AnnotationTag.objects.create(
            dataset=self.dataset, name=name, scope=scope,
            widget=AnnotationTag.RADIO, choices=list(choices))

    def _sidecar(self, document):
        (self.source / "ds1" / "labels" / "annotation_tags.json").write_text(
            json.dumps(document), encoding="utf-8")

    def _availability(self):
        return tag_availability.availability(self.dataset, EvalRun.SOURCE)


class AvailabilityTests(TagAvailabilitySetup):
    def test_the_computed_tags_are_listed_even_with_no_annotations_anywhere(self):
        availability = self._availability()

        self.assertEqual(availability["declared"], [])
        self.assertIn("crowding", [tag["label"] for tag in availability["auto"]])

    def test_a_tag_with_answers_reports_how_many(self):
        self._tag("weather")
        self._sidecar({
            "version": 1,
            "tags": [{"name": "weather", "scope": "frame", "widget": "radio",
                      "choices": ["sun", "rain"]}],
            "images": {"a.jpg": {"frame": {"weather": ["rain"]}},
                       "b.jpg": {"frame": {"weather": ["sun"]}}},
        })

        weather = self._availability()["declared"][0]

        self.assertEqual(weather["state"], "answered")
        self.assertIn("2", weather["detail"])

    def test_a_tag_defined_after_the_last_sync_says_so(self):
        self._tag("weather")
        self._tag("shift")
        self._sidecar({
            "version": 1,
            "tags": [{"name": "weather", "scope": "frame", "widget": "radio",
                      "choices": ["sun", "rain"]}],
            "images": {"a.jpg": {"frame": {"weather": ["rain"]}}},
        })

        states = {tag["name"]: tag["state"] for tag in self._availability()["declared"]}

        self.assertEqual(states["shift"], "unsynced")

    def test_no_sidecar_at_all_is_reported_rather_than_raised(self):
        self._tag("weather")

        weather = self._availability()["declared"][0]

        self.assertEqual(weather["state"], "unknown")
        self.assertIn("synced", weather["detail"])

    def test_box_tags_are_counted_over_boxes_not_images(self):
        self._tag("occluded", scope=AnnotationTag.REGION, choices=("yes", "no"))
        self._sidecar({
            "version": 1,
            "tags": [{"name": "occluded", "scope": "region", "widget": "radio",
                      "choices": ["yes", "no"]}],
            "images": {"a.jpg": {"boxes": [
                {"row": 0, "class_id": 0, "tags": {"occluded": ["yes"]}},
                {"row": 1, "class_id": 0, "tags": {"occluded": ["no"]}},
            ]}},
        })

        occluded = self._availability()["declared"][0]

        self.assertEqual(occluded["scope"], "box-wide")
        self.assertIn("2", occluded["detail"])

    def test_an_unresolvable_label_source_is_a_message_not_an_exception(self):
        # "annotator output" with no annotator chosen: an ordinary state of a
        # half-filled form, which the panel prints rather than 500-ing on.
        availability = tag_availability.availability(
            self.dataset, EvalRun.ANNOTATOR, None)

        self.assertTrue(availability["labels_error"])


class TagPanelEndpointTests(TagAvailabilitySetup):
    def _get(self, **params):
        request = RequestFactory().get("/", {"dataset": self.dataset.pk, **params})
        request.user = mock.Mock(is_active=True, is_staff=True)
        return dataset_tags_view(request)

    def test_it_renders_the_panel_for_a_dataset(self):
        self._tag("weather")

        response = self._get(label_source=EvalRun.SOURCE)

        self.assertEqual(response.status_code, 200)
        html = json.loads(response.content)["html"]
        self.assertIn("weather", html)
        self.assertIn("crowding", html)

    def test_without_a_dataset_it_asks_for_one_instead_of_failing(self):
        request = RequestFactory().get("/", {})
        request.user = mock.Mock(is_active=True, is_staff=True)

        response = dataset_tags_view(request)

        self.assertEqual(response.status_code, 400)
        self.assertIn("dataset", json.loads(response.content)["error"].lower())


class EvaluateFormTests(TagAvailabilitySetup):
    """Both evaluate forms carry the panel's container."""

    def test_the_model_side_form_declares_the_panel(self):
        from training.admin import TrainedModelAdmin
        from training.models import TrainedModel

        admin = TrainedModelAdmin(TrainedModel, AdminSite())
        request = RequestFactory().post("/")
        request.user = mock.Mock(is_active=True, is_staff=True)
        model = TrainedModel.objects.create(
            name="m1", arch="retinanet", checkpoint_path="/tmp/best.pt",
            num_classes=2, classes=["helmet", "person"])

        with mock.patch.object(admin, "message_user"):
            response = admin.evaluate(request, TrainedModel.objects.filter(pk=model.pk))

        self.assertIn("tag_availability_url", response.context_data)
