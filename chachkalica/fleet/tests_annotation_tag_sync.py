"""Annotation tag answers travelling out of Label Studio with the labels.

The sidecar's whole job is a join: an image's frame answers by filename, a
box's answers by its line in the ``.txt`` next to it. These tests are mostly
about that join surviving the things that would break it — an object dropped
between the LS result and the file, a promote, a second annotator.
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

from django.test import TestCase

from fleet.models import AnnotationTag, Annotator, Dataset, FleetSettings, Project
from fleet.reconcile import tag_values, txt_format, validate
from fleet.services import datasets as datasets_svc
from fleet.services import sync as sync_svc


def _rect(region_id, label, x=10.0, y=10.0, w=20.0, h=20.0):
    return {
        "id": region_id, "from_name": "bbox", "to_name": "image",
        "type": "rectanglelabels", "original_width": 640, "original_height": 480,
        "value": {"x": x, "y": y, "width": w, "height": h, "rectanglelabels": [label]},
    }


def _choice(name, values, region_id=None):
    item = {"from_name": name, "to_name": "image", "type": "choices",
            "value": {"choices": list(values)}}
    if region_id is not None:
        item["id"] = region_id
    return item


class TagValueExtractionTests(TestCase):
    """``reconcile.tag_values`` — pure, no Label Studio and no database."""

    specs = [
        {"name": "weather", "scope": "frame", "widget": "radio", "choices": ["sun", "rain"]},
        {"name": "quality", "scope": "frame", "widget": "rating", "max_rating": 5},
        {"name": "notes", "scope": "frame", "widget": "text"},
        {"name": "occluded", "scope": "region", "widget": "checkbox", "choices": ["a", "b"]},
    ]

    def test_each_widget_keeps_its_own_natural_type(self):
        frame, regions = tag_values.collect_answers([
            _choice("weather", ["rain"]),
            {"from_name": "quality", "type": "rating", "value": {"rating": 4}},
            {"from_name": "notes", "type": "textarea", "value": {"text": ["blurry"]}},
        ], self.specs)

        self.assertEqual(frame, {"weather": ["rain"], "quality": 4, "notes": "blurry"})
        self.assertEqual(regions, {})

    def test_scope_comes_from_the_dataset_not_the_payload(self):
        # A frame answer and a region answer are the same kind of result entry;
        # only the tag definition says which control was declared perRegion.
        frame, regions = tag_values.collect_answers([
            _choice("weather", ["sun"], region_id="r1"),
            _choice("occluded", ["a", "b"], region_id="r1"),
        ], self.specs)

        self.assertEqual(frame, {"weather": ["sun"]})
        self.assertEqual(regions, {"r1": {"occluded": ["a", "b"]}})

    def test_unanswered_and_foreign_controls_are_left_out(self):
        frame, regions = tag_values.collect_answers([
            _choice("weather", []),                       # touched, then cleared
            {"from_name": "notes", "type": "textarea", "value": {"text": ["   "]}},
            _choice("sam_point", ["x"]),                   # not one of ours
        ], self.specs)

        self.assertEqual(frame, {})
        self.assertEqual(regions, {})

    def test_a_box_row_is_its_line_in_the_txt_not_its_place_in_the_result(self):
        # The middle region names a class the dataset does not have, so it never
        # reaches the file — and every later box moves up a line with it.
        result = [
            _rect("r1", "person"),
            _rect("r2", "ghost"),
            _rect("r3", "person"),
            _choice("occluded", ["a"], region_id="r3"),
        ]
        _w, _h, objects = txt_format.result_to_image(result, {"person": 0})
        objects, _warnings = validate.clean_objects(objects, 1)
        frame, regions = tag_values.collect_answers(result, self.specs)

        entry = tag_values.image_entry(objects, frame, regions)

        self.assertEqual(len(entry["boxes"]), 2)
        self.assertEqual(entry["boxes"][1], {"row": 1, "class_id": 0, "tags": {"occluded": ["a"]}})

    def test_an_image_nobody_answered_anything_on_is_omitted(self):
        _w, _h, objects = txt_format.result_to_image([_rect("r1", "person")], {"person": 0})
        self.assertIsNone(tag_values.image_entry(objects, {}, {}))


class SyncWritesTagSidecarTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.src = root / "source"
        self.tgt = root / "target"
        (self.src / "ds").mkdir(parents=True)
        (self.src / "ds" / "classes.txt").write_text("# tools: bbox\nperson\n", encoding="utf-8")
        self.tgt.mkdir()
        fs = FleetSettings.load()
        fs.source_dir = str(self.src)
        fs.target_dir = str(self.tgt)
        fs.save()

        self.dataset = Dataset.objects.create(name="ds", storage_type=Dataset.LOCAL)
        self.annotator = Annotator.objects.create(username="ann1", token="t", port=9000)
        self.project = Project.objects.create(
            annotator=self.annotator, dataset=self.dataset, ls_project_id=7, title="ds — ann1",
        )
        self.labels_dir = self.tgt / "ds" / "ann1"

    def tearDown(self):
        self.tmp.cleanup()

    def _sync(self, tasks):
        with mock.patch.object(sync_svc.lsapi, "fetch_project_annotations", return_value=tasks):
            return sync_svc.sync_project(self.project)

    def _task(self, filename, result):
        return {"data": {"image": f"/data/local-files/?d=source/ds/{filename}"},
                "annotations": [{"id": 1, "result": result}]}

    def test_no_tags_on_the_dataset_writes_no_sidecar(self):
        result = self._sync([self._task("a.jpg", [_rect("r1", "person")])])

        self.assertEqual(result["tagged_images"], 0)
        self.assertFalse(tag_values.tags_path(self.labels_dir).exists())

    def test_answers_land_beside_the_labels_they_describe(self):
        AnnotationTag.objects.create(
            dataset=self.dataset, scope=AnnotationTag.FRAME, name="weather",
            widget=AnnotationTag.RADIO, choices=["sun", "rain"],
        )
        AnnotationTag.objects.create(
            dataset=self.dataset, scope=AnnotationTag.REGION, name="occluded",
            widget=AnnotationTag.CHECKBOX, choices=["partial", "heavy"],
        )

        result = self._sync([
            self._task("a.jpg", [
                _rect("r1", "person"),
                _rect("r2", "person", x=40.0),
                _choice("weather", ["rain"]),
                _choice("occluded", ["heavy"], region_id="r2"),
            ]),
            self._task("b.jpg", [_rect("r9", "person")]),
        ])

        self.assertEqual(result["tagged_images"], 1)
        document = json.loads(tag_values.tags_path(self.labels_dir).read_text())
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["annotator"], "ann1")
        self.assertEqual([tag["name"] for tag in document["tags"]], ["weather", "occluded"])
        # b.jpg had boxes but no answers, so it is not in the document at all.
        self.assertEqual(list(document["images"]), ["a.jpg"])

        entry = document["images"]["a.jpg"]
        self.assertEqual(entry["frame"], {"weather": ["rain"]})
        self.assertEqual(entry["boxes"], [
            {"row": 0, "class_id": 0, "tags": {}},
            {"row": 1, "class_id": 0, "tags": {"occluded": ["heavy"]}},
        ])

        # And the rows really are the lines of the file next to it.
        lines = (self.labels_dir / "a.txt").read_text(encoding="utf-8").splitlines()[1:]
        self.assertEqual(len(lines), 2)

    def test_removing_every_tag_clears_a_stale_sidecar(self):
        tag = AnnotationTag.objects.create(
            dataset=self.dataset, scope=AnnotationTag.FRAME, name="weather",
            widget=AnnotationTag.RADIO, choices=["sun"],
        )
        self._sync([self._task("a.jpg", [_rect("r1", "person"), _choice("weather", ["sun"])])])
        self.assertTrue(tag_values.tags_path(self.labels_dir).exists())

        tag.delete()
        self._sync([self._task("a.jpg", [_rect("r1", "person")])])

        self.assertFalse(tag_values.tags_path(self.labels_dir).exists())


class PromoteCarriesTagsTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.src = root / "source"
        self.tgt = root / "target"
        (self.src / "ds").mkdir(parents=True)
        (self.src / "ds" / "classes.txt").write_text("# tools: bbox\nperson\n", encoding="utf-8")
        self.tgt.mkdir()
        fs = FleetSettings.load()
        fs.source_dir = str(self.src)
        fs.target_dir = str(self.tgt)
        fs.save()
        self.dataset = Dataset.objects.create(name="ds", storage_type=Dataset.LOCAL)

    def tearDown(self):
        self.tmp.cleanup()

    def _annotator_with_labels(self, username, tags: dict | None):
        annotator = Annotator.objects.create(username=username, token="t")
        directory = self.tgt / "ds" / username
        directory.mkdir(parents=True)
        (directory / "a.jpg.txt").write_text("640 480\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        if tags is not None:
            tag_values.tags_path(directory).write_text(json.dumps(tags), encoding="utf-8")
        return annotator

    def test_the_sidecar_follows_the_labels_into_source(self):
        annotator = self._annotator_with_labels("ann1", {"version": 1, "images": {"a.jpg": {}}})

        result = datasets_svc.promote_annotator_labels(self.dataset, annotator)

        self.assertTrue(result["tags_promoted"])
        promoted = tag_values.tags_path(self.src / "ds" / "labels")
        self.assertEqual(json.loads(promoted.read_text())["images"], {"a.jpg": {}})
        # Copied, not moved: the annotator's own output keeps describing itself.
        self.assertTrue(tag_values.tags_path(self.tgt / "ds" / "ann1").exists())

    def test_promoting_an_untagged_annotator_removes_the_previous_sidecar(self):
        first = self._annotator_with_labels("ann1", {"version": 1, "images": {"a.jpg": {}}})
        datasets_svc.promote_annotator_labels(self.dataset, first)
        second = self._annotator_with_labels("ann2", None)

        result = datasets_svc.promote_annotator_labels(self.dataset, second)

        # ann1's rows describe ann1's boxes; leaving them next to ann2's labels
        # would attribute answers to whatever box happens to be on that line.
        self.assertFalse(result["tags_promoted"])
        self.assertFalse(tag_values.tags_path(self.src / "ds" / "labels").exists())
