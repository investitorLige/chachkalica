"""Measuring a dataset's frames, and slicing an eval by the result.

The numbers here are chosen so a failure says which step drifted: a flat grey
frame has a known mean and no spread at all, a half-black/half-white one has a
known mean and the full spread.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from django.test import TestCase

from fleet.models import Dataset, DatasetMeasureRun, FleetSettings
from fleet.services import measures
from training.services import tag_analytics


class MeasuresSetup(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.source = root / "source"
        self.images = self.source / "ds1" / "images"
        self.images.mkdir(parents=True)
        fs = FleetSettings.load()
        fs.source_dir = str(self.source)
        fs.target_dir = str(root / "target")
        fs.save()
        self.dataset = Dataset.objects.create(name="ds1")

    def tearDown(self):
        self._tmp.cleanup()

    def _flat(self, name, level, size=64):
        path = self.images / name
        cv2.imwrite(str(path), np.full((size, size), level, dtype=np.uint8))
        return path

    def _split(self, name, size=64):
        """Half black, half white: mean 127.5, full percentile spread."""
        frame = np.zeros((size, size), dtype=np.uint8)
        frame[:, size // 2:] = 255
        path = self.images / name
        cv2.imwrite(str(path), frame)
        return path


class MeasureImageTests(MeasuresSetup):
    def test_a_flat_frame_has_its_own_level_and_no_spread(self):
        result = measures.measure_image(self._flat("grey.png", 120))

        self.assertAlmostEqual(result["brightness"], 120.0, places=3)
        self.assertAlmostEqual(result["contrast"], 0.0, places=3)

    def test_a_half_black_half_white_frame_spans_the_whole_range(self):
        result = measures.measure_image(self._split("split.png"))

        self.assertAlmostEqual(result["brightness"], 127.5, delta=1.0)
        self.assertGreater(result["contrast"], 250)

    def test_the_frame_size_comes_back_with_it(self):
        result = measures.measure_image(self._flat("sized.png", 10, size=48))

        self.assertEqual((result["width"], result["height"]), (48, 48))

    def test_subsampling_a_large_frame_preserves_the_statistics(self):
        # This is the test that documents why the pass strides rather than
        # resizes: a stride is an unbiased sample of the pixel distribution, so
        # the numbers survive it. An area-resize would shrink the spread by an
        # amount that depends on the source resolution, quietly turning
        # "contrast" into a partial proxy for image size.
        small = measures.measure_image(self._split("small.png", size=64))
        large = measures.measure_image(self._split("large.png", size=2048))

        self.assertAlmostEqual(small["brightness"], large["brightness"], delta=2.0)
        self.assertAlmostEqual(small["contrast"], large["contrast"], delta=2.0)

    def test_an_unreadable_file_is_none_rather_than_an_exception(self):
        broken = self.images / "broken.png"
        broken.write_bytes(b"not an image")

        self.assertIsNone(measures.measure_image(broken))


class MeasureDatasetTests(MeasuresSetup):
    def test_it_measures_every_frame_and_records_what_it_skipped(self):
        self._flat("a.png", 20)
        self._flat("b.png", 200)
        (self.images / "c.png").write_bytes(b"not an image")
        run = DatasetMeasureRun.objects.create(dataset=self.dataset)

        result = measures.measure_dataset(self.dataset, run=run)

        self.assertEqual(result["images"], 3)
        self.assertEqual(result["measured"], 2)
        self.assertEqual(result["skipped"], 1)
        run.refresh_from_db()
        self.assertEqual(run.images_processed, 3)
        self.assertEqual(run.images_failed, 1)

    def test_nested_frames_are_measured_too(self):
        # The trainer's loader recurses; a flat walk here would leave every
        # nested frame unmeasured and silently unbucketed.
        (self.images / "cam1").mkdir()
        self._flat("cam1/nested.png", 90)
        self._flat("top.png", 90)

        result = measures.measure_dataset(self.dataset)

        self.assertEqual(result["measured"], 2)

    def test_a_basename_under_two_directories_is_named_as_ambiguous(self):
        (self.images / "cam1").mkdir()
        (self.images / "cam2").mkdir()
        self._flat("cam1/frame.png", 10)
        self._flat("cam2/frame.png", 240)

        measures.measure_dataset(self.dataset)
        document = measures.load("ds1")

        self.assertEqual(document["basename_collisions"], ["frame.png"])

    def test_a_changed_dataset_reads_as_stale_not_wrong(self):
        self._flat("a.png", 20)
        measures.measure_dataset(self.dataset)
        self.assertEqual(measures.freshness("ds1", measures.load("ds1"))[0], "ok")

        self._flat("b.png", 30)

        state, detail = measures.freshness("ds1", measures.load("ds1"))
        self.assertEqual(state, "stale")
        self.assertIn("Measure image statistics", detail)

    def test_a_dataset_never_measured_says_which_action_measures_it(self):
        state, detail = measures.freshness("ds1", measures.load("ds1"))

        self.assertEqual(state, "missing")
        self.assertIn("Measure image statistics", detail)

    def test_a_dataset_with_no_images_says_so_instead_of_writing_nothing(self):
        with self.assertRaises(FileNotFoundError):
            measures.measure_dataset(self.dataset)


class MeasureSliceTests(MeasuresSetup):
    """The end of the loop: measurements become slices of an eval."""

    def _table(self, names):
        return {
            "version": 2, "iou_thresholds": [0.5], "score_threshold": 0.25,
            "score_floor": 0.001, "operating_nms_threshold": None,
            "classes": {"0": "person"},
            "images": names,
            "gt": {"image": list(range(len(names))), "class": [0] * len(names),
                   "row": [0] * len(names),
                   "w": [0.2] * len(names), "h": [0.2] * len(names)},
            "pred": {"image": [0], "class": [0], "score": [0.9], "iou": [0.9], "gt": [0]},
            "pred_operating": None,
        }

    def test_brightness_becomes_a_slice_of_the_eval(self):
        for index, level in enumerate([10, 20, 120, 130, 240, 250]):
            self._flat(f"f{index}.png", level)
        measures.measure_dataset(self.dataset)
        names = [f"f{i}.png" for i in range(6)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval_matches.json"
            path.write_text(json.dumps(self._table(names)), encoding="utf-8")
            table = tag_analytics.load_table(path)
        index = tag_analytics.build_tag_index(
            table, measures=measures.load("ds1"))

        brightness = index.tag(tag_analytics.BRIGHTNESS_TAG)
        self.assertEqual(brightness.status, "ok")
        self.assertEqual(brightness.source, "auto")
        # Six frames in three brightness pairs, so one pair per bucket.
        self.assertEqual([cut["count"] for cut in brightness.cuts], [2, 2, 2])

    def test_without_a_sidecar_the_tag_names_the_action_that_makes_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval_matches.json"
            path.write_text(json.dumps(self._table(["a.png"])), encoding="utf-8")
            table = tag_analytics.load_table(path)
        index = tag_analytics.build_tag_index(table)

        brightness = index.tag(tag_analytics.BRIGHTNESS_TAG)
        self.assertEqual(brightness.status, "unavailable")
        self.assertIn("Measure image statistics", brightness.status_detail)

    def test_frames_the_sidecar_does_not_know_are_unmeasured_not_guessed(self):
        self._flat("known.png", 40)
        self._flat("other.png", 200)
        measures.measure_dataset(self.dataset)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval_matches.json"
            path.write_text(
                json.dumps(self._table(["known.png", "other.png", "ghost.png"])),
                encoding="utf-8")
            table = tag_analytics.load_table(path)
        index = tag_analytics.build_tag_index(
            table, measures=measures.load("ds1"))

        brightness = index.tag(tag_analytics.BRIGHTNESS_TAG)
        self.assertEqual(brightness.counts[tag_analytics.UNMEASURED], 1)

    def test_an_ambiguous_basename_is_counted_rather_than_attributed(self):
        (self.images / "cam1").mkdir()
        (self.images / "cam2").mkdir()
        self._flat("cam1/frame.png", 10)
        self._flat("cam2/frame.png", 240)
        measures.measure_dataset(self.dataset)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval_matches.json"
            path.write_text(json.dumps(self._table(["frame.png"])), encoding="utf-8")
            table = tag_analytics.load_table(path)
        index = tag_analytics.build_tag_index(
            table, measures=measures.load("ds1"))

        self.assertEqual(index.ambiguous_filenames, 1)


class BoxMeasureRequestTests(MeasuresSetup):
    """The request the trainer is handed for a box-measure pass."""

    def setUp(self):
        super().setUp()
        (self.source / "ds1" / "labels").mkdir(parents=True, exist_ok=True)
        (self.source / "ds1" / "classes.txt").write_text("helmet\nperson\n", encoding="utf-8")
        self._flat("a.png", 100)
        self.bundle = Path(self._tmp.name) / "bundles"
        engine = self.bundle / "rtmo-posture-bundle" / "models"
        engine.mkdir(parents=True)
        (engine / "rtmo.engine").write_bytes(b"stub")
        from training.models import TrainingSettings
        ts = TrainingSettings.load()
        ts.bundles_root = str(self.bundle)
        # Point the generated-config roots at the fixture too: the real ones are
        # owned by the containers that created them and are not writable here.
        ts.configs_root = str(Path(self._tmp.name) / "configs")
        ts.runs_root = str(Path(self._tmp.name) / "runs")
        ts.save()

    def _run(self, **kwargs):
        from training.models import EvalRun
        return DatasetMeasureRun.objects.create(
            dataset=self.dataset, kind=DatasetMeasureRun.BOX,
            label_source=kwargs.pop("label_source", EvalRun.SOURCE), **kwargs)

    def test_a_dataset_labelling_people_uses_its_own_boxes(self):
        from training.services import config_gen

        _path, text = config_gen.write_measures_request(self._run())

        self.assertIn("kind: ground_truth", text)
        self.assertIn("rtmo.engine", text)

    def test_a_dataset_without_a_person_class_falls_back_to_the_posture_engine(self):
        (self.source / "ds1" / "classes.txt").write_text("helmet\nvest\n", encoding="utf-8")
        from training.services import config_gen

        _path, text = config_gen.write_measures_request(self._run())

        self.assertIn("kind: rtmo", text)

    def test_the_request_records_where_it_will_write(self):
        from training.services import config_gen

        run = self._run()
        config_gen.write_measures_request(run)
        run.refresh_from_db()

        self.assertTrue(run.labels_dir_snapshot.endswith("ds1/labels"))
        self.assertEqual(run.person_source, "ground_truth")

    def test_a_pass_with_no_label_source_is_refused_with_the_reason(self):
        from training.models import EvalRun
        from training.services import config_gen

        run = self._run(label_source=EvalRun.NONE)

        with self.assertRaises(ValueError) as caught:
            config_gen.write_measures_request(run)

        self.assertIn("label source", str(caught.exception))

    def test_a_missing_posture_engine_says_so_rather_than_failing_in_the_trainer(self):
        (self.bundle / "rtmo-posture-bundle" / "models" / "rtmo.engine").unlink()
        from training.services import config_gen

        with self.assertRaises(FileNotFoundError) as caught:
            config_gen.write_measures_request(self._run())

        self.assertIn("rtmo-posture-bundle", str(caught.exception))
