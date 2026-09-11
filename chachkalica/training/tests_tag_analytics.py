"""Scoring an eval over the slices its annotation tags define.

The numbers in these tests are worked out by hand from a four-image table, so a
failure says which step of the accumulation drifted rather than only that two
implementations disagree. Parity with the trainer's own ``evaluate_detection``
over random data is checked separately, where torch is available (see
``docs/annotation-tags.md``).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from django.test import TestCase

from training.services import tag_analytics


# Four images, one class, one IoU threshold. Ground truth, in order:
#   gt0  img0 row0   found by a 0.90-confidence box at IoU 0.90
#   gt1  img0 row1   found by a 0.80-confidence box at IoU 0.80
#   gt2  img1 row0   a 0.70-confidence box lands nearby (IoU 0.10) but misses it
#   gt3  img2 row0   nothing predicted at all
#   gt4  img3 row0   found by a 0.30-confidence box at IoU 0.60
def _table(**overrides) -> dict:
    table = {
        "version": 1,
        "iou_thresholds": [0.5],
        "score_threshold": 0.25,
        "score_floor": 0.001,
        "operating_nms_threshold": None,
        "classes": {"0": "person"},
        "images": ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"],
        "gt": {"image": [0, 0, 1, 2, 3], "class": [0, 0, 0, 0, 0], "row": [0, 1, 0, 0, 0]},
        "pred": {
            "image": [0, 0, 1, 3],
            "class": [0, 0, 0, 0],
            "score": [0.9, 0.8, 0.7, 0.3],
            "iou": [0.9, 0.8, 0.1, 0.6],
            "gt": [0, 1, 2, 4],
        },
        "pred_operating": None,
    }
    table.update(overrides)
    return table


def _tags(**overrides) -> dict:
    document = {
        "version": 1,
        "dataset": "ds",
        "annotator": "ann1",
        "tags": [
            {"name": "weather", "scope": "frame", "widget": "radio",
             "choices": ["sun", "rain"], "max_rating": 5},
            {"name": "occluded", "scope": "region", "widget": "radio",
             "choices": ["yes", "no"], "max_rating": 5},
        ],
        "images": {
            "img0.jpg": {"frame": {"weather": ["rain"]},
                         "boxes": [{"row": 0, "class_id": 0, "tags": {"occluded": ["no"]}},
                                   {"row": 1, "class_id": 0, "tags": {"occluded": ["yes"]}}]},
            "img1.jpg": {"frame": {"weather": ["rain"]},
                         "boxes": [{"row": 0, "class_id": 0, "tags": {"occluded": ["no"]}}]},
            "img2.jpg": {"frame": {"weather": ["sun"]},
                         "boxes": [{"row": 0, "class_id": 0, "tags": {"occluded": ["yes"]}}]},
            "img3.jpg": {"frame": {"weather": ["sun"]}},
        },
    }
    document.update(overrides)
    return document


class TagAnalyticsSetup(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def load(self, table=None, tags=None):
        path = self.root / "eval_matches.json"
        path.write_text(json.dumps(table or _table()), encoding="utf-8")
        loaded = tag_analytics.load_table(path)
        return loaded, tag_analytics.build_tag_index(loaded, tags or _tags())


class UnfilteredTests(TagAnalyticsSetup):
    def test_the_whole_table_scores_to_the_hand_computed_numbers(self):
        table, _index = self.load()

        metrics = tag_analytics.image_slice_metrics(table, np.ones(4, dtype=bool))

        # Three of four predictions win a box; five boxes exist.
        self.assertEqual(metrics["predictions"], 4)
        self.assertEqual(metrics["gt"], 5)
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 0.6)
        # Precision envelope 1, 1, .75 over recall steps .2, .2, .2.
        self.assertAlmostEqual(metrics["map50"], 0.55)

    def test_the_page_reports_agreement_with_the_evals_own_metrics(self):
        table, index = self.load()

        report = tag_analytics.report(
            table, index, {"map50": 0.55, "precision": 0.75, "recall": 0.6})

        self.assertTrue(report["check"]["agrees"])

    def test_a_table_built_with_other_thresholds_is_called_out(self):
        table, index = self.load()

        report = tag_analytics.report(table, index, {"map50": 0.91})

        self.assertFalse(report["check"]["agrees"])


class FrameTagTests(TagAnalyticsSetup):
    def test_a_frame_value_is_scored_as_its_own_little_dataset(self):
        table, index = self.load()

        rain = tag_analytics.slice_metrics(table, index, [("weather", "rain")])

        self.assertEqual(rain["kind"], "frame")
        self.assertEqual(rain["images"], 2)
        self.assertEqual(rain["gt"], 3)
        self.assertEqual(rain["predictions"], 3)
        self.assertAlmostEqual(rain["precision"], 2 / 3)
        self.assertAlmostEqual(rain["recall"], 2 / 3)
        self.assertAlmostEqual(rain["map50"], 2 / 3)

    def test_the_breakdown_measures_each_value_against_the_whole_dataset(self):
        table, index = self.load()
        definition = index.tag("weather")

        breakdown = tag_analytics.tag_breakdown(table, index, definition)

        rows = {row["value"]: row for row in breakdown["rows"]}
        self.assertEqual(set(rows), {"sun", "rain"})
        self.assertAlmostEqual(rows["rain"]["delta"], 2 / 3 - 0.55)
        self.assertLess(rows["sun"]["delta"], 0)
        # Declared order wins over observed frequency, so the reader sees the
        # options in the order the annotator saw them.
        self.assertEqual([row["value"] for row in breakdown["rows"]], ["sun", "rain"])

    def test_images_nobody_answered_become_their_own_named_group(self):
        tags = _tags()
        del tags["images"]["img3.jpg"]
        table, index = self.load(tags=tags)

        unanswered = tag_analytics.slice_metrics(
            table, index, [("weather", tag_analytics.UNANSWERED)])

        self.assertEqual(unanswered["images"], 1)


class BoxTagTests(TagAnalyticsSetup):
    def test_a_box_value_is_scored_on_whether_its_boxes_were_found(self):
        table, index = self.load()

        occluded = tag_analytics.slice_metrics(table, index, [("occluded", "yes")])

        # gt1 (claimed at 0.80) and gt3 (never predicted).
        self.assertEqual(occluded["kind"], "box")
        self.assertEqual(occluded["gt"], 2)
        self.assertEqual(occluded["found"], 1)
        self.assertEqual(occluded["missed"], 1)
        self.assertAlmostEqual(occluded["recall"], 0.5)
        self.assertAlmostEqual(occluded["mean_iou"], 0.8)
        self.assertAlmostEqual(occluded["mean_score"], 0.8)

    def test_a_box_slice_offers_no_precision_or_map(self):
        table, index = self.load()

        occluded = tag_analytics.slice_metrics(table, index, [("occluded", "yes")])

        self.assertNotIn("map50", occluded)
        self.assertNotIn("precision", occluded)

    def test_a_box_found_only_below_the_operating_confidence_counts_as_missed(self):
        table, index = self.load()

        # img3 answered no box tags, so its gt4 falls in the unanswered group.
        # Its only claimant scores 0.30: found at the eval's 0.25 operating
        # point, and not found at 0.5.
        clause = [("occluded", tag_analytics.UNANSWERED)]
        self.assertEqual(tag_analytics.slice_metrics(table, index, clause)["found"], 1)

        table.score_threshold = 0.5
        table.claims.clear()
        self.assertEqual(tag_analytics.slice_metrics(table, index, clause)["found"], 0)

    def test_boxes_nobody_answered_for_are_a_group_rather_than_dropped(self):
        table, index = self.load()

        # "The model is worse on the boxes nobody tagged" is a finding about the
        # annotation job; excluding them silently would hide it.
        answered = tag_analytics.slice_metrics(table, index, [("occluded", "no")])
        unanswered = tag_analytics.slice_metrics(
            table, index, [("occluded", tag_analytics.UNANSWERED)])

        self.assertEqual(answered["gt"], 2)      # gt0, gt2
        self.assertEqual(answered["found"], 1)   # gt2's nearest box misses at IoU 0.1
        self.assertEqual(unanswered["gt"], 1)    # gt4, on the untagged image


class CrossUnionTests(TagAnalyticsSetup):
    def test_a_frame_clause_narrows_the_boxes_a_box_clause_selects(self):
        table, index = self.load()

        both = tag_analytics.slice_metrics(
            table, index, [("weather", "rain"), ("occluded", "yes")])

        # Only gt1 is both occluded and in a rainy frame.
        self.assertEqual(both["kind"], "box")
        self.assertEqual(both["gt"], 1)
        self.assertEqual(both["found"], 1)

    def test_two_frame_clauses_keep_the_full_metric_set(self):
        tags = _tags()
        tags["tags"].append({"name": "shift", "scope": "frame", "widget": "radio",
                             "choices": ["day", "night"], "max_rating": 5})
        for name in ("img0.jpg", "img1.jpg"):
            tags["images"][name]["frame"]["shift"] = ["night"]
        table, index = self.load(tags=tags)

        both = tag_analytics.slice_metrics(
            table, index, [("weather", "rain"), ("shift", "night")])

        self.assertEqual(both["kind"], "frame")
        self.assertEqual(both["images"], 2)
        self.assertAlmostEqual(both["map50"], 2 / 3)

    def test_an_impossible_combination_is_an_empty_slice_not_an_error(self):
        table, index = self.load()

        empty = tag_analytics.slice_metrics(
            table, index, [("weather", "sun"), ("weather", "rain")])

        self.assertEqual(empty["images"], 0)
        self.assertEqual(empty["gt"], 0)
        self.assertEqual(empty["map50"], 0.0)

    def test_a_cross_tab_fills_every_combination_of_two_tags(self):
        table, index = self.load()

        grid = tag_analytics.cross_tab(table, index, "weather", "occluded", "recall")

        self.assertEqual(grid["kind"], "box")
        # img3's boxes were never answered, so the grid carries that column too.
        self.assertEqual(grid["columns"], ["yes", "no", tag_analytics.UNANSWERED])
        cells = {(row["label"], cell["column"]): cell
                 for row in grid["rows"] for cell in row["cells"]}
        self.assertAlmostEqual(cells[("rain", "yes")]["value"], 1.0)
        self.assertAlmostEqual(cells[("sun", "yes")]["value"], 0.0)

    def test_a_metric_the_slice_kind_cannot_answer_falls_back(self):
        table, index = self.load()

        grid = tag_analytics.cross_tab(table, index, "weather", "occluded", "map50")

        self.assertEqual(grid["metric"], "recall")

    def test_a_value_nothing_carries_is_refused_rather_than_scored_as_empty(self):
        table, index = self.load()

        with self.assertRaises(tag_analytics.UnknownClause):
            tag_analytics.slice_metrics(table, index, [("weather", "hail")])


class JoinIntegrityTests(TagAnalyticsSetup):
    def test_box_tags_are_dropped_when_the_rows_no_longer_describe_the_labels(self):
        tags = _tags()
        # The labels were re-promoted and img0's second box is a different class
        # now — its row means something else than when the tags were answered.
        tags["images"]["img0.jpg"]["boxes"][1]["class_id"] = 7
        table, index = self.load(tags=tags)

        self.assertEqual(index.row_mismatches, 1)
        occluded = tag_analytics.slice_metrics(table, index, [("occluded", "yes")])
        self.assertEqual(occluded["gt"], 1)  # only img2's box survives
        # The frame answer on the same image is untouched by a box-level drift.
        rain = tag_analytics.slice_metrics(table, index, [("weather", "rain")])
        self.assertEqual(rain["images"], 2)

    def test_tagged_images_the_eval_never_saw_are_counted_not_joined(self):
        tags = _tags()
        tags["images"]["elsewhere.jpg"] = {"frame": {"weather": ["sun"]}}
        table, index = self.load(tags=tags)

        self.assertEqual(index.unmatched_images, 1)
        self.assertEqual(index.matched_images, 4)

    def test_a_future_table_version_is_refused_rather_than_misread(self):
        path = self.root / "eval_matches.json"
        path.write_text(json.dumps(_table(version=2)), encoding="utf-8")

        with self.assertRaises(ValueError):
            tag_analytics.load_table(path)


class ArtifactDiscoveryTests(TagAnalyticsSetup):
    def test_either_evals_naming_of_the_artifact_is_found(self):
        (self.root / "predictions_matches.json").write_text("{}", encoding="utf-8")

        found = tag_analytics.match_table_artifact(self.root)

        self.assertEqual(found.name, "predictions_matches.json")

    def test_a_directory_without_one_is_simply_none(self):
        self.assertIsNone(tag_analytics.match_table_artifact(self.root))
        self.assertIsNone(tag_analytics.match_table_artifact(""))
