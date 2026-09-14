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


def _table_v2(**overrides) -> dict:
    """The same table, at the version that also records each box's extent."""
    table = _table(version=2)
    table["gt"] = {
        **table["gt"],
        "w": [0.1, 0.2, 0.3, 0.4, 0.5],
        "h": [0.1, 0.2, 0.3, 0.4, 0.5],
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
        path.write_text(json.dumps(_table(version=3)), encoding="utf-8")

        with self.assertRaises(ValueError):
            tag_analytics.load_table(path)

    def test_the_version_that_carries_box_geometry_is_read(self):
        table, _index = self.load(table=_table_v2())

        self.assertEqual(table.version, 2)
        self.assertTrue(table.has_geometry)

    def test_a_version_one_table_reads_with_the_geometry_simply_unknown(self):
        table, _index = self.load()

        self.assertEqual(table.version, 1)
        self.assertFalse(table.has_geometry)
        self.assertEqual(table.gt_w.size, table.num_gt)

    def test_a_truncated_ground_truth_column_is_refused_not_zipped_past(self):
        # A short column would silently pair a class with another box's row and
        # mis-tag every box after it, which no later check would notice.
        broken = _table()
        broken["gt"]["row"] = broken["gt"]["row"][:-1]
        path = self.root / "eval_matches.json"
        path.write_text(json.dumps(broken), encoding="utf-8")

        with self.assertRaises(ValueError) as caught:
            tag_analytics.load_table(path)

        self.assertIn("gt.row", str(caught.exception))


class ArtifactDiscoveryTests(TagAnalyticsSetup):
    def test_either_evals_naming_of_the_artifact_is_found(self):
        (self.root / "predictions_matches.json").write_text("{}", encoding="utf-8")

        found = tag_analytics.match_table_artifact(self.root)

        self.assertEqual(found.name, "predictions_matches.json")

    def test_a_directory_without_one_is_simply_none(self):
        self.assertIsNone(tag_analytics.match_table_artifact(self.root))
        self.assertIsNone(tag_analytics.match_table_artifact(""))


class TercileBucketTests(TestCase):
    """Cutting a measure into low/medium/high without inventing empty rows."""

    def test_a_spiky_discrete_measure_still_yields_three_populated_buckets(self):
        # 80% of frames hold exactly one box, so the 33rd and 66th percentiles
        # are both 1.0 and a naive tercile leaves the middle bucket empty.
        result = tag_analytics.tercile_buckets(
            np.array([1.0] * 80 + [2.0] * 10 + [3.0] * 10))

        counts = {cut["value"]: cut["count"] for cut in result["cuts"]}
        self.assertEqual(counts, {"low": 80, "medium": 10, "high": 10})

    def test_a_continuous_measure_cuts_where_the_percentiles_do(self):
        values = np.linspace(0.0, 1.0, 3000)

        result = tag_analytics.tercile_buckets(values)

        self.assertAlmostEqual(result["cuts"][0]["max"],
                               float(np.quantile(values, 1 / 3)), places=2)
        self.assertAlmostEqual(result["cuts"][1]["max"],
                               float(np.quantile(values, 2 / 3)), places=2)

    def test_a_constant_measure_says_so_rather_than_showing_one_full_width_row(self):
        result = tag_analytics.tercile_buckets(np.full(50, 4.0))

        self.assertEqual(result["degenerate"], "constant")
        self.assertEqual(len(result["cuts"]), 1)

    def test_two_distinct_values_give_two_buckets_not_an_empty_middle(self):
        result = tag_analytics.tercile_buckets(np.array([1.0] * 30 + [7.0] * 70))

        self.assertEqual([cut["value"] for cut in result["cuts"]], ["low", "high"])
        self.assertTrue(all(cut["count"] for cut in result["cuts"]))

    def test_entries_with_no_value_are_left_unbucketed(self):
        result = tag_analytics.tercile_buckets(
            np.array([1.0, 5.0, 9.0, 0.0]),
            np.array([True, True, True, False]),
        )

        self.assertEqual(int(result["assignment"][3]), -1)


class AutomaticTagTests(TagAnalyticsSetup):
    """Slices nobody had to annotate."""

    def _bare(self, table=None):
        path = self.root / "eval_matches.json"
        path.write_text(json.dumps(table or _table()), encoding="utf-8")
        loaded = tag_analytics.load_table(path)
        return loaded, tag_analytics.build_tag_index(loaded)

    def test_crowding_works_with_no_tag_sidecar_at_all(self):
        _table_obj, index = self._bare()

        crowding = index.tag("auto:crowding")

        self.assertEqual(crowding.source, "auto")
        # img0 holds two boxes; the other three hold one each.
        self.assertEqual(crowding.counts["low"], 3)
        self.assertEqual(crowding.counts["high"], 1)

    def test_a_crowding_slice_scores_the_images_it_names(self):
        table, index = self._bare()

        busiest = tag_analytics.slice_metrics(table, index, [("auto:crowding", "high")])

        self.assertEqual(busiest["images"], 1)
        self.assertEqual(busiest["gt"], 2)

    def test_box_size_says_which_action_unlocks_it_when_geometry_is_missing(self):
        _table_obj, index = self.load()

        box_size = index.tag("auto:box size")

        self.assertEqual(box_size.status, "unavailable")
        self.assertFalse(box_size.usable)
        self.assertIn("Build tag analytics data", box_size.status_detail)

    def test_box_size_buckets_every_box_once_the_table_carries_geometry(self):
        _table_obj, index = self.load(table=_table_v2())

        box_size = index.tag("auto:box size")

        self.assertEqual(box_size.status, "ok")
        self.assertEqual(sum(box_size.counts.values()), 5)
        self.assertTrue(box_size.cuts)

    def test_the_agreement_check_still_holds_with_computed_tags_present(self):
        table, index = self.load(table=_table_v2())

        report = tag_analytics.report(
            table, index, {"map50": 0.55, "precision": 0.75, "recall": 0.6})

        self.assertTrue(report["check"]["agrees"])


class EveryTagIsListedTests(TagAnalyticsSetup):
    """What could this eval have been sliced by — including the answer 'nothing'."""

    _SHIFT = [{"name": "shift", "scope": "frame", "widget": "radio",
               "choices": ["day", "night"], "max_rating": 5}]

    def _with_choices(self, *choices):
        tags = _tags()
        tags["tags"][0]["choices"] = list(choices)
        return self.load(tags=tags)

    def test_a_declared_option_nobody_picked_is_a_zero_row_not_a_missing_one(self):
        _table_obj, index = self._with_choices("sun", "rain", "hail")

        weather = index.tag("weather")

        self.assertIn("hail", weather.values)
        self.assertEqual(weather.counts["hail"], 0)

    def test_a_zero_row_can_still_be_scored_rather_than_refused(self):
        table, index = self._with_choices("sun", "rain", "hail")

        empty = tag_analytics.slice_metrics(table, index, [("weather", "hail")])

        self.assertEqual(empty["images"], 0)

    def test_a_tag_defined_on_the_dataset_but_never_synced_is_listed_with_why(self):
        table, _index = self.load()
        index = tag_analytics.build_tag_index(
            table, _tags(), declared_specs=self._SHIFT)

        shift = index.tag("shift")

        self.assertEqual(shift.status, "unsynced")
        self.assertFalse(shift.usable)
        self.assertIn("sync", shift.status_detail.lower())

    def test_a_tag_with_no_data_is_reported_rather_than_dropped(self):
        table, _index = self.load()
        index = tag_analytics.build_tag_index(
            table, _tags(), declared_specs=self._SHIFT)

        report = tag_analytics.report(table, index)

        self.assertIn("shift", {e["name"] for e in report["tags_without_data"]})
        self.assertNotIn("shift", {b["name"] for b in report["breakdowns"]})

    def test_a_value_neither_declared_nor_observed_is_still_refused(self):
        table, index = self.load()

        with self.assertRaises(tag_analytics.UnknownClause):
            tag_analytics.slice_metrics(table, index, [("weather", "hail")])

    def test_a_frame_slice_with_no_ground_truth_reports_undefined_not_zero(self):
        # Empty frames are real and attract false positives, so the row stays --
        # but a mean over no boxes is 0.0, which would read as a catastrophic
        # mAP rather than an undefined one.
        only_img0_has_boxes = _table()
        only_img0_has_boxes["gt"] = {"image": [0, 0], "class": [0, 0], "row": [0, 1]}
        only_img0_has_boxes["pred"] = {"image": [0, 3], "class": [0, 0],
                                       "score": [0.9, 0.8], "iou": [0.9, 0.0],
                                       "gt": [0, -1]}
        table, index = self.load(table=only_img0_has_boxes)

        breakdown = tag_analytics.tag_breakdown(
            table, index, index.tag("auto:crowding"))
        barren = next(r for r in breakdown["rows"] if r["gt"] == 0)

        self.assertTrue(barren["no_ground_truth"])
        self.assertIsNone(barren["delta"])


class BoxMeasureTests(TagAnalyticsSetup):
    """Person size and pose, joined to ground-truth boxes by row."""

    def _document(self, **overrides):
        document = {
            "version": 1, "kind": "box", "complete": True,
            "person_source": "rtmo",
            "measures": [{"name": "pose", "kind": "categorical",
                          "choices": ["standing", "sitting", "lying", "(no person)"]},
                         {"name": "person size ratio", "kind": "numeric"}],
            "images": {
                "img0.jpg": {"boxes": [
                    {"row": 0, "class_id": 0,
                     "values": {"pose": "standing", "person size ratio": 0.05}},
                    {"row": 1, "class_id": 0,
                     "values": {"pose": "sitting", "person size ratio": 0.30}},
                ]},
                "img1.jpg": {"boxes": [
                    {"row": 0, "class_id": 0,
                     "values": {"pose": "standing", "person size ratio": 0.12}},
                ]},
                "img2.jpg": {"boxes": [
                    {"row": 0, "class_id": 0, "values": {"pose": "(no person)"}},
                ]},
                "img3.jpg": {"boxes": [
                    {"row": 0, "class_id": 0,
                     "values": {"pose": "lying", "person size ratio": 0.45}},
                ]},
            },
        }
        document.update(overrides)
        return document

    def _index(self, document=None):
        table, _index = self.load()
        return table, tag_analytics.build_tag_index(
            table, _tags(), box_measures=document or self._document())

    def test_pose_becomes_a_box_slice(self):
        _table, index = self._index()

        pose = index.tag(tag_analytics.POSE_TAG)

        self.assertEqual(pose.source, "auto")
        self.assertEqual(pose.counts["standing"], 2)
        self.assertEqual(pose.counts["sitting"], 1)
        self.assertEqual(pose.counts["lying"], 1)

    def test_a_box_with_nobody_around_it_is_its_own_bucket(self):
        _table, index = self._index()

        pose = index.tag(tag_analytics.POSE_TAG)

        # Distinct from "not measured": the pass looked and found nobody.
        self.assertEqual(pose.counts[tag_analytics.NO_PERSON], 1)
        self.assertNotIn(tag_analytics.UNMEASURED, pose.counts)

    def test_a_posture_nobody_struck_is_a_zero_row(self):
        document = self._document()
        for entry in document["images"].values():
            for box in entry["boxes"]:
                box["values"]["pose"] = "standing"
        _table, index = self._index(document)

        pose = index.tag(tag_analytics.POSE_TAG)

        self.assertIn("lying", pose.values)
        self.assertEqual(pose.counts["lying"], 0)

    def test_person_size_is_bucketed_over_the_boxes_that_have_one(self):
        _table, index = self._index()

        size = index.tag(tag_analytics.PERSON_SIZE_TAG)

        self.assertEqual(size.status, "ok")
        # Four ratios, one box with no person and therefore no ratio.
        self.assertEqual(size.counts[tag_analytics.UNMEASURED], 1)
        self.assertEqual(sum(cut["count"] for cut in size.cuts), 4)

    def test_a_pose_slice_can_be_scored(self):
        table, index = self._index()

        standing = tag_analytics.slice_metrics(
            table, index, [(tag_analytics.POSE_TAG, "standing")])

        self.assertEqual(standing["kind"], "box")
        self.assertEqual(standing["gt"], 2)

    def test_a_drifted_row_drops_that_image_rather_than_mislabelling_it(self):
        document = self._document()
        # The sidecar thinks img0's boxes are class 7; the labels say 0.
        for box in document["images"]["img0.jpg"]["boxes"]:
            box["class_id"] = 7
        _table, index = self._index(document)

        pose = index.tag(tag_analytics.POSE_TAG)

        # img0's two boxes are left unmeasured rather than attributed to
        # whatever now sits on those lines.
        self.assertEqual(pose.counts["standing"], 1)
        self.assertEqual(pose.counts["sitting"], 0)

    def test_the_no_person_bucket_keeps_its_exact_spelling(self):
        # friendy_chachkalica/ml/build_measures.py writes this string and pins
        # the same literal. The two cannot import each other (no torch here, no
        # Django there), so a drift would split one bucket into two that look
        # identical on the page.
        self.assertEqual(tag_analytics.NO_PERSON, "(no person)")

    def test_without_a_sidecar_both_tags_name_the_pass_that_makes_one(self):
        table, _index = self.load()
        index = tag_analytics.build_tag_index(table, _tags())

        for name in (tag_analytics.POSE_TAG, tag_analytics.PERSON_SIZE_TAG):
            definition = index.tag(name)
            self.assertEqual(definition.status, "unavailable")
            self.assertIn("Measure box statistics", definition.status_detail)


class ScoreCutTests(TagAnalyticsSetup):
    """Scoring at a confidence other than the one the eval ran at.

    The match table keeps every prediction down to its score floor, so a
    different operating point is a filter over rows already on disk rather than
    a re-run. These pin the parts of that which are easy to get wrong.
    """

    def test_absent_cut_is_the_evals_own_operating_point(self):
        table, _index = self.load()

        self.assertEqual(tag_analytics.resolve_score_cut(table, None),
                         table.score_threshold)

    def test_a_cut_below_the_floor_clamps_to_it(self):
        # Below the floor there are no stored predictions to read, so honouring
        # the request would report a slice as emptier than it really is.
        table, _index = self.load()

        self.assertEqual(tag_analytics.resolve_score_cut(table, 0.0), table.score_floor)
        self.assertEqual(tag_analytics.resolve_score_cut(table, -5.0), table.score_floor)

    def test_a_cut_above_one_clamps_to_one(self):
        table, _index = self.load()

        self.assertEqual(tag_analytics.resolve_score_cut(table, 4.2), 1.0)

    def test_raising_the_cut_drops_predictions_and_recall(self):
        table, _index = self.load()
        mask = np.ones(table.num_images, dtype=bool)

        low = tag_analytics.image_slice_metrics(table, mask, 0.001)
        high = tag_analytics.image_slice_metrics(table, mask, 0.75)

        self.assertGreater(low["predictions"], high["predictions"])
        self.assertGreater(low["recall"], high["recall"])

    def test_average_precision_does_not_move_with_the_cut(self):
        # AP integrates the whole precision-recall curve down to the score
        # floor, so the operating confidence cannot change it. The page marks
        # those columns "fixed" on the strength of this.
        table, _index = self.load()
        mask = np.ones(table.num_images, dtype=bool)

        low = tag_analytics.image_slice_metrics(table, mask, 0.001)
        high = tag_analytics.image_slice_metrics(table, mask, 0.75)

        self.assertAlmostEqual(low["map50"], high["map50"], places=9)

    def test_a_box_slice_finds_fewer_boxes_as_the_cut_rises(self):
        table, _index = self.load()
        mask = np.ones(table.num_gt, dtype=bool)

        low = tag_analytics.gt_slice_metrics(table, mask, 0.001)
        high = tag_analytics.gt_slice_metrics(table, mask, 0.75)

        self.assertGreaterEqual(low["found"], high["found"])
        self.assertGreaterEqual(low["recall"], high["recall"])


class ScorecardTests(TagAnalyticsSetup):
    """Every tag value in one table, for "which slice is worst anywhere"."""

    def test_it_reports_every_usable_tag(self):
        table, index = self.load()

        card = tag_analytics.scorecard(table, index)

        usable = {tag.name for tag in index.tags if tag.usable}
        self.assertEqual({group["name"] for group in card["groups"]}, usable)

    def test_it_says_which_columns_the_slider_moves(self):
        table, index = self.load()

        card = tag_analytics.scorecard(table, index)

        swept = {column["key"]: column["swept"] for column in card["columns"]}
        self.assertFalse(swept["map50"])
        self.assertFalse(swept["map50_95"])
        self.assertTrue(swept["precision"])
        self.assertTrue(swept["recall"])

    def test_it_reports_the_cut_it_actually_used(self):
        table, index = self.load()

        card = tag_analytics.scorecard(table, index, 0.0)

        self.assertEqual(card["score_cut"], table.score_floor)
        self.assertFalse(card["at_eval_threshold"])
        self.assertTrue(tag_analytics.scorecard(table, index)["at_eval_threshold"])

    def test_shading_is_bounded_per_column_within_one_tag(self):
        # A column is shaded against itself: precision and recall move in
        # opposite directions as the cut rises, so one scale for the tag would
        # flatten whichever of them happened to span less.
        table, index = self.load()

        card = tag_analytics.scorecard(table, index)

        for group in card["groups"]:
            for key, bounds in group["bounds"].items():
                if bounds is None:
                    continue
                shades = [row["intensity"][key] for row in group["rows"]
                          if row["intensity"][key] is not None]
                if len(set(shades)) > 1:
                    self.assertAlmostEqual(min(shades), 0.0, places=6)
                    self.assertAlmostEqual(max(shades), 1.0, places=6)

    def test_a_flat_column_is_left_unshaded_rather_than_painted(self):
        table, index = self.load()

        card = tag_analytics.scorecard(table, index)

        for group in card["groups"]:
            for key, bounds in group["bounds"].items():
                if bounds is None or bounds["low"] != bounds["high"]:
                    continue
                self.assertTrue(all(row["intensity"][key] is None
                                    for row in group["rows"]))

    def test_a_frame_slice_with_no_ground_truth_reports_no_metrics(self):
        # mAP and recall over an empty ground-truth set come out 0.0, which
        # reads as a catastrophic score rather than an undefined one. The row
        # stays -- it is where false positives live -- but the numbers go.
        # Ground truth only on img0/img1, so img3 keeps its prediction and has
        # nothing to score it against. Its claim is -1: nothing matched it.
        table = _table()
        table["gt"] = {"image": [0, 0, 1], "class": [0, 0, 0], "row": [0, 1, 0]}
        table["pred"] = {
            "image": [0, 0, 1, 3], "class": [0, 0, 0, 0],
            "score": [0.9, 0.8, 0.7, 0.3], "iou": [0.9, 0.8, 0.1, 0.0],
            "gt": [0, 1, 2, -1],
        }
        loaded, index = self.load(table)

        card = tag_analytics.scorecard(loaded, index)
        crowding = next(g for g in card["groups"] if g["name"] == "auto:crowding")
        bare = [row for row in crowding["rows"] if row["no_ground_truth"]]

        self.assertTrue(bare)
        for row in bare:
            self.assertIsNone(row["headline"])
            self.assertTrue(all(value is None for value in row["metrics"].values()))

    def test_a_box_tag_carries_only_the_metrics_a_box_slice_can_answer(self):
        table, index = self.load(_table_v2())

        card = tag_analytics.scorecard(table, index)
        box_groups = [g for g in card["groups"] if g["kind"] == "box"]

        self.assertTrue(box_groups)
        for group in box_groups:
            for row in group["rows"]:
                if row["empty"]:
                    continue
                self.assertIsNone(row["metrics"]["precision"])
                self.assertIsNone(row["metrics"]["map50"])
                self.assertIsNotNone(row["metrics"]["recall"])
