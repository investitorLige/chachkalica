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
        path.write_text(json.dumps(_table(version=4)), encoding="utf-8")

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


# Five people over four frames, one scored class. Laid out so every number
# below can be read off the picture:
#
#   img0  P0 standing, wearing a helmet, and the model says so      -> hit
#         P1 lying,    wearing a helmet, and the model says so      -> hit
#   img1  P2 standing, wearing nothing, and the model says helmet   -> false alarm
#         plus a helmet lying on the floor, on nobody               -> an orphan
#   img2  P3 lying,    wearing a helmet, and the model says nothing -> miss
#   img3  P4 standing, wearing nothing, and the model agrees        -> correct reject
def _person_table(**overrides) -> dict:
    table = {
        "version": 3,
        "iou_thresholds": [0.5],
        "score_threshold": 0.25,
        "score_floor": 0.001,
        "operating_nms_threshold": None,
        # "person" is in the class space and is deliberately never asked about:
        # "does this person have a person on them" is not a question.
        "classes": {"0": "helmet", "1": "person"},
        "images": ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"],
        "gt": {
            "image": [0, 0, 1, 2], "class": [0, 0, 0, 0], "row": [0, 1, 0, 0],
            "w": [0.1, 0.1, 0.1, 0.1], "h": [0.1, 0.1, 0.1, 0.1],
        },
        "pred": {
            "image": [0, 0, 1],
            "class": [0, 0, 0],
            "score": [0.9, 0.8, 0.7],
            "iou": [0.9, 0.8, 0.1],
            "gt": [0, 1, 2],
            "geom": {
                "cut": 0.25,
                "index": [0, 1, 2],
                "x0": [0.05, 0.55, 0.15], "y0": [0.05, 0.05, 0.15],
                "x1": [0.15, 0.65, 0.25], "y1": [0.15, 0.15, 0.25],
            },
        },
        "pred_operating": None,
    }
    table.update(overrides)
    return table


def _person_measures(**overrides) -> dict:
    document = {
        "version": 2,
        "kind": "box",
        "person_source": "detector",
        "containment_min": 0.7,
        "person_iou_min": 0.5,
        "person_class_names": ["person", "people", "pedestrian"],
        "person_measures": [
            {"name": "pose", "kind": "categorical",
             "choices": ["standing", "sitting", "lying"]},
            {"name": "size ratio", "kind": "numeric"},
        ],
        "images": {
            "img0.jpg": {
                "people": [
                    {"box": [0.0, 0.0, 0.4, 1.0], "pose": "standing", "size ratio": 0.40},
                    {"box": [0.5, 0.0, 0.9, 1.0], "pose": "lying", "size ratio": 0.40},
                ],
                "boxes": [
                    {"row": 0, "class_id": 0, "person": 0, "values": {"pose": "standing"}},
                    {"row": 1, "class_id": 0, "person": 1, "values": {"pose": "lying"}},
                ],
            },
            "img1.jpg": {
                "people": [
                    {"box": [0.1, 0.1, 0.5, 0.9], "pose": "standing", "size ratio": 0.32},
                ],
                # The helmet on the floor: measured, and on nobody.
                "boxes": [{"row": 0, "class_id": 0, "person": None,
                           "values": {"pose": "(no person)"}}],
            },
            "img2.jpg": {
                "people": [
                    {"box": [0.0, 0.0, 0.5, 1.0], "pose": "lying", "size ratio": 0.50},
                ],
                "boxes": [{"row": 0, "class_id": 0, "person": 0,
                           "values": {"pose": "lying"}}],
            },
            # No labels at all, which is exactly why this frame matters: the
            # person in it can only ever be a correct rejection.
            "img3.jpg": {
                "people": [
                    {"box": [0.2, 0.2, 0.6, 0.8], "pose": "standing", "size ratio": 0.24},
                ],
                "boxes": [],
            },
        },
    }
    document.update(overrides)
    return document


class PersonScoringTests(TagAnalyticsSetup):
    """The population where a false positive has an owner."""

    def load_persons(self, table=None, measures=None):
        path = self.root / "eval_matches.json"
        path.write_text(json.dumps(table or _person_table()), encoding="utf-8")
        loaded = tag_analytics.load_table(path)
        persons = tag_analytics.build_persons(
            loaded, _person_measures() if measures is None else measures)
        index = tag_analytics.build_tag_index(
            loaded, {}, box_measures=_person_measures() if measures is None else measures,
            persons=persons)
        return loaded, persons, index

    def test_everybody_the_pass_found_is_a_row_including_the_empty_handed(self):
        _table, persons, _index = self.load_persons()

        # Four ground-truth boxes but five people: the two the model was right
        # about carrying nothing have no box to appear as.
        self.assertEqual(persons.count, 5)

    def test_the_confusion_matches_the_picture_in_the_fixture(self):
        table, persons, index = self.load_persons()

        metrics = tag_analytics.overall_metrics(table, index, "person")

        self.assertEqual(metrics["hit"], 2)
        self.assertEqual(metrics["miss"], 1)
        self.assertEqual(metrics["false_alarm"], 1)
        self.assertEqual(metrics["correct_reject"], 1)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        # Two people carry nothing; the model fired on one of them.
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.6)

    def test_pose_slices_people_rather_than_boxes(self):
        table, _persons, index = self.load_persons()

        pose = index.tag(tag_analytics.POSE_TAG)
        standing = tag_analytics.slice_metrics(
            table, index, [(tag_analytics.POSE_TAG, "standing")])

        self.assertEqual(pose.scope, tag_analytics.PERSON_SCOPE)
        self.assertEqual(pose.kind, "person")
        self.assertEqual(standing["kind"], "person")
        # P0 (hit), P2 (false alarm), P4 (correct reject).
        self.assertEqual(standing["persons"], 3)
        self.assertEqual(standing["hit"], 1)
        self.assertEqual(standing["false_alarm"], 1)
        self.assertEqual(standing["correct_reject"], 1)
        self.assertAlmostEqual(standing["precision"], 0.5)
        self.assertAlmostEqual(standing["recall"], 1.0)

    def test_the_lying_slice_is_the_one_the_model_is_worse_on(self):
        table, _persons, index = self.load_persons()

        lying = tag_analytics.slice_metrics(
            table, index, [(tag_analytics.POSE_TAG, "lying")])

        self.assertEqual(lying["persons"], 2)
        self.assertEqual(lying["hit"], 1)
        self.assertEqual(lying["miss"], 1)
        self.assertAlmostEqual(lying["recall"], 0.5)
        # Nothing was predicted wrongly here, so precision is a clean 1.0 --
        # the failure is entirely on the recall side.
        self.assertAlmostEqual(lying["precision"], 1.0)

    def test_a_person_class_is_never_asked_about_itself(self):
        table, persons, _index = self.load_persons()

        self.assertEqual(
            [table.classes[c] for c in tag_analytics.scored_class_ids(table, persons)],
            ["helmet"])

    def test_a_box_on_nobody_is_counted_and_scored_apart(self):
        table, persons, _index = self.load_persons()

        orphans = tag_analytics.person_orphans(table, persons)

        self.assertEqual(orphans["gt_on_person"], 3)
        self.assertEqual(orphans["gt_on_nobody"], 1)
        self.assertEqual(orphans["gt_unmeasured"], 0)
        helmet = orphans["per_class"][0]
        self.assertEqual(helmet["on_nobody"], 1)
        # The 0.7 prediction lands near it at IoU 0.1, which is a miss.
        self.assertEqual(helmet["found"], 0)
        self.assertAlmostEqual(helmet["recall"], 0.0)

    def test_predictions_are_accounted_for_three_ways(self):
        table, persons, _index = self.load_persons()

        predictions = tag_analytics.person_orphans(table, persons)["predictions"]

        self.assertEqual(predictions["total"], 3)
        self.assertEqual(predictions["on_person"], 3)
        self.assertEqual(predictions["on_nobody"], 0)
        self.assertEqual(predictions["no_geometry"], 0)

    def test_a_prediction_landing_on_no_one_is_not_silently_dropped(self):
        table = _person_table()
        # Move the img1 prediction into a corner no person occupies.
        table["pred"]["geom"]["x0"][2] = 0.90
        table["pred"]["geom"]["y0"][2] = 0.90
        table["pred"]["geom"]["x1"][2] = 0.98
        table["pred"]["geom"]["y1"][2] = 0.98
        loaded, persons, index = self.load_persons(table=table)

        orphans = tag_analytics.person_orphans(loaded, persons)
        metrics = tag_analytics.overall_metrics(loaded, index, "person")

        self.assertEqual(orphans["predictions"]["on_nobody"], 1)
        # It stops being a false alarm against P2, so P2 becomes a correct
        # rejection -- and the prediction is now only visible in the orphan
        # panel, which is why that panel has to exist.
        self.assertEqual(metrics["false_alarm"], 0)
        self.assertEqual(metrics["correct_reject"], 2)

    def test_geometry_survives_the_sort_into_score_order(self):
        table = _person_table()
        # Same three predictions, written to the file in ascending score, so a
        # loader that forgot to permute the sparse index would hang each box on
        # the wrong prediction.
        table["pred"] = {
            "image": [1, 0, 0],
            "class": [0, 0, 0],
            "score": [0.7, 0.8, 0.9],
            "iou": [0.1, 0.8, 0.9],
            "gt": [2, 1, 0],
            "geom": {
                "cut": 0.25,
                "index": [0, 1, 2],
                "x0": [0.15, 0.55, 0.05], "y0": [0.15, 0.05, 0.05],
                "x1": [0.25, 0.65, 0.15], "y1": [0.25, 0.15, 0.15],
            },
        }
        loaded, _persons, index = self.load_persons(table=table)

        metrics = tag_analytics.overall_metrics(loaded, index, "person")

        self.assertEqual(metrics["hit"], 2)
        self.assertEqual(metrics["false_alarm"], 1)

    def test_a_frame_clause_narrows_the_people(self):
        table, _persons, index = self.load_persons()
        tags = {
            "tags": [{"name": "weather", "scope": "frame", "widget": "radio",
                      "choices": ["sun", "rain"]}],
            "images": {"img0.jpg": {"frame": {"weather": ["rain"]}},
                       "img1.jpg": {"frame": {"weather": ["sun"]}},
                       "img2.jpg": {"frame": {"weather": ["sun"]}},
                       "img3.jpg": {"frame": {"weather": ["sun"]}}},
        }
        index = tag_analytics.build_tag_index(
            table, tags, box_measures=_person_measures(), persons=index.persons)

        rainy = tag_analytics.slice_metrics(
            table, index, [(tag_analytics.POSE_TAG, "standing"), ("weather", "rain")])

        # Only P0 stands in the rain.
        self.assertEqual(rainy["kind"], "person")
        self.assertEqual(rainy["persons"], 1)
        self.assertEqual(rainy["hit"], 1)

    def test_a_table_without_prediction_geometry_says_so_instead_of_scoring(self):
        table = _person_table(version=2)
        table["pred"].pop("geom")
        loaded, persons, index = self.load_persons(table=table)

        section = tag_analytics.person_section(loaded, index)

        self.assertIsNone(persons)
        self.assertFalse(section["available"])
        self.assertIn("Build tag analytics data", section["detail"])

    def test_an_older_sidecar_keeps_the_box_scope_tables(self):
        # No "people" key: the measures pass predates per-person rows. The tag
        # does not vanish, it just answers the smaller question it always did.
        measures = _person_measures()
        for entry in measures["images"].values():
            entry.pop("people")
        _loaded, persons, index = self.load_persons(measures=measures)

        pose = index.tag(tag_analytics.POSE_TAG)

        self.assertIsNone(persons)
        self.assertEqual(pose.kind, "box")

    def test_the_section_names_the_pass_that_would_produce_people(self):
        loaded, _persons, index = self.load_persons(measures={})

        section = tag_analytics.person_section(loaded, index)

        self.assertFalse(section["available"])
        self.assertIn("Measure box statistics", section["detail"])


class AttributionTests(TestCase):
    """The rule that decides which person a box belongs to."""

    def test_a_small_box_goes_to_the_smallest_person_containing_it(self):
        people = np.array([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.3, 0.3]])
        boxes = np.array([[0.15, 0.15, 0.2, 0.2]])

        owner = tag_analytics.attribute_boxes(
            boxes, np.array([False]), people,
            containment_min=0.7, person_iou_min=0.5)

        # Both contain it; the tighter enclosure is the wearer far more often
        # than the large figure in the foreground.
        self.assertEqual(int(owner[0]), 1)

    def test_a_box_that_is_itself_a_person_is_matched_by_overlap(self):
        people = np.array([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.3, 0.3]])
        boxes = np.array([[0.1, 0.1, 0.3, 0.3]])

        owner = tag_analytics.attribute_boxes(
            boxes, np.array([True]), people,
            containment_min=0.7, person_iou_min=0.5)

        # Containment would have handed this to the full-frame person, which is
        # a different human being.
        self.assertEqual(int(owner[0]), 1)

    def test_a_box_inside_nobody_belongs_to_nobody(self):
        people = np.array([[0.0, 0.0, 0.2, 0.2]])
        boxes = np.array([[0.8, 0.8, 0.9, 0.9]])

        owner = tag_analytics.attribute_boxes(
            boxes, np.array([False]), people,
            containment_min=0.7, person_iou_min=0.5)

        self.assertEqual(int(owner[0]), -1)

    def test_a_box_half_out_of_a_person_is_not_theirs(self):
        people = np.array([[0.0, 0.0, 0.5, 1.0]])
        boxes = np.array([[0.4, 0.4, 0.6, 0.6]])

        owner = tag_analytics.attribute_boxes(
            boxes, np.array([False]), people,
            containment_min=0.7, person_iou_min=0.5)

        # Half its area falls outside, which is under the 0.7 containment floor.
        self.assertEqual(int(owner[0]), -1)

    def test_the_rule_constants_match_the_pass_that_wrote_the_sidecar(self):
        # friendy_chachkalica/ml/build_measures.py pins the same two numbers and
        # writes them into every sidecar it produces. These defaults are only
        # reached for a sidecar that predates that, so they have to agree.
        self.assertEqual(tag_analytics.DEFAULT_CONTAINMENT_MIN, 0.7)
        self.assertEqual(tag_analytics.DEFAULT_PERSON_IOU_MIN, 0.5)


class AllTagsGridTests(TagAnalyticsSetup):
    """The one grid at the end that lays every tag's values side by side."""

    def test_each_population_gets_its_own_table(self):
        table, index = self.load()
        report = tag_analytics.report(table, index)

        kinds = [group["kind"] for group in report["all_tags"]]

        # Frame and box tags never share a metric set, so they never share a
        # table either.
        self.assertIn("frame", kinds)
        self.assertIn("box", kinds)
        for group in report["all_tags"]:
            self.assertEqual(
                [key for key, _label in group["metrics"]],
                [key for key, _label in tag_analytics.METRICS_BY_KIND[group["kind"]][0]])

    def test_shading_is_normalized_inside_one_tag_not_across_them(self):
        table, index = self.load()
        report = tag_analytics.report(table, index)

        for group in report["all_tags"]:
            for tag in group["tags"]:
                for position in range(len(group["metrics"])):
                    shares = [row["cells"][position]["share"] for row in tag["rows"]
                              if row["cells"][position]["share"] is not None]
                    if not shares:
                        continue
                    # Worst and best of *this tag's* values pin the ends.
                    self.assertAlmostEqual(min(shares), 0.0)
                    self.assertAlmostEqual(max(shares), 1.0)

    def test_a_single_valued_column_is_left_unshaded(self):
        table, index = self.load()
        report = tag_analytics.report(table, index)

        for group in report["all_tags"]:
            for tag in group["tags"]:
                for position in range(len(group["metrics"])):
                    values = [row["cells"][position]["value"] for row in tag["rows"]
                              if row["cells"][position]["value"] is not None]
                    if len(set(values)) > 1:
                        continue
                    # One number is not a spread; shading it would imply a
                    # comparison nobody made.
                    for row in tag["rows"]:
                        self.assertIsNone(row["cells"][position]["share"])


class PersonPageTests(TagAnalyticsSetup):
    """The page itself, which is where a scope mistake actually shows up."""

    def _report(self, table=None, measures=None):
        path = self.root / "eval_matches.json"
        path.write_text(json.dumps(table or _person_table()), encoding="utf-8")
        loaded = tag_analytics.load_table(path)
        document = _person_measures() if measures is None else measures
        persons = tag_analytics.build_persons(loaded, document)
        index = tag_analytics.build_tag_index(
            loaded, _tags(), box_measures=document, persons=persons)
        return loaded, index, tag_analytics.report(loaded, index)

    def _render(self, report):
        from django.template.loader import render_to_string
        return render_to_string("admin/eval_pipelines/tag_analytics.html", {
            "report": report, "sources": [], "subject_name": "eval", "dataset": "ds",
            "fatal": "", "data_url": "/data", "query": "id=1", "title": "t",
        })

    def test_the_person_section_reaches_the_page(self):
        _table, _index, report = self._report()

        html = self._render(report)

        self.assertIn("<h3>Boxes on nobody</h3>", html)
        self.assertIn("correct reject", html)
        self.assertIn("specificity", html)

    def test_an_eval_too_old_for_people_gets_the_reason_not_an_empty_table(self):
        table = _person_table(version=2)
        table["pred"].pop("geom")
        _table, _index, report = self._report(table=table)

        html = self._render(report)

        self.assertFalse(report["persons"]["available"])
        self.assertIn("Build tag analytics data", html)
        self.assertNotIn("<h3>Boxes on nobody</h3>", html)

    def test_person_breakdowns_are_kept_out_of_the_box_section(self):
        _table, _index, report = self._report()

        kinds = {breakdown["kind"] for breakdown in report["breakdowns"]}
        person_kinds = {b["kind"] for b in report["person_breakdowns"]}

        # Otherwise the same tag would render twice, once under a metric set it
        # does not have.
        self.assertNotIn("person", kinds)
        self.assertEqual(person_kinds, {"person"})

    def test_the_all_tags_grid_gains_a_person_table(self):
        _table, _index, report = self._report()

        groups = {group["kind"]: group for group in report["all_tags"]}

        self.assertIn("person", groups)
        self.assertEqual(groups["person"]["population_label"], "persons")
        self.assertIn("specificity",
                      [label for _key, label in groups["person"]["metrics"]])

    def test_a_cross_tab_pairing_a_person_tag_scores_people(self):
        table, index, _report = self._report()

        grid = tag_analytics.cross_tab(
            table, index, tag_analytics.POSE_TAG, "weather", "f1")

        self.assertEqual(grid["kind"], "person")
        self.assertEqual(grid["population_label"], "persons")

    def test_a_person_slice_survives_json_serialization(self):
        from django.core.serializers.json import DjangoJSONEncoder
        table, index, _report = self._report()

        payload = tag_analytics.slice_metrics(
            table, index, [(tag_analytics.POSE_TAG, "standing")])

        # numpy scalars serialize to nothing useful, so the endpoint would 500
        # long after this module looked correct.
        json.dumps(payload, cls=DjangoJSONEncoder)

    def test_comparing_evals_that_disagree_about_the_population_blanks_the_cell(self):
        old = _person_table(version=2)
        old["pred"].pop("geom")
        new_table, new_index, _ = self._report()
        old_table, old_index, _ = self._report(table=old)

        grid = tag_analytics.compare_tag(
            [{"label": "new", "dataset": "ds", "table": new_table, "index": new_index},
             {"label": "old", "dataset": "ds", "table": old_table, "index": old_index}],
            tag_analytics.POSE_TAG, "f1")

        mismatched = [cell for row in grid["rows"] for cell in row["cells"]
                      if cell.get("mismatched_kind")]
        self.assertTrue(mismatched)
        self.assertTrue(all(cell["value"] is None for cell in mismatched))
        self.assertTrue(any("different population" in warning
                            for warning in grid["warnings"]))


class PersonJoinIntegrityTests(TagAnalyticsSetup):
    """Ways the two sidecars can disagree, and what each one costs."""

    def _load(self, measures):
        path = self.root / "eval_matches.json"
        path.write_text(json.dumps(_person_table()), encoding="utf-8")
        table = tag_analytics.load_table(path)
        return table, tag_analytics.build_persons(table, measures)

    def test_a_malformed_person_does_not_shift_the_ones_after_it(self):
        measures = _person_measures()
        # A first person with no box at all. The helmet on img0 row1 still
        # names person 1, and person 1 is still the one lying down.
        measures["images"]["img0.jpg"]["people"].insert(0, {"pose": "sitting"})
        for box in measures["images"]["img0.jpg"]["boxes"]:
            box["person"] += 1
        table, persons = self._load(measures)

        owners = persons.gt_person[:2]

        self.assertEqual(persons.count, 5)
        self.assertEqual([persons.pose[owner] for owner in owners],
                         ["standing", "lying"])

    def test_a_box_naming_a_person_that_is_not_there_lands_on_nobody(self):
        measures = _person_measures()
        measures["images"]["img0.jpg"]["boxes"][0]["person"] = 9
        _table, persons = self._load(measures)

        # Attributed to nobody rather than to whoever happens to be at index 9
        # once the frames are concatenated -- which would be someone in another
        # image entirely.
        self.assertEqual(int(persons.gt_person[0]), -1)

    def test_a_drifted_frame_contributes_neither_people_nor_boxes(self):
        measures = _person_measures()
        for box in measures["images"]["img0.jpg"]["boxes"]:
            box["class_id"] = 7
        table, persons = self._load(measures)

        self.assertEqual(persons.count, 3)
        self.assertEqual(persons.drifted_images, 1)
        self.assertFalse(bool(persons.measured_mask[0]))
        # Its boxes are "not measured", not "on nobody" -- a different fact,
        # calling for a different fix.
        orphans = tag_analytics.person_orphans(table, persons)
        self.assertEqual(orphans["gt_unmeasured"], 2)
        self.assertEqual(orphans["gt_on_nobody"], 1)
