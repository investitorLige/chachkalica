"""Tests for scoring a dataset whose class names are not the model's.

``evaluate_detection`` matches predictions to ground truth **by class name** and
drops both sides of anything the two spaces don't share. Two consequences are
pinned here:

* two *disjoint* spaces are refused outright, because the alternative is a run
  that reports 0.0 for every metric over zero classes — a number that reads
  exactly like a model detecting nothing, and which cost a real debugging
  session to tell apart;
* a ``class_map`` translates the dataset's names into the model's first, which
  is what makes a fine-grained test set scorable against a coarse model at all.

The arithmetic is deliberately trivial (two boxes, exact overlaps) so a failure
here is a failure of the class handling and not of the matcher.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch
import yaml

from friendy_chachkalica.ml import eval_checkpoint as eval_checkpoint_mod

from friendy_chachkalica.metrics import apply_class_map, evaluate_detection

#: A four-class weapon test set scored against a single-class "gun" detector —
#: the case this whole path exists for.
DATASET = ["handgun", "rifle", "knife", "bat"]
ONE_CLASS_MODEL = {0: "gun"}


def _targets():
    """One image, two boxes: a rifle (id 1) and a knife (id 2)."""
    return [{
        "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 80.0, 80.0]]),
        "labels": torch.tensor([1, 2]),
        "orig_size": torch.tensor([100, 100]),
    }]


def _predictions():
    """Two confident class-0 detections, each exactly on one of the targets."""
    return [torch.tensor([
        [0.20, 0.20, 0.20, 0.20, 0.90, 0.0],
        [0.70, 0.70, 0.20, 0.20, 0.80, 0.0],
    ])]


def _evaluate(class_map):
    target_classes, eval_classes = apply_class_map(DATASET, class_map)
    return evaluate_detection(
        _predictions(), _targets(),
        score_threshold=0.5,
        prediction_classes=ONE_CLASS_MODEL,
        target_classes=target_classes,
        eval_classes=eval_classes,
    )


class ApplyClassMapTest(unittest.TestCase):
    def test_no_map_is_an_exact_identity(self):
        """The default has to leave every existing eval byte-for-byte unchanged."""
        expected = ({0: "handgun", 1: "rifle", 2: "knife", 3: "bat"},
                    {0: "handgun", 1: "rifle", 2: "knife", 3: "bat"})
        self.assertEqual(apply_class_map(DATASET, None), expected)
        self.assertEqual(apply_class_map(DATASET, {}), expected)

    def test_collapses_several_dataset_classes_onto_one_model_class(self):
        target_classes, eval_classes = apply_class_map(
            DATASET, {"handgun": "gun", "rifle": "gun", "knife": "gun", "bat": "gun"})
        # Every dataset id still resolves — to the same, single eval class.
        self.assertEqual(target_classes, {0: "gun", 1: "gun", 2: "gun", 3: "gun"})
        self.assertEqual(eval_classes, {0: "gun"})

    def test_none_and_empty_string_both_drop_a_class(self):
        """YAML writes a drop as null, an HTML form as ""; both must mean the same."""
        by_none = apply_class_map(DATASET, {"handgun": "gun", "rifle": None,
                                            "knife": None, "bat": None})
        by_blank = apply_class_map(DATASET, {"handgun": "gun", "rifle": "",
                                             "knife": "", "bat": ""})
        self.assertEqual(by_none, by_blank)
        self.assertEqual(by_none, ({0: "gun"}, {0: "gun"}))

    def test_unmentioned_classes_pass_through(self):
        target_classes, eval_classes = apply_class_map(DATASET, {"handgun": "gun"})
        self.assertEqual(target_classes, {0: "gun", 1: "rifle", 2: "knife", 3: "bat"})
        self.assertEqual(sorted(eval_classes.values()), ["bat", "gun", "knife", "rifle"])

    def test_accepts_a_mapping_with_non_contiguous_ids(self):
        """Class spaces come from classes.txt *and* from checkpoints; both shapes work."""
        self.assertEqual(
            apply_class_map({0: "a", 5: "b"}, {"a": "z", "b": "z"}),
            ({0: "z", 5: "z"}, {0: "z"}),
        )


class DisjointClassSpaceTest(unittest.TestCase):
    def test_refuses_rather_than_scoring_zero(self):
        with self.assertRaises(ValueError) as caught:
            _evaluate(None)
        message = str(caught.exception)
        # The two spaces have to be *in* the message: "which two?" is the whole
        # question the operator is left with otherwise.
        self.assertIn("gun", message)
        self.assertIn("handgun", message)
        self.assertIn("class map", message)

    def test_a_map_onto_a_shared_name_is_enough_to_run(self):
        metrics = _evaluate({name: "gun" for name in DATASET})
        self.assertEqual(metrics["num_eval_classes"], 1)
        self.assertEqual(metrics["num_targets"], 2)
        self.assertEqual(metrics["num_predictions"], 2)
        self.assertAlmostEqual(metrics["map50"], 1.0, places=5)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=5)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=5)


class DroppedClassTest(unittest.TestCase):
    def test_a_dropped_class_is_neither_scored_nor_counted_as_a_miss(self):
        """The knife target disappears; the prediction that hit it becomes a FP.

        Which is the honest reading: the operator said this eval is not about
        knives, so the box is not ground truth — but the detection is still a
        detection the model made, and recall must not be inflated by pretending
        otherwise.
        """
        metrics = _evaluate({"handgun": None, "rifle": "gun",
                             "knife": None, "bat": None})
        self.assertEqual(metrics["num_targets"], 1)          # the rifle only
        self.assertEqual(metrics["num_predictions"], 2)      # both still counted
        self.assertAlmostEqual(metrics["recall"], 1.0, places=5)
        self.assertAlmostEqual(metrics["precision"], 0.5, places=5)


class NoRegressionTest(unittest.TestCase):
    def test_matching_taxonomies_are_unaffected_by_the_new_code(self):
        """The overwhelmingly common case: model and dataset already agree."""
        model = {0: "handgun", 1: "rifle", 2: "knife", 3: "bat"}
        target_classes, eval_classes = apply_class_map(DATASET, None)
        metrics = evaluate_detection(
            [torch.tensor([[0.20, 0.20, 0.20, 0.20, 0.90, 1.0],
                           [0.70, 0.70, 0.20, 0.20, 0.80, 2.0]])],
            _targets(),
            score_threshold=0.5,
            prediction_classes=model,
            target_classes=target_classes,
            eval_classes=eval_classes,
        )
        self.assertEqual(metrics["num_eval_classes"], 4)
        self.assertEqual(metrics["num_targets"], 2)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=5)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=5)


class RequestWiringTest(unittest.TestCase):
    """``class_map`` has to survive the request YAML, not just the function call.

    Pinned separately because the rest of this file exercises the metrics
    directly: a working ``apply_class_map`` reached through an
    ``eval_from_request`` that forgets to forward the field is a map that
    silently does nothing, and the run then fails (or scores the wrong thing)
    with everything downstream looking correct.
    """

    def _request(self, body):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(body, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    BASE = {
        "checkpoint_path": "/x.pt",
        "images": "/imgs",
        "labels": "/lbls",
        "classes": ["handgun", "rifle"],
        "output_dir": "/out",
        "class_map": {"handgun": "gun", "rifle": None},
    }

    def test_a_single_model_request_forwards_it(self):
        with mock.patch.object(eval_checkpoint_mod, "eval_checkpoint") as run:
            eval_checkpoint_mod.eval_from_request(self._request(self.BASE))
        self.assertEqual(run.call_args.kwargs["class_map"],
                         {"handgun": "gun", "rifle": None})

    def test_a_combined_request_forwards_it(self):
        body = {**self.BASE, "extra_checkpoints": ["/y.pt"]}
        with mock.patch.object(eval_checkpoint_mod, "eval_combined_checkpoints") as run:
            eval_checkpoint_mod.eval_from_request(self._request(body))
        self.assertEqual(run.call_args.kwargs["class_map"],
                         {"handgun": "gun", "rifle": None})

    def test_a_request_without_one_forwards_none(self):
        body = {k: v for k, v in self.BASE.items() if k != "class_map"}
        with mock.patch.object(eval_checkpoint_mod, "eval_checkpoint") as run:
            eval_checkpoint_mod.eval_from_request(self._request(body))
        self.assertIsNone(run.call_args.kwargs["class_map"])


if __name__ == "__main__":
    unittest.main()
