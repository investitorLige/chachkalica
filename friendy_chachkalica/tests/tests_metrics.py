import json
import tempfile
import unittest
from pathlib import Path

import torch

from friendy_chachkalica.metrics import (
    _matrix_from_confusion_data,
    evaluate_detection,
    remap_raw_predictions_to_eval_classes,
    select_hard_images,
)
from friendy_chachkalica.ml.train import _write_hard_images


class RemapRawPredictionsToEvalClassesTests(unittest.TestCase):
    """The raw-(N,6)-tensor remap combined multi-model eval uses to merge
    each model's predictions into one eval class space before concatenating."""

    def test_remaps_by_name_and_drops_unmatched(self):
        # Model trains class id 0 as "person", 1 as "dog"; eval space only
        # knows "car" (0) and "person" (1) — "dog" has no match and is dropped.
        predictions = torch.tensor([
            [0.5, 0.5, 0.1, 0.1, 0.9, 0.0],
            [0.2, 0.2, 0.1, 0.1, 0.8, 1.0],
        ])
        out = remap_raw_predictions_to_eval_classes(
            predictions, {0: "person", 1: "dog"}, {0: "car", 1: "person"}
        )
        self.assertEqual(out.shape, (1, 6))
        self.assertEqual(int(out[0, 5]), 1)
        self.assertAlmostEqual(float(out[0, 4]), 0.9)

    def test_two_models_with_colliding_raw_ids_land_on_distinct_eval_ids(self):
        # Both models use raw id 0 for their one class, but different names —
        # this is exactly the case combining relies on to merge safely.
        cat_model = torch.tensor([[0.5, 0.5, 0.4, 0.4, 0.9, 0.0]])
        dog_model = torch.tensor([[0.5, 0.5, 0.4, 0.4, 0.8, 0.0]])
        eval_classes = {0: "cat", 1: "dog"}
        remapped_cat = remap_raw_predictions_to_eval_classes(
            cat_model, {0: "cat"}, eval_classes
        )
        remapped_dog = remap_raw_predictions_to_eval_classes(
            dog_model, {0: "dog"}, eval_classes
        )
        merged = torch.cat([remapped_cat, remapped_dog], dim=0)
        self.assertEqual(sorted(int(row[5]) for row in merged), [0, 1])

    def test_empty_predictions_stay_empty(self):
        out = remap_raw_predictions_to_eval_classes(
            torch.empty((0, 6)), {0: "person"}, {0: "person"}
        )
        self.assertEqual(out.shape, (0, 6))

    def test_no_matches_returns_empty(self):
        predictions = torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.9, 0.0]])
        out = remap_raw_predictions_to_eval_classes(
            predictions, {0: "dog"}, {0: "cat"}
        )
        self.assertEqual(out.shape, (0, 6))


class EvaluateDetectionConfusionMatrixTests(unittest.TestCase):
    def test_confusion_matrix_uses_operating_threshold_not_map_floor(self):
        target = {
            "orig_size": torch.tensor([100, 100]),
            "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
            "labels": torch.tensor([0]),
        }
        prediction = torch.tensor([
            [0.5, 0.5, 0.2, 0.2, 0.9, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.2, 0.0],
        ])

        metrics = evaluate_detection(
            [prediction],
            [target],
            iou_thresholds=[0.5],
            map_score_threshold=0.1,
            score_threshold=0.5,
            num_classes=1,
        )

        self.assertEqual(metrics["num_predictions"], 1)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["conf_threshold"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["matrix"], [[1, 0], [0, 0]])

    def test_confusion_matrix_data_rethresholds_interactively(self):
        """The stored event histogram rebuilds the matrix at any threshold >= floor.

        Two ground truths (class 0 and class 1). A high-confidence correct class-0
        detection, a mid-confidence class-0 prediction landing on the class-1 box
        (a confusion), and a low-confidence class-1 false positive. Raising the
        threshold should first drop the false positive, then turn the mid-confidence
        confusion into a miss.
        """
        target = {
            "orig_size": torch.tensor([100, 100]),
            "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0], [10.0, 10.0, 20.0, 20.0]]),
            "labels": torch.tensor([0, 1]),
        }
        prediction = torch.tensor([
            [0.50, 0.50, 0.20, 0.20, 0.90, 0.0],  # class0 @0.90 -> box0 correct
            [0.15, 0.15, 0.10, 0.10, 0.60, 0.0],  # class0 @0.60 -> box1 location, wrong class
            [0.90, 0.90, 0.05, 0.05, 0.20, 1.0],  # class1 @0.20 -> false positive
        ])

        metrics = evaluate_detection(
            [prediction],
            [target],
            iou_thresholds=[0.5],
            map_score_threshold=0.01,
            score_threshold=0.25,
            num_classes=2,
        )

        data = metrics["confusion_matrix_data"]
        self.assertEqual(data["floor"], 0.01)
        # Matrix rows: [truth cat, truth dog, background]; cols mirror + background.
        # At the operating threshold the stored matrix and a reconstruction agree.
        self.assertEqual(
            metrics["confusion_matrix"]["matrix"],
            _matrix_from_confusion_data(data, 0.25)["matrix"],
        )
        # Below every prediction: correct hit, the confusion, and the false positive.
        self.assertEqual(
            _matrix_from_confusion_data(data, 0.01)["matrix"],
            [[1, 0, 0], [1, 0, 0], [0, 1, 0]],
        )
        # Above the false positive (0.20) but below the confusion (0.60): FP gone.
        self.assertEqual(
            _matrix_from_confusion_data(data, 0.25)["matrix"],
            [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
        )
        # Above the confusion too: its ground truth becomes a miss (background col).
        self.assertEqual(
            _matrix_from_confusion_data(data, 0.70)["matrix"],
            [[1, 0, 0], [0, 0, 1], [0, 0, 0]],
        )


class EvaluateDetectionOperatingNMSTests(unittest.TestCase):
    """operating_nms_threshold dedupes the operating-point metrics; mAP never changes."""

    def _target(self):
        return {
            "orig_size": torch.tensor([100, 100]),
            "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
            "labels": torch.tensor([0]),
        }

    def _duplicate_predictions(self):
        # Two near-identical confident boxes on one object (a DETR duplicate,
        # IoU ~0.91) — without NMS the second is a guaranteed false positive.
        return torch.tensor([
            [0.50, 0.50, 0.20, 0.20, 0.90, 0.0],
            [0.50, 0.50, 0.21, 0.21, 0.80, 0.0],
        ])

    def _evaluate(self, **overrides):
        kwargs = dict(
            iou_thresholds=[0.5],
            map_score_threshold=0.01,
            score_threshold=0.5,
            num_classes=1,
        )
        kwargs.update(overrides)
        return evaluate_detection(
            [self._duplicate_predictions()], [self._target()], **kwargs
        )

    def test_duplicate_counts_as_fp_without_operating_nms(self):
        metrics = self._evaluate()
        self.assertEqual(metrics["num_predictions"], 2)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["per_class"][0]["precision"], 0.5)
        self.assertIsNone(metrics["operating_nms_threshold"])

    def test_operating_nms_dedupes_all_operating_metrics_but_not_map(self):
        base = self._evaluate()
        nms = self._evaluate(operating_nms_threshold=0.5)

        self.assertEqual(nms["num_predictions"], 1)
        self.assertEqual(nms["precision"], 1.0)
        self.assertEqual(nms["recall"], 1.0)
        self.assertEqual(nms["per_class"][0]["precision"], 1.0)
        self.assertEqual(nms["per_class"][0]["prediction_count"], 1)
        self.assertEqual(nms["operating_nms_threshold"], 0.5)
        # The duplicate FP disappears from the confusion matrix's background row.
        self.assertEqual(base["confusion_matrix"]["matrix"], [[1, 0], [1, 0]])
        self.assertEqual(nms["confusion_matrix"]["matrix"], [[1, 0], [0, 0]])
        # mAP integrates the raw NMS-free ranking either way.
        self.assertEqual(base["map50"], nms["map50"])
        self.assertEqual(base["map50_95"], nms["map50_95"])

    def test_operating_nms_keeps_separate_objects(self):
        # Two confident boxes on two disjoint ground truths must both survive.
        target = {
            "orig_size": torch.tensor([100, 100]),
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 80.0, 80.0]]),
            "labels": torch.tensor([0, 0]),
        }
        prediction = torch.tensor([
            [0.20, 0.20, 0.20, 0.20, 0.90, 0.0],
            [0.70, 0.70, 0.20, 0.20, 0.80, 0.0],
        ])
        metrics = evaluate_detection(
            [prediction], [target],
            iou_thresholds=[0.5], map_score_threshold=0.01,
            score_threshold=0.5, num_classes=1,
            operating_nms_threshold=0.5,
        )
        self.assertEqual(metrics["num_predictions"], 2)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)


class SelectHardImagesTests(unittest.TestCase):
    @staticmethod
    def _target(count=1):
        return {
            "orig_size": torch.tensor([100, 100]),
            "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]] * count),
            "labels": torch.zeros((count,), dtype=torch.long),
        }

    @staticmethod
    def _perfect_predictions(count=1):
        return torch.tensor(
            [[0.5, 0.5, 0.2, 0.2, 0.9, 0.0]] * count,
            dtype=torch.float32,
        ).reshape(-1, 6)

    def test_operating_threshold_excludes_low_confidence_noise(self):
        prediction = torch.cat([
            self._perfect_predictions(),
            torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.0]]),
        ])
        images = select_hard_images(
            [prediction],
            [self._target()],
            [{"image_path": "/data/a.jpg"}],
            score_threshold=0.25,
        )

        self.assertEqual(images[0]["num_predictions"], 1)
        self.assertEqual(images[0]["f1"], 1.0)
        self.assertEqual(images[0]["difficulty"], 0.0)

    def test_normalized_score_surfaces_complete_single_object_failure(self):
        predictions = [
            self._perfect_predictions(5),
            torch.empty((0, 6)),
        ]
        targets = [self._target(10), self._target(1)]
        infos = [
            {"image_path": "/data/crowded.jpg"},
            {"image_path": "/data/single.jpg"},
        ]

        images = select_hard_images(predictions, targets, infos, score_threshold=0.25)

        self.assertEqual(images[0]["image_name"], "single.jpg")
        self.assertEqual(images[0]["f1"], 0.0)
        self.assertGreater(images[0]["difficulty"], images[1]["difficulty"])

    def test_wrong_class_counts_as_effective_fp_and_fn(self):
        prediction = self._perfect_predictions()
        prediction[:, 5] = 1
        image = select_hard_images(
            [prediction],
            [self._target()],
            [{"image_path": "/data/wrong.jpg"}],
            score_threshold=0.25,
        )[0]

        self.assertEqual(image["wrong_class"], 1)
        self.assertEqual(image["total_errors"], 2)
        self.assertEqual(image["f1"], 0.0)

    def test_wrong_class_box_does_not_steal_lower_scored_correct_match(self):
        prediction = torch.cat([
            self._perfect_predictions(),
            self._perfect_predictions(),
        ])
        prediction[0, 5] = 1
        prediction[1, 4] = 0.8
        image = select_hard_images(
            [prediction],
            [self._target()],
            [{"image_path": "/data/confusion.jpg"}],
            score_threshold=0.25,
        )[0]

        self.assertEqual(image["wrong_class"], 0)
        self.assertEqual(image["false_positives"], 1)
        self.assertEqual(image["missed"], 0)
        self.assertEqual(image["f1"], 0.6667)

    def test_rejects_mismatched_batch_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            select_hard_images(
                [self._perfect_predictions()],
                [self._target()],
                [],
            )

    def test_artifact_uses_operating_nms_and_atomic_json(self):
        duplicate_predictions = torch.cat([
            self._perfect_predictions(),
            self._perfect_predictions(),
        ])
        duplicate_predictions[1, 2:4] *= 1.05
        duplicate_predictions[1, 4] = 0.8

        with tempfile.TemporaryDirectory() as temporary_dir:
            predictions_path = Path(temporary_dir) / "val_predictions.pt"
            _write_hard_images(
                predictions_path,
                [duplicate_predictions],
                [self._target()],
                [{"image_path": "/data/a.jpg"}],
                config=None,
                prediction_classes={0: "object"},
                target_classes={0: "object"},
                eval_classes={0: "object"},
                operating_nms_threshold=0.5,
            )
            payload = json.loads(
                (Path(temporary_dir) / "val_hard_images.json").read_text()
            )

        self.assertEqual(payload["score_threshold"], 0.25)
        self.assertEqual(payload["operating_nms_threshold"], 0.5)
        self.assertEqual(payload["images"][0]["num_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
