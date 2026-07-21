"""Tests for eval_combined_checkpoints: merging 2+ checkpoints' predictions.

Uses a stub model adapter (a duck-typed ``.predict``) and mocks
``_load_checkpoint_adapter_for_eval``/``build_eval_dataloader`` so nothing needs
a real checkpoint file or a GPU.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from friendy_chachkalica import eval_checkpoint as eval_checkpoint_mod
from friendy_chachkalica.eval_checkpoint import eval_combined_checkpoints


class StubModel:
    """Emits one fixed box per input image."""

    def __init__(self, box):
        self.box = box

    def predict(self, images, score_threshold=None):
        return [torch.tensor([list(self.box)]) for _ in images]


def sample_batch():
    images = [torch.rand(3, 32, 32), torch.rand(3, 32, 32)]
    targets = [
        {"image_path": "img1.jpg", "label_path": "img1.txt",
         "orig_size": torch.tensor([32, 32]),
         "boxes": torch.tensor([[5.0, 5.0, 10.0, 10.0]]), "labels": torch.tensor([0])},
        {"image_path": "img2.jpg", "label_path": "img2.txt",
         "orig_size": torch.tensor([32, 32]),
         "boxes": torch.tensor([[5.0, 5.0, 10.0, 10.0]]), "labels": torch.tensor([1])},
    ]
    return images, targets


class EvalCombinedCheckpointsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmp.name) / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, adapters_and_classes, loader, **overrides):
        calls = iter(adapters_and_classes)

        def fake_load(checkpoint_path, device):
            return next(calls)

        kwargs = dict(
            checkpoint_paths=["/dev/null/a.pt", "/dev/null/b.pt"],
            images="/dev/null/images",
            labels="/dev/null/labels",
            classes={0: "cat", 1: "dog"},
            output_dir=self.output_dir,
        )
        kwargs.update(overrides)

        with mock.patch.object(
            eval_checkpoint_mod, "_load_checkpoint_adapter_for_eval", side_effect=fake_load
        ), mock.patch.object(eval_checkpoint_mod, "build_eval_dataloader", return_value=loader):
            return eval_combined_checkpoints(**kwargs)

    def test_merges_disjoint_class_predictions_with_correct_eval_ids(self):
        images, targets = sample_batch()
        adapters_and_classes = [
            (StubModel((0.5, 0.5, 0.4, 0.4, 0.9, 0)), {0: "cat"}),
            (StubModel((0.5, 0.5, 0.4, 0.4, 0.8, 0)), {0: "dog"}),
        ]
        result = self._run(adapters_and_classes, loader=[(images, targets)])

        self.assertIn("map50", result["metrics"])
        self.assertEqual(result["metrics"]["num_predictions"], 4)
        self.assertEqual(result["checkpoint"], ["/dev/null/a.pt", "/dev/null/b.pt"])

        records = torch.load(self.output_dir / "eval_predictions.pt")
        self.assertEqual(len(records), 2)
        for record in records:
            labels = sorted(int(row[5]) for row in record["predictions"])
            self.assertEqual(labels, [0, 1])

    def test_prediction_only_when_labels_is_none(self):
        images, targets = sample_batch()
        adapters_and_classes = [
            (StubModel((0.5, 0.5, 0.4, 0.4, 0.9, 0)), {0: "cat"}),
            (StubModel((0.5, 0.5, 0.4, 0.4, 0.8, 0)), {0: "dog"}),
        ]
        result = self._run(adapters_and_classes, loader=[(images, targets)], labels=None)
        self.assertTrue(result["metrics"].get("prediction_only"))
        self.assertIsNone(result["labels"])
        # Predictions are still written even without ground truth.
        records = torch.load(self.output_dir / "eval_predictions.pt")
        self.assertEqual(len(records), 2)

    def test_requires_at_least_two_checkpoints(self):
        with self.assertRaises(ValueError):
            eval_combined_checkpoints(
                checkpoint_paths=["/dev/null/a.pt"],
                images="/dev/null/images",
                labels="/dev/null/labels",
                classes={0: "cat"},
                output_dir=self.output_dir,
            )


if __name__ == "__main__":
    unittest.main()
