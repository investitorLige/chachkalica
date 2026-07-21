import tempfile
import unittest
from pathlib import Path

import torch

from friendy_chachkalica.promote_labels import (
    build_class_remap,
    promote_labels,
    record_to_lines,
    _label_name,
)


class ClassRemapTests(unittest.TestCase):
    def test_remaps_by_name_and_drops_unmatched(self):
        # Model trained on [head, helmet, vest]; dataset is [helmet, head] only.
        remap = build_class_remap(["head", "helmet", "vest"], ["helmet", "head"])
        # head(0)->dataset head(1), helmet(1)->dataset helmet(0); vest has no match.
        self.assertEqual(remap, {0: 1, 1: 0})

    def test_accepts_mapping_form(self):
        remap = build_class_remap({0: "a", 1: "b"}, {0: "b", 1: "a"})
        self.assertEqual(remap, {0: 1, 1: 0})


class RecordToLinesTests(unittest.TestCase):
    def test_threshold_filters_and_class_is_remapped(self):
        preds = [
            [0.5, 0.5, 0.2, 0.2, 0.9, 0],   # kept, class 0 -> 1
            [0.1, 0.1, 0.1, 0.1, 0.1, 1],   # dropped: below threshold
            [0.2, 0.2, 0.1, 0.1, 0.8, 2],   # dropped: class 2 not in dataset
        ]
        lines, dropped = record_to_lines(preds, {0: 1, 1: 0}, score_threshold=0.25)
        self.assertEqual(lines, ["1 0.500000 0.500000 0.200000 0.200000"])
        self.assertEqual(dropped, 1)  # the above-threshold, unmapped class-2 box

    def test_empty_when_nothing_survives(self):
        lines, dropped = record_to_lines([[0, 0, 0, 0, 0.1, 0]], {0: 0}, 0.25)
        self.assertEqual(lines, [])
        self.assertEqual(dropped, 0)


class LabelNameTests(unittest.TestCase):
    def test_relative_stem(self):
        images = Path("/data/source/ds/images")
        self.assertEqual(_label_name("/data/source/ds/images/a/b.jpg", images), "a/b.txt")

    def test_falls_back_to_basename_when_not_under_root(self):
        self.assertEqual(_label_name("/elsewhere/c.png", Path("/data/images")), "c.txt")


class PromoteLabelsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        (self.images / "img1.jpg").write_bytes(b"")
        (self.images / "img2.jpg").write_bytes(b"")
        self.labels = self.root / "labels"
        self.backup = self.root / "backup_labels" / "pipeline-1"

        # A checkpoint that records its training class names (head, helmet).
        self.ckpt = self.root / "model.pt"
        torch.save({"train_dataset": {"classes": ["head", "helmet"]}}, self.ckpt)

        # Predictions in the model's train-class space.
        records = [
            {"image_path": str(self.images / "img1.jpg"),
             "predictions": torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.9, 1]])},  # helmet
            {"image_path": str(self.images / "img2.jpg"),
             "predictions": torch.tensor([[0.4, 0.4, 0.1, 0.1, 0.1, 0]])},  # below threshold
        ]
        self.preds = self.root / "predictions.pt"
        torch.save(records, self.preds)

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_labels_remapped_to_dataset_space(self):
        # Dataset class order differs from the model: [helmet, head].
        summary = promote_labels(
            self.preds, self.ckpt, self.images, ["helmet", "head"],
            self.labels, self.backup, score_threshold=0.25,
        )
        self.assertEqual(summary["labels_written"], 2)
        self.assertEqual(summary["boxes_written"], 1)
        # helmet is dataset index 0 here.
        self.assertEqual((self.labels / "img1.txt").read_text().strip(),
                         "0 0.500000 0.500000 0.200000 0.200000")
        # Below-threshold image still gets an (empty) label file.
        self.assertEqual((self.labels / "img2.txt").read_text().strip(), "")

    def test_backs_up_existing_labels(self):
        self.labels.mkdir()
        (self.labels / "img1.txt").write_text("9 0.1 0.1 0.1 0.1\n")
        summary = promote_labels(
            self.preds, self.ckpt, self.images, ["helmet", "head"],
            self.labels, self.backup, score_threshold=0.25,
        )
        self.assertEqual(summary["backed_up"], 1)
        self.assertEqual((self.backup / "img1.txt").read_text().strip(), "9 0.1 0.1 0.1 0.1")
        # And the new label replaced the old in place.
        self.assertTrue((self.labels / "img1.txt").read_text().startswith("0 "))


class PromoteCombinedLabelsTests(unittest.TestCase):
    """A combined multi-model eval's predictions are already indexed in the
    eval dataset's own class space, so promote takes `prediction_classes`
    instead of a checkpoint and skips loading one entirely."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        (self.images / "img1.jpg").write_bytes(b"")
        self.labels = self.root / "labels"
        self.backup = self.root / "backup_labels" / "eval-1"

        # Predictions already merged/remapped into the eval dataset's class
        # space {0: cat, 1: dog} — see eval_checkpoint.eval_combined_checkpoints.
        records = [
            {"image_path": str(self.images / "img1.jpg"),
             "predictions": torch.tensor([
                 [0.5, 0.5, 0.4, 0.4, 0.9, 0.0],
                 [0.3, 0.3, 0.1, 0.1, 0.8, 1.0],
             ])},
        ]
        self.preds = self.root / "eval_predictions.pt"
        torch.save(records, self.preds)

    def tearDown(self):
        self._tmp.cleanup()

    def test_identity_remap_writes_correct_indices_without_a_checkpoint(self):
        summary = promote_labels(
            self.preds,
            checkpoint_path=None,
            images_dir=self.images,
            dataset_classes=["cat", "dog"],
            labels_dir=self.labels,
            backup_dir=self.backup,
            score_threshold=0.25,
            prediction_classes=["cat", "dog"],
        )
        self.assertEqual(summary["labels_written"], 1)
        self.assertEqual(summary["boxes_written"], 2)
        self.assertEqual(summary["dropped_unmapped"], 0)
        lines = (self.labels / "img1.txt").read_text().strip().splitlines()
        self.assertEqual(lines[0], "0 0.500000 0.500000 0.400000 0.400000")
        self.assertEqual(lines[1], "1 0.300000 0.300000 0.100000 0.100000")

    def test_no_checkpoint_file_needs_to_exist(self):
        # checkpoint_path is None and nothing under self.root is a checkpoint —
        # if promote_labels tried to load one this would raise FileNotFoundError.
        promote_labels(
            self.preds,
            checkpoint_path=None,
            images_dir=self.images,
            dataset_classes=["cat", "dog"],
            labels_dir=self.labels,
            backup_dir=self.backup,
            score_threshold=0.25,
            prediction_classes=["cat", "dog"],
        )


if __name__ == "__main__":
    unittest.main()
