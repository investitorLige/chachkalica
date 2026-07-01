"""Tests for the per-epoch training-progress reader (training.services.progress)."""

import tempfile
from pathlib import Path

import yaml
from django.test import TestCase

from training.services import progress


def _write_history(run_dir: Path, epochs: list[dict]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "history.yaml").write_text(yaml.safe_dump(epochs), encoding="utf-8")


class ProgressTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_or_empty_output_dir(self):
        self.assertEqual(progress.run_histories(None), [])
        self.assertEqual(progress.run_histories(self.out / "nope"), [])
        self.assertEqual(progress.run_histories(self.out), [])  # no run dirs yet

    def test_reads_and_sorts_histories(self):
        _write_history(self.out / "01-b", [{"epoch": 1, "train": {"loss": 0.5}, "is_best": True}])
        _write_history(self.out / "00-a", [
            {"epoch": 1, "train": {"loss": 0.9}, "val": {"loss": 0.8}, "lr": 1e-4, "is_best": True},
            {"epoch": 2, "train": {"loss": 0.7}, "val": {"loss": 0.85}, "lr": 1e-4, "is_best": False},
        ])
        histories = progress.run_histories(self.out)

        self.assertEqual([h["run_name"] for h in histories], ["00-a", "01-b"])  # sorted
        self.assertEqual(len(histories[0]["epochs"]), 2)

    def test_epoch_line_formats_values_and_best_marker(self):
        line = progress.epoch_line(
            {"epoch": 3, "train": {"loss": 0.1234}, "val": {"loss": 0.099}, "lr": 1e-4, "is_best": True}
        )
        self.assertIn("epoch 3", line)
        self.assertIn("train_loss=0.1234", line)
        self.assertIn("val_loss=0.0990", line)
        self.assertIn("lr=1.00e-04", line)
        self.assertIn("best", line)

    def test_epoch_line_handles_missing_val(self):
        line = progress.epoch_line({"epoch": 1, "train": {"loss": 0.5}, "val": None, "lr": None})
        self.assertIn("val_loss=—", line)
        self.assertIn("lr=—", line)
        self.assertNotIn("best", line)

    def test_best_epoch_entry_matches_best_epoch_number(self):
        _write_history(self.out / "run", [
            {"epoch": 1, "train": {"loss": 0.9}, "val": {"loss": 0.8}, "is_best": True},
            {"epoch": 2, "train": {"loss": 0.7}, "val": {"loss": 0.6}, "is_best": True},
            {"epoch": 3, "train": {"loss": 0.65}, "val": {"loss": 0.7}, "is_best": False},
        ])
        entry = progress.best_epoch_entry(self.out / "run", best_epoch=2)
        self.assertEqual(entry["epoch"], 2)
        self.assertEqual(entry["val"]["loss"], 0.6)

    def test_best_epoch_entry_falls_back_to_last_is_best(self):
        _write_history(self.out / "run", [
            {"epoch": 1, "train": {"loss": 0.9}, "is_best": True},
            {"epoch": 2, "train": {"loss": 0.7}, "is_best": True},
            {"epoch": 3, "train": {"loss": 0.8}, "is_best": False},
        ])
        entry = progress.best_epoch_entry(self.out / "run")  # no best_epoch given
        self.assertEqual(entry["epoch"], 2)  # last flagged is_best

    def test_best_epoch_entry_none_when_no_history(self):
        self.assertIsNone(progress.best_epoch_entry(None))
        self.assertIsNone(progress.best_epoch_entry(self.out / "missing"))
        _write_history(self.out / "nobest", [{"epoch": 1, "train": {"loss": 0.5}, "is_best": False}])
        self.assertIsNone(progress.best_epoch_entry(self.out / "nobest"))
