"""Tests for ingesting a finished run's trainer output into RunResult rows."""

import tempfile
from pathlib import Path

import yaml
from django.test import TestCase

from training.models import Experiment, RunResult, TrainingRun
from training.services import ingest


def _write(path: Path, value):
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(value, fh, sort_keys=False)


class IngestTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.exp = Experiment.objects.create(name="exp1")
        self.run = TrainingRun.objects.create(experiment=self.exp, output_dir=str(self.out))

        _write(self.out / "results.yaml", [
            {
                "run_index": 0, "model": "retinanet", "train_dataset": "ds1",
                "run_name": "run0_retinanet_ds1", "run_dir": str(self.out / "run0"),
                "best_epoch": 7, "best_loss": 0.42,
                "best_checkpoint": str(self.out / "run0/best.pt"),
                "last_checkpoint": str(self.out / "run0/last.pt"),
                "test_metrics": None,
            },
            {
                "run_index": 1, "model": "yolox", "train_dataset": "ds1",
                "run_name": "run1_yolox_ds1", "run_dir": str(self.out / "run1"),
                "best_epoch": None, "best_loss": None,
                "best_checkpoint": None,
                "last_checkpoint": str(self.out / "run1/last.pt"),
                "test_metrics": None,
            },
        ])
        _write(self.out / "test_results.yaml", [
            {"run_index": 0, "run_name": "run0_retinanet_ds1", "model": "retinanet",
             "eval_dataset": "holdout", "metrics": {"map50": 0.81, "map50_95": 0.55}},
            {"run_index": 1, "run_name": "run1_yolox_ds1", "model": "yolox",
             "eval_dataset": "holdout", "metrics": {"map50": 0.77, "map50_95": 0.49}},
        ])
        _write(self.out / "run_summary.yaml", {
            "output_dir": str(self.out), "num_train_runs": 2, "checkpoint": "best",
        })

    def tearDown(self):
        self._tmp.cleanup()

    def test_is_complete(self):
        self.assertTrue(ingest.is_complete(self.out))

    def test_ingest_creates_rows_with_metrics(self):
        result = ingest.ingest_run(self.run)
        self.assertEqual(result["run_results"], 2)
        self.assertEqual(RunResult.objects.filter(run=self.run).count(), 2)

        r0 = RunResult.objects.get(run=self.run, run_name="run0_retinanet_ds1")
        self.assertEqual(r0.model_arch, "retinanet")
        self.assertEqual(r0.train_dataset_name, "ds1")
        self.assertEqual(r0.best_epoch, 7)
        self.assertEqual(r0.test_metrics["map50_95"], 0.55)
        self.assertEqual(r0.metric("map50"), 0.81)

        self.run.refresh_from_db()
        self.assertEqual(self.run.results["num_train_runs"], 2)

    def test_ingest_is_idempotent(self):
        ingest.ingest_run(self.run)
        ingest.ingest_run(self.run)
        self.assertEqual(RunResult.objects.filter(run=self.run).count(), 2)

    def test_incomplete_when_no_summary(self):
        (self.out / "run_summary.yaml").unlink()
        self.assertFalse(ingest.is_complete(self.out))

    def test_val_only_metrics_fallback(self):
        (self.out / "test_results.yaml").unlink()
        _write(self.out / "val_results.yaml", [
            {"run_index": 0, "run_name": "run0_retinanet_ds1", "model": "retinanet",
             "eval_dataset": "ds1", "metrics": {"map50": 0.6}},
        ])
        ingest.ingest_run(self.run)
        r0 = RunResult.objects.get(run=self.run, run_name="run0_retinanet_ds1")
        self.assertIsNone(r0.test_metrics)
        self.assertEqual(r0.val_metrics["map50"], 0.6)
        self.assertEqual(r0.metric("map50"), 0.6)  # primary falls back to val
