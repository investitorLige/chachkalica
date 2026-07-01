"""Tests for the model registry: promotion, eval request generation, eval ingest."""

import tempfile
from pathlib import Path
from unittest import mock

import yaml
from django.test import TestCase

from fleet.models import Dataset, FleetSettings
from training.models import (
    EvalRun,
    Experiment,
    ExperimentDataset,
    RunResult,
    TrainedModel,
    TrainingRun,
    TrainingSettings,
)
from training.services import autoeval, config_gen, ingest, promote


def _make_dataset_on_disk(source_root: Path, name: str, classes: list[str]) -> None:
    ds = source_root / name
    (ds / "images").mkdir(parents=True)
    (ds / "images" / "img1.jpg").write_bytes(b"")
    (ds / "labels").mkdir()
    (ds / "labels" / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (ds / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")


class RegistryTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.source = root / "source"
        self.source.mkdir()
        fs = FleetSettings.load()
        fs.source_dir = str(self.source)
        fs.target_dir = str(root / "target")
        fs.save()

        _make_dataset_on_disk(self.source, "ds1", ["helmet", "head", "vest"])
        self.ds1 = Dataset.objects.create(name="ds1")

        exp = Experiment.objects.create(name="exp1")
        run = TrainingRun.objects.create(experiment=exp, output_dir=str(root / "out"))
        self.rr = RunResult.objects.create(
            run=run, run_name="00-ds1-00-retinanet", model_arch="retinanet",
            train_dataset_name="ds1", best_epoch=5, best_loss=0.3,
            best_checkpoint=str(root / "out/run0/best.pt"),
            last_checkpoint=str(root / "out/run0/last.pt"),
            test_metrics={"map50": 0.8, "map50_95": 0.5},
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_promote_copies_checkpoint_classes_metrics(self):
        tm = promote.promote_run_result(self.rr, name="helmet-v1", stage=TrainedModel.STAGING)
        self.assertEqual(tm.arch, "retinanet")
        self.assertEqual(tm.checkpoint_path, self.rr.best_checkpoint)
        self.assertEqual(tm.classes, ["helmet", "head", "vest"])
        self.assertEqual(tm.num_classes, 3)
        self.assertEqual(tm.metrics["map50_95"], 0.5)
        self.assertEqual(tm.stage, TrainedModel.STAGING)
        self.assertEqual(tm.source_run_result, self.rr)

    def test_promote_dedupes_name(self):
        promote.promote_run_result(self.rr, name="dup")
        rr2 = RunResult.objects.create(
            run=self.rr.run, run_name="other", model_arch="yolox",
            train_dataset_name="ds1", last_checkpoint="/x/last.pt",
        )
        tm2 = promote.promote_run_result(rr2, name="dup")
        self.assertNotEqual(tm2.name, "dup")
        self.assertTrue(tm2.name.startswith("dup-"))

    def test_build_eval_request_shape(self):
        tm = promote.promote_run_result(self.rr, name="m1")
        eval_run = EvalRun.objects.create(
            trained_model=tm, dataset=self.ds1, label_source=EvalRun.SOURCE,
        )
        req = config_gen.build_eval_request(eval_run, "/out/eval-1")
        self.assertEqual(req["checkpoint_path"], tm.checkpoint_path)
        self.assertTrue(req["images"].endswith("ds1/images"))
        self.assertTrue(req["labels"].endswith("ds1/labels"))
        self.assertEqual(req["classes"], ["helmet", "head", "vest"])
        self.assertEqual(req["output_dir"], "/out/eval-1")

    def test_ingest_eval(self):
        tm = promote.promote_run_result(self.rr, name="m2")
        out = Path(self._tmp.name) / "evalout"
        out.mkdir()
        eval_run = EvalRun.objects.create(
            trained_model=tm, dataset=self.ds1, output_dir=str(out),
        )
        with open(out / "eval_result.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump({"metrics": {"map50": 0.66, "map50_95": 0.44}}, fh)

        self.assertTrue(ingest.eval_is_complete(out))
        ingest.ingest_eval(eval_run)
        eval_run.refresh_from_db()
        self.assertEqual(eval_run.metric("map50_95"), 0.44)


class AutoEvalTests(TestCase):
    """After-training auto-eval: promote every trained model, eval on the test set."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.source = root / "source"
        self.source.mkdir()
        fs = FleetSettings.load()
        fs.source_dir = str(self.source)
        fs.target_dir = str(root / "target")
        fs.save()

        ts = TrainingSettings.load()
        ts.configs_root = str(root / "configs")
        ts.runs_root = str(root / "runs")
        ts.save()

        _make_dataset_on_disk(self.source, "train_ds", ["object"])
        _make_dataset_on_disk(self.source, "test_ds", ["object"])
        self.train_ds = Dataset.objects.create(name="train_ds")
        self.test_ds = Dataset.objects.create(name="test_ds")

        self.exp = Experiment.objects.create(name="exp-auto")
        ExperimentDataset.objects.create(
            experiment=self.exp, dataset=self.train_ds, role=ExperimentDataset.TRAIN,
        )
        self.run = TrainingRun.objects.create(experiment=self.exp, output_dir=str(root / "out"))
        # Two trained models (checkpoints) + one that failed (no checkpoint).
        RunResult.objects.create(
            run=self.run, run_name="00-train_ds-00-yolox", model_arch="yolox",
            train_dataset_name="train_ds", best_checkpoint=str(root / "out/0/best.pt"),
        )
        RunResult.objects.create(
            run=self.run, run_name="01-train_ds-01-rtdetr", model_arch="rtdetr",
            train_dataset_name="train_ds", last_checkpoint=str(root / "out/1/last.pt"),
        )
        RunResult.objects.create(
            run=self.run, run_name="02-train_ds-02-rfdetr", model_arch="rfdetr",
            train_dataset_name="train_ds",  # no checkpoint — training failed
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _add_test_dataset(self):
        ExperimentDataset.objects.create(
            experiment=self.exp, dataset=self.test_ds, role=ExperimentDataset.TEST,
        )

    def test_noop_without_test_dataset(self):
        with mock.patch.object(autoeval, "_queue") as q:
            self.assertEqual(autoeval.schedule_test_evals(self.run), [])
            q.return_value.enqueue.assert_not_called()
        self.assertEqual(EvalRun.objects.count(), 0)
        self.assertEqual(TrainedModel.objects.count(), 0)

    def test_promotes_and_evals_every_checkpointed_model(self):
        self._add_test_dataset()
        with mock.patch.object(autoeval, "_queue") as q:
            queued = autoeval.schedule_test_evals(self.run)

        # Two models had checkpoints; the checkpoint-less one is skipped.
        self.assertEqual(len(queued), 2)
        self.assertEqual(TrainedModel.objects.count(), 2)
        self.assertEqual(EvalRun.objects.count(), 2)
        # Every eval targets the test dataset and got queued.
        for ev in EvalRun.objects.all():
            self.assertEqual(ev.dataset, self.test_ds)
            self.assertEqual(ev.status, EvalRun.QUEUED)
            self.assertTrue(ev.request_yaml_path)
        self.assertEqual(q.return_value.enqueue.call_count, 2)

    def test_carries_test_dataset_label_source(self):
        ExperimentDataset.objects.create(
            experiment=self.exp, dataset=self.test_ds, role=ExperimentDataset.TEST,
            label_source=ExperimentDataset.EXPLICIT,
            explicit_labels_path=str(self.source / "test_ds" / "labels"),
        )
        with mock.patch.object(autoeval, "_queue"):
            autoeval.schedule_test_evals(self.run)
        ev = EvalRun.objects.first()
        self.assertEqual(ev.label_source, ExperimentDataset.EXPLICIT)
        self.assertTrue(ev.explicit_labels_path.endswith("test_ds/labels"))
