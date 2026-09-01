"""Tests for the "Extend run (more epochs)…" action (training.services.extend)."""

import tempfile
from pathlib import Path
from unittest import mock

import yaml
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.contrib.auth.models import User
from django.test import TestCase

from fleet.models import Dataset, FleetSettings
from training import admin as training_admin  # noqa: F401 - registers the admin action
from training import jobs
from training.models import (
    Experiment,
    ExperimentDataset,
    ExperimentModel,
    TrainingRun,
    TrainingSettings,
)
from training.services import config_gen, extend
from training.tests import _make_dataset_on_disk


class ExtendRunTests(TestCase):
    """extend.extend_run: bump epochs/lr, rewrite config, clear the stale result."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        source, target = root / "source", root / "target"
        source.mkdir()
        target.mkdir()

        fs = FleetSettings.load()
        fs.source_dir, fs.target_dir = str(source), str(target)
        fs.save()

        ts = TrainingSettings.load()
        ts.configs_root = str(root / "configs")
        ts.runs_root = str(root / "runs")
        ts.save()

        _make_dataset_on_disk(source, "ext-ds", ["helmet"])
        dataset = Dataset.objects.create(name="ext-ds")

        self.experiment = Experiment.objects.create(name="ext-exp", epochs=10, lr=1e-4)
        ExperimentDataset.objects.create(
            experiment=self.experiment, dataset=dataset, role=ExperimentDataset.TRAIN,
        )
        ExperimentModel.objects.create(
            experiment=self.experiment, arch=ExperimentModel.YOLOX, num_classes=1,
        )
        self.run = TrainingRun.objects.create(experiment=self.experiment)
        config_gen.write_config(self.experiment, self.run)

        # write_config only lays down the YAML; simulate what a finished
        # training run actually leaves on disk under its output_dir.
        self.sub_run_dir = Path(self.run.output_dir) / "ext-exp"
        self.sub_run_dir.mkdir(parents=True)
        (self.sub_run_dir / "last.pt").write_bytes(b"fake-checkpoint")
        self.result_path = self.sub_run_dir / "result.yaml"
        self.result_path.write_text("best_epoch: 10\n", encoding="utf-8")

    def test_no_checkpoint_refuses(self):
        (self.sub_run_dir / "last.pt").unlink()
        with self.assertRaises(ValueError):
            extend.extend_run(self.run, 5)

    def test_no_output_dir_refuses(self):
        run = TrainingRun.objects.create(experiment=self.experiment)  # never write_config'd
        with self.assertRaises(ValueError):
            extend.extend_run(run, 5)

    def test_zero_or_negative_epochs_refuses(self):
        with self.assertRaises(ValueError):
            extend.extend_run(self.run, 0)

    def test_bumps_epochs_and_rewrites_config(self):
        extend.extend_run(self.run, 5)

        self.experiment.refresh_from_db()
        self.assertEqual(self.experiment.epochs, 15)
        self.assertEqual(self.experiment.lr, 1e-4)  # untouched when new_lr is omitted

        data = yaml.safe_load(Path(self.run.config_yaml_path).read_text())
        self.assertEqual(data["training"]["epochs"], 15)
        self.assertEqual(data["training"]["optimizer"]["lr"], 1e-4)

    def test_new_lr_updates_experiment_and_config(self):
        extend.extend_run(self.run, 5, new_lr=1e-6)

        self.experiment.refresh_from_db()
        self.assertEqual(self.experiment.lr, 1e-6)
        data = yaml.safe_load(Path(self.run.config_yaml_path).read_text())
        self.assertEqual(data["training"]["optimizer"]["lr"], 1e-6)

    def test_stale_result_yaml_is_moved_aside(self):
        extend.extend_run(self.run, 5)

        self.assertFalse(self.result_path.exists())
        leftovers = list(self.sub_run_dir.glob("result.yaml.pre-extend-*"))
        self.assertEqual(len(leftovers), 1)
        self.assertEqual(leftovers[0].read_text(), "best_epoch: 10\n")


class ExtendRunAdminActionTests(TestCase):
    """The "Extend run…" action end-to-end through the admin, job mocked out."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        source, target = root / "source", root / "target"
        source.mkdir()
        target.mkdir()

        fs = FleetSettings.load()
        fs.source_dir, fs.target_dir = str(source), str(target)
        fs.save()

        ts = TrainingSettings.load()
        ts.configs_root = str(root / "configs")
        ts.runs_root = str(root / "runs")
        ts.save()

        _make_dataset_on_disk(source, "ext-ds2", ["helmet"])
        dataset = Dataset.objects.create(name="ext-ds2")

        self.experiment = Experiment.objects.create(name="ext-exp2", epochs=10, lr=1e-4)
        ExperimentDataset.objects.create(
            experiment=self.experiment, dataset=dataset, role=ExperimentDataset.TRAIN,
        )
        ExperimentModel.objects.create(
            experiment=self.experiment, arch=ExperimentModel.YOLOX, num_classes=1,
        )
        self.run = TrainingRun.objects.create(experiment=self.experiment, status=TrainingRun.OK)
        config_gen.write_config(self.experiment, self.run)
        sub_run_dir = Path(self.run.output_dir) / "ext-exp2"
        sub_run_dir.mkdir(parents=True)
        (sub_run_dir / "last.pt").write_bytes(b"fake-checkpoint")

        self.admin_user = User.objects.create_superuser("admin2", "admin2@example.com", "pw")
        self.client.force_login(self.admin_user)

    def _post(self, **extra):
        return self.client.post(
            "/admin/training/trainingrun/",
            {
                "action": "extend_run",
                ACTION_CHECKBOX_NAME: [str(self.run.pk)],
                "index": "0",
                **extra,
            },
        )

    def test_first_click_renders_the_form_without_side_effects(self):
        resp = self._post()
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, "admin/training/extend_run.html")
        self.experiment.refresh_from_db()
        self.assertEqual(self.experiment.epochs, 10)

    def test_apply_extends_and_queues_a_resume(self):
        with mock.patch.object(training_admin, "_queue") as queue_factory:
            resp = self._post(apply="1", additional_epochs="7", new_lr="1e-6")

        self.assertEqual(resp.status_code, 302)
        queue_factory.return_value.enqueue.assert_called_once_with(
            jobs.run_training, self.run.pk, resume=True, job_timeout=jobs.JOB_TIMEOUT,
        )
        self.experiment.refresh_from_db()
        self.assertEqual(self.experiment.epochs, 17)
        self.assertEqual(self.experiment.lr, 1e-6)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, TrainingRun.QUEUED)

    def test_apply_without_additional_epochs_is_rejected(self):
        with mock.patch.object(training_admin, "_queue") as queue_factory:
            resp = self._post(apply="1")
        self.assertEqual(resp.status_code, 302)
        queue_factory.return_value.enqueue.assert_not_called()
        self.experiment.refresh_from_db()
        self.assertEqual(self.experiment.epochs, 10)

    def test_running_run_is_rejected(self):
        self.run.status = TrainingRun.RUNNING
        self.run.save(update_fields=["status"])
        with mock.patch.object(training_admin, "_queue") as queue_factory:
            resp = self._post(apply="1", additional_epochs="5")
        self.assertEqual(resp.status_code, 302)
        queue_factory.return_value.enqueue.assert_not_called()
