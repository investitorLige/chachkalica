"""Tests for the "Fine-tune model…" action (training.services.finetune) and its
knock-on effects: auto-promotion (training.jobs) and the split between the
"Training runs" and "Fine-tuning runs" admin tabs.
"""

import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.contrib.auth.models import User
from django.test import TestCase

from fleet.models import Dataset, FleetSettings
from training import admin as training_admin
from training import jobs, model_specs, pipelines
from training.models import (
    Experiment,
    ExperimentDataset,
    ExperimentModel,
    FineTuningRun,
    RunResult,
    TrainedModel,
    TrainingRun,
    TrainingSettings,
)
from training.services import finetune, pipeline_meta, promote
from training.tests import _make_dataset_on_disk


class FreezeParamsTests(TestCase):
    def test_unchecked_is_a_no_op_for_every_arch(self):
        for arch, _label in ExperimentModel.ARCH_CHOICES:
            self.assertEqual(model_specs.freeze_params(arch, False), {})

    def test_rfdetr_uses_its_own_bool_kwarg(self):
        self.assertEqual(model_specs.freeze_params("rfdetr", True), {"freeze_backbone": True})

    def test_every_other_arch_floors_trainable_backbone_layers(self):
        for arch, _label in ExperimentModel.ARCH_CHOICES:
            if arch == "rfdetr":
                continue
            self.assertEqual(
                model_specs.freeze_params(arch, True), {"trainable_backbone_layers": 0}
            )


class CreateRunTests(TestCase):
    """finetune.create_run assembles the Experiment/dataset/model/run rows."""

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

        _make_dataset_on_disk(source, "ft-ds", ["helmet", "vest"])
        self.dataset = Dataset.objects.create(name="ft-ds")

        self.source_model = TrainedModel.objects.create(
            name="base-model", arch=ExperimentModel.YOLOX, checkpoint_path="/ckpts/base/best.pt",
            num_classes=2,
            pipeline_metadata=pipeline_meta.normalize({
                "pipeline": pipelines.BATCH_DETECT, "tile_size_px": 640,
            }),
        )

    def _request(self, **overrides):
        fields = dict(train_dataset=self.dataset, epochs=5, batch_size=2, lr=1e-5)
        fields.update(overrides)
        return finetune.FineTuneRequest(**fields)

    def test_no_checkpoint_refuses(self):
        self.source_model.checkpoint_path = ""
        self.source_model.save()
        with self.assertRaises(ValueError):
            finetune.create_run(self.source_model, self._request())

    def test_builds_a_flagged_experiment_seeded_from_the_source_model(self):
        with mock.patch.object(finetune, "django_rq") as rq:
            run = finetune.create_run(self.source_model, self._request())

        rq.get_queue.return_value.enqueue.assert_called_once_with(
            jobs.run_training, run.pk, job_timeout=jobs.JOB_TIMEOUT
        )

        exp = run.experiment
        self.assertTrue(exp.is_finetune)
        self.assertEqual(exp.finetune_source, self.source_model)
        self.assertEqual(exp.epochs, 5)
        self.assertEqual(exp.batch_size, 2)
        self.assertEqual(exp.lr, 1e-5)
        # Pipeline geometry carried over from the source model's frozen metadata,
        # not asked for again.
        self.assertEqual(exp.pipeline, pipelines.BATCH_DETECT)
        self.assertEqual(exp.tile_size_px, 640)

        model_row = exp.models.get()
        self.assertEqual(model_row.arch, ExperimentModel.YOLOX)
        self.assertEqual(model_row.num_classes, 2)
        self.assertEqual(
            model_row.params[model_specs.INIT_CHECKPOINT_KEY], "/ckpts/base/best.pt"
        )
        self.assertNotIn("trainable_backbone_layers", model_row.params)

        dataset_row = exp.datasets.get()
        self.assertEqual(dataset_row.dataset, self.dataset)
        self.assertEqual(dataset_row.role, ExperimentDataset.TRAIN)

        self.assertEqual(run.status, TrainingRun.QUEUED)
        self.assertTrue(run.config_yaml_path)
        self.assertTrue(Path(run.config_yaml_path).is_file())

    def test_freeze_backbone_checkbox_sets_the_arch_appropriate_param(self):
        with mock.patch.object(finetune, "django_rq"):
            run = finetune.create_run(self.source_model, self._request(freeze_backbone=True))
        self.assertEqual(run.experiment.models.get().params["trainable_backbone_layers"], 0)

    def test_val_dataset_is_an_optional_second_row(self):
        with mock.patch.object(finetune, "django_rq"):
            run = finetune.create_run(
                self.source_model, self._request(val_dataset=self.dataset)
            )
        roles = sorted(d.role for d in run.experiment.datasets.all())
        self.assertEqual(roles, [ExperimentDataset.TRAIN, ExperimentDataset.VAL])

    def test_experiment_name_is_deduped(self):
        Experiment.objects.create(name="base-model-ft")
        with mock.patch.object(finetune, "django_rq"):
            run = finetune.create_run(self.source_model, self._request())
        self.assertEqual(run.experiment.name, "base-model-ft-2")


class AutoPromoteFinetuneTests(TestCase):
    """jobs._auto_promote_finetune registers a new TrainedModel with no click."""

    def _run_result(self, *, is_finetune, source=None, error=""):
        exp = Experiment.objects.create(
            name=f"exp-{'ft' if is_finetune else 'plain'}-{RunResult.objects.count()}",
            is_finetune=is_finetune, finetune_source=source,
        )
        run = TrainingRun.objects.create(experiment=exp)
        rr = RunResult.objects.create(
            run=run, run_name="r0", model_arch=ExperimentModel.YOLOX,
            best_checkpoint="/runs/r0/best.pt", error=error,
        )
        return run, rr

    def test_finetune_run_promotes_automatically_with_parent_lineage(self):
        source = TrainedModel.objects.create(
            name="parent", arch=ExperimentModel.YOLOX, checkpoint_path="/ckpts/parent.pt",
        )
        run, rr = self._run_result(is_finetune=True, source=source)

        jobs._auto_promote_finetune(run)

        child = TrainedModel.objects.get(source_run_result=rr)
        self.assertEqual(child.parent_model, source)
        self.assertEqual(child.name, run.experiment.name)

    def test_failed_sub_run_is_not_promoted(self):
        source = TrainedModel.objects.create(
            name="parent2", arch=ExperimentModel.YOLOX, checkpoint_path="/ckpts/parent2.pt",
        )
        run, _rr = self._run_result(is_finetune=True, source=source, error="crashed")

        jobs._auto_promote_finetune(run)

        self.assertFalse(TrainedModel.objects.filter(source_run_result__run=run).exists())

    def test_a_promotion_failure_is_recorded_but_does_not_raise(self):
        run, rr = self._run_result(is_finetune=True)
        rr.best_checkpoint = ""  # promote_run_result raises ValueError: no checkpoint
        rr.save()

        jobs._auto_promote_finetune(run)  # must not raise

        run.refresh_from_db()
        self.assertIn("auto-promote failed", run.last_error)


class FineTuneAdminActionTests(TestCase):
    """The 'Fine-tune model…' action end-to-end through the admin, job mocked out."""

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

        _make_dataset_on_disk(source, "ft-ds", ["helmet"])
        self.dataset = Dataset.objects.create(name="ft-ds")

        self.source_model = TrainedModel.objects.create(
            name="base-model", arch=ExperimentModel.YOLOX, checkpoint_path="/ckpts/base/best.pt",
        )
        self.admin_user = User.objects.create_superuser("admin", "admin@example.com", "pw")
        self.client.force_login(self.admin_user)

    def _post(self, **extra):
        return self.client.post(
            "/admin/training/trainedmodel/",
            {
                "action": "fine_tune",
                ACTION_CHECKBOX_NAME: [str(self.source_model.pk)],
                "index": "0",
                **extra,
            },
        )

    def test_first_click_renders_the_form_without_side_effects(self):
        resp = self._post()
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, "admin/training/fine_tune.html")
        self.assertEqual(TrainingRun.objects.count(), 0)

    def test_apply_creates_and_queues_a_run(self):
        with mock.patch.object(finetune, "django_rq") as rq:
            resp = self._post(apply="1", train_dataset=str(self.dataset.pk))

        self.assertEqual(resp.status_code, 302)
        rq.get_queue.return_value.enqueue.assert_called_once()
        run = TrainingRun.objects.get()
        self.assertTrue(run.experiment.is_finetune)
        self.assertEqual(run.experiment.finetune_source, self.source_model)

    def test_apply_without_a_dataset_is_rejected(self):
        # A rejected action message_user()s a warning and returns None, which
        # Django admin turns into a redirect back to the changelist — same as
        # the sibling "evaluate" action's own validation failures.
        with mock.patch.object(finetune, "django_rq"):
            resp = self._post(apply="1")
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(TrainingRun.objects.count(), 0)

    def test_early_stopping_without_a_val_dataset_is_rejected(self):
        with mock.patch.object(finetune, "django_rq"):
            resp = self._post(
                apply="1", train_dataset=str(self.dataset.pk), early_stopping_patience="3",
            )
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(TrainingRun.objects.count(), 0)


class TabSeparationTests(TestCase):
    """Fine-tune experiments/runs show up under their own tab, not the main ones."""

    def test_experiment_admin_hides_finetune_rows(self):
        Experiment.objects.create(name="hand-built")
        Experiment.objects.create(name="auto-ft", is_finetune=True)
        names = set(training_admin.ExperimentAdmin(Experiment, None).get_queryset(None)
                    .values_list("name", flat=True))
        self.assertEqual(names, {"hand-built"})

    def test_training_run_admin_hides_finetune_rows_finetuning_run_admin_shows_only_them(self):
        plain = TrainingRun.objects.create(experiment=Experiment.objects.create(name="plain-exp"))
        ft_run = TrainingRun.objects.create(
            experiment=Experiment.objects.create(name="ft-exp", is_finetune=True)
        )

        plain_ids = set(
            training_admin.TrainingRunAdmin(TrainingRun, None).get_queryset(None)
            .values_list("pk", flat=True)
        )
        ft_ids = set(
            training_admin.FineTuningRunAdmin(FineTuningRun, None).get_queryset(None)
            .values_list("pk", flat=True)
        )
        self.assertEqual(plain_ids, {plain.pk})
        self.assertEqual(ft_ids, {ft_run.pk})
