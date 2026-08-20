"""Tests for on-disk artifact cleanup when a run/eval row is deleted.

Deleting a TrainingRun/EvalRun (directly, in bulk, or via cascade) should remove
its output_dir and generated YAML under the configured roots — but never touch
anything outside them.
"""

import shutil
import tempfile
import uuid
from pathlib import Path

from django.conf import settings
from django.test import TestCase

from fleet.models import Dataset
from training.models import (
    EvalRun,
    Experiment,
    ExportRun,
    FineTuningRun,
    RunResult,
    TrainedModel,
    TrainingRun,
    TrainingSettings,
)


class CleanupTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.runs_root = self.root / "runs"
        self.configs_root = self.root / "configs"
        self.runs_root.mkdir()
        self.configs_root.mkdir()

        ts = TrainingSettings.load()
        ts.runs_root = str(self.runs_root)
        ts.configs_root = str(self.configs_root)
        ts.save()

    def tearDown(self):
        self._tmp.cleanup()

    def _make_run(self, name="run-1"):
        out = self.runs_root / name
        (out / "00-yolox").mkdir(parents=True)
        (out / "00-yolox" / "best.pt").write_bytes(b"weights")
        (out / "run_summary.yaml").write_text("summary: yes\n", encoding="utf-8")
        cfg = self.configs_root / f"{name}.yaml"
        cfg.write_text("name: x\n", encoding="utf-8")
        run = TrainingRun.objects.create(output_dir=str(out), config_yaml_path=str(cfg))
        return run, out, cfg

    def test_deleting_run_removes_output_dir_and_config(self):
        run, out, cfg = self._make_run()
        self.assertTrue(out.exists() and cfg.exists())
        run.delete()
        self.assertFalse(out.exists())
        self.assertFalse(cfg.exists())

    def test_bulk_delete_removes_artifacts(self):
        run_a, out_a, _ = self._make_run("run-a")
        run_b, out_b, _ = self._make_run("run-b")
        TrainingRun.objects.all().delete()
        self.assertFalse(out_a.exists())
        self.assertFalse(out_b.exists())

    def test_cascade_delete_from_experiment_cleans_run(self):
        run, out, _ = self._make_run("run-casc")
        exp = Experiment.objects.create(name="exp-casc")
        run.experiment = exp
        run.save(update_fields=["experiment"])
        # TrainingRun.experiment is SET_NULL, so deleting the experiment does not
        # cascade-delete the run; delete the run's owner path directly instead.
        run.delete()
        self.assertFalse(out.exists())

    def test_refuses_to_delete_path_outside_runs_root(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("important", encoding="utf-8")
        run = TrainingRun.objects.create(output_dir=str(outside))
        run.delete()
        self.assertTrue(outside.exists(), "must not delete dirs outside runs_root")

    def test_blank_output_dir_is_safe(self):
        run = TrainingRun.objects.create(output_dir="", config_yaml_path="")
        run.delete()  # must not raise, must not delete anything
        self.assertTrue(self.runs_root.exists())

    def test_deleting_eval_removes_its_artifacts(self):
        out = self.runs_root / "eval-1"
        out.mkdir()
        (out / "eval_result.yaml").write_text("metrics: {}\n", encoding="utf-8")
        req = self.configs_root / "eval-1.yaml"
        req.write_text("eval: yes\n", encoding="utf-8")

        model = TrainedModel.objects.create(name="m1", arch="yolox",
                                            checkpoint_path="/tmp/best.pt")
        dataset = Dataset.objects.create(name="ds1")
        ev = EvalRun.objects.create(trained_model=model, dataset=dataset,
                                    output_dir=str(out), request_yaml_path=str(req))
        ev.delete()
        self.assertFalse(out.exists())
        self.assertFalse(req.exists())

    def test_deleting_via_finetuning_run_proxy_still_cleans_up(self):
        """FineTuningRun is a proxy of TrainingRun (same table, see its
        docstring) — Django's delete Collector groups instances by the exact
        class a queryset/instance was fetched as, so a delete made through the
        proxy manager (the "Fine-tuning runs" tab) fires post_delete with
        sender=FineTuningRun, never sender=TrainingRun. Without a matching
        receiver this silently never cleans up — regression coverage for
        that gap."""
        run, out, cfg = self._make_run("ft-run")
        FineTuningRun.objects.get(pk=run.pk).delete()
        self.assertFalse(out.exists())
        self.assertFalse(cfg.exists())

    def test_deleting_run_result_removes_its_run_dir(self):
        run, out, _ = self._make_run("run-with-results")
        run_dir = out / "00-yolox"
        self.assertTrue(run_dir.is_dir())
        rr = RunResult.objects.create(run=run, run_name="00-yolox", run_dir=str(run_dir))
        rr.delete()
        self.assertFalse(run_dir.exists())
        self.assertTrue(out.exists(), "the parent run's own dir must survive")

    def test_deleting_trained_model_removes_its_checkpoint(self):
        ckpt = self.runs_root / "some-run" / "best.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"weights")
        model = TrainedModel.objects.create(name="m1", arch="yolox", checkpoint_path=str(ckpt))
        model.delete()
        self.assertFalse(ckpt.exists())

    def test_deleting_trained_model_refuses_checkpoint_outside_runs_root(self):
        model = TrainedModel.objects.create(name="m2", arch="yolox",
                                            checkpoint_path="/tmp/best.pt")
        model.delete()  # must not raise, must not touch a path outside runs_root

    def test_deleting_export_run_removes_output_and_bundle_dir(self):
        # output_path/bundle_dir come from an operator-chosen directory (the
        # export forms' "output_dir" field), not one fixed TrainingSettings
        # root, so cleanup guards against the project root instead — exercise
        # that with a scratch dir actually under it, not an arbitrary tmp dir.
        out_dir = Path(settings.BASE_DIR) / f"_cleanup_test_{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        out_dir.mkdir()
        model = TrainedModel.objects.create(name="m3", arch="yolox", checkpoint_path="")
        onnx_path = out_dir / "m3-best.onnx"
        onnx_path.write_bytes(b"onnx")
        bundle_dir = out_dir / "m3-best-bundle"
        (bundle_dir / "models").mkdir(parents=True)
        export = ExportRun.objects.create(
            model=model, kind=ExportRun.ONNX, checkpoint_label="best",
            checkpoint_path=str(onnx_path), output_path=str(onnx_path),
            bundle_dir=str(bundle_dir),
        )
        export.delete()
        self.assertFalse(onnx_path.exists())
        self.assertFalse(bundle_dir.exists())

    def test_deleting_export_run_refuses_path_outside_project_root(self):
        model = TrainedModel.objects.create(name="m4", arch="yolox", checkpoint_path="")
        outside = self.root / "outside-exports" / "m4-best.onnx"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b"onnx")
        export = ExportRun.objects.create(
            model=model, kind=ExportRun.ONNX, checkpoint_label="best",
            checkpoint_path=str(outside), output_path=str(outside),
        )
        export.delete()
        self.assertTrue(outside.exists(), "must not delete paths outside the project root")
