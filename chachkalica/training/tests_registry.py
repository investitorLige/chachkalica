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

    def test_resolved_checkpoint_path_passes_through_when_file_exists(self):
        run_dir = Path(self._tmp.name) / "out/run0"
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_bytes(b"")
        (run_dir / "last.pt").write_bytes(b"")
        tm = promote.promote_run_result(self.rr, name="both-present")
        self.assertEqual(tm.resolved_checkpoint_path(), run_dir / "best.pt")

    def test_resolved_checkpoint_path_falls_back_to_sibling_last_pt(self):
        # No val dataset -> best_checkpoint is empty and only last.pt is ever
        # written, same as friendy_chachkalica/ml/train.py when best_epoch is
        # None. Simulates a promoted best.pt that has since gone missing (e.g.
        # pruned to save disk) while last.pt in the same run dir survives.
        run_dir = Path(self._tmp.name) / "out/run0"
        run_dir.mkdir(parents=True)
        (run_dir / "last.pt").write_bytes(b"")
        tm = promote.promote_run_result(self.rr, name="best-pruned")
        self.assertEqual(tm.checkpoint_path, str(run_dir / "best.pt"))
        self.assertEqual(tm.resolved_checkpoint_path(), run_dir / "last.pt")

    def test_resolved_checkpoint_path_reports_missing_when_neither_exists(self):
        tm = promote.promote_run_result(self.rr, name="both-missing")
        self.assertEqual(
            tm.resolved_checkpoint_path(), Path(self.rr.best_checkpoint))

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


class AutoEvalPipelineGeometryTests(TestCase):
    """The automatic test eval runs the geometry the model was *trained* with.

    An experiment with a pipeline gets PipelineEvalRuns instead of EvalRuns, and
    those rows have to carry every knob the training YAML did. This scheduler used
    to copy seven of the eleven by hand, so `tile_size_px`, `detector_min_box_size`,
    `detector_expand_ratio` and `merge_nms_iou` silently fell back to defaults —
    person crops with no size floor, percentage tiling for a model trained on
    fixed-pixel tiles — and the automatic number disagreed with the one the manual
    admin action produced for the same model.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
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

        _make_dataset_on_disk(self.source, "train_ds", ["helmet"])
        _make_dataset_on_disk(self.source, "test_ds", ["helmet"])
        self.train_ds = Dataset.objects.create(name="train_ds")
        self.test_ds = Dataset.objects.create(name="test_ds")

    def _run_with(self, **experiment_fields):
        from training import pipelines

        exp = Experiment.objects.create(
            name="exp-geo", pipeline=pipelines.PEOPLE_DETECT_FIRST, **experiment_fields)
        ExperimentDataset.objects.create(
            experiment=exp, dataset=self.train_ds, role=ExperimentDataset.TRAIN)
        ExperimentDataset.objects.create(
            experiment=exp, dataset=self.test_ds, role=ExperimentDataset.TEST)
        run = TrainingRun.objects.create(experiment=exp, output_dir=str(Path(self._tmp.name) / "out"))
        RunResult.objects.create(
            run=run, run_name="00-train_ds-00-yolox", model_arch="yolox",
            train_dataset_name="train_ds",
            best_checkpoint=str(Path(self._tmp.name) / "out/0/best.pt"),
        )
        with mock.patch.object(autoeval, "_queue"):
            autoeval.schedule_test_evals(run)
        from eval_pipelines.models import PipelineEvalRun

        return PipelineEvalRun.objects.get()

    def test_every_pipeline_knob_reaches_the_eval_row(self):
        pe = self._run_with(
            detector_expand_ratio=0.3,
            detector_min_box_size=320.0,
            merge_nms_iou=0.7,
            eval_score_threshold=0.4,
        )
        self.assertEqual(pe.detector_expand_ratio, 0.3)
        self.assertEqual(pe.detector_min_box_size, 320.0)
        self.assertEqual(pe.merge_nms_iou, 0.7)
        self.assertEqual(pe.score_threshold, 0.4)
        self.assertEqual(pe.pipeline, "people_detect_first")

    def test_the_min_box_size_floor_is_not_silently_dropped(self):
        # The Experiment default is 224 while the eval row's is blank ("chachak's
        # default", i.e. no floor at all), so the two disagree unless it's copied.
        pe = self._run_with()
        self.assertEqual(pe.detector_min_box_size, 224.0)

    def test_fixed_pixel_tiling_survives(self):
        from training import pipelines

        pe = self._run_with(tile_size_px=640, overlap=0.25)
        self.assertEqual(pe.tile_size_px, 640)
        self.assertEqual(pe.overlap, 0.25)
        # Percentage tiling stays unset, so it can't override the pixel size.
        self.assertIsNone(pe.tile_width_pct)
        self.assertEqual(pe.pipeline, pipelines.PEOPLE_DETECT_FIRST)

    def test_the_request_yaml_carries_the_trained_geometry(self):
        # The row is only a means to an end: what the eval actually runs is the
        # generated chachak request.
        pe = self._run_with(detector_min_box_size=320.0, merge_nms_iou=0.7)
        request = config_gen.build_pipeline_request(pe, Path(self._tmp.name) / "o")
        self.assertEqual(request["detector"]["min_box_size"], 320.0)
        self.assertEqual(request["merge_nms_iou"], 0.7)


class EvalCompareTests(TestCase):
    """The EvalRun analyze/compare action's underlying reshape."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.source = root / "source"
        self.source.mkdir()
        fs = FleetSettings.load()
        fs.source_dir = str(self.source)
        fs.target_dir = str(root / "target")
        fs.save()
        _make_dataset_on_disk(self.source, "testset", ["object"])
        self.ds = Dataset.objects.create(name="testset")
        self.run = TrainingRun.objects.create(
            experiment=Experiment.objects.create(name="e"), output_dir=str(root / "o"),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _eval(self, model_name, arch, metrics):
        rr = RunResult.objects.create(
            run=self.run, run_name=model_name, model_arch=arch,
            train_dataset_name="testset", best_checkpoint=f"/ck/{model_name}.pt",
        )
        tm = TrainedModel.objects.create(name=model_name, arch=arch, checkpoint_path=rr.best_checkpoint)
        return EvalRun.objects.create(trained_model=tm, dataset=self.ds, metrics=metrics)

    def test_compare_orders_and_flags_winner(self):
        from training.services import eval_analytics

        weak = self._eval("weak", "yolox", {"map50": 0.1, "map50_95": 0.05, "precision": 0.2})
        strong = self._eval("strong", "rfdetr", {"map50": 0.8, "map50_95": 0.5, "precision": 0.1})
        result = eval_analytics.compare([weak, strong])

        # Ordered best-map50 first.
        self.assertEqual([c["model"] for c in result["columns"]], ["strong", "weak"])
        # map50 row: strong (col 0) wins.
        map50_row = next(r for r in result["overall_rows"] if r["label"] == "mAP@50")
        self.assertTrue(map50_row["cells"][0]["is_best"])
        self.assertFalse(map50_row["cells"][1]["is_best"])
        # precision row: weak (now col 1) actually has the higher precision → its cell wins.
        prec_row = next(r for r in result["overall_rows"] if r["label"] == "Precision")
        self.assertTrue(prec_row["cells"][1]["is_best"])

    def test_missing_metric_and_single_column_no_false_winner(self):
        from training.services import eval_analytics

        only = self._eval("solo", "yolox", {"map50": 0.4})  # no map50_95 key
        result = eval_analytics.compare([only])
        map95_row = next(r for r in result["overall_rows"] if r["label"] == "mAP@50-95")
        self.assertIsNone(map95_row["cells"][0]["value"])
        # A single column is never flagged as "best" (nothing to compare against).
        map50_row = next(r for r in result["overall_rows"] if r["label"] == "mAP@50")
        self.assertFalse(map50_row["cells"][0]["is_best"])

    def test_eval_time_surfaced_in_comparison(self):
        from training.services import eval_analytics

        slow = self._eval("slow", "yolox", {"map50": 0.5, "eval_seconds": 42.0,
                                            "evaluated_at": "2026-07-03T10:00:00+00:00"})
        fast = self._eval("fast", "rfdetr", {"map50": 0.4, "eval_seconds": 7.5,
                                             "evaluated_at": "2026-07-03T11:00:00+00:00"})
        result = eval_analytics.compare([slow, fast])

        # The evaluated-at timestamp rides along on each column for the header.
        by_model = {c["model"]: c for c in result["columns"]}
        self.assertEqual(by_model["fast"]["evaluated_at"], "2026-07-03T11:00:00+00:00")

        # Eval time is a lower-is-better row: the faster eval's cell wins.
        time_row = next(r for r in result["overall_rows"] if r["label"] == "Eval time (s)")
        cells = {c["model"]: cell for c, cell in zip(result["columns"], time_row["cells"])}
        self.assertTrue(cells["fast"]["is_best"])
        self.assertFalse(cells["slow"]["is_best"])

        per_frame_row = next(r for r in result["overall_rows"] if r["label"] == "Eval per frame time (s)")
        per_frame_cells = {c["model"]: cell for c, cell in zip(result["columns"], per_frame_row["cells"])}
        self.assertIsNone(per_frame_cells["fast"]["value"])

    def test_eval_per_frame_time_derived_from_eval_seconds_and_images(self):
        from training.services import eval_analytics

        slow = self._eval("slow", "yolox", {"map50": 0.5, "eval_seconds": 20.0, "num_images": 4})
        fast = self._eval("fast", "rfdetr", {"map50": 0.4, "eval_seconds": 12.0, "num_images": 6})
        result = eval_analytics.compare([slow, fast])

        labels = [r["label"] for r in result["overall_rows"]]
        self.assertLess(labels.index("Eval time (s)"), labels.index("Eval per frame time (s)"))

        row = next(r for r in result["overall_rows"] if r["label"] == "Eval per frame time (s)")
        cells = {c["model"]: cell for c, cell in zip(result["columns"], row["cells"])}
        self.assertEqual(cells["slow"]["value"], 5.0)
        self.assertEqual(cells["fast"]["value"], 2.0)
        self.assertTrue(cells["fast"]["is_best"])
        self.assertFalse(cells["slow"]["is_best"])

    def test_per_class_rows_union_classes(self):
        from training.services import eval_analytics

        a = self._eval("a", "yolox", {"map50": 0.5, "per_class": {
            "0": {"class_name": "object", "ap50": 0.5, "ap50_95": 0.3}}})
        b = self._eval("b", "rtdetr", {"map50": 0.6, "per_class": {
            "0": {"class_name": "object", "ap50": 0.7, "ap50_95": 0.2}}})
        result = eval_analytics.compare([a, b])
        self.assertEqual(len(result["class_rows"]), 1)
        row = result["class_rows"][0]
        self.assertEqual(row["class_name"], "object")
        # b (col 0 after sort by map50) has higher ap50 → its cell wins.
        self.assertTrue(row["ap50"][0]["is_best"])

    def test_confusion_matrix_reshaped_for_heatmap(self):
        from training.services import eval_analytics

        # 2 classes (cat, dog) + background. 3 cats correct, 1 cat called dog, 1 cat
        # missed; 4 dogs correct; 2 spurious dog predictions (background row).
        run = self._eval("m", "rfdetr", {"map50": 0.5, "confusion_matrix": {
            "labels": ["cat", "dog"],
            "background_index": 2,
            "iou_threshold": 0.5,
            "conf_threshold": 0.25,
            "matrix": [
                [3, 1, 1],   # truth=cat
                [0, 4, 0],   # truth=dog
                [0, 2, 0],   # truth=background (false positives)
            ],
        }})
        result = eval_analytics.compare([run])
        self.assertEqual(len(result["confusion_matrices"]), 1)
        cm = result["confusion_matrices"][0]
        self.assertEqual(cm["axis"], ["cat", "dog", "background"])
        self.assertEqual(cm["iou_threshold"], 0.5)

        cat_row = cm["rows"][0]
        # Diagonal cat/cat is flagged and shaded by its row share (3 of 5).
        self.assertTrue(cat_row["cells"][0]["is_diagonal"])
        self.assertAlmostEqual(cat_row["cells"][0]["intensity"], 0.6)
        # The cat-called-dog cell is a confusion, not a diagonal.
        self.assertFalse(cat_row["cells"][1]["is_diagonal"])
        self.assertTrue(cat_row["cells"][2]["is_background"])
        # The background row (false positives) is flagged as such.
        self.assertTrue(cm["rows"][2]["is_background"])

    def test_confusion_matrix_absent_when_metric_missing(self):
        from training.services import eval_analytics

        run = self._eval("old", "yolox", {"map50": 0.5})  # pre-confusion-matrix eval
        result = eval_analytics.compare([run])
        self.assertEqual(result["confusion_matrices"], [])
