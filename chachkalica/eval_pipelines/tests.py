"""Tests for chachak pipeline evals: request generation and metric ingest."""

import tempfile
from pathlib import Path
from unittest import mock

import yaml
from django.contrib.admin.sites import AdminSite
from django.test import RequestFactory, TestCase

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
from training.services import autoeval, config_gen, ingest, promote, runner

from eval_pipelines.admin import CombinedEvalAdmin
from eval_pipelines.models import (
    BaseEval,
    BatchDetectEval,
    BatchPeopleEval,
    ChainEval,
    CombinedEval,
    PeopleDetectFirstEval,
    PipelineEvalRun,
)


def _make_dataset_on_disk(source_root: Path, name: str, classes: list[str]) -> None:
    ds = source_root / name
    (ds / "images").mkdir(parents=True)
    (ds / "images" / "img1.jpg").write_bytes(b"")
    (ds / "labels").mkdir()
    (ds / "labels" / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (ds / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")


class PipelineEvalSetup(TestCase):
    """Shared fixture: an on-disk ``ds1`` dataset + a promoted model ``m1``."""

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
        rr = RunResult.objects.create(
            run=run, run_name="00-ds1-00-retinanet", model_arch="retinanet",
            train_dataset_name="ds1", best_epoch=5, best_loss=0.3,
            best_checkpoint=str(root / "out/run0/best.pt"),
            last_checkpoint=str(root / "out/run0/last.pt"),
            test_metrics={"map50": 0.8, "map50_95": 0.5},
        )
        self.tm = promote.promote_run_result(rr, name="m1")

    def tearDown(self):
        self._tmp.cleanup()

    def _make(self, **kwargs):
        return PipelineEvalRun.objects.create(
            trained_model=self.tm, dataset=self.ds1,
            label_source=PipelineEvalRun.SOURCE, **kwargs,
        )


class PipelineRequestTests(PipelineEvalSetup):
    def test_build_request_batch_detect(self):
        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT)
        req = config_gen.build_pipeline_request(pe, "/out/pipeline-1")
        self.assertEqual(req["pipeline"], "batch_detect")
        self.assertEqual(req["model_checkpoint"], self.tm.checkpoint_path)
        self.assertTrue(req["images"].endswith("ds1/images"))
        self.assertTrue(req["labels"].endswith("ds1/labels"))
        self.assertEqual(req["classes"], ["helmet", "head", "vest"])
        self.assertEqual(req["output_dir"], "/out/pipeline-1")
        self.assertNotIn("detector", req)
        self.assertNotIn("tiling", req)

    def test_detector_pipeline_falls_back_to_default_checkpoint(self):
        pe = self._make(pipeline=PipelineEvalRun.PEOPLE_DETECT_FIRST)
        req = config_gen.build_pipeline_request(pe, "/out/pipeline-2")
        self.assertTrue(
            req["detector"]["checkpoint"].endswith("models/people/best_ckpt.engine")
        )

    def test_detector_and_tiling_emitted(self):
        pe = self._make(
            pipeline=PipelineEvalRun.BATCH_PEOPLE,
            detector_checkpoint="/models/person.pt",
            tile_width_pct=25, tile_height_pct=40, overlap=0.25,
        )
        req = config_gen.build_pipeline_request(pe, "/out/pipeline-3")
        # expand_ratio rides along with the detector block (default 0.10).
        self.assertEqual(
            req["detector"], {"checkpoint": "/models/person.pt", "expand_ratio": 0.10}
        )
        self.assertEqual(
            req["tiling"],
            {"tile_width_pct": 25, "tile_height_pct": 40, "overlap": 0.25},
        )

    def test_custom_expand_ratio_emitted(self):
        pe = self._make(
            pipeline=PipelineEvalRun.PEOPLE_DETECT_FIRST,
            detector_checkpoint="/models/person.pt",
            detector_expand_ratio=0.2,
        )
        req = config_gen.build_pipeline_request(pe, "/out/pipeline-5")
        self.assertEqual(req["detector"]["expand_ratio"], 0.2)

    def test_tile_size_px_overrides_the_percentages(self):
        pe = self._make(
            pipeline=PipelineEvalRun.BATCH_DETECT, tile_size_px=640, overlap=0.2)
        req = config_gen.build_pipeline_request(pe, "/out/pipeline-6")
        self.assertEqual(req["tiling"], {"tile_size_px": 640, "overlap": 0.2})

    def test_min_box_size_emitted_for_both_person_crop_pipelines(self):
        # Matches pipeline_block: batch_people crops the original frame (it only
        # *finds* people in tiles), so its crops are as small as
        # people_detect_first's and chachak applies the floor to both. An eval
        # that dropped it here measured different geometry than training used.
        for pipeline in (PipelineEvalRun.PEOPLE_DETECT_FIRST, PipelineEvalRun.BATCH_PEOPLE):
            with self.subTest(pipeline=pipeline):
                pe = self._make(
                    pipeline=pipeline,
                    detector_checkpoint="/models/person.pt", detector_min_box_size=96,
                )
                self.assertEqual(
                    config_gen.build_pipeline_request(pe, "/out/p")["detector"]["min_box_size"],
                    96,
                )

    def test_merge_nms_iou_emitted_only_when_set(self):
        # chachak parses it with an unconditional float(), so an explicit null
        # would crash rather than fall through to the default.
        blank = self._make(pipeline=PipelineEvalRun.BATCH_DETECT)
        self.assertNotIn(
            "merge_nms_iou", config_gen.build_pipeline_request(blank, "/out/p"))

        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT, merge_nms_iou=0.55)
        self.assertEqual(
            config_gen.build_pipeline_request(pe, "/out/p")["merge_nms_iou"], 0.55)

    def test_chain_pipeline_carries_children(self):
        pe = self._make(
            pipeline=PipelineEvalRun.CHAIN,
            chain=[PipelineEvalRun.BATCH_DETECT],
        )
        req = config_gen.build_pipeline_request(pe, "/out/pipeline-4")
        self.assertEqual(req["chain"], ["batch_detect"])

    def test_write_request_persists_paths(self):
        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT)
        request_path, text = config_gen.write_pipeline_request(pe)
        pe.refresh_from_db()
        self.assertEqual(pe.request_yaml_path, str(request_path))
        self.assertTrue(pe.output_dir)
        self.assertTrue(Path(request_path).exists())
        self.assertEqual(yaml.safe_load(text)["pipeline"], "batch_detect")

    def test_ingest_pipeline_eval(self):
        out = Path(self._tmp.name) / "pipeout"
        out.mkdir()
        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT, output_dir=str(out))
        with open(out / "result.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump({"metrics": {"map50": 0.66, "map50_95": 0.44}}, fh)

        self.assertTrue(ingest.pipeline_is_complete(out))
        ingest.ingest_pipeline_eval(pe)
        pe.refresh_from_db()
        self.assertEqual(pe.metric("map50_95"), 0.44)


class PromotePayloadTests(PipelineEvalSetup):
    """build_promote_payload picks the right predictions file + source paths."""

    def test_pipeline_payload(self):
        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT, output_dir="/out/pipeline-9")
        payload = config_gen.build_promote_payload(pe, "pipeline", 0.3)
        self.assertTrue(payload["predictions_path"].endswith("pipeline-9/predictions.pt"))
        self.assertEqual(payload["checkpoint_path"], self.tm.checkpoint_path)
        self.assertTrue(payload["images_dir"].endswith("ds1/images"))
        self.assertTrue(payload["labels_dir"].endswith("ds1/labels"))
        self.assertTrue(payload["backup_dir"].endswith("ds1/backup_labels/pipeline-9"))
        self.assertEqual(payload["dataset_classes"], ["helmet", "head", "vest"])
        self.assertEqual(payload["score_threshold"], 0.3)

    def test_base_payload_uses_eval_predictions(self):
        er = EvalRun.objects.create(
            trained_model=self.tm, dataset=self.ds1,
            label_source=EvalRun.SOURCE, output_dir="/out/eval-3",
        )
        payload = config_gen.build_promote_payload(er, "base", 0.25)
        self.assertTrue(payload["predictions_path"].endswith("eval-3/eval_predictions.pt"))
        self.assertTrue(payload["backup_dir"].endswith("ds1/backup_labels/eval-3"))

    def test_requires_output_dir(self):
        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT)  # no output_dir
        with self.assertRaises(ValueError):
            config_gen.build_promote_payload(pe, "pipeline", 0.25)


class CombinedEvalViewTests(PipelineEvalSetup):
    """The union view lists both base and pipeline evals in one changelist."""

    def test_view_unions_base_and_pipeline(self):
        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT, status=PipelineEvalRun.OK,
                        metrics={"map50": 0.7})
        er = EvalRun.objects.create(
            trained_model=self.tm, dataset=self.ds1, status=EvalRun.OK,
            metrics={"map50": 0.5},
        )

        by_id = {row.id: row for row in CombinedEval.objects.all()}
        self.assertIn(f"pe-{pe.pk}", by_id)
        self.assertIn(f"be-{er.pk}", by_id)

        pipeline_row = by_id[f"pe-{pe.pk}"]
        self.assertEqual(pipeline_row.kind, CombinedEval.PIPELINE)
        self.assertEqual(pipeline_row.pipeline, "batch_detect")
        self.assertEqual(pipeline_row.orig_id, pe.pk)
        self.assertEqual(pipeline_row.metric("map50"), 0.7)
        self.assertEqual(pipeline_row.trained_model.name, "m1")

        base_row = by_id[f"be-{er.pk}"]
        self.assertEqual(base_row.kind, CombinedEval.BASE)
        self.assertEqual(base_row.pipeline, "base")
        self.assertEqual(base_row.metric("map50"), 0.5)

    def test_admin_promote_routes_to_real_pipeline_run(self):
        pe = self._make(pipeline=PipelineEvalRun.BATCH_DETECT, output_dir="/out/pipeline-7")
        admin = CombinedEvalAdmin(CombinedEval, AdminSite())
        qs = CombinedEval.objects.filter(id=f"pe-{pe.pk}")
        request = RequestFactory().post("/", {"apply": "1", "score_threshold": "0.4"})

        with mock.patch.object(runner, "promote_labels", return_value={
            "labels_written": 2, "boxes_written": 3, "backed_up": 0,
            "backup_dir": "", "dropped_unmapped": 0, "labels_dir": "/x/labels",
        }) as promote_call, mock.patch.object(admin, "message_user"):
            admin.promote_labels(request, qs)

        promote_call.assert_called_once()
        payload = promote_call.call_args.args[0]
        self.assertTrue(payload["predictions_path"].endswith("pipeline-7/predictions.pt"))
        self.assertEqual(payload["score_threshold"], 0.4)


class AutoEvalPipelineTests(TestCase):
    """When the experiment has a pipeline, the post-training test eval is a
    PipelineEvalRun (lands in the Eval Pipelines tab), not a plain EvalRun."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "source").mkdir()
        fs = FleetSettings.load()
        fs.source_dir = str(self.root / "source")
        fs.target_dir = str(self.root / "target")
        fs.save()

        self.ds = Dataset.objects.create(name="testds")
        self.exp = Experiment.objects.create(
            name="exp1", pipeline=PipelineEvalRun.BATCH_DETECT,
            tile_width_pct=50, overlap=0.2,
        )
        ExperimentDataset.objects.create(
            experiment=self.exp, dataset=self.ds, role=ExperimentDataset.TEST,
        )
        self.run = TrainingRun.objects.create(experiment=self.exp)
        RunResult.objects.create(
            run=self.run, run_name="00-x-00-retinanet", model_arch="retinanet",
            train_dataset_name="x",
            best_checkpoint=str(self.root / "best.pt"),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_pipeline_eval_run_with_copied_config(self):
        with mock.patch.object(autoeval, "_queue") as queue, \
                mock.patch.object(autoeval.config_gen, "write_pipeline_request"):
            queued = autoeval.schedule_test_evals(self.run)

        self.assertEqual(len(queued), 1)
        pe = PipelineEvalRun.objects.get(pk=queued[0])
        self.assertEqual(pe.pipeline, PipelineEvalRun.BATCH_DETECT)
        self.assertEqual(pe.tile_width_pct, 50)
        self.assertEqual(pe.overlap, 0.2)
        self.assertEqual(pe.dataset, self.ds)
        self.assertEqual(pe.status, PipelineEvalRun.QUEUED)
        queue.return_value.enqueue.assert_called_once()
        self.assertEqual(
            queue.return_value.enqueue.call_args.args[0], "training.jobs.run_pipeline_eval"
        )


class ProxyCleanupTests(TestCase):
    """Every list an operator actually deletes from is a *proxy* model — Base
    Eval proxies training.EvalRun, and each per-pipeline list (Batch detect,
    People detect first, Batch people, Chain) proxies PipelineEvalRun. Django's
    delete Collector groups instances by the exact class fetched, so a delete
    made through a proxy's own manager fires post_delete with that proxy as
    sender, never the concrete model underneath — a receiver registered only
    on the concrete model silently never cleans up. See eval_pipelines.signals.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir()
        ts = TrainingSettings.load()
        ts.runs_root = str(self.runs_root)
        ts.save()

        self.dataset = Dataset.objects.create(name="ds1")
        self.model = TrainedModel.objects.create(name="m1", arch="yolox", checkpoint_path="")

    def tearDown(self):
        self._tmp.cleanup()

    def _artifacts(self, name):
        out = self.runs_root / name
        out.mkdir()
        (out / "result.yaml").write_text("metrics: {}\n", encoding="utf-8")
        return out

    def test_deleting_via_base_eval_proxy_cleans_up(self):
        out = self._artifacts("base-eval-1")
        ev = BaseEval.objects.create(
            trained_model=self.model, dataset=self.dataset, output_dir=str(out),
        )
        BaseEval.objects.get(pk=ev.pk).delete()
        self.assertFalse(out.exists())

    def test_deleting_via_each_pipeline_proxy_cleans_up(self):
        proxies = [
            (BatchDetectEval, PipelineEvalRun.BATCH_DETECT),
            (PeopleDetectFirstEval, PipelineEvalRun.PEOPLE_DETECT_FIRST),
            (BatchPeopleEval, PipelineEvalRun.BATCH_PEOPLE),
            (ChainEval, PipelineEvalRun.CHAIN),
        ]
        for proxy_cls, pipeline in proxies:
            with self.subTest(proxy=proxy_cls.__name__):
                out = self._artifacts(f"pipeline-eval-{pipeline}")
                pe = PipelineEvalRun.objects.create(
                    trained_model=self.model, dataset=self.dataset, pipeline=pipeline,
                    output_dir=str(out),
                )
                proxy_cls.objects.get(pk=pe.pk).delete()
                self.assertFalse(out.exists())

    def test_bulk_delete_via_proxy_cleans_up(self):
        out = self._artifacts("pipeline-eval-bulk")
        PipelineEvalRun.objects.create(
            trained_model=self.model, dataset=self.dataset,
            pipeline=PipelineEvalRun.BATCH_DETECT, output_dir=str(out),
        )
        BatchDetectEval.objects.all().delete()
        self.assertFalse(out.exists())
