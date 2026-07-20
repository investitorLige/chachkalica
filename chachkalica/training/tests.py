"""Tests for training config generation.

These exercise the pure ``config_gen`` surface against on-disk dataset fixtures,
so we know the YAML we hand friendy_chachkalica has the right shape and paths without
needing the trainer itself.
"""

import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.test import TestCase

from fleet.models import Annotator, Dataset, FleetSettings
from training import admin as training_admin
from training import model_specs
from training.forms import ExperimentModelForm
from training.models import (
    EvalRun,
    Experiment,
    ExperimentDataset,
    ExperimentModel,
    TrainedModel,
    TrainingSettings,
)
from training.services import combine, config_gen


def _make_dataset_on_disk(source_root: Path, name: str, classes: list[str]) -> None:
    ds = source_root / name
    (ds / "images").mkdir(parents=True)
    (ds / "images" / "img1.jpg").write_bytes(b"")
    (ds / "labels").mkdir()
    (ds / "labels" / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (ds / "classes.txt").write_text("# tools: bbox\n" + "\n".join(classes) + "\n", encoding="utf-8")


class ConfigGenTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.source = root / "source"
        self.target = root / "target"
        self.source.mkdir()
        self.target.mkdir()

        fs = FleetSettings.load()
        fs.source_dir = str(self.source)
        fs.target_dir = str(self.target)
        fs.save()

        _make_dataset_on_disk(self.source, "ds1", ["helmet", "head", "vest"])
        self.ds1 = Dataset.objects.create(name="ds1")

        self.exp = Experiment.objects.create(name="exp1", scheduler_name="cosine")
        ExperimentDataset.objects.create(
            experiment=self.exp, dataset=self.ds1, role=ExperimentDataset.TRAIN
        )
        ExperimentModel.objects.create(
            experiment=self.exp, arch=ExperimentModel.RETINANET,
            params={"variant": "resnet50_fpn_v2"},
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_build_experiment_dict_shape(self):
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertEqual(data["name"], "exp1")
        self.assertEqual(data["output_dir"], "/out/exp1")

        train = data["datasets"]["train"]
        self.assertEqual(len(train), 1)
        entry = train[0]
        self.assertEqual(entry["name"], "ds1")
        self.assertEqual(entry["classes"], ["helmet", "head", "vest"])
        self.assertTrue(entry["images"].endswith("ds1/images"))
        self.assertTrue(entry["labels"].endswith("ds1/labels"))

        model = data["models"][0]
        self.assertEqual(model["name"], "retinanet")
        self.assertEqual(model["num_classes"], "auto")
        self.assertEqual(model["variant"], "resnet50_fpn_v2")

        self.assertEqual(data["evaluation"]["map_score_threshold"], 0.001)

        # cosine scheduler -> a dict with name; none -> None
        self.assertEqual(data["training"]["scheduler"], {"name": "cosine"})

    def test_pretrained_checkbox_sets_weights_true(self):
        m = self.exp.models.first()
        self.assertNotIn("weights", config_gen.model_entry(m))  # off by default

        m.pretrained = True
        m.save()
        self.assertIs(config_gen.model_entry(m)["weights"], True)

    def test_explicit_weights_in_params_overrides_checkbox(self):
        m = self.exp.models.first()
        m.pretrained = True
        m.params = {"weights": "/ckpts/custom.pth"}
        m.save()
        # An explicit path in params wins; the checkbox does not clobber it.
        self.assertEqual(config_gen.model_entry(m)["weights"], "/ckpts/custom.pth")

    def test_label_dir_source_vs_annotator(self):
        ed = self.exp.datasets.first()
        self.assertEqual(config_gen.label_dir(ed).name, "labels")

        alice = Annotator.objects.create(username="alice")
        ed.label_source = ExperimentDataset.ANNOTATOR
        ed.annotator = alice
        ed.save()
        self.assertEqual(
            config_gen.label_dir(ed),
            (self.target / "ds1" / "alice"),
        )

    def test_requires_train_dataset(self):
        self.exp.datasets.all().delete()
        with self.assertRaises(ValueError):
            config_gen.build_experiment_dict(self.exp, "/out")

    def test_no_pipeline_block_when_unset(self):
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertNotIn("pipeline", data)

    def test_batch_detect_pipeline_block(self):
        self.exp.pipeline = "batch_detect"
        self.exp.tile_size_px = 640
        self.exp.tile_width_pct = 50
        self.exp.tile_height_pct = 40
        self.exp.overlap = 0.2
        self.exp.merge_nms_iou = 0.5
        self.exp.save()
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertEqual(data["pipeline"]["name"], "batch_detect")
        self.assertEqual(
            data["pipeline"]["tiling"],
            {
                "tile_size_px": 640,
                "tile_width_pct": 50,
                "tile_height_pct": 40,
                "overlap": 0.2,
            },
        )
        self.assertEqual(data["pipeline"]["merge_nms_iou"], 0.5)
        self.assertNotIn("detector", data["pipeline"])

    def test_detector_pipeline_requires_checkpoint(self):
        self.exp.pipeline = "people_detect_first"
        self.exp.save()
        with self.assertRaises(ValueError):
            config_gen.build_experiment_dict(self.exp, "/out/exp1")

        self.exp.detector_checkpoint = "/ckpts/person.pt"
        self.exp.save()
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertEqual(data["pipeline"]["detector"], {"checkpoint": "/ckpts/person.pt"})

    def test_at_most_one_val(self):
        _make_dataset_on_disk(self.source, "ds2", ["helmet"])
        ds2 = Dataset.objects.create(name="ds2")
        ExperimentDataset.objects.create(
            experiment=self.exp, dataset=self.ds1, role=ExperimentDataset.VAL
        )
        ExperimentDataset.objects.create(
            experiment=self.exp, dataset=ds2, role=ExperimentDataset.VAL
        )
        with self.assertRaises(ValueError):
            config_gen.build_experiment_dict(self.exp, "/out")

    def test_requires_model(self):
        self.exp.models.all().delete()
        with self.assertRaises(ValueError):
            config_gen.build_experiment_dict(self.exp, "/out")

    def test_clean_requires_annotator(self):
        ed = ExperimentDataset(
            experiment=self.exp, dataset=self.ds1,
            role=ExperimentDataset.TRAIN, label_source=ExperimentDataset.ANNOTATOR,
        )
        with self.assertRaises(ValidationError):
            ed.clean()

    def test_augmentation_block_emitted_for_enabled_train_flags(self):
        ed = self.exp.datasets.first()
        ed.aug_hflip = True
        ed.aug_hflip_fraction = 0.5
        ed.aug_scale_crop = True
        ed.aug_scale_crop_fraction = 0.3
        ed.save()
        entry = config_gen.build_experiment_dict(self.exp, "/out")["datasets"]["train"][0]
        self.assertEqual(entry["augmentation"], {"hflip": 0.5, "scale_crop": 0.3})

    def test_augmentation_block_absent_when_disabled(self):
        # Checkboxes off (the default) -> no augmentation key at all, so the
        # YAML stays identical to the pre-augmentation format.
        entry = config_gen.build_experiment_dict(self.exp, "/out")["datasets"]["train"][0]
        self.assertNotIn("augmentation", entry)

    def test_augmentation_only_partially_enabled(self):
        ed = self.exp.datasets.first()
        ed.aug_scale_crop = True
        ed.aug_scale_crop_fraction = 0.25
        ed.save()
        entry = config_gen.build_experiment_dict(self.exp, "/out")["datasets"]["train"][0]
        self.assertEqual(entry["augmentation"], {"scale_crop": 0.25})

    def test_augmentation_never_emitted_for_val_rows(self):
        # A stale non-train row with flags set (predating clean()'s guard) must
        # not leak an augmentation block the trainer would reject.
        ExperimentDataset.objects.create(
            experiment=self.exp, dataset=self.ds1, role=ExperimentDataset.VAL,
            aug_hflip=True,
        )
        data = config_gen.build_experiment_dict(self.exp, "/out")
        self.assertNotIn("augmentation", data["datasets"]["val"])

    def test_clean_rejects_augmentation_on_non_train_role(self):
        ed = ExperimentDataset(
            experiment=self.exp, dataset=self.ds1,
            role=ExperimentDataset.VAL, aug_hflip=True,
        )
        with self.assertRaises(ValidationError):
            ed.clean()

    def test_clean_rejects_out_of_range_fraction(self):
        for bad in (0, -0.1, 1.5, None):
            ed = ExperimentDataset(
                experiment=self.exp, dataset=self.ds1,
                role=ExperimentDataset.TRAIN,
                aug_hflip=True, aug_hflip_fraction=bad,
            )
            with self.assertRaises(ValidationError, msg=f"fraction={bad}"):
                ed.clean()

    def test_clean_ignores_fraction_when_augmentation_off(self):
        ed = ExperimentDataset(
            experiment=self.exp, dataset=self.ds1,
            role=ExperimentDataset.TRAIN,
            aug_hflip=False, aug_hflip_fraction=7.0,
        )
        ed.clean()  # must not raise: the fraction is inert while unticked

    def test_best_metric_and_val_interval_flow_into_training_dict(self):
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertEqual(data["training"]["best_metric"], ["map50"])  # default
        self.assertEqual(data["training"]["val_interval"], 1)

        self.exp.best_metric = "f1+map50"
        self.exp.val_interval = 5
        self.exp.save()
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertEqual(data["training"]["best_metric"], ["f1", "map50"])
        self.assertEqual(data["training"]["val_interval"], 5)

    def test_operating_nms_threshold_flows_into_evaluation_dict(self):
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertIsNone(data["evaluation"]["operating_nms_threshold"])  # default off

        self.exp.eval_operating_nms_threshold = 0.7
        self.exp.save()
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertEqual(data["evaluation"]["operating_nms_threshold"], 0.7)

    def test_early_stopping_patience_flows_into_training_dict(self):
        self.exp.early_stopping_patience = 10
        self.exp.save()
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertEqual(data["training"]["early_stopping_patience"], 10)
        # Blank means "train all epochs" — passed through as None.
        self.exp.early_stopping_patience = None
        self.exp.save()
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertIsNone(data["training"]["early_stopping_patience"])


class ExperimentAdminRenderTests(TestCase):
    """The dataset inline must render the augmentation widgets and their JS."""

    def test_add_form_renders_augmentation_fields(self):
        from django.contrib.auth.models import User

        user = User.objects.create_superuser("admin", "admin@example.com", "pw")
        self.client.force_login(user)
        resp = self.client.get("/admin/training/experiment/add/")
        self.assertEqual(resp.status_code, 200)
        for name in ["aug_hflip", "aug_hflip_fraction", "aug_scale_crop",
                     "aug_scale_crop_fraction"]:
            self.assertContains(resp, name)
        self.assertContains(resp, "training/experiment_dataset_aug.js")
        # Help must be hoverable on the widget itself, not only on the tabular
        # header's 10px icon (formfield_for_dbfield mirrors it into title=).
        html = resp.content.decode()
        checkbox = next(
            line for line in html.splitlines()
            if 'id="id_datasets-0-aug_hflip"' in line and 'type="checkbox"' in line
        )
        self.assertIn('title="Randomly mirror images', checkbox)

    def test_add_form_renders_pipeline_section(self):
        from django.contrib.auth.models import User

        user = User.objects.create_superuser("admin2", "admin2@example.com", "pw")
        self.client.force_login(user)
        resp = self.client.get("/admin/training/experiment/add/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Training / Eval pipeline")
        self.assertContains(resp, 'id="id_pipeline"')
        for name in [
            "detector_checkpoint", "tile_size_px", "tile_width_pct",
            "tile_height_pct", "overlap", "merge_nms_iou",
        ]:
            self.assertContains(resp, name)
        self.assertContains(resp, "Fixed square source-image tile size")
        self.assertContains(resp, "training/experiment_pipeline_form.js")
        # Every trainable pipeline (train + val + test end-to-end) is offered;
        # only `chain` stays filtered out of the dropdown.
        self.assertContains(resp, 'value="batch_detect"')
        self.assertContains(resp, 'value="people_detect_first"')
        self.assertContains(resp, 'value="batch_people"')
        self.assertNotContains(resp, 'value="chain"')


class RFDETRResolutionValidationTests(TestCase):
    """The admin form must reject an RF-DETR resolution not divisible by the
    selected variant's patch stride (56 for base, 32 for the others)."""

    def _form(self, resolution, variant=None):
        rfield = model_specs.field_name(ExperimentModel.RFDETR, "resolution")
        vfield = model_specs.field_name(ExperimentModel.RFDETR, "variant")
        data = {"arch": ExperimentModel.RFDETR, "params": "{}", rfield: str(resolution)}
        if variant is not None:
            data[vfield] = variant
        form = ExperimentModelForm(data=data, instance=ExperimentModel())
        return form, rfield

    def test_rejects_non_multiple_of_56(self):
        # Blank variant → base (multiple 56); 900 % 56 != 0.
        form, rfield = self._form(900)
        self.assertFalse(form.is_valid())
        self.assertIn(rfield, form.errors)

    def test_accepts_multiple_of_56(self):
        form, rfield = self._form(896)
        self.assertTrue(form.is_valid(), form.errors)

    def test_accepts_nano_native_resolution(self):
        # nano's native 384 is a multiple of 32 but not of 56 — the old validator
        # wrongly rejected it; the per-variant multiple must accept it.
        form, _ = self._form(384, variant="nano")
        self.assertTrue(form.is_valid(), form.errors)

    def test_rejects_non_multiple_of_32_for_nano(self):
        form, rfield = self._form(400, variant="nano")
        self.assertFalse(form.is_valid())
        self.assertIn(rfield, form.errors)


class WeightsDropdownTests(TestCase):
    """The pretrained-weights dropdown resolves into params['weights'] and keeps
    the legacy ``pretrained`` column in sync."""

    def _save(self, arch, weights_value, custom="", extra=None):
        wfield = model_specs.weights_field_name(arch)
        data = {"arch": arch, "params": "{}", wfield: weights_value}
        if custom:
            data["weights_custom"] = custom
        if extra:
            data.update(extra)
        form = ExperimentModelForm(data=data, instance=ExperimentModel())
        self.assertTrue(form.is_valid(), form.errors)
        return form.save(commit=False)

    def test_default_option_sets_weights_true(self):
        obj = self._save(ExperimentModel.YOLOX, model_specs.WEIGHTS_DEFAULT)
        self.assertIs(obj.params["weights"], True)
        self.assertTrue(obj.pretrained)

    def test_none_option_sets_explicit_scratch_and_clears_pretrained(self):
        obj = self._save(ExperimentModel.YOLOX, model_specs.WEIGHTS_NONE)
        self.assertIs(obj.params["weights"], False)
        self.assertFalse(obj.pretrained)

    def test_catalog_option_passes_through(self):
        obj = self._save(ExperimentModel.RTDETR, "PekingU/rtdetr_v2_r50vd")
        self.assertEqual(obj.params["weights"], "PekingU/rtdetr_v2_r50vd")
        self.assertFalse(obj.pretrained)

    def test_custom_option_uses_text_field(self):
        obj = self._save(
            ExperimentModel.YOLOX, model_specs.WEIGHTS_CUSTOM,
            custom="/ckpts/mine.pth",
        )
        self.assertEqual(obj.params["weights"], "/ckpts/mine.pth")

    def test_custom_without_text_is_rejected(self):
        wfield = model_specs.weights_field_name(ExperimentModel.YOLOX)
        form = ExperimentModelForm(
            data={"arch": ExperimentModel.YOLOX, "params": "{}",
                  wfield: model_specs.WEIGHTS_CUSTOM},
            instance=ExperimentModel(),
        )
        self.assertFalse(form.is_valid())
        self.assertIn("weights_custom", form.errors)

    def test_torchvision_arches_do_not_offer_unsupported_custom_paths(self):
        for arch in (ExperimentModel.RETINANET, ExperimentModel.FASTERRCNN):
            choices = dict(model_specs.weights_base_choices(arch))
            self.assertNotIn(model_specs.WEIGHTS_CUSTOM, choices)

    def test_rtdetr_has_no_default_option_but_lists_v1_and_v2(self):
        choices = dict(model_specs.weights_base_choices(ExperimentModel.RTDETR))
        self.assertNotIn(model_specs.WEIGHTS_DEFAULT, choices)
        self.assertIn("PekingU/rtdetr_r50vd", choices)
        self.assertIn("PekingU/rtdetr_v2_r18vd", choices)

    def test_rfdetr_o365_is_variant_tagged_to_base(self):
        vmap = model_specs.weights_variant_map(ExperimentModel.RFDETR)
        self.assertEqual(vmap.get("rf-detr-base-o365.pth"), "base")

    def test_trained_model_appears_as_option(self):
        trained = TrainedModel.objects.create(
            name="my-yolox", arch=ExperimentModel.YOLOX,
            checkpoint_path="/runs/best.pt",
        )
        form = ExperimentModelForm(instance=ExperimentModel(arch=ExperimentModel.YOLOX))
        choices = dict(form.fields[
            model_specs.weights_field_name(ExperimentModel.YOLOX)
        ].choices)
        encoded = model_specs.friendy_weights_value(trained.checkpoint_path)
        self.assertIn(encoded, choices)
        # A model of a different arch must not leak into another arch's dropdown.
        rtdetr_choices = dict(form.fields[
            model_specs.weights_field_name(ExperimentModel.RTDETR)
        ].choices)
        self.assertNotIn(encoded, rtdetr_choices)

    def test_trained_model_selection_saves_init_checkpoint_not_weights(self):
        trained = TrainedModel.objects.create(
            name="my-yolox", arch=ExperimentModel.YOLOX,
            checkpoint_path="/runs/best.pt",
        )

        obj = self._save(
            ExperimentModel.YOLOX,
            model_specs.friendy_weights_value(trained.checkpoint_path),
        )

        self.assertEqual(obj.params["init_checkpoint"], "/runs/best.pt")
        self.assertNotIn("weights", obj.params)
        self.assertFalse(obj.pretrained)

    def test_existing_init_checkpoint_preselects_friendy_option(self):
        checkpoint = "/runs/deleted-registry-model/best.pt"
        model = ExperimentModel(
            arch=ExperimentModel.YOLOX,
            params={"init_checkpoint": checkpoint},
        )

        form = ExperimentModelForm(instance=model)
        field = form.fields[model_specs.weights_field_name(ExperimentModel.YOLOX)]

        self.assertEqual(
            field.initial,
            model_specs.friendy_weights_value(checkpoint),
        )
        self.assertIn(field.initial, dict(field.choices))

    def test_existing_string_weights_preselects_dropdown(self):
        m = ExperimentModel(arch=ExperimentModel.RTDETR,
                            params={"weights": "PekingU/rtdetr_v2_r34vd"})
        form = ExperimentModelForm(instance=m)
        field = form.fields[model_specs.weights_field_name(ExperimentModel.RTDETR)]
        self.assertEqual(field.initial, "PekingU/rtdetr_v2_r34vd")

    def test_existing_custom_path_selects_custom_and_fills_text(self):
        m = ExperimentModel(arch=ExperimentModel.YOLOX,
                            params={"weights": "/some/where.pth"})
        form = ExperimentModelForm(instance=m)
        field = form.fields[model_specs.weights_field_name(ExperimentModel.YOLOX)]
        self.assertEqual(field.initial, model_specs.WEIGHTS_CUSTOM)
        self.assertEqual(form.fields["weights_custom"].initial, "/some/where.pth")

    def test_bytetrack_option_is_variant_tagged_when_present(self):
        from unittest import mock

        fake = [{
            "value": "/app/data/training/weights/bytetrack_s_mot17.pth.tar",
            "label": "ByteTrack person — CrowdHuman+MOT17 (s)",
            "variant": "yolox-s",
        }]
        with mock.patch(
            "training.forms.model_specs.bytetrack_yolox_options", return_value=fake
        ):
            form = ExperimentModelForm(
                instance=ExperimentModel(arch=ExperimentModel.YOLOX)
            )
        field = form.fields[model_specs.weights_field_name(ExperimentModel.YOLOX)]
        self.assertIn(fake[0]["value"], dict(field.choices))
        # Tagged so the JS shows it only for the yolox-s variant.
        self.assertEqual(field.widget.variant_map.get(fake[0]["value"]), "yolox-s")
        # The trailing "Custom path or URL…" sentinel stays last.
        self.assertEqual(field.choices[-1][0], model_specs.WEIGHTS_CUSTOM)


class CombinedEvalTests(TestCase):
    """Evaluate 2+ models combined: the class-overlap guard and request/promote
    payloads that carry the extra models through to the trainer."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.source = root / "source"
        self.target = root / "target"
        self.source.mkdir()
        self.target.mkdir()

        fs = FleetSettings.load()
        fs.source_dir = str(self.source)
        fs.target_dir = str(self.target)
        fs.save()

        _make_dataset_on_disk(self.source, "combo_ds", ["cat", "dog"])
        self.dataset = Dataset.objects.create(name="combo_ds")

        self.model_a = TrainedModel.objects.create(
            name="model-a", arch="retinanet", checkpoint_path="/ckpts/a.pt",
            classes=["cat"],
        )
        self.model_b = TrainedModel.objects.create(
            name="model-b", arch="retinanet", checkpoint_path="/ckpts/b.pt",
            classes=["dog"],
        )
        self.model_c_overlap = TrainedModel.objects.create(
            name="model-c", arch="retinanet", checkpoint_path="/ckpts/c.pt",
            classes=["cat", "bird"],
        )

        ts = TrainingSettings.load()
        ts.configs_root = str(root / "configs")
        ts.runs_root = str(root / "runs")
        ts.save()

        self.admin_user = User.objects.create_superuser("admin", "admin@example.com", "pw")
        self.client.force_login(self.admin_user)

    def tearDown(self):
        self._tmp.cleanup()

    # --- overlapping_class_names guard ---

    def test_disjoint_classes_have_no_overlap(self):
        self.assertEqual(
            combine.overlapping_class_names([self.model_a, self.model_b]), set()
        )

    def test_shared_class_name_is_reported(self):
        self.assertEqual(
            combine.overlapping_class_names([self.model_a, self.model_c_overlap]),
            {"cat"},
        )

    def test_single_model_never_overlaps(self):
        self.assertEqual(combine.overlapping_class_names([self.model_a]), set())

    # --- config_gen: extra_checkpoints ---

    def test_build_eval_request_omits_extra_checkpoints_when_not_combined(self):
        eval_run = EvalRun.objects.create(trained_model=self.model_a, dataset=self.dataset)
        data = config_gen.build_eval_request(eval_run, "/out/eval-1")
        self.assertNotIn("extra_checkpoints", data)

    def test_build_eval_request_emits_extra_checkpoints_when_combined(self):
        eval_run = EvalRun.objects.create(trained_model=self.model_a, dataset=self.dataset)
        eval_run.combined_models.set([self.model_b])
        data = config_gen.build_eval_request(eval_run, "/out/eval-1")
        self.assertEqual(data["extra_checkpoints"], ["/ckpts/b.pt"])

    def test_build_eval_request_rejects_combined_model_without_checkpoint(self):
        no_checkpoint = TrainedModel.objects.create(
            name="model-d", arch="retinanet", checkpoint_path=""
        )
        eval_run = EvalRun.objects.create(trained_model=self.model_a, dataset=self.dataset)
        eval_run.combined_models.set([no_checkpoint])
        with self.assertRaises(ValueError):
            config_gen.build_eval_request(eval_run, "/out/eval-1")

    def test_build_pipeline_request_emits_extra_checkpoints_when_combined(self):
        from eval_pipelines.models import PipelineEvalRun

        pe = PipelineEvalRun.objects.create(
            trained_model=self.model_a, dataset=self.dataset,
            pipeline=PipelineEvalRun.BATCH_DETECT,
        )
        pe.combined_models.set([self.model_b])
        data = config_gen.build_pipeline_request(pe, "/out/pipeline-1")
        self.assertEqual(data["extra_checkpoints"], ["/ckpts/b.pt"])

    def test_build_pipeline_request_omits_extra_checkpoints_when_not_combined(self):
        from eval_pipelines.models import PipelineEvalRun

        pe = PipelineEvalRun.objects.create(
            trained_model=self.model_a, dataset=self.dataset,
            pipeline=PipelineEvalRun.BATCH_DETECT,
        )
        data = config_gen.build_pipeline_request(pe, "/out/pipeline-1")
        self.assertNotIn("extra_checkpoints", data)

    # --- config_gen: build_promote_payload ---

    def test_promote_payload_single_model_uses_checkpoint(self):
        eval_run = EvalRun.objects.create(
            trained_model=self.model_a, dataset=self.dataset, output_dir="/out/eval-1"
        )
        payload = config_gen.build_promote_payload(eval_run, "base", 0.25)
        self.assertEqual(payload["checkpoint_path"], "/ckpts/a.pt")
        self.assertNotIn("prediction_classes", payload)

    def test_promote_payload_combined_uses_prediction_classes(self):
        eval_run = EvalRun.objects.create(
            trained_model=self.model_a, dataset=self.dataset, output_dir="/out/eval-1"
        )
        eval_run.combined_models.set([self.model_b])
        payload = config_gen.build_promote_payload(eval_run, "base", 0.25)
        self.assertNotIn("checkpoint_path", payload)
        self.assertEqual(payload["prediction_classes"], ["cat", "dog"])

    # --- __str__ / is_combined ---

    def test_str_and_is_combined_reflect_combined_models(self):
        eval_run = EvalRun.objects.create(trained_model=self.model_a, dataset=self.dataset)
        self.assertFalse(eval_run.is_combined)
        self.assertIn("model-a", str(eval_run))

        eval_run.combined_models.set([self.model_b])
        self.assertTrue(eval_run.is_combined)
        self.assertIn("model-a + model-b", str(eval_run))

    # --- TrainedModelAdmin.evaluate action, end-to-end via the admin client ---

    def _post_evaluate(self, model_pks, **extra):
        return self.client.post(
            "/admin/training/trainedmodel/",
            {
                "action": "evaluate",
                ACTION_CHECKBOX_NAME: model_pks,
                "index": "0",
                **extra,
            },
        )

    def test_evaluate_action_blocks_models_with_overlapping_classes(self):
        from django.contrib.messages import get_messages

        resp = self._post_evaluate([self.model_a.pk, self.model_c_overlap.pk])
        self.assertEqual(resp.status_code, 302)  # redirects back to the changelist
        messages = [str(m) for m in get_messages(resp.wsgi_request)]
        self.assertTrue(any("cat" in m for m in messages))
        self.assertEqual(EvalRun.objects.count(), 0)

    def test_evaluate_action_renders_combined_form_for_disjoint_models(self):
        resp = self._post_evaluate([self.model_a.pk, self.model_b.pk])
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "model-a")
        self.assertContains(resp, "model-b")
        self.assertContains(resp, "combined")

    def test_evaluate_action_queues_one_combined_eval_run(self):
        with mock.patch.object(training_admin, "_queue") as queue:
            resp = self._post_evaluate(
                [self.model_a.pk, self.model_b.pk],
                apply="1",
                dataset=str(self.dataset.pk),
                map_score_threshold="0.001",
                score_threshold="0.25",
                pipeline="",
                label_source="source",
                annotator="",
                explicit_labels_path="",
            )
        self.assertEqual(resp.status_code, 302)
        queue.return_value.enqueue.assert_called_once()

        self.assertEqual(EvalRun.objects.count(), 1)
        eval_run = EvalRun.objects.get()
        self.assertEqual(eval_run.trained_model, self.model_a)
        self.assertEqual(list(eval_run.combined_models.all()), [self.model_b])
        self.assertEqual(eval_run.status, EvalRun.QUEUED)
        self.assertTrue(eval_run.request_yaml_path)
