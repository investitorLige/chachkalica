"""Tests for training config generation.

These exercise the pure ``config_gen`` surface against on-disk dataset fixtures,
so we know the YAML we hand friendy_chachkalica has the right shape and paths without
needing the trainer itself.
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.urls import reverse

from fleet.models import Annotator, Dataset, FleetSettings
from training import admin as training_admin
from training import jobs
from training import model_specs
from training import pipelines
from training.forms import ExperimentModelForm
from training.models import (
    DEFAULT_PERSON_DETECTOR_CHECKPOINT,
    BuildNode,
    EvalRun,
    Experiment,
    ExperimentDataset,
    ExperimentModel,
    ExportRun,
    RunResult,
    TrainedModel,
    TrainingRun,
    TrainingSettings,
)
from training.services import (
    buildnode,
    bundles,
    combine,
    config_gen,
    exports,
    pipeline_meta,
    promote,
    runner,
)


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

    def test_people_detect_first_injects_rtdetr_num_queries_default(self):
        self.exp.pipeline = pipelines.PEOPLE_DETECT_FIRST
        self.exp.save()
        m = ExperimentModel.objects.create(experiment=self.exp, arch=ExperimentModel.RTDETR)
        entry = config_gen.model_entry(m, pipeline_name=self.exp.pipeline)
        self.assertEqual(
            entry["num_queries"], config_gen.PEOPLE_DETECT_FIRST_RTDETR_NUM_QUERIES_DEFAULT
        )

    def test_explicit_num_queries_overrides_people_detect_first_default(self):
        self.exp.pipeline = pipelines.PEOPLE_DETECT_FIRST
        self.exp.save()
        m = ExperimentModel.objects.create(
            experiment=self.exp, arch=ExperimentModel.RTDETR, params={"num_queries": 10},
        )
        self.assertEqual(config_gen.model_entry(m, pipeline_name=self.exp.pipeline)["num_queries"], 10)

    def test_num_queries_not_injected_off_people_detect_first(self):
        # Blank pipeline, and batch_people: rtdetr keeps its own (HF) default.
        m = ExperimentModel.objects.create(experiment=self.exp, arch=ExperimentModel.RTDETR)
        self.assertNotIn("num_queries", config_gen.model_entry(m, pipeline_name=self.exp.pipeline))

        self.exp.pipeline = pipelines.BATCH_PEOPLE
        self.exp.save()
        self.assertNotIn("num_queries", config_gen.model_entry(m, pipeline_name=self.exp.pipeline))

    def test_min_box_size_emitted_for_both_person_crop_pipelines(self):
        """The floor reaches the training YAML for batch_people too.

        It was once scoped to people_detect_first on the theory that
        batch_people crops fixed-size tiles — but BatchPeoplePipeline only
        *finds* people in tiles and crops the original frame, so its crops are
        just as small, and chachak applies the floor for both regardless
        (``crop_regions`` has no pipeline gate). Scoping it here meant a
        batch_people model trained with no floor and was served with one.
        """
        for pipeline in (pipelines.PEOPLE_DETECT_FIRST, pipelines.BATCH_PEOPLE):
            with self.subTest(pipeline=pipeline):
                self.exp.pipeline = pipeline
                self.exp.save()
                data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
                self.assertEqual(
                    data["pipeline"]["detector"]["min_box_size"],
                    self.exp.detector_min_box_size,
                )

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

    def test_detector_pipeline_uses_default_checkpoint_when_blank(self):
        self.exp.pipeline = "people_detect_first"
        self.exp.detector_checkpoint = ""
        self.exp.save()
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertEqual(
            data["pipeline"]["detector"]["checkpoint"],
            str(config_gen._resolve("models/people/best_ckpt.engine")),
        )

        self.exp.detector_checkpoint = "/ckpts/person.pt"
        self.exp.save()
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        # expand_ratio and min_box_size (people_detect_first only) ride along
        # with the detector block — 0.10 and 224.0 are the field defaults.
        self.assertEqual(
            data["pipeline"]["detector"],
            {"checkpoint": "/ckpts/person.pt", "expand_ratio": 0.10, "min_box_size": 224.0},
        )

    def test_custom_detector_expand_ratio_emitted(self):
        self.exp.pipeline = "batch_people"
        self.exp.detector_checkpoint = "/ckpts/person.pt"
        self.exp.detector_expand_ratio = 0.25
        self.exp.save()
        data = config_gen.build_experiment_dict(self.exp, "/out/exp1")
        self.assertEqual(data["pipeline"]["detector"]["expand_ratio"], 0.25)

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


class PipelineMetadataTests(TestCase):
    """The frozen pipeline record: one schema, every model, every export format.

    Covers what used to be three hand-copied field lists (``videos.admin``,
    ``cameras.admin``, the export sidecar writer) and the resolution order that
    keeps a promoted model describing the run it came from.
    """

    def _experiment(self, **overrides) -> Experiment:
        fields = {
            "name": "exp-pipeline",
            "pipeline": pipelines.PEOPLE_DETECT_FIRST,
            "detector_checkpoint": "/models/person.pt",
            "detector_expand_ratio": 0.15,
            "detector_min_box_size": 96.0,
            "merge_nms_iou": 0.55,
            "eval_score_threshold": 0.3,
        }
        fields.update(overrides)
        return Experiment.objects.create(**fields)

    def _promoted(self, experiment) -> TrainedModel:
        run = TrainingRun.objects.create(experiment=experiment)
        result = RunResult.objects.create(
            run=run, run_name="r0", model_arch=ExperimentModel.YOLOX,
            best_checkpoint="/runs/r0/best.pt",
        )
        return promote.promote_run_result(result)

    def test_normalize_fills_missing_keys_from_a_partial_sidecar(self):
        # A sidecar written before a field existed omits it; that means "chachak's
        # default", so it must read back as the default rather than KeyError.
        blob = pipeline_meta.normalize({"pipeline": pipelines.BATCH_DETECT,
                                        "tile_size_px": 640})
        self.assertEqual(blob["pipeline"], pipelines.BATCH_DETECT)
        self.assertEqual(blob["tile_size_px"], 640)
        self.assertIsNone(blob["merge_nms_iou"])
        self.assertEqual(blob["chain"], [])
        self.assertEqual(blob["detector_checkpoint"], "")
        self.assertEqual(blob["schema_version"], pipeline_meta.SCHEMA_VERSION)

    def test_normalize_survives_junk(self):
        self.assertEqual(pipeline_meta.normalize(None), pipeline_meta.raw())
        self.assertEqual(pipeline_meta.normalize({"chain": "nope"})["chain"], [])

    def test_blank_experiment_pipeline_is_recorded_as_raw(self):
        blob = pipeline_meta.from_experiment(self._experiment(pipeline=""))
        self.assertEqual(blob["pipeline"], pipeline_meta.RAW)

    def test_blank_detector_records_the_default_the_run_actually_used(self):
        """A blank ``detector_checkpoint`` must freeze as the resolved default.

        ``config_gen.pipeline_block`` substitutes
        DEFAULT_PERSON_DETECTOR_CHECKPOINT when it writes the training YAML, so
        freezing the blank recorded a pipeline the model was never trained
        through. Every consumer then had to re-guess the fallback, and
        ``build_predict_request`` didn't — it raised, so video/camera inference
        refused to run any model whose experiment hadn't overridden the default.
        """
        blob = pipeline_meta.from_experiment(self._experiment(detector_checkpoint=""))
        self.assertEqual(blob["detector_checkpoint"], DEFAULT_PERSON_DETECTOR_CHECKPOINT)

    def test_blank_detector_stays_blank_when_the_pipeline_has_no_detector(self):
        # Nothing crops here, so a detector path on the record would be a lie —
        # and would leak a detector block into a plain tiling run.
        blob = pipeline_meta.from_experiment(
            self._experiment(pipeline=pipelines.BATCH_DETECT, detector_checkpoint="")
        )
        self.assertEqual(blob["detector_checkpoint"], "")

    def test_an_explicit_detector_is_never_overridden_by_the_default(self):
        blob = pipeline_meta.from_experiment(
            self._experiment(detector_checkpoint="/models/mine.engine")
        )
        self.assertEqual(blob["detector_checkpoint"], "/models/mine.engine")

    def test_promotion_freezes_the_experiment_pipeline(self):
        model = self._promoted(self._experiment())

        self.assertEqual(model.pipeline_metadata["pipeline"], pipelines.PEOPLE_DETECT_FIRST)
        self.assertEqual(model.pipeline_metadata["detector_expand_ratio"], 0.15)
        self.assertEqual(model.pipeline_metadata["detector_min_box_size"], 96.0)
        self.assertEqual(model.pipeline_metadata["merge_nms_iou"], 0.55)
        self.assertEqual(model.pipeline_metadata["score_threshold"], 0.3)

    def test_frozen_record_survives_the_experiment_changing(self):
        experiment = self._experiment()
        model = self._promoted(experiment)

        experiment.pipeline = pipelines.BATCH_DETECT
        experiment.detector_expand_ratio = 0.99
        experiment.save()

        blob = pipeline_meta.for_trained_model(TrainedModel.objects.get(pk=model.pk))
        self.assertEqual(blob["pipeline"], pipelines.PEOPLE_DETECT_FIRST)
        self.assertEqual(blob["detector_expand_ratio"], 0.15)

    def test_frozen_record_survives_the_experiment_being_deleted(self):
        experiment = self._experiment()
        model = self._promoted(experiment)
        experiment.delete()

        blob = pipeline_meta.for_trained_model(TrainedModel.objects.get(pk=model.pk))
        self.assertEqual(blob["pipeline"], pipelines.PEOPLE_DETECT_FIRST)

    def test_model_with_no_frozen_record_falls_back_to_its_experiment(self):
        # Covers a row created outside promote_run_result (a fixture, or a promotion
        # that predates the field and escaped the backfill migration).
        experiment = self._experiment()
        model = self._promoted(experiment)
        TrainedModel.objects.filter(pk=model.pk).update(pipeline_metadata={})

        blob = pipeline_meta.for_trained_model(TrainedModel.objects.get(pk=model.pk))
        self.assertEqual(blob["pipeline"], pipelines.PEOPLE_DETECT_FIRST)

    def test_model_with_no_experiment_at_all_reads_as_raw(self):
        model = TrainedModel.objects.create(
            name="hand-registered", arch=ExperimentModel.YOLOX,
            checkpoint_path="/models/best.pt",
        )
        self.assertEqual(
            pipeline_meta.for_trained_model(model)["pipeline"], pipeline_meta.RAW)


class ExportPipelineMetadataTests(TestCase):
    """Exported ``.onnx``/``.engine`` artifacts carry the same record as the ``.pt``."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        ts = TrainingSettings.load()
        ts.exports_root = str(self.root)
        ts.save()

        experiment = Experiment.objects.create(
            name="exp-export", pipeline=pipelines.BATCH_DETECT,
            tile_size_px=640, overlap=0.2, eval_score_threshold=0.35,
        )
        run = TrainingRun.objects.create(experiment=experiment)
        result = RunResult.objects.create(
            run=run, run_name="r0", model_arch=ExperimentModel.YOLOX,
            best_checkpoint="/runs/r0/best.pt",
        )
        self.model = promote.promote_run_result(result, name="exp-export")

    def _artifact(self, name: str) -> Path:
        path = self.root / name
        path.write_bytes(b"")
        return path

    def test_sidecar_carries_the_models_frozen_record(self):
        artifact = self._artifact("exp-export-best.onnx")
        exports.export_pipeline_sidecar(self.model, artifact)

        blob = exports.read_pipeline_defaults("exp-export-best.onnx")
        self.assertEqual(blob["pipeline"], pipelines.BATCH_DETECT)
        self.assertEqual(blob["tile_size_px"], 640)
        self.assertEqual(blob["overlap"], 0.2)
        self.assertEqual(blob["score_threshold"], 0.35)

    def test_sidecar_is_written_even_for_a_full_frame_model(self):
        # "raw" on record is prefillable; a missing sidecar is indistinguishable
        # from an export made before sidecars existed, and sends the operator back
        # to filling the form by hand.
        raw_model = TrainedModel.objects.create(
            name="raw-model", arch=ExperimentModel.YOLOX,
            checkpoint_path="/models/best.pt",
        )
        artifact = self._artifact("raw-model-best.onnx")
        exports.export_pipeline_sidecar(raw_model, artifact)

        blob = exports.read_pipeline_defaults("raw-model-best.onnx")
        self.assertIsNotNone(blob)
        self.assertEqual(blob["pipeline"], pipeline_meta.RAW)

    def test_sidecarless_artifact_falls_back_to_the_catalogued_model(self):
        # Everything exported before sidecars were written — matched by the
        # "<model name>-best" / "-last" stem the export actions produce.
        self._artifact("exp-export-last.engine")

        blob = exports.read_pipeline_defaults("exp-export-last.engine")
        self.assertIsNotNone(blob)
        self.assertEqual(blob["pipeline"], pipelines.BATCH_DETECT)
        self.assertEqual(blob["tile_size_px"], 640)

    def test_unknown_artifact_has_nothing_on_record(self):
        self._artifact("something-nobody-catalogued.onnx")
        self.assertIsNone(
            exports.read_pipeline_defaults("something-nobody-catalogued.onnx"))

    def test_bundle_request_is_built_from_the_frozen_record(self):
        artifact = self._artifact("exp-export-best.onnx")
        self.model.classes = ["helmet", "head"]
        self.model.save()

        request = exports.build_bundle_request(self.model, artifact)
        self.assertEqual(request["pipeline"], pipelines.BATCH_DETECT)
        self.assertEqual(request["tiling"], {"tile_size_px": 640, "overlap": 0.2})
        self.assertEqual(request["score_threshold"], 0.35)

    def test_bundle_request_still_builds_after_the_experiment_is_gone(self):
        artifact = self._artifact("exp-export-best.onnx")
        self.model.classes = ["helmet"]
        self.model.save()
        Experiment.objects.all().delete()

        request = exports.build_bundle_request(
            TrainedModel.objects.get(pk=self.model.pk), artifact)
        self.assertIsNotNone(request)
        self.assertEqual(request["pipeline"], pipelines.BATCH_DETECT)


class ExportedDetectorCopyTests(TestCase):
    """A person-crop export carries its detector as a loadable, non-selectable copy.

    Two things have to hold at once: the copy must be complete enough to load
    (chachak resolves an artifact's ``.meta.json`` by name, so the pair travels
    together), and it must not show up as a model an operator can pick — it is a
    single-class person detector, not this export.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.exports_dir = self.root / "exports"
        self.exports_dir.mkdir()
        self.detectors = self.root / "detectors"
        self.detectors.mkdir()

        ts = TrainingSettings.load()
        ts.exports_root = str(self.exports_dir)
        ts.save()

        experiment = Experiment.objects.create(
            name="ppe", pipeline=pipelines.PEOPLE_DETECT_FIRST,
            detector_checkpoint="", eval_score_threshold=0.4,
        )
        run = TrainingRun.objects.create(experiment=experiment)
        result = RunResult.objects.create(
            run=run, run_name="r0", model_arch=ExperimentModel.YOLOX,
            best_checkpoint="/runs/r0/best.pt",
        )
        self.model = promote.promote_run_result(result, name="ppe")
        self.artifact = self.exports_dir / "ppe-best.onnx"
        self.artifact.write_bytes(b"")

    def _detector(self, name: str, *, meta: bool = True) -> Path:
        path = self.detectors / name
        path.write_bytes(b"")
        if meta:
            path.with_suffix(".meta.json").write_text('{"arch": "yolox"}')
        return path

    def _sidecar(self, detector: Path) -> dict:
        # The experiment leaves detector_checkpoint blank, so the copy is driven by
        # DEFAULT_PERSON_DETECTOR_CHECKPOINT — patched to the fixture detector.
        # That fallback is resolved when the record is frozen
        # (pipeline_meta.from_experiment), so the metadata is re-frozen here under
        # the patch: setUp promoted the model before the fixture detector existed.
        with mock.patch(
            "training.models.DEFAULT_PERSON_DETECTOR_CHECKPOINT", str(detector)
        ):
            self.model.pipeline_metadata = pipeline_meta.from_experiment(
                pipeline_meta.source_experiment(self.model)
            )
            self.model.save(update_fields=["pipeline_metadata"])
            return exports.export_pipeline_sidecar(self.model, self.artifact)

    def _pin_detector(self, path: Path) -> None:
        """Freeze an explicit detector path on the row, as an experiment that set
        one does — a blank field means "whatever the default is" and stays blank."""
        meta = dict(self.model.pipeline_metadata)
        meta["detector_checkpoint"] = str(path)
        self.model.pipeline_metadata = meta
        self.model.save(update_fields=["pipeline_metadata"])

    def test_engine_detector_is_copied_with_the_sidecar_its_loader_needs(self):
        defaults = self._sidecar(self._detector("person.engine"))

        copy = Path(defaults["detector_checkpoint"])
        self.assertEqual(copy, self.exports_dir / "ppe-best.detector.engine")
        self.assertTrue(copy.is_file())
        # trt_infer/onnx_infer resolve the meta as artifact.with_suffix(".meta.json"),
        # so this exact name is what the copy will be loaded through.
        self.assertTrue(copy.with_suffix(".meta.json").is_file())

    def test_detector_without_a_sidecar_leaves_the_record_on_the_original(self):
        # Copying it would replace a path that loads with one that raises.
        detector = self._detector("person.engine", meta=False)
        self._pin_detector(detector)

        defaults = exports.export_pipeline_sidecar(self.model, self.artifact)

        self.assertEqual(defaults["detector_checkpoint"], str(detector))
        self.assertFalse((self.exports_dir / "ppe-best.detector.engine").exists())

    def test_pt_detector_carries_the_onnx_chachak_would_have_preferred(self):
        detector = self._detector("person.pt", meta=False)
        self._detector("person.onnx")  # sibling chachak loads instead of the .pt

        copy = Path(self._sidecar(detector)["detector_checkpoint"])
        self.assertEqual(copy, self.exports_dir / "ppe-best.detector.pt")
        self.assertTrue(copy.with_suffix(".onnx").is_file())
        self.assertTrue(copy.with_suffix(".meta.json").is_file())

    def test_missing_detector_leaves_the_record_alone(self):
        gone = self.detectors / "nope.engine"
        self._pin_detector(gone)

        defaults = exports.export_pipeline_sidecar(self.model, self.artifact)

        self.assertEqual(defaults["detector_checkpoint"], str(gone))
        self.assertEqual(list(self.exports_dir.glob("*.detector.*")), [])

    def test_the_copy_is_not_offered_as_an_exported_model(self):
        self._sidecar(self._detector("person.engine"))

        self.assertEqual(
            [a["relpath"] for a in exports.list_artifacts()], ["ppe-best.onnx"])
        with self.assertRaises(ValueError):
            exports.resolve("ppe-best.detector.engine")

    def test_a_full_frame_export_copies_no_detector(self):
        raw = TrainedModel.objects.create(
            name="raw", arch=ExperimentModel.YOLOX, checkpoint_path="/models/best.pt")
        artifact = self.exports_dir / "raw-best.onnx"
        artifact.write_bytes(b"")
        detector = self._detector("person.engine")

        with mock.patch(
            "training.services.exports.DEFAULT_PERSON_DETECTOR_CHECKPOINT", str(detector)
        ):
            exports.export_pipeline_sidecar(raw, artifact)

        self.assertEqual(list(self.exports_dir.glob("*.detector.*")), [])


class ModelActionPrefillTests(TestCase):
    """The eval and preview forms prefill from the same frozen record.

    Both used to derive their defaults independently — the eval action by walking
    back to the experiment, the preview action not at all — so the same model gave
    three different answers depending on which action you opened.
    """

    def setUp(self):
        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")
        self.url = reverse("admin:training_trainedmodel_changelist")

        self.model = TrainedModel.objects.create(
            name="cropped", arch=ExperimentModel.YOLOX,
            checkpoint_path="/runs/best.pt",
            pipeline_metadata=pipeline_meta.normalize({
                "pipeline": pipelines.PEOPLE_DETECT_FIRST,
                "detector_checkpoint": "/models/person.engine",
                "detector_expand_ratio": 0.18,
                "detector_min_box_size": 128,
                "merge_nms_iou": 0.55,
                "score_threshold": 0.4,
            }),
        )

    def _open(self, action):
        return self.client.post(self.url, {
            "action": action,
            ACTION_CHECKBOX_NAME: [str(self.model.pk)],
        }, follow=True)

    def test_preview_form_defaults_to_the_trained_pipeline(self):
        resp = self._open("preview_on_dataset")
        self.assertContains(resp, 'value="people_detect_first" selected')
        self.assertContains(resp, 'value="/models/person.engine"')
        self.assertContains(resp, 'value="0.18"')
        # tile_size_px and detector_min_box_size are new inputs on this form; both
        # are knobs /predict_image already accepted but the page never offered.
        self.assertContains(resp, 'name="detector_min_box_size"')
        self.assertContains(resp, 'value="128"')
        self.assertContains(resp, 'name="tile_size_px"')

    def test_evaluate_form_defaults_to_the_trained_pipeline(self):
        resp = self._open("evaluate")
        self.assertContains(resp, 'value="people_detect_first" selected')
        self.assertContains(resp, 'value="/models/person.engine"')
        self.assertContains(resp, 'value="0.18"')

    def test_evaluate_form_prefills_a_zero_expand_ratio_as_zero(self):
        """An expand ratio of exactly 0 must survive the render.

        ``0 crops the detector box exactly`` is a documented setting, but the
        template used ``|default:'0.10'`` and Django's ``default`` filter fires on
        *any* falsy value — so a model trained with 0 offered the operator 0.10 to
        confirm, and the eval silently ran 10% wider crops than training.
        """
        self.model.pipeline_metadata = pipeline_meta.normalize({
            "pipeline": pipelines.PEOPLE_DETECT_FIRST,
            "detector_checkpoint": "/models/person.engine",
            "detector_expand_ratio": 0.0,
        })
        self.model.save(update_fields=["pipeline_metadata"])

        resp = self._open("evaluate")
        self.assertContains(resp, 'id="detector_expand_ratio"\n           value="0.0"')
        self.assertNotContains(resp, 'value="0.10"')

    def test_both_forms_prefill_merge_nms_iou(self):
        # Recorded all along, but neither form could act on it until
        # PipelineEvalRun and /predict_image gained the field.
        for action in ("preview_on_dataset", "evaluate"):
            with self.subTest(action=action):
                resp = self._open(action)
                self.assertContains(resp, 'name="merge_nms_iou"')
                self.assertContains(resp, 'step="0.05"\n           value="0.55"')

    def test_full_frame_model_leaves_both_forms_on_no_pipeline(self):
        plain = TrainedModel.objects.create(
            name="plain", arch=ExperimentModel.YOLOX, checkpoint_path="/runs/p.pt")
        for action in ("preview_on_dataset", "evaluate"):
            resp = self.client.post(self.url, {
                "action": action,
                ACTION_CHECKBOX_NAME: [str(plain.pk)],
            }, follow=True)
            self.assertNotContains(resp, 'value="people_detect_first" selected')


class ExportActionsQueueJobsTests(TestCase):
    """export_onnx/export_trt create ExportRun rows and enqueue jobs instead of
    blocking the admin request on the trainer service (see training.jobs)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        ts = TrainingSettings.load()
        ts.exports_root = str(self.root / "exports")
        ts.save()

        self.model = TrainedModel.objects.create(
            name="export-me", arch=ExperimentModel.YOLOX, checkpoint_path="/ckpts/best.pt",
        )

        self.admin_user = User.objects.create_superuser("admin", "admin@example.com", "pw")
        self.client.force_login(self.admin_user)

    def _post(self, action, **extra):
        return self.client.post(
            "/admin/training/trainedmodel/",
            {
                "action": action,
                ACTION_CHECKBOX_NAME: [str(self.model.pk)],
                "index": "0",
                **extra,
            },
        )

    def test_export_onnx_queues_one_job_and_creates_a_queued_row(self):
        with mock.patch.object(training_admin, "_queue") as queue:
            resp = self._post("export_onnx", apply="1", output_dir=str(self.root / "out"))

        self.assertEqual(resp.status_code, 302)
        queue.return_value.enqueue.assert_called_once()
        args, kwargs = queue.return_value.enqueue.call_args
        self.assertEqual(args[0], jobs.run_export_onnx)
        self.assertEqual(kwargs.get("job_timeout"), jobs.EXPORT_ONNX_JOB_TIMEOUT)

        export_run = ExportRun.objects.get()
        self.assertEqual(export_run.model, self.model)
        self.assertEqual(export_run.kind, ExportRun.ONNX)
        self.assertEqual(export_run.checkpoint_label, "best")
        self.assertEqual(export_run.checkpoint_path, "/ckpts/best.pt")
        self.assertEqual(export_run.status, ExportRun.QUEUED)
        self.assertTrue(export_run.output_path.endswith("export-me-best.onnx"))

    def test_export_trt_queues_with_precision_and_parsed_input_size(self):
        with mock.patch.object(training_admin, "_queue") as queue:
            resp = self._post(
                "export_trt", apply="1", output_dir=str(self.root / "out"),
                precision="fp32", input_size="640x480",
            )

        self.assertEqual(resp.status_code, 302)
        queue.return_value.enqueue.assert_called_once()
        args, kwargs = queue.return_value.enqueue.call_args
        self.assertEqual(args[0], jobs.run_export_trt)
        self.assertEqual(kwargs.get("job_timeout"), jobs.EXPORT_TRT_JOB_TIMEOUT)

        export_run = ExportRun.objects.get()
        self.assertEqual(export_run.kind, ExportRun.TRT)
        self.assertEqual(export_run.precision, "fp32")
        self.assertEqual(export_run.input_hw, [640, 480])
        self.assertEqual(export_run.status, ExportRun.QUEUED)

    def test_export_trt_rejects_invalid_input_size_without_queuing(self):
        from django.contrib.messages import get_messages

        with mock.patch.object(training_admin, "_queue") as queue:
            resp = self._post(
                "export_trt", apply="1", output_dir=str(self.root / "out"),
                input_size="not-a-size",
            )

        self.assertEqual(resp.status_code, 302)  # returns None -> redirects to the changelist
        queue.return_value.enqueue.assert_not_called()
        self.assertEqual(ExportRun.objects.count(), 0)
        messages = [str(m) for m in get_messages(resp.wsgi_request)]
        self.assertTrue(any("Invalid input size" in m for m in messages))

    def test_export_onnx_form_shows_the_trained_size_it_will_export_at(self):
        with mock.patch(
            "training.services.runner.inspect_checkpoint",
            return_value={"arch": "yolox", "trained_size": [640, 640]},
        ):
            resp = self._post("export_onnx")
        self.assertContains(resp, "640×640")

    def test_export_trt_form_prefills_input_size_from_the_trained_checkpoint(self):
        with mock.patch(
            "training.services.runner.inspect_checkpoint",
            return_value={"arch": "yolox", "trained_size": [640, 640]},
        ):
            resp = self._post("export_trt")
        self.assertContains(resp, 'value="640x640"')

    def test_export_trt_form_leaves_input_size_blank_for_a_variable_size_arch(self):
        with mock.patch(
            "training.services.runner.inspect_checkpoint",
            return_value={"arch": "fasterrcnn", "trained_size": None},
        ):
            resp = self._post("export_trt")
        self.assertContains(resp, 'value=""')

    def test_export_trt_form_survives_a_trainer_service_hiccup(self):
        with mock.patch(
            "training.services.runner.inspect_checkpoint",
            side_effect=RuntimeError("trainer unreachable"),
        ):
            resp = self._post("export_trt")
        self.assertEqual(resp.status_code, 200)


class ExportJobsTests(TestCase):
    """training.jobs.run_export_onnx/run_export_trt: export -> sidecar -> bundle,
    with a non-fatal bundle step, run on the django_rq worker."""

    def setUp(self):
        self.model = TrainedModel.objects.create(
            name="export-me", arch=ExperimentModel.YOLOX, checkpoint_path="/ckpts/best.pt",
        )

    def _export_run(self, **overrides) -> ExportRun:
        fields = dict(
            model=self.model, kind=ExportRun.ONNX, checkpoint_label="best",
            checkpoint_path="/ckpts/best.pt", output_path="/out/export-me-best.onnx",
        )
        fields.update(overrides)
        return ExportRun.objects.create(**fields)

    def test_run_export_onnx_success_populates_result_and_bundle(self):
        run = self._export_run()
        onnx_result = {"onnx_path": run.output_path, "meta_path": run.output_path + ".meta.json"}
        with mock.patch(
            "training.jobs.runner.export_onnx", return_value=onnx_result,
        ) as export_onnx, mock.patch(
            "training.jobs.exports.export_pipeline_sidecar",
        ) as sidecar, mock.patch(
            "training.jobs.exports.build_bundle_request", return_value={"pipeline": "raw"},
        ), mock.patch(
            "training.jobs.runner.export_bundle", return_value={"bundle_dir": "/out/bundle"},
        ) as export_bundle:
            result = jobs.run_export_onnx(run.pk)

        export_onnx.assert_called_once_with(run.checkpoint_path, run.output_path)
        sidecar.assert_called_once_with(self.model, Path(run.output_path))
        export_bundle.assert_called_once()
        self.assertEqual(result, onnx_result)

        run.refresh_from_db()
        self.assertEqual(run.status, ExportRun.OK)
        self.assertEqual(run.result, onnx_result)
        self.assertEqual(run.bundle_dir, "/out/bundle")
        self.assertEqual(run.bundle_error, "")
        self.assertIsNotNone(run.started_at)
        self.assertIsNotNone(run.finished_at)

    def test_run_export_onnx_primary_failure_marks_error_and_reraises(self):
        run = self._export_run()
        with mock.patch(
            "training.jobs.runner.export_onnx", side_effect=RuntimeError("trainer down"),
        ):
            with self.assertRaises(RuntimeError):
                jobs.run_export_onnx(run.pk)

        run.refresh_from_db()
        self.assertEqual(run.status, ExportRun.ERROR)
        self.assertIn("trainer down", run.last_error)
        self.assertIsNone(run.result)

    def test_run_export_onnx_bundle_failure_is_non_fatal(self):
        run = self._export_run()
        with mock.patch(
            "training.jobs.runner.export_onnx", return_value={"onnx_path": run.output_path},
        ), mock.patch(
            "training.jobs.exports.export_pipeline_sidecar",
        ), mock.patch(
            "training.jobs.exports.build_bundle_request", return_value={"pipeline": "raw"},
        ), mock.patch(
            "training.jobs.runner.export_bundle", side_effect=RuntimeError("bundle broke"),
        ):
            jobs.run_export_onnx(run.pk)

        run.refresh_from_db()
        self.assertEqual(run.status, ExportRun.OK)
        self.assertEqual(run.bundle_dir, "")
        self.assertIn("bundle broke", run.bundle_error)

    def test_run_export_onnx_sidecar_failure_does_not_strand_the_row(self):
        # The artifact is already on disk by this point, so the row must reach a
        # terminal state carrying the reason — it used to be left RUNNING with an
        # empty last_error, and nothing reconciles ExportRun.
        run = self._export_run()
        with mock.patch(
            "training.jobs.runner.export_onnx", return_value={"onnx_path": run.output_path},
        ), mock.patch(
            "training.jobs.exports.export_pipeline_sidecar",
            side_effect=OSError("no space left on device"),
        ), mock.patch(
            "training.jobs.exports.build_bundle_request", return_value=None,
        ):
            jobs.run_export_onnx(run.pk)

        run.refresh_from_db()
        self.assertEqual(run.status, ExportRun.OK)
        self.assertIn("pipeline sidecar", run.bundle_error)
        self.assertIn("no space left on device", run.bundle_error)
        self.assertIsNotNone(run.finished_at)

    def test_run_export_onnx_bundle_request_failure_does_not_strand_the_row(self):
        # build_bundle_request resolves paths and reads sidecars, so it can raise
        # for the same environmental reasons the bundle call can; it used to sit
        # outside _bundle_after_export's guard.
        run = self._export_run()
        with mock.patch(
            "training.jobs.runner.export_onnx", return_value={"onnx_path": run.output_path},
        ), mock.patch(
            "training.jobs.exports.export_pipeline_sidecar",
        ), mock.patch(
            "training.jobs.exports.build_bundle_request",
            side_effect=ValueError("detector checkpoint is missing"),
        ), mock.patch("training.jobs.runner.export_bundle") as export_bundle:
            jobs.run_export_onnx(run.pk)

        export_bundle.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.status, ExportRun.OK)
        self.assertIn("detector checkpoint is missing", run.bundle_error)
        self.assertIsNotNone(run.finished_at)

    def test_run_export_trt_sidecar_failure_does_not_strand_the_row(self):
        run = self._export_run(kind=ExportRun.TRT, output_path="/out/export-me-best.engine")
        with mock.patch(
            "training.jobs.runner.export_trt", return_value={"engine_path": run.output_path},
        ), mock.patch(
            "training.jobs.exports.export_pipeline_sidecar", side_effect=OSError("disk"),
        ), mock.patch(
            "training.jobs.exports.build_bundle_request", return_value=None,
        ):
            jobs.run_export_trt(run.pk)

        run.refresh_from_db()
        self.assertEqual(run.status, ExportRun.OK)
        self.assertIn("pipeline sidecar", run.bundle_error)

    def test_run_export_onnx_skips_bundle_when_theres_nothing_to_bundle(self):
        run = self._export_run()
        with mock.patch(
            "training.jobs.runner.export_onnx", return_value={"onnx_path": run.output_path},
        ), mock.patch(
            "training.jobs.exports.export_pipeline_sidecar",
        ), mock.patch(
            "training.jobs.exports.build_bundle_request", return_value=None,
        ), mock.patch("training.jobs.runner.export_bundle") as export_bundle:
            jobs.run_export_onnx(run.pk)

        export_bundle.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.status, ExportRun.OK)
        self.assertEqual(run.bundle_dir, "")

    def test_run_export_trt_passes_precision_and_input_hw_through(self):
        run = self._export_run(
            kind=ExportRun.TRT, output_path="/out/export-me-best.engine",
            precision="fp16", input_hw=[640, 640],
        )
        with mock.patch(
            "training.jobs.runner.export_trt",
            return_value={"engine_path": run.output_path, "precision": "fp16"},
        ) as export_trt, mock.patch(
            "training.jobs.exports.export_pipeline_sidecar",
        ), mock.patch(
            "training.jobs.exports.build_bundle_request", return_value=None,
        ):
            jobs.run_export_trt(run.pk)

        export_trt.assert_called_once_with(
            run.checkpoint_path, run.output_path, precision="fp16", input_hw=(640, 640))
        run.refresh_from_db()
        self.assertEqual(run.status, ExportRun.OK)

    def test_run_export_trt_dynamic_profile_passes_none_input_hw(self):
        run = self._export_run(kind=ExportRun.TRT, output_path="/out/export-me-best.engine",
                                precision="fp32", input_hw=None)
        with mock.patch(
            "training.jobs.runner.export_trt",
            return_value={"engine_path": run.output_path, "precision": "fp32"},
        ) as export_trt, mock.patch(
            "training.jobs.exports.export_pipeline_sidecar",
        ), mock.patch(
            "training.jobs.exports.build_bundle_request", return_value=None,
        ):
            jobs.run_export_trt(run.pk)

        export_trt.assert_called_once_with(
            run.checkpoint_path, run.output_path, precision="fp32", input_hw=None)


class RunnerInspectCheckpointTests(TestCase):
    """training.services.runner.inspect_checkpoint against a stubbed trainer response."""

    def test_returns_the_trainer_services_json_body(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"arch": "rfdetr", "trained_size": [560, 560]}
        with mock.patch("training.services.runner.requests.post", return_value=response) as post:
            info = runner.inspect_checkpoint("/ckpts/best.pt")

        self.assertEqual(info, {"arch": "rfdetr", "trained_size": [560, 560]})
        args, kwargs = post.call_args
        self.assertTrue(args[0].endswith("/checkpoint_info"))
        self.assertEqual(kwargs["json"], {"checkpoint_path": "/ckpts/best.pt"})

    def test_raises_runtimeerror_with_the_trainers_detail_on_http_error(self):
        response = mock.Mock(status_code=400, text="bad request")
        response.json.return_value = {"detail": "checkpoint not found: /ckpts/missing.pt"}
        with mock.patch("training.services.runner.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "checkpoint not found"):
                runner.inspect_checkpoint("/ckpts/missing.pt")


# ---------------------------------------------------------------- build nodes


class BuildNodeHealthTests(TestCase):
    """``buildnode.record_health`` — the ping behind the admin action.

    It must never raise: an unreachable node is an ordinary thing to show in a
    changelist, and the action pings a whole queryset.
    """

    def setUp(self):
        self.node = BuildNode.objects.create(
            name="gpu-box-a", base_url="http://10.0.0.9:8300/", token="tok")

    def test_records_snapshot_from_a_healthy_node(self):
        body = {
            "status": "ok", "gpu_name": "NVIDIA RTX 4090", "compute_capability": "8.9",
            "tensorrt_version": "11.2.1.2", "driver_version": "580.173.02",
            "busy": False, "queue_len": 0,
            "person_detector": {"present": True, "onnx": True, "trt_graph": True},
        }
        response = mock.Mock(status_code=200)
        response.json.return_value = body
        with mock.patch("training.services.buildnode.requests.get", return_value=response):
            returned = buildnode.record_health(self.node)

        self.assertEqual(returned, body)
        self.node.refresh_from_db()
        self.assertEqual(self.node.last_status, BuildNode.OK)
        self.assertEqual(self.node.gpu_name, "NVIDIA RTX 4090")
        self.assertEqual(self.node.tensorrt_version, "11.2.1.2")
        self.assertEqual(self.node.last_health, body)
        self.assertEqual(self.node.last_error, "")
        self.assertIsNotNone(self.node.last_seen_at)

    def test_unreachable_node_is_recorded_not_raised(self):
        with mock.patch(
            "training.services.buildnode.requests.get",
            side_effect=OSError("connection refused"),
        ):
            self.assertEqual(buildnode.record_health(self.node), {})

        self.node.refresh_from_db()
        self.assertEqual(self.node.last_status, BuildNode.ERROR)
        self.assertIn("connection refused", self.node.last_error)
        self.assertIsNone(self.node.last_seen_at)
        # A failed ping still records that we tried — otherwise the admin can't
        # distinguish "never pinged" from "pinged, and it's down".
        self.assertIsNotNone(self.node.last_checked_at)

    def test_degraded_node_is_not_ok(self):
        """A node that answers but has no GPU must not read as healthy."""
        response = mock.Mock(status_code=200)
        response.json.return_value = {"status": "degraded", "gpu_name": None,
                                      "tensorrt_version": None}
        with mock.patch("training.services.buildnode.requests.get", return_value=response):
            buildnode.record_health(self.node)

        self.node.refresh_from_db()
        self.assertEqual(self.node.last_status, BuildNode.ERROR)
        self.assertIn("degraded", self.node.last_error)

    def test_bearer_token_is_sent(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"status": "ok"}
        with mock.patch(
            "training.services.buildnode.requests.get", return_value=response
        ) as get:
            buildnode.health(self.node)
        self.assertEqual(
            get.call_args.kwargs["headers"], {"Authorization": "Bearer tok"})
        # base_url's trailing slash must not produce a double slash.
        self.assertEqual(get.call_args.args[0], "http://10.0.0.9:8300/health")

    def test_poll_treats_404_as_unknown(self):
        """A node that lost the build is terminal, not an exception."""
        response = mock.Mock(status_code=404, text="gone")
        with mock.patch("training.services.buildnode.requests.get", return_value=response):
            self.assertEqual(buildnode.poll(self.node, "abc"), {"status": "unknown"})

    def test_wait_raises_when_the_node_lost_the_build(self):
        with mock.patch.object(buildnode, "poll", return_value={"status": "unknown"}):
            with self.assertRaisesRegex(buildnode.BuildNodeError, "lost build"):
                buildnode.wait(self.node, "abc")


class BuildNodeArchiveTests(TestCase):
    """``extract_bundle`` — the archive came off another machine, so it's untrusted."""

    def _tar(self, tmp: Path, build) -> Path:
        import tarfile

        payload = tmp / "payload"
        payload.mkdir()
        build(payload)
        archive = tmp / "bundle.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for child in payload.iterdir():
                tar.add(child, arcname=child.name)
        return archive

    def test_extracts_a_normal_bundle(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)

            def build(root: Path):
                bundle = root / "thing-bundle"
                (bundle / "models").mkdir(parents=True)
                (bundle / "pipeline.json").write_text("{}")
                (bundle / "models" / "model.engine").write_bytes(b"x")

            archive = self._tar(tmp, build)
            dest = tmp / "out"
            result = buildnode.extract_bundle(archive, dest)

            self.assertEqual(result, dest / "thing-bundle")
            self.assertTrue((result / "pipeline.json").is_file())
            self.assertTrue((result / "models" / "model.engine").is_file())
            # The staging dir must not survive.
            self.assertEqual([p.name for p in dest.iterdir()], ["thing-bundle"])

    def test_rejects_a_traversing_member(self):
        import tarfile

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            escape = tmp / "escape.txt"
            escape.write_text("pwned")
            archive = tmp / "evil.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(escape, arcname="../escape.txt")

            with self.assertRaisesRegex(buildnode.BuildNodeError, "outside the destination"):
                buildnode.extract_bundle(archive, tmp / "out")
            self.assertFalse((tmp / "out" / "escape.txt").exists())

    def test_rejects_a_symlink_member(self):
        import tarfile

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            link = tmp / "link"
            link.symlink_to("/etc/passwd")
            archive = tmp / "evil.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(link, arcname="thing-bundle/link")

            with self.assertRaisesRegex(buildnode.BuildNodeError, "contains a link"):
                buildnode.extract_bundle(archive, tmp / "out")

    def test_rejects_an_archive_without_one_bundle_root(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)

            def build(root: Path):
                (root / "a-bundle").mkdir()
                (root / "b-bundle").mkdir()

            archive = self._tar(tmp, build)
            with self.assertRaisesRegex(buildnode.BuildNodeError, "exactly one bundle"):
                buildnode.extract_bundle(archive, tmp / "out")


class RemoteExportActionTests(TestCase):
    """The "Build on" selector routes to the remote job instead of the local one."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        ts = TrainingSettings.load()
        ts.exports_root = str(self.root / "exports")
        ts.bundles_root = str(self.root / "bundles")
        ts.save()

        self.model = TrainedModel.objects.create(
            name="export-me", arch=ExperimentModel.YOLOX, checkpoint_path="/ckpts/best.pt")
        self.node = BuildNode.objects.create(
            name="gpu-box-a", base_url="http://10.0.0.9:8300", token="tok")

        self.admin_user = User.objects.create_superuser("admin", "a@example.com", "pw")
        self.client.force_login(self.admin_user)

    def _post(self, **extra):
        return self.client.post(
            "/admin/training/trainedmodel/",
            {"action": "export_trt", ACTION_CHECKBOX_NAME: [str(self.model.pk)],
             "index": "0", **extra},
        )

    def test_choosing_a_node_queues_the_remote_job(self):
        with mock.patch.object(training_admin, "_queue") as queue:
            resp = self._post(apply="1", output_dir=str(self.root / "out"),
                              precision="fp16", node=str(self.node.pk))

        self.assertEqual(resp.status_code, 302)
        args, kwargs = queue.return_value.enqueue.call_args
        self.assertEqual(args[0], jobs.run_export_remote)
        self.assertEqual(kwargs.get("job_timeout"), jobs.EXPORT_REMOTE_JOB_TIMEOUT)
        self.assertEqual(ExportRun.objects.get().node, self.node)

    def test_no_node_still_queues_the_local_job(self):
        """The pre-existing path must be untouched when nothing is selected."""
        with mock.patch.object(training_admin, "_queue") as queue:
            self._post(apply="1", output_dir=str(self.root / "out"),
                       precision="fp16", node="")

        args, _ = queue.return_value.enqueue.call_args
        self.assertEqual(args[0], jobs.run_export_trt)
        self.assertIsNone(ExportRun.objects.get().node)

    def test_a_retired_node_queues_nothing(self):
        self.node.status = BuildNode.RETIRED
        self.node.save(update_fields=["status"])
        with mock.patch.object(training_admin, "_queue") as queue:
            self._post(apply="1", output_dir=str(self.root / "out"),
                       precision="fp16", node=str(self.node.pk))

        queue.return_value.enqueue.assert_not_called()
        self.assertEqual(ExportRun.objects.count(), 0)

    def test_form_lists_active_nodes(self):
        resp = self._post()
        self.assertContains(resp, "gpu-box-a")
        self.assertContains(resp, "Local trainer")


class RemoteExportJobTests(TestCase):
    """``jobs.run_export_remote`` — the whole round trip, with the node mocked out."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        ts = TrainingSettings.load()
        ts.exports_root = str(self.root / "exports")
        ts.bundles_root = str(self.root / "bundles")
        ts.save()

        # A person-crop model, so the detector branch is exercised.
        self.model = TrainedModel.objects.create(
            name="ppe", arch=ExperimentModel.RFDETR, checkpoint_path="/ckpts/best.pt",
            classes=["helmet", "vest"],
            pipeline_metadata={"pipeline": "people_detect_first", "chain": [],
                               "score_threshold": 0.4},
        )
        self.node = BuildNode.objects.create(
            name="gpu-box-a", base_url="http://10.0.0.9:8300", token="tok")

        # The graph the local trainer would have produced.
        self.graph = self.root / "exports" / "_remote" / "ppe-best.trt.onnx"
        self.graph.parent.mkdir(parents=True, exist_ok=True)
        self.graph.write_bytes(b"onnx")
        self.graph.with_suffix(".meta.json").write_text('{"arch": "rfdetr"}')

        self.run = ExportRun.objects.create(
            model=self.model, kind=ExportRun.TRT, checkpoint_label="best",
            checkpoint_path="/ckpts/best.pt",
            output_path=str(self.root / "out" / "ppe-best.engine"),
            precision="fp16", node=self.node,
        )

    def _bundle_on_disk(self, *_args, **_kwargs):
        """Stand in for extract_bundle: put a bundle where the real one would land."""
        bundle = bundles.bundles_root() / self.node.name / "ppe-best-bundle"
        (bundle / "models").mkdir(parents=True, exist_ok=True)
        return bundle

    def _run(self, status=None):
        status = status or {"status": "ok", "result": {"gpu_name": "RTX 4090",
                                                       "tensorrt_version": "11.2.1.2"}}
        with mock.patch.object(
            runner, "export_trt_onnx",
            return_value={"onnx_path": str(self.graph), "prepared": True, "arch": "rfdetr"},
        ), mock.patch.object(
            buildnode, "submit_build", return_value="bid123"
        ) as submit, mock.patch.object(
            buildnode, "wait", return_value=status
        ), mock.patch.object(
            buildnode, "download_artifact"
        ), mock.patch.object(
            buildnode, "extract_bundle", side_effect=self._bundle_on_disk
        ), mock.patch.object(buildnode, "cleanup") as cleanup:
            try:
                result = jobs.run_export_remote(self.run.pk)
            except Exception as exc:  # surfaced to the caller by design
                result = exc
        return result, submit, cleanup

    def test_happy_path_records_the_bundle_and_the_node(self):
        result, submit, cleanup = self._run()
        self.assertNotIsInstance(result, Exception)

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ExportRun.OK)
        self.assertEqual(self.run.remote_build_id, "bid123")
        self.assertTrue(self.run.bundle_dir.endswith("gpu-box-a/ppe-best-bundle"))
        # output_path points into the bundle: the engine only exists in there.
        self.assertTrue(self.run.output_path.endswith("ppe-best-bundle/models/model.engine"))
        self.assertEqual(self.run.result["node"], "gpu-box-a")
        self.assertEqual(self.run.result["gpu_name"], "RTX 4090")
        cleanup.assert_called()

    def test_spec_uses_bare_names_and_never_uploads_the_local_detector(self):
        _, submit, _ = self._run()
        spec = submit.call_args.args[1]
        files = submit.call_args.args[2]

        self.assertEqual(spec["request"]["model_checkpoint"], "model.onnx")
        self.assertTrue(spec["model_prepared"])
        self.assertEqual(spec["fmt"], "engine")

        # The shipped detector is an engine built for THIS box's GPU. It must never
        # be uploaded — the node compiles its own from the graph baked into it.
        detector = spec["request"]["detector"]
        self.assertEqual(detector["checkpoint"], "builtin")
        self.assertNotIn("detector", files)
        blob = json.dumps(spec)
        self.assertNotIn(DEFAULT_PERSON_DETECTOR_CHECKPOINT, blob)
        self.assertNotIn("best_ckpt.engine", blob)

        # Only the model graph + its meta go up.
        self.assertEqual(set(files), {"model", "model_meta"})

    def test_a_failed_remote_build_lands_on_the_row_with_the_node_log(self):
        result, _, cleanup = self._run(
            {"status": "error", "error": "TrtBuildError: out of memory",
             "log_tail": "[build] compiling model.onnx"})
        self.assertIsInstance(result, Exception)

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ExportRun.ERROR)
        self.assertIn("out of memory", self.run.last_error)
        self.assertIn("[build] compiling", self.run.last_error)
        cleanup.assert_called()

    def test_a_raw_frame_model_is_refused_before_uploading_anything(self):
        self.model.pipeline_metadata = {"pipeline": "", "chain": []}
        self.model.save(update_fields=["pipeline_metadata"])

        result, submit, _ = self._run()
        self.assertIsInstance(result, Exception)
        submit.assert_not_called()
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, ExportRun.ERROR)
        self.assertIn("no pipeline geometry", self.run.last_error)


class ForeignBundleLoadTestTests(TestCase):
    """A bundle built on a node must not be reported as broken here.

    Its engine cannot load on this machine by design, so running the load test
    would fail truthfully but read as "this bundle is bad".
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _engine(self, provenance: dict | None) -> Path:
        engine = self.root / "model.engine"
        engine.write_bytes(b"plan")
        if provenance is not None:
            Path(str(engine) + ".json").write_text(json.dumps(provenance))
        return engine

    def test_a_node_built_engine_is_flagged_as_foreign(self):
        engine = self._engine({"built_by": "buildnode", "gpu_name": "RTX 4090",
                               "tensorrt_version": "11.2.1.2"})
        reason = bundles._foreign_build(engine)
        self.assertIsNotNone(reason)
        self.assertIn("RTX 4090", reason)
        self.assertIn("11.2.1.2", reason)

    def test_a_locally_built_engine_is_not_flagged(self):
        self.assertIsNone(self._foreign({"precision": "fp16", "arch": "rfdetr"}))

    def test_an_engine_without_provenance_is_not_flagged(self):
        self.assertIsNone(self._foreign(None))

    def test_an_onnx_artifact_is_never_flagged(self):
        onnx = self.root / "model.onnx"
        onnx.write_bytes(b"onnx")
        Path(str(onnx) + ".json").write_text(json.dumps({"built_by": "buildnode"}))
        self.assertIsNone(bundles._foreign_build(onnx))

    def _foreign(self, provenance):
        return bundles._foreign_build(self._engine(provenance))
