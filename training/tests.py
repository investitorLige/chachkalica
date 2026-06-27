"""Tests for training config generation.

These exercise the pure ``config_gen`` surface against on-disk dataset fixtures,
so we know the YAML we hand friendy_mercury has the right shape and paths without
needing the trainer itself.
"""

import tempfile
from pathlib import Path

from django.core.exceptions import ValidationError
from django.test import TestCase

from fleet.models import Annotator, Dataset, FleetSettings
from training.models import Experiment, ExperimentDataset, ExperimentModel
from training.services import config_gen


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

        # cosine scheduler -> a dict with name; none -> None
        self.assertEqual(data["training"]["scheduler"], {"name": "cosine"})

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
