"""Tests for run_combined_pipeline: merging 2+ checkpoints' predictions.

Uses stub model adapters (a duck-typed ``.predict``) and a mocked
``load_checkpoint_adapter``/``build_eval_dataloader`` so nothing needs a real
checkpoint file or a GPU. Run from the chachak directory:

    python -m unittest tests.test_run
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_CHACHAK_DIR = Path(__file__).resolve().parent.parent
if str(_CHACHAK_DIR) not in sys.path:
    sys.path.insert(0, str(_CHACHAK_DIR))

import torch  # noqa: E402

from config import DetectorConfig, PipelineConfig, TilingConfig  # noqa: E402
import run as run_mod  # noqa: E402


class StubModel:
    """Emits one fixed box per input image."""

    def __init__(self, box):
        self.box = box

    def predict(self, images, score_threshold=None):
        return [torch.tensor([list(self.box)]) for _ in images]


def make_combined_config(**overrides):
    kwargs = dict(
        name="combo",
        pipeline="batch_detect",
        model_checkpoint=Path("/dev/null/a.pt"),
        extra_checkpoints=[Path("/dev/null/b.pt")],
        images=Path("/dev/null"),
        labels=Path("/dev/null"),
        classes={0: "cat", 1: "dog"},
        output_dir=Path(tempfile.mkdtemp()),
        device="cpu",
        infer_batch_size=2,
        score_threshold=0.05,
        tiling=TilingConfig(tile_width_pct=100, tile_height_pct=100, overlap=0.0, nms_iou=0.5),
        detector=DetectorConfig(),
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def sample_frames():
    images = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
    targets = [
        {"image_path": "img1.jpg", "label_path": "img1.txt",
         "orig_size": torch.tensor([64, 64]),
         "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]), "labels": torch.tensor([0])},
        {"image_path": "img2.jpg", "label_path": "img2.txt",
         "orig_size": torch.tensor([64, 64]),
         "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]), "labels": torch.tensor([1])},
    ]
    return images, targets


class RunCombinedPipelineTest(unittest.TestCase):
    def _run(self, config, adapters_info, loader):
        calls = iter(adapters_info)

        def fake_load_checkpoint_adapter(checkpoint, device):
            return next(calls)

        with mock.patch.object(
            run_mod, "load_checkpoint_adapter", side_effect=fake_load_checkpoint_adapter
        ), mock.patch.object(run_mod, "build_eval_dataloader", return_value=loader):
            return run_mod.run_combined_pipeline(config)

    def test_merges_disjoint_class_predictions_with_correct_eval_ids(self):
        images, targets = sample_frames()
        config = make_combined_config()
        adapters_info = [
            (StubModel((0.5, 0.5, 0.4, 0.4, 0.9, 0)), {"train_classes": {0: "cat"}}),
            (StubModel((0.5, 0.5, 0.4, 0.4, 0.8, 0)), {"train_classes": {0: "dog"}}),
        ]
        result = self._run(config, adapters_info, loader=[(images, targets)])

        self.assertIn("map50", result["metrics"])
        self.assertEqual(result["metrics"]["num_predictions"], 4)  # 2 images x 2 boxes
        self.assertEqual(result["model_checkpoint"], ["/dev/null/a.pt", "/dev/null/b.pt"])

        records = torch.load(Path(config.output_dir) / "predictions.pt")
        self.assertEqual(len(records), 2)
        for record in records:
            labels = sorted(int(row[5]) for row in record["predictions"])
            self.assertEqual(labels, [0, 1])  # both models' boxes, in eval-space ids

    def test_unmapped_class_from_one_model_is_dropped_not_merged(self):
        images, targets = sample_frames()
        config = make_combined_config()
        adapters_info = [
            (StubModel((0.5, 0.5, 0.4, 0.4, 0.9, 0)), {"train_classes": {0: "cat"}}),
            # "bird" isn't in the eval class space -> dropped after remap.
            (StubModel((0.5, 0.5, 0.4, 0.4, 0.8, 0)), {"train_classes": {0: "bird"}}),
        ]
        result = self._run(config, adapters_info, loader=[(images, targets)])
        records = torch.load(Path(config.output_dir) / "predictions.pt")
        for record in records:
            self.assertEqual(record["predictions"].shape[0], 1)
            self.assertEqual(int(record["predictions"][0, 5]), 0)

    def test_rejects_checkpoint_without_train_classes(self):
        images, targets = sample_frames()
        config = make_combined_config()
        adapters_info = [
            (StubModel((0.5, 0.5, 0.4, 0.4, 0.9, 0)), {"train_classes": {0: "cat"}}),
            (StubModel((0.5, 0.5, 0.4, 0.4, 0.8, 0)), {"train_classes": {}}),
        ]
        with self.assertRaises(ValueError):
            self._run(config, adapters_info, loader=[(images, targets)])


if __name__ == "__main__":
    unittest.main()
