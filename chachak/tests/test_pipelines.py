"""Tests for the pipeline classes and the base Pipeline.run() loop.

Uses stub model/detector adapters (a duck-typed ``.predict``) so nothing needs a
trained checkpoint or a GPU. Run from the chachak directory:

    python -m unittest tests.test_pipelines
"""

import sys
import tempfile
import unittest
from pathlib import Path

_CHACHAK_DIR = Path(__file__).resolve().parent.parent
if str(_CHACHAK_DIR) not in sys.path:
    sys.path.insert(0, str(_CHACHAK_DIR))

import torch  # noqa: E402

from config import DetectorConfig, PipelineConfig, TilingConfig  # noqa: E402
from infer import infer_in_chunks  # noqa: E402
from pipeline import (  # noqa: E402
    BatchDetectPipeline,
    BatchPeoplePipeline,
    ChainedPipeline,
    PeopleDetectFirstPipeline,
)

DEVICE = torch.device("cpu")


class StubModel:
    """Emits one centered box per input image (tile or crop)."""

    def __init__(self, box=(0.5, 0.5, 0.4, 0.4, 0.9, 0.0)):
        self.box = box

    def predict(self, images, score_threshold=None):
        return [torch.tensor([list(self.box)]) for _ in images]


class EmptyModel:
    def predict(self, images, score_threshold=None):
        return [torch.zeros((0, 6)) for _ in images]


class ShortBatchModel:
    def predict(self, images, score_threshold=None):
        return [torch.zeros((0, 6)) for _ in images[:-1]]


class TileAwareStubModel:
    """Emits one centered box per tile, class_id cycling 0, 1, 2... in call order.

    Lets a test assert that a tile's class survives remap + merge intact,
    instead of every tile emitting the same hardcoded class.
    """

    def __init__(self, box_wh=(0.5, 0.5, 0.3, 0.3, 0.9)):
        self.box_wh = box_wh
        self._next_class = 0

    def predict(self, images, score_threshold=None):
        preds = []
        for _ in images:
            preds.append(torch.tensor([[*self.box_wh, float(self._next_class)]]))
            self._next_class += 1
        return preds


class StubDetector:
    """One centered person box per frame (normalized to each input)."""

    def predict(self, images):
        return [torch.tensor([[0.5, 0.5, 0.6, 0.6, 0.99, 1.0]]) for _ in images]


def make_config(pipeline, **overrides):
    kwargs = dict(
        name="t",
        pipeline=pipeline,
        model_checkpoint=Path("/dev/null"),
        images=Path("/dev/null"),
        labels=Path("/dev/null"),
        classes={0: "a", 1: "b"},
        output_dir=Path("/tmp/chachak_test_out"),
        device="cpu",
        infer_batch_size=2,
        score_threshold=0.05,
        tiling=TilingConfig(tile_width_pct=40, tile_height_pct=40, overlap=0.2, nms_iou=0.5),
        detector=DetectorConfig(score_threshold=0.0, expand_ratio=0.1, nms_iou=0.5),
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def sample_frames():
    images = [torch.rand(3, 120, 160), torch.rand(3, 96, 144)]
    targets = [
        {
            "image_path": "a.jpg",
            "label_path": "a.txt",
            "orig_size": torch.tensor([120, 160]),
            "boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]),
            "labels": torch.tensor([0]),
        },
        {
            "image_path": "b.jpg",
            "label_path": "b.txt",
            "orig_size": torch.tensor([96, 144]),
            "boxes": torch.tensor([[5.0, 5.0, 60.0, 60.0]]),
            "labels": torch.tensor([1]),
        },
    ]
    return images, targets


def assert_valid_preds(testcase, outputs, n_frames):
    testcase.assertEqual(len(outputs), n_frames)
    for preds in outputs:
        testcase.assertEqual(preds.ndim, 2)
        testcase.assertEqual(preds.shape[1], 6)
        if preds.numel():
            boxes = preds[:, :4]
            testcase.assertGreaterEqual(float(boxes.min()), -1e-4)
            testcase.assertLessEqual(float(boxes.max()), 1.0 + 1e-4)


class InferInChunksTest(unittest.TestCase):
    def test_rejects_adapter_output_count_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "prediction count mismatch"):
            infer_in_chunks(
                ShortBatchModel(),
                [torch.rand(3, 10, 10), torch.rand(3, 10, 10)],
                chunk_size=2,
            )


class ProcessBatchTest(unittest.TestCase):
    def test_batch_detect_tiles_and_produces_boxes(self):
        images, targets = sample_frames()
        pipe = BatchDetectPipeline(StubModel(), DEVICE, make_config("batch_detect"))
        out = pipe.process_batch(images, targets)
        assert_valid_preds(self, out, len(images))
        # 40% tiles over 120x160 / 96x144 yield several tiles -> several boxes.
        self.assertTrue(all(p.shape[0] >= 1 for p in out))

    def test_batch_detect_empty_model_yields_empty(self):
        images, targets = sample_frames()
        pipe = BatchDetectPipeline(EmptyModel(), DEVICE, make_config("batch_detect"))
        out = pipe.process_batch(images, targets)
        assert_valid_preds(self, out, len(images))
        self.assertTrue(all(p.shape[0] == 0 for p in out))

    def test_batch_detect_preserves_labels_per_tile(self):
        # 50%-wide, full-height, no-overlap tiling on a 100x100 frame yields
        # exactly two side-by-side tiles: [0, 50) and [50, 100).
        image = torch.rand(3, 100, 100)
        target = {
            "image_path": "a.jpg",
            "label_path": "a.txt",
            "orig_size": torch.tensor([100, 100]),
            "boxes": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
            "labels": torch.tensor([0]),
        }
        config = make_config(
            "batch_detect",
            tiling=TilingConfig(tile_width_pct=50, tile_height_pct=100, overlap=0.0, nms_iou=0.5),
        )
        pipe = BatchDetectPipeline(TileAwareStubModel(), DEVICE, config)
        out = pipe.process_batch([image], [target])[0]

        self.assertEqual(out.shape[0], 2)
        # Sort by x-center so the left tile's box is compared first.
        sorted_out = out[out[:, 0].argsort()]
        self.assertEqual(sorted_out[0, 5].item(), 0.0)  # left tile -> class 0
        self.assertEqual(sorted_out[1, 5].item(), 1.0)  # right tile -> class 1

    def test_people_detect_first_crops_and_infers(self):
        images, targets = sample_frames()
        pipe = PeopleDetectFirstPipeline(
            StubModel(), DEVICE, make_config("people_detect_first"), detector=StubDetector()
        )
        out = pipe.process_batch(images, targets)
        assert_valid_preds(self, out, len(images))
        self.assertTrue(all(p.shape[0] == 1 for p in out))  # one person -> one crop box

    def test_people_detect_first_without_detector_raises(self):
        images, targets = sample_frames()
        pipe = PeopleDetectFirstPipeline(StubModel(), DEVICE, make_config("people_detect_first"))
        with self.assertRaises(ValueError):
            pipe.process_batch(images, targets)

    def test_batch_people_tiles_detects_then_crops(self):
        images, targets = sample_frames()
        pipe = BatchPeoplePipeline(
            StubModel(), DEVICE, make_config("batch_people"), detector=StubDetector()
        )
        out = pipe.process_batch(images, targets)
        assert_valid_preds(self, out, len(images))
        # Person boxes from overlapping tiles collapse; each yields a crop box.
        self.assertTrue(all(p.shape[0] >= 1 for p in out))

    def test_chain_merges_child_outputs(self):
        images, targets = sample_frames()
        bd = BatchDetectPipeline(StubModel(), DEVICE, make_config("batch_detect"))
        pdf = PeopleDetectFirstPipeline(
            StubModel(), DEVICE, make_config("people_detect_first"), detector=StubDetector()
        )
        chain = ChainedPipeline(
            StubModel(), DEVICE, make_config("batch_detect"), detector=StubDetector(),
            pipelines=[bd, pdf],
        )
        out = chain.process_batch(images, targets)
        assert_valid_preds(self, out, len(images))
        self.assertIn("chain[", chain.name)


class RunLoopTest(unittest.TestCase):
    def test_run_writes_predictions_and_metrics(self):
        images, targets = sample_frames()
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config("batch_detect", output_dir=Path(tmp))
            pipe = BatchDetectPipeline(StubModel(), DEVICE, config)
            # A "loader" is anything iterable of (images, targets) batches.
            loader = [(images, targets)]
            result = pipe.run(
                loader, tmp,
                num_classes=2,
                prediction_classes=config.classes,
                target_classes=config.classes,
                eval_classes=config.classes,
            )
            pred_path = Path(tmp) / "predictions.pt"
            self.assertTrue(pred_path.exists())
            records = torch.load(pred_path)
            self.assertEqual(len(records), len(images))
            self.assertIn("image_path", records[0])
            self.assertIn("predictions", records[0])
            self.assertIn("map50", result["metrics"])
            self.assertIn("eval_seconds", result["metrics"])


if __name__ == "__main__":
    unittest.main()
