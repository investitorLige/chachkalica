import unittest
from types import SimpleNamespace

import torch

from friendy_chachkalica.adapters.fasterrcnn import FasterRCNNAdapter
from friendy_chachkalica.adapters.retinanet import RetinaNetAdapter
from friendy_chachkalica.train import _predict_with_config


def _torchvision_prediction(label: int):
    return {
        "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([label]),
    }


class _ThresholdCapturingRetinaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score_thresh = 0.05
        self.seen_thresholds = []
        self.fail = False

    def forward(self, images):
        self.seen_thresholds.append(self.score_thresh)
        if self.fail:
            raise RuntimeError("inference failed")
        return [_torchvision_prediction(label=0) for _ in images]


class _ThresholdCapturingFasterRCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roi_heads = SimpleNamespace(score_thresh=0.05)
        self.seen_thresholds = []

    def forward(self, images):
        self.seen_thresholds.append(self.roi_heads.score_thresh)
        return [_torchvision_prediction(label=1) for _ in images]


class TorchvisionAdapterThresholdTests(unittest.TestCase):
    def test_retinanet_uses_requested_floor_and_restores_serving_threshold(self):
        model = _ThresholdCapturingRetinaNet()
        adapter = RetinaNetAdapter(model=model, num_classes=1)

        predictions = adapter.predict(
            [torch.zeros(3, 100, 200)],
            score_threshold=0.001,
        )

        self.assertEqual(model.seen_thresholds, [0.001])
        self.assertEqual(model.score_thresh, 0.05)
        self.assertEqual(predictions[0].shape, (1, 6))
        self.assertEqual(predictions[0][0, 5].item(), 0.0)

    def test_retinanet_restores_threshold_when_inference_fails(self):
        model = _ThresholdCapturingRetinaNet()
        model.fail = True
        adapter = RetinaNetAdapter(model=model, num_classes=1)

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            adapter.predict(
                [torch.zeros(3, 100, 200)],
                score_threshold=0.001,
            )

        self.assertEqual(model.seen_thresholds, [0.001])
        self.assertEqual(model.score_thresh, 0.05)

    def test_fasterrcnn_uses_requested_floor_and_restores_serving_threshold(self):
        model = _ThresholdCapturingFasterRCNN()
        adapter = FasterRCNNAdapter(model=model, num_classes=1)

        predictions = adapter.predict(
            [torch.zeros(3, 100, 200)],
            score_threshold=0.001,
        )

        self.assertEqual(model.seen_thresholds, [0.001])
        self.assertEqual(model.roi_heads.score_thresh, 0.05)
        self.assertEqual(predictions[0].shape, (1, 6))
        self.assertEqual(predictions[0][0, 5].item(), 0.0)


class _ThresholdCapturingAdapter:
    def __init__(self):
        self.seen_threshold = None

    def predict(self, images, score_threshold=None):
        self.seen_threshold = score_threshold
        return []


class _NoThresholdAdapter:
    def predict(self, images):
        return []


class TrainerThresholdRoutingTests(unittest.TestCase):
    def test_map_floor_takes_precedence_over_operating_threshold(self):
        adapter = _ThresholdCapturingAdapter()
        config = SimpleNamespace(
            evaluation=SimpleNamespace(
                map_score_threshold=0.002,
                score_threshold=0.5,
            )
        )

        _predict_with_config(adapter, [], config)

        self.assertEqual(adapter.seen_threshold, 0.002)

    def test_operating_threshold_is_used_when_map_floor_is_unspecified(self):
        adapter = _ThresholdCapturingAdapter()
        config = SimpleNamespace(
            evaluation=SimpleNamespace(
                map_score_threshold=None,
                score_threshold=0.25,
            )
        )

        _predict_with_config(adapter, [], config)

        self.assertEqual(adapter.seen_threshold, 0.25)

    def test_adapter_that_cannot_honor_threshold_fails_loudly(self):
        config = SimpleNamespace(
            evaluation=SimpleNamespace(
                map_score_threshold=0.001,
                score_threshold=0.5,
            )
        )

        with self.assertRaises(TypeError):
            _predict_with_config(_NoThresholdAdapter(), [], config)


if __name__ == "__main__":
    unittest.main()
