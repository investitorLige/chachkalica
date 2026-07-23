import math
import unittest
from types import SimpleNamespace

import torch

from friendy_chachkalica.ml.adapters.fasterrcnn import _build_fasterrcnn_model
from friendy_chachkalica.ml.adapters.retinanet import (
    RetinaNetAdapter,
    _build_retinanet_model,
)


class _FakeRetinaNet(torch.nn.Module):
    def __init__(self, num_classes=91, num_anchors=3):
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 8, kernel_size=1)
        classification_tower = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        classification_head = SimpleNamespace(
            conv=classification_tower,
            cls_logits=torch.nn.Conv2d(
                8,
                num_anchors * num_classes,
                kernel_size=3,
                padding=1,
            ),
            num_anchors=num_anchors,
            num_classes=num_classes,
        )
        self.head = SimpleNamespace(classification_head=classification_head)


class _FakeFastRCNNPredictor(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.cls_score = torch.nn.Linear(in_features, num_classes)
        self.bbox_pred = torch.nn.Linear(in_features, num_classes * 4)


class _FakeFasterRCNN(torch.nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.roi_heads = SimpleNamespace(
            box_predictor=_FakeFastRCNNPredictor(16, num_classes)
        )


class _TargetCapturingRetinaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_targets = None

    def forward(self, images, targets):
        self.received_targets = targets
        return {"loss": torch.tensor(1.0)}


class RetinaNetPretrainedTransferTests(unittest.TestCase):
    def test_loads_native_pretrained_model_then_replaces_only_logits(self):
        weights = object()
        model = _FakeRetinaNet()
        original_backbone = model.backbone
        original_tower = model.head.classification_head.conv
        calls = []

        def builder(**kwargs):
            calls.append(kwargs)
            return model

        result = _build_retinanet_model(
            builder=builder,
            model_weights=weights,
            backbone_weights=None,
            model_num_classes=5,
            trainable_backbone_layers=3,
            builder_kwargs={"min_size": 640},
        )

        self.assertIs(result, model)
        self.assertIs(model.backbone, original_backbone)
        self.assertIs(model.head.classification_head.conv, original_tower)
        self.assertEqual(model.head.classification_head.num_classes, 5)
        self.assertEqual(model.head.classification_head.cls_logits.out_channels, 15)
        self.assertIs(calls[0]["weights"], weights)
        self.assertIsNone(calls[0]["weights_backbone"])
        self.assertNotIn("num_classes", calls[0])
        expected_bias = -math.log(99.0)
        self.assertTrue(
            torch.allclose(
                model.head.classification_head.cls_logits.bias,
                torch.full((15,), expected_bias),
            )
        )

    def test_scratch_build_receives_exact_foreground_class_count(self):
        calls = []
        sentinel = object()

        def builder(**kwargs):
            calls.append(kwargs)
            return sentinel

        result = _build_retinanet_model(
            builder=builder,
            model_weights=None,
            backbone_weights="imagenet-weights",
            model_num_classes=5,
            trainable_backbone_layers=None,
            builder_kwargs={},
        )

        self.assertIs(result, sentinel)
        self.assertEqual(calls[0]["num_classes"], 5)
        self.assertEqual(calls[0]["weights_backbone"], "imagenet-weights")

    def test_training_keeps_zero_based_dataset_labels(self):
        model = _TargetCapturingRetinaNet()
        adapter = RetinaNetAdapter(model=model, num_classes=2)
        targets = [{"labels": torch.tensor([0, 1])}]

        adapter.training_step([torch.zeros(3, 8, 8)], targets)

        self.assertIs(model.received_targets, targets)
        self.assertEqual(model.received_targets[0]["labels"].tolist(), [0, 1])


class FasterRCNNPretrainedTransferTests(unittest.TestCase):
    def test_loads_native_pretrained_model_then_replaces_only_predictor(self):
        weights = object()
        model = _FakeFasterRCNN()
        original_backbone = model.backbone
        calls = []

        def builder(**kwargs):
            calls.append(kwargs)
            return model

        result = _build_fasterrcnn_model(
            builder=builder,
            model_weights=weights,
            backbone_weights=None,
            model_num_classes=6,
            trainable_backbone_layers=3,
            predictor_factory=_FakeFastRCNNPredictor,
            builder_kwargs={"box_score_thresh": 0.01},
        )

        self.assertIs(result, model)
        self.assertIs(model.backbone, original_backbone)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 6)
        self.assertEqual(model.roi_heads.box_predictor.bbox_pred.out_features, 24)
        self.assertIs(calls[0]["weights"], weights)
        self.assertIsNone(calls[0]["weights_backbone"])
        self.assertNotIn("num_classes", calls[0])

    def test_scratch_build_receives_background_inclusive_class_count(self):
        calls = []
        sentinel = object()

        def builder(**kwargs):
            calls.append(kwargs)
            return sentinel

        result = _build_fasterrcnn_model(
            builder=builder,
            model_weights=None,
            backbone_weights="imagenet-weights",
            model_num_classes=6,
            trainable_backbone_layers=None,
            predictor_factory=_FakeFastRCNNPredictor,
            builder_kwargs={},
        )

        self.assertIs(result, sentinel)
        self.assertEqual(calls[0]["num_classes"], 6)
        self.assertEqual(calls[0]["weights_backbone"], "imagenet-weights")


if __name__ == "__main__":
    unittest.main()
