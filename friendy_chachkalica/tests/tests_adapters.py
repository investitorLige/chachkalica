import math
import unittest
from types import SimpleNamespace

import torch

from friendy_chachkalica.ml.adapters.fasterrcnn import _build_fasterrcnn_model
from friendy_chachkalica.ml.adapters.retinanet import (
    RetinaNetAdapter,
    _build_retinanet_model,
)
from friendy_chachkalica.ml.adapters.dfine import DFineAdapter
from friendy_chachkalica.ml.adapters.ecdet import ECDetAdapter
from friendy_chachkalica.ml.adapters.rfdetr import RFDETRAdapter
from friendy_chachkalica.ml.adapters.rtdetr import RTDETRAdapter
from friendy_chachkalica.ml.adapters.yolox import YOLOXAdapter, yolox_detection_to_friendy


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


class _DummyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch, *args, **kwargs):
        return {"batch": batch}


class _FixedRFPostprocess:
    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, outputs, target_sizes):
        return [{
            "boxes": self.boxes.to(target_sizes.device),
            "scores": torch.tensor([0.9], device=target_sizes.device),
            "labels": torch.tensor([0], device=target_sizes.device),
        }]


class _ConfiguredDetector(_DummyDetector):
    """A dummy whose ``.config`` RTDETRAdapter.predict reads for use_focal_loss."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_focal_loss=True)

    def forward(self, **kwargs):
        return {"batch": kwargs}


class _FixedRTPostprocess:
    """Stands in for the HF image processor: returns boxes in model-input pixels."""

    def __init__(self, boxes):
        self.boxes = boxes

    def post_process_object_detection(self, outputs, threshold, target_sizes, use_focal_loss):
        return [{
            "boxes": self.boxes.to(target_sizes.device),
            "scores": torch.tensor([0.9], device=target_sizes.device),
            "labels": torch.tensor([0], device=target_sizes.device),
        }]


class _FixedECDetHead(_DummyDetector):
    """Stands in for the ECDet decoder: fixed ``pred_logits``/``pred_boxes``.

    ``pred_boxes`` is normalized ``cxcywh`` over the model input, exactly what
    ``ECTransformer`` emits, so ``predict``'s sigmoid + top-k + clip is exercised
    against known values.
    """

    def __init__(self, logits, boxes):
        super().__init__()
        self.logits = logits
        self.boxes = boxes

    def forward(self, batch, targets=None):
        return {
            "pred_logits": self.logits.to(batch.device),
            "pred_boxes": self.boxes.to(batch.device),
        }


class AdapterResizeRoundTripTests(unittest.TestCase):
    IMAGE_H = 333
    IMAGE_W = 1000
    BOX = torch.tensor([[100.0, 50.0, 500.0, 300.0]])

    def _target(self):
        return {"boxes": self.BOX.clone(), "labels": torch.tensor([0])}

    def test_yolox_uses_actual_rounded_resize_scales_in_both_directions(self):
        adapter = YOLOXAdapter(model=_DummyDetector(), num_classes=1, input_max_size=640)
        image = torch.zeros(3, self.IMAGE_H, self.IMAGE_W)
        resized, targets = adapter._resize_training_inputs([image], [self._target()])
        scale_y = resized[0].shape[-2] / self.IMAGE_H
        scale_x = resized[0].shape[-1] / self.IMAGE_W
        expected = self.BOX.clone()
        expected[:, [0, 2]] *= scale_x
        expected[:, [1, 3]] *= scale_y
        self.assertTrue(torch.allclose(targets[0]["boxes"], expected))

        detection = torch.cat([expected, torch.tensor([[0.9, 1.0, 0.0]])], dim=1)
        pred = yolox_detection_to_friendy(detection, image, scale_y, scale_x)
        self.assertTrue(torch.allclose(pred[0, :4], torch.tensor([0.3, 175.0 / 333.0, 0.4, 250.0 / 333.0])))

    def test_rtdetr_stretches_to_a_square_canvas_with_no_padding(self):
        adapter = RTDETRAdapter(
            model=_DummyDetector(), image_processor=None, num_classes=1, input_max_size=640
        )
        image = torch.zeros(3, self.IMAGE_H, self.IMAGE_W)
        resized, _targets = adapter._resize_training_inputs([image], [self._target()])
        # No letterbox margin: the whole model input is real content, so nothing
        # the decoder is trained to point at can land on padding.
        self.assertEqual(tuple(resized[0].shape[-2:]), (640, 640))

    def test_rtdetr_labels_normalize_over_the_model_input(self):
        adapter = RTDETRAdapter(
            model=_DummyDetector(), image_processor=None, num_classes=1, input_max_size=640
        )
        image = torch.zeros(3, self.IMAGE_H, self.IMAGE_W)
        resized, targets = adapter._resize_training_inputs([image], [self._target()])
        labels = adapter._prepare_labels(targets, resized[0].shape[-2:])
        # Stretching to the square canvas leaves normalized coordinates equal to
        # the original image's own fractions — which is what makes the exported
        # graph's `box_coords: "input_normalized"` contract hold.
        self.assertTrue(torch.allclose(labels[0]["boxes"], torch.tensor([[0.3, 175.0 / 333.0, 0.4, 250.0 / 333.0]])))

    def test_rfdetr_round_trip_uses_actual_per_axis_scale_after_rounding(self):
        image = torch.zeros(3, self.IMAGE_H, self.IMAGE_W)
        adapter = RFDETRAdapter(
            model=_DummyDetector(),
            criterion=None,
            postprocess=_FixedRFPostprocess(torch.tensor([[64.0, 50.0 * 213.0 / 333.0, 320.0, 300.0 * 213.0 / 333.0]])),
            num_classes=1,
            resolution=640,
        )
        _batch, scales = adapter._prepare_batch([image])
        scale_y, scale_x = scales[0]
        self.assertEqual(scale_x, 0.64)
        self.assertEqual(scale_y, 213 / 333)

        labels = adapter._prepare_labels([self._target()], scales)
        expected_canvas = self.BOX.clone()
        expected_canvas[:, [0, 2]] *= scale_x
        expected_canvas[:, [1, 3]] *= scale_y
        expected_label = torch.tensor([[
            (expected_canvas[0, 0] + expected_canvas[0, 2]) / 2 / 640,
            (expected_canvas[0, 1] + expected_canvas[0, 3]) / 2 / 640,
            (expected_canvas[0, 2] - expected_canvas[0, 0]) / 640,
            (expected_canvas[0, 3] - expected_canvas[0, 1]) / 640,
        ]])
        self.assertTrue(torch.allclose(labels[0]["boxes"], expected_label))

        pred = adapter.predict([image], score_threshold=0.0)[0]
        # The canvas box corresponds to [100, 50, 500, 300] using the actual
        # (rounded) resize dimensions: 640x213.
        self.assertTrue(torch.allclose(pred[0, :4], torch.tensor([0.3, 175.0 / 333.0, 0.4, 250.0 / 333.0])))

    def test_rtdetr_pad_margin_prediction_stays_inside_the_image(self):
        # Resize disabled: _prepare_batch pads 100x100 up to the 32-multiple
        # 128x128, and RT-DETR can score a box in that margin. The result must
        # still be normalized within [0, 1] against the real image.
        adapter = RTDETRAdapter(
            model=_ConfiguredDetector(),
            image_processor=_FixedRTPostprocess(torch.tensor([[90.0, 90.0, 120.0, 120.0]])),
            num_classes=1,
            input_max_size=None,
        )
        pred = adapter.predict([torch.zeros(3, 100, 100)], score_threshold=0.0)[0]

        cx, cy, w, h = pred[0, :4].tolist()
        self.assertTrue(all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)), pred[0, :4])
        # Clipped to [90, 90, 100, 100] of a 100x100 image.
        self.assertTrue(torch.allclose(pred[0, :4], torch.tensor([0.95, 0.95, 0.1, 0.1])))

    def test_dfine_stretches_to_a_square_canvas_with_no_padding(self):
        # DFineAdapter is a near-verbatim clone of RTDETRAdapter (same shared
        # transformers-DETR geometry, confirmed against modeling_d_fine.py).
        adapter = DFineAdapter(
            model=_DummyDetector(), image_processor=None, num_classes=1, input_max_size=640
        )
        image = torch.zeros(3, self.IMAGE_H, self.IMAGE_W)
        resized, _targets = adapter._resize_training_inputs([image], [self._target()])
        self.assertEqual(tuple(resized[0].shape[-2:]), (640, 640))

    def test_dfine_labels_normalize_over_the_model_input(self):
        adapter = DFineAdapter(
            model=_DummyDetector(), image_processor=None, num_classes=1, input_max_size=640
        )
        image = torch.zeros(3, self.IMAGE_H, self.IMAGE_W)
        resized, targets = adapter._resize_training_inputs([image], [self._target()])
        labels = adapter._prepare_labels(targets, resized[0].shape[-2:])
        self.assertTrue(torch.allclose(
            labels[0]["boxes"], torch.tensor([[0.3, 175.0 / 333.0, 0.4, 250.0 / 333.0]])
        ))

    def test_dfine_pad_margin_prediction_stays_inside_the_image(self):
        # Resize disabled: _prepare_batch pads 100x100 up to the 32-multiple
        # 128x128, and D-FINE's decoder can score a box in that margin (its
        # padding-blind pixel_mask handling matches RT-DETR's). The result must
        # still be normalized within [0, 1] against the real image.
        adapter = DFineAdapter(
            model=_ConfiguredDetector(),
            image_processor=_FixedRTPostprocess(torch.tensor([[90.0, 90.0, 120.0, 120.0]])),
            num_classes=1,
            input_max_size=None,
        )
        pred = adapter.predict([torch.zeros(3, 100, 100)], score_threshold=0.0)[0]

        cx, cy, w, h = pred[0, :4].tolist()
        self.assertTrue(all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)), pred[0, :4])
        self.assertTrue(torch.allclose(pred[0, :4], torch.tensor([0.95, 0.95, 0.1, 0.1])))

    def test_ecdet_stretches_to_a_square_canvas_with_no_padding(self):
        adapter = ECDetAdapter(
            model=_DummyDetector(), criterion=None, num_classes=1, input_max_size=640
        )
        image = torch.zeros(3, self.IMAGE_H, self.IMAGE_W)
        resized, _targets = adapter._resize_training_inputs([image], [self._target()])
        self.assertEqual(tuple(resized[0].shape[-2:]), (640, 640))

    def test_ecdet_labels_normalize_over_the_model_input(self):
        adapter = ECDetAdapter(
            model=_DummyDetector(), criterion=None, num_classes=1, input_max_size=640
        )
        image = torch.zeros(3, self.IMAGE_H, self.IMAGE_W)
        resized, targets = adapter._resize_training_inputs([image], [self._target()])
        labels = adapter._prepare_labels(targets, resized[0].shape[-2:])
        # Same identity RT-DETR relies on: stretching to the square canvas leaves
        # normalized coordinates equal to the original image's own fractions, which
        # is what makes the exported graph's `box_coords: "input_normalized"` hold.
        self.assertTrue(torch.allclose(
            labels[0]["boxes"],
            torch.tensor([[0.3, 175.0 / 333.0, 0.4, 250.0 / 333.0]]),
        ))

    def test_ecdet_predict_round_trips_normalized_boxes_and_clips_the_overflow(self):
        # Two queries: one comfortably inside the frame, one whose sigmoid-decoded
        # centre puts its right/bottom edge past the canvas (cx + w/2 = 1.05). The
        # first must round-trip exactly; the second must come back clipped, matching
        # the exported graph's clip_boxes=True.
        boxes = torch.tensor([[[0.3, 0.5, 0.4, 0.2], [0.95, 0.95, 0.2, 0.2]]])
        logits = torch.tensor([[[4.0], [3.0]]])
        adapter = ECDetAdapter(
            model=_FixedECDetHead(logits, boxes),
            criterion=None,
            num_classes=1,
            input_max_size=640,
            num_top_queries=2,
        )
        pred = adapter.predict([torch.zeros(3, self.IMAGE_H, self.IMAGE_W)], score_threshold=0.0)

        rows = pred[0]
        self.assertEqual(rows.shape, (2, 6))
        by_score = rows[rows[:, 4].argsort(descending=True)]
        self.assertTrue(torch.allclose(
            by_score[0, :4], torch.tensor([0.3, 0.5, 0.4, 0.2]), atol=1e-6
        ))
        # x2/y2 clipped from 1.05 to 1.0, so the box becomes 0.15 wide/high and its
        # centre shifts to 0.925 — never normalizing past 1.0 against the image.
        self.assertTrue(torch.allclose(
            by_score[1, :4], torch.tensor([0.925, 0.925, 0.15, 0.15]), atol=1e-6
        ))
        self.assertTrue((rows[:, :4] <= 1.0 + 1e-6).all())


if __name__ == "__main__":
    unittest.main()
