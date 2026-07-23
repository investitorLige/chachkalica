from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

try:
    from ...formats import xyxy_prediction_to_friendy
except ImportError:
    from formats import xyxy_prediction_to_friendy


# torchvision's Faster R-CNN reserves label 0 for the background class, so real
# (foreground) classes must occupy labels 1..num_classes. The rest of the
# pipeline (datasets, class dicts, metrics) works in 0-indexed dataset ids, so
# the adapter shifts labels by this offset on the way in/out to hide the
# background slot from everything outside torchvision.
_BACKGROUND_CLASS_OFFSET = 1


@dataclass
class FasterRCNNAdapter:
    model: torch.nn.Module
    num_classes: int
    name: str = "fasterrcnn"

    def to(self, device):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def training_step(self, images, targets):
        self.model.train()
        return self._loss_forward(images, _shift_targets_to_model_labels(targets))

    def validation_step(self, images, targets):
        was_training = self.model.training
        self.model.train()
        _set_batch_norm_eval(self.model)
        try:
            return self._loss_forward(images, _shift_targets_to_model_labels(targets))
        finally:
            self.model.train(was_training)

    def _loss_forward(self, images, targets):
        losses = self.model(images, targets)
        return sum(loss for loss in losses.values()), losses

    @torch.no_grad()
    def predict(self, images, score_threshold: Optional[float] = None):
        self.model.eval()
        original_threshold = self.model.roi_heads.score_thresh
        if score_threshold is not None:
            self.model.roi_heads.score_thresh = float(score_threshold)
        try:
            predictions = self.model(images)
        finally:
            # Evaluation can request a low mAP floor without permanently
            # changing the model's configured serving threshold.
            self.model.roi_heads.score_thresh = original_threshold
        return [
            fasterrcnn_prediction_to_friendy(_shift_prediction_to_dataset_labels(prediction), image)
            for prediction, image in zip(predictions, images)
        ]


# variant -> ImageNet backbone family, so the right backbone weight enum is picked
# when the caller supplies `weights_backbone` instead of the full COCO `weights`.
_VARIANT_BACKBONES = {
    "resnet50_fpn": "resnet50",
    "resnet50_fpn_v2": "resnet50",
    "mobilenet_v3_large_fpn": "mobilenet_v3_large",
    "mobilenet_v3_large_320_fpn": "mobilenet_v3_large",
}


def build_fasterrcnn(
    num_classes: int,
    weights: Optional[str] = None,
    weights_backbone: Optional[str] = None,
    trainable_backbone_layers: Optional[int] = None,
    variant: str = "resnet50_fpn_v2",
    box_score_thresh: Optional[float] = None,
    box_nms_thresh: Optional[float] = None,
    box_detections_per_img: Optional[int] = None,
    rpn_pre_nms_top_n_test: Optional[int] = None,
    rpn_post_nms_top_n_test: Optional[int] = None,
    rpn_nms_thresh: Optional[float] = None,
    rpn_score_thresh: Optional[float] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNNAdapter:
    if variant not in _VARIANT_BACKBONES:
        raise ValueError(f"Unsupported Faster R-CNN variant: {variant}")

    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_mobilenet_v3_large_fpn,
        fasterrcnn_resnet50_fpn,
        fasterrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights

    builder_by_variant = {
        "resnet50_fpn": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights),
        "resnet50_fpn_v2": (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights),
        "mobilenet_v3_large_fpn": (
            fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        ),
        "mobilenet_v3_large_320_fpn": (
            fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        ),
    }
    backbone_weight_enum_by_family = {
        "resnet50": ResNet50_Weights,
        "mobilenet_v3_large": MobileNet_V3_Large_Weights,
    }

    # torchvision counts the background as one of num_classes, so add a slot for
    # it on top of the foreground classes the caller asked for.
    model_num_classes = num_classes + _BACKGROUND_CLASS_OFFSET

    builder, weight_enum = builder_by_variant[variant]
    backbone_weight_enum = backbone_weight_enum_by_family[_VARIANT_BACKBONES[variant]]

    model_weights = _resolve_weights(weight_enum, weights)
    backbone_weights = (
        None
        if model_weights is not None
        else _resolve_weights(backbone_weight_enum, weights_backbone)
    )

    # Only forward RoI-head / RPN knobs the caller actually set — torchvision's own
    # defaults (box_score_thresh=0.05, rpn_post_nms_top_n_test=1000, ...) already
    # apply when omitted; passing an explicit None would stomp them with None.
    head_kwargs = {
        "box_score_thresh": box_score_thresh,
        "box_nms_thresh": box_nms_thresh,
        "box_detections_per_img": box_detections_per_img,
        "rpn_pre_nms_top_n_test": rpn_pre_nms_top_n_test,
        "rpn_post_nms_top_n_test": rpn_post_nms_top_n_test,
        "rpn_nms_thresh": rpn_nms_thresh,
        "rpn_score_thresh": rpn_score_thresh,
        "min_size": min_size,
        "max_size": max_size,
    }
    head_kwargs = {k: v for k, v in head_kwargs.items() if v is not None}

    try:
        model = _build_fasterrcnn_model(
            builder=builder,
            model_weights=model_weights,
            backbone_weights=backbone_weights,
            model_num_classes=model_num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            predictor_factory=FastRCNNPredictor,
            builder_kwargs={**head_kwargs, **kwargs},
        )
    except Exception as exc:  # noqa: BLE001
        if model_weights is not None or backbone_weights is not None:
            raise RuntimeError(
                "Faster R-CNN pretrained weights were requested but could not be loaded; "
                "refusing to silently train from random initialization."
            ) from exc
        raise
    return FasterRCNNAdapter(model=model, num_classes=num_classes)


def _build_fasterrcnn_model(
    builder,
    model_weights,
    backbone_weights,
    model_num_classes: int,
    trainable_backbone_layers: Optional[int],
    predictor_factory,
    builder_kwargs: Dict[str, Any],
):
    if model_weights is not None:
        # Load the native COCO predictor first so torchvision can restore the
        # complete detector state dict, then replace only the task-specific box
        # predictor. The backbone, FPN, RPN, and RoI feature layers stay pretrained.
        model = builder(
            weights=model_weights,
            weights_backbone=None,
            trainable_backbone_layers=trainable_backbone_layers,
            **builder_kwargs,
        )
        old_predictor = model.roi_heads.box_predictor
        in_features = old_predictor.cls_score.in_features
        old_num_classes = old_predictor.cls_score.out_features
        model.roi_heads.box_predictor = predictor_factory(
            in_features,
            model_num_classes,
        )
        print(
            "[fasterrcnn] Loaded pretrained detector weights and reinitialized "
            f"the box predictor ({old_num_classes} -> {model_num_classes} classes)."
        )
        return model

    return builder(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=model_num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        **builder_kwargs,
    )


def fasterrcnn_prediction_to_friendy(
    prediction: Dict[str, torch.Tensor], image: torch.Tensor
) -> torch.Tensor:
    image_height, image_width = image.shape[-2:]
    return xyxy_prediction_to_friendy(
        prediction["boxes"],
        prediction["scores"],
        prediction["labels"],
        image_width=image_width,
        image_height=image_height,
    )


def _shift_targets_to_model_labels(targets):
    """Shift 0-indexed dataset labels up to torchvision's 1-indexed foreground labels."""
    shifted = []
    for target in targets:
        shifted_target = dict(target)
        shifted_target["labels"] = target["labels"] + _BACKGROUND_CLASS_OFFSET
        shifted.append(shifted_target)
    return shifted


def _shift_prediction_to_dataset_labels(prediction):
    """Shift torchvision's 1-indexed foreground labels back to 0-indexed dataset ids.

    torchvision's postprocessing already drops the background class, so predicted
    labels are always >= 1 and this never produces a negative id.
    """
    shifted = dict(prediction)
    shifted["labels"] = prediction["labels"] - _BACKGROUND_CLASS_OFFSET
    return shifted


def _resolve_weights(enum_cls, value):
    if value is None:
        return None

    if value is True:
        value = "DEFAULT"
    elif value is False:
        return None

    return enum_cls.verify(value)


def _set_batch_norm_eval(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()
