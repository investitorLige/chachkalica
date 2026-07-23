import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

try:
    from ...formats import xyxy_prediction_to_friendy
except ImportError:
    from formats import xyxy_prediction_to_friendy


@dataclass
class RetinaNetAdapter:
    model: torch.nn.Module
    num_classes: int
    name: str = "retinanet"

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
        return self._loss_forward(images, targets)

    def validation_step(self, images, targets):
        was_training = self.model.training
        self.model.train()
        _set_batch_norm_eval(self.model)
        try:
            return self._loss_forward(images, targets)
        finally:
            self.model.train(was_training)

    def _loss_forward(self, images, targets):
        losses = self.model(images, targets)
        return sum(loss for loss in losses.values()), losses

    @torch.no_grad()
    def predict(self, images, score_threshold: Optional[float] = None):
        self.model.eval()
        original_threshold = self.model.score_thresh
        if score_threshold is not None:
            self.model.score_thresh = float(score_threshold)
        try:
            predictions = self.model(images)
        finally:
            # Evaluation can request a low mAP floor without permanently
            # changing the model's configured serving threshold.
            self.model.score_thresh = original_threshold
        return [
            retinanet_prediction_to_friendy(prediction, image)
            for prediction, image in zip(predictions, images)
        ]


def build_retinanet(
    num_classes: int,
    weights: Optional[str] = None,
    weights_backbone: Optional[str] = None,
    trainable_backbone_layers: Optional[int] = None,
    variant: str = "resnet50_fpn_v2",
    **kwargs: Any,
) -> RetinaNetAdapter:
    if variant not in {"resnet50_fpn", "resnet50_fpn_v2"}:
        raise ValueError(f"Unsupported RetinaNet variant: {variant}")

    from torchvision.models.detection import (
        RetinaNet_ResNet50_FPN_V2_Weights,
        RetinaNet_ResNet50_FPN_Weights,
        retinanet_resnet50_fpn,
        retinanet_resnet50_fpn_v2,
    )
    from torchvision.models import ResNet50_Weights

    if variant == "resnet50_fpn_v2":
        builder = retinanet_resnet50_fpn_v2
        weight_enum = RetinaNet_ResNet50_FPN_V2_Weights
    else:
        builder = retinanet_resnet50_fpn
        weight_enum = RetinaNet_ResNet50_FPN_Weights

    model_weights = _resolve_weights(weight_enum, weights)
    backbone_weights = (
        None
        if model_weights is not None
        else _resolve_weights(ResNet50_Weights, weights_backbone)
    )

    try:
        model = _build_retinanet_model(
            builder=builder,
            model_weights=model_weights,
            backbone_weights=backbone_weights,
            model_num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            builder_kwargs=kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        if model_weights is not None or backbone_weights is not None:
            raise RuntimeError(
                "RetinaNet pretrained weights were requested but could not be loaded; "
                "refusing to silently train from random initialization."
            ) from exc
        raise
    return RetinaNetAdapter(model=model, num_classes=num_classes)


def _build_retinanet_model(
    builder,
    model_weights,
    backbone_weights,
    model_num_classes: int,
    trainable_backbone_layers: Optional[int],
    builder_kwargs: Dict[str, Any],
):
    if model_weights is not None:
        # torchvision requires the pretrained detector's native COCO class count
        # while loading its state dict. Load it first, then replace only the final
        # classification convolution so the backbone, FPN, regression head, and
        # classification feature tower retain their pretrained parameters.
        model = builder(
            weights=model_weights,
            weights_backbone=None,
            trainable_backbone_layers=trainable_backbone_layers,
            **builder_kwargs,
        )
        old_num_classes = model.head.classification_head.num_classes
        _replace_retinanet_classifier(model, model_num_classes)
        print(
            "[retinanet] Loaded pretrained detector weights and reinitialized "
            f"the classifier ({old_num_classes} -> {model_num_classes} classes)."
        )
        return model

    return builder(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=model_num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        **builder_kwargs,
    )


def _replace_retinanet_classifier(model, num_classes: int) -> None:
    classification_head = model.head.classification_head
    old_logits = classification_head.cls_logits
    num_anchors = classification_head.num_anchors

    new_logits = torch.nn.Conv2d(
        in_channels=old_logits.in_channels,
        out_channels=num_anchors * num_classes,
        kernel_size=old_logits.kernel_size,
        stride=old_logits.stride,
        padding=old_logits.padding,
        dilation=old_logits.dilation,
        groups=old_logits.groups,
        bias=old_logits.bias is not None,
        padding_mode=old_logits.padding_mode,
    ).to(device=old_logits.weight.device, dtype=old_logits.weight.dtype)

    torch.nn.init.normal_(new_logits.weight, std=0.01)
    if new_logits.bias is not None:
        prior_probability = 0.01
        torch.nn.init.constant_(
            new_logits.bias,
            -math.log((1 - prior_probability) / prior_probability),
        )

    classification_head.cls_logits = new_logits
    classification_head.num_classes = num_classes


def retinanet_prediction_to_friendy(
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
