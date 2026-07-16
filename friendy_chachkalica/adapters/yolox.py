from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

try:
    from ..formats import clip_xyxy, xyxy_prediction_to_friendy, xyxy_to_xywh
except ImportError:
    from formats import clip_xyxy, xyxy_prediction_to_friendy, xyxy_to_xywh

try:
    from .retinanet import _set_batch_norm_eval
except ImportError:
    from adapters.retinanet import _set_batch_norm_eval


DEFAULT_YOLOX_VARIANT = "yolox-s"

# Official COCO-pretrained YOLOX checkpoints, keyed by variant. Passing
# ``weights=True`` fetches the one matching ``variant`` — the yolox counterpart
# to RF-DETR's default download and RT-DETR's ``weights=True``.
#
# LICENSE: every variant here is released by Megvii under Apache-2.0 (both the
# code and these weights) and is free for commercial, closed-source use. YOLOX
# has no restricted tier — unlike RF-DETR, whose `[plus]` XLarge/2XLarge weights
# are PML-1.0 (non-commercial) and are deliberately excluded there. So this whole
# map is safe to ship; do not add third-party YOLOX weights without checking
# their license first.
YOLOX_PRETRAINED_URLS = {
    "yolox-nano": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
    "yolox-tiny": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
    "yolox-s": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
    "yolox-m": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth",
    "yolox-l": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
    "yolox-x": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
}


@dataclass
class YOLOXAdapter:
    model: torch.nn.Module
    num_classes: int
    score_threshold: float = 0.3
    nms_threshold: float = 0.45
    name: str = "yolox"

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
        batch = self._prepare_batch(images)
        yolox_targets = self._prepare_targets(targets, images)
        outputs = self.model(batch, yolox_targets)
        losses = {k: v for k, v in outputs.items() if k != "total_loss"}
        return outputs["total_loss"], losses

    @torch.no_grad()
    def predict(
        self,
        images,
        score_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
    ):
        try:
            from ..vendor.yolox.utils import postprocess
        except ImportError:
            from vendor.yolox.utils import postprocess

        self.model.eval()
        batch = self._prepare_batch(images)
        outputs = self.model(batch)
        detections = postprocess(
            outputs,
            num_classes=self.num_classes,
            conf_thre=self.score_threshold if score_threshold is None else score_threshold,
            nms_thre=self.nms_threshold if nms_threshold is None else nms_threshold,
        )
        return [
            yolox_detection_to_friendy(detection, image)
            for detection, image in zip(detections, images)
        ]

    def _prepare_batch(self, images):
        device = next(self.model.parameters()).device
        prepared_images = [image.to(device).float() for image in images]
        max_height = _make_divisible(
            max(image.shape[-2] for image in prepared_images),
            32,
        )
        max_width = _make_divisible(
            max(image.shape[-1] for image in prepared_images),
            32,
        )

        padded_images = []
        for image in prepared_images:
            height, width = image.shape[-2:]
            padded_images.append(
                F.pad(
                    image,
                    (0, max_width - width, 0, max_height - height),
                    value=0.0,
                )
            )

        return torch.stack(padded_images)

    def _prepare_targets(self, targets, images):
        device = next(self.model.parameters()).device
        max_objects = max((len(target["labels"]) for target in targets), default=0)
        yolox_targets = torch.zeros(
            (len(targets), max_objects, 5),
            dtype=torch.float32,
            device=device,
        )

        for batch_index, target in enumerate(targets):
            labels = target["labels"].to(device).float()
            boxes = target["boxes"].to(device).float()
            if boxes.numel() == 0:
                continue

            xywh = xyxy_to_xywh(boxes)
            yolox_targets[batch_index, : len(labels), 0] = labels
            yolox_targets[batch_index, : len(labels), 1:5] = xywh

        return yolox_targets


def build_yolox(
    num_classes: int,
    weights=None,
    variant: str = DEFAULT_YOLOX_VARIANT,
    score_threshold: float = 0.3,
    nms_threshold: float = 0.45,
    **builder_options: Any,
) -> YOLOXAdapter:
    try:
        from ..vendor.yolox.models import build_yolox_model
    except ImportError:
        from vendor.yolox.models import build_yolox_model

    # weights: True -> the variant's default COCO-pretrained checkpoint (mirrors
    # rtdetr); False/None -> train from scratch; a str -> a URL or local path.
    if weights is True:
        try:
            weights = YOLOX_PRETRAINED_URLS[variant]
        except KeyError as exc:
            available = ", ".join(sorted(YOLOX_PRETRAINED_URLS))
            raise ValueError(
                f"No default pretrained weights for YOLOX variant {variant!r}. "
                f"Variants with published weights: {available}."
            ) from exc
    elif weights is False:
        weights = None

    model = build_yolox_model(
        num_classes=num_classes,
        variant=variant,
        **builder_options,
    )
    if weights:
        try:
            _load_checkpoint(model, weights)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"YOLOX pretrained weights {weights!r} were requested but could "
                "not be loaded; refusing to silently train from random initialization."
            ) from exc

    return YOLOXAdapter(
        model=model,
        num_classes=num_classes,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )


def yolox_detection_to_friendy(
    detection: Optional[torch.Tensor],
    image: torch.Tensor,
) -> torch.Tensor:
    if detection is None or detection.numel() == 0:
        return image.new_zeros((0, 6))

    image_height, image_width = image.shape[-2:]
    boxes = clip_xyxy(
        detection[:, 0:4],
        image_width=image_width,
        image_height=image_height,
    )
    confidence = detection[:, 4] * detection[:, 5]

    return xyxy_prediction_to_friendy(
        boxes,
        confidence,
        detection[:, 6],
        image_width=image_width,
        image_height=image_height,
    )


def _make_divisible(value: int, divisor: int) -> int:
    return int((value + divisor - 1) // divisor * divisor)


def _load_checkpoint(model: torch.nn.Module, checkpoint_ref: str) -> None:
    """Load pretrained YOLOX weights from a URL or local path.

    Tolerates a class-count mismatch: tensors whose shape differs from the model
    (e.g. the ``head.cls_preds`` classification layers when fine-tuning an
    80-class COCO checkpoint onto a different class count) are skipped and left
    at their fresh init, exactly like RF-DETR / RT-DETR's head re-initialization.
    Backbone, neck, and the class-agnostic box/objectness heads still load.
    """
    if isinstance(checkpoint_ref, str) and checkpoint_ref.startswith(("http://", "https://")):
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_ref, map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_ref, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raise ValueError(
            "This is a Friendy training checkpoint. Select it as 'Your model' "
            "so the checked warm-start loader is used."
        )
    state_dict = (
        checkpoint.get("model", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise ValueError(
            "YOLOX pretrained checkpoint does not contain a state dict")

    model_state = model.state_dict()
    compatible = {
        key: tensor for key, tensor in state_dict.items()
        if key in model_state and torch.is_tensor(tensor) and tensor.shape == model_state[key].shape
    }
    reinit = [key for key in state_dict if key not in compatible]
    if not compatible:
        raise ValueError(
            "YOLOX pretrained checkpoint has no tensors compatible with this model"
        )
    model.load_state_dict(compatible, strict=False)
    if reinit:
        print(
            f"[yolox] Loaded {len(compatible)}/{len(state_dict)} pretrained tensor(s); "
            f"re-initialized {len(reinit)} with a mismatched shape "
            f"(the detection head for a new class count): {reinit[:6]}"
            + (" …" if len(reinit) > 6 else "")
        )


