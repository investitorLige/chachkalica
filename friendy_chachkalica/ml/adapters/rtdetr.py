from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

try:
    from ...formats import xyxy_prediction_to_friendy, xyxy_to_xywhn
except ImportError:
    from formats import xyxy_prediction_to_friendy, xyxy_to_xywhn


DEFAULT_RTDETR_WEIGHTS = "PekingU/rtdetr_r50vd"


@dataclass
class RTDETRAdapter:
    model: torch.nn.Module
    image_processor: Any
    num_classes: int
    score_threshold: float = 0.5
    # NOT applied in predict() — DETRs are set-based and run NMS-free. The
    # trainer reads this as the IoU for the operating-point val/test metrics
    # (precision/recall/F1/confusion); see train.resolve_operating_nms_threshold.
    nms_threshold: Optional[float] = None
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)
    input_max_size: Optional[int] = 640
    input_size_multiple: int = 32
    name: str = "rtdetr"
    # HF RT-DETR overflows to NaN under fp16 autocast (decoder + GIoU loss),
    # so the trainer must always run this adapter in full precision.
    supports_amp: bool = False

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
        images, targets = self._resize_training_inputs(images, targets)
        batch = self._prepare_batch(images)
        labels = self._prepare_labels(targets, images)
        outputs = self.model(**batch, labels=labels)
        losses = (
            dict(outputs.loss_dict)
            if outputs.loss_dict is not None
            else {"loss": outputs.loss}
        )
        return outputs.loss, losses

    def validation_step(self, images, targets):
        was_training = self.model.training
        self.model.eval()
        try:
            images, targets = self._resize_training_inputs(images, targets)
            batch = self._prepare_batch(images)
            labels = self._prepare_labels(targets, images)
            outputs = self.model(**batch, labels=labels)
            losses = (
                dict(outputs.loss_dict)
                if outputs.loss_dict is not None
                else {"loss": outputs.loss}
            )
            return outputs.loss, losses
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def predict(self, images, score_threshold: Optional[float] = None):
        self.model.eval()
        batch = self._prepare_batch([self._resize_image(image) for image in images])
        outputs = self.model(**batch)
        target_sizes = torch.tensor(
            [[image.shape[-2], image.shape[-1]] for image in images],
            dtype=torch.long,
            device=batch["pixel_values"].device,
        )
        predictions = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.score_threshold if score_threshold is None else score_threshold,
            target_sizes=target_sizes,
            use_focal_loss=getattr(self.model.config, "use_focal_loss", True),
        )
        return [
            rtdetr_prediction_to_friendy(prediction, image)
            for prediction, image in zip(predictions, images)
        ]

    def _prepare_batch(self, images):
        device = next(self.model.parameters()).device
        image_mean = torch.tensor(self.image_mean, device=device).view(3, 1, 1)
        image_std = torch.tensor(self.image_std, device=device).view(3, 1, 1)

        prepared_images = [
            ((image.to(device).float() - image_mean) / image_std)
            for image in images
        ]
        canvas_size = self._fixed_canvas_size()
        if canvas_size is not None:
            max_height = max_width = canvas_size
        else:
            max_height = max(image.shape[-2] for image in prepared_images)
            max_width = max(image.shape[-1] for image in prepared_images)
            max_height = _ceil_to_multiple(max_height, self.input_size_multiple)
            max_width = _ceil_to_multiple(max_width, self.input_size_multiple)

        pixel_values = []
        pixel_masks = []
        for image in prepared_images:
            height, width = image.shape[-2:]
            pixel_values.append(
                F.pad(image, (0, max_width - width, 0, max_height - height))
            )

            mask = torch.zeros((max_height, max_width), dtype=torch.long, device=device)
            mask[:height, :width] = 1
            pixel_masks.append(mask)

        return {
            "pixel_values": torch.stack(pixel_values),
            "pixel_mask": torch.stack(pixel_masks),
        }

    def _fixed_canvas_size(self) -> Optional[int]:
        """Square token-grid side every batch pads/letterboxes to, or ``None``
        to fall back to the old batch-derived size (only when resizing is
        disabled via ``input_max_size``).

        HF's RT-DETR never applies ``pixel_mask`` to encoder attention or to
        the two-stage anchor/topk selection (confirmed against
        ``transformers.models.rt_detr.modeling_rt_detr``: the mask is
        downsampled in the backbone and then discarded, and the anchor
        ``valid_mask`` used at the topk step is purely geometric) — padding is
        scored identically to real content. Deriving the canvas from the
        actual images in a batch therefore let the real-content-vs-padding
        token ratio (and even the raw token count relative to
        ``num_queries``) swing with whatever crops happened to land in the
        same micro-batch — most visible with the person crops
        ``people_detect_first`` produces, whose aspect ratios vary far more
        than tiles/full frames. A canvas fixed to ``input_max_size`` makes the
        token count, and so the topk selection's real-vs-padding odds, depend
        only on that setting, never on batch composition.
        """
        if self.input_max_size is None or self.input_max_size <= 0:
            return None
        return _ceil_to_multiple(self.input_max_size, self.input_size_multiple)

    def _resize_training_inputs(self, images, targets):
        resized_images = []
        resized_targets = []
        for image, target in zip(images, targets):
            resized_image, scale_y, scale_x = self._resize_image_with_scale(image)
            resized_target = dict(target)
            if scale_y != 1.0 or scale_x != 1.0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                resized_target["boxes"] = boxes
            resized_images.append(resized_image)
            resized_targets.append(resized_target)
        return resized_images, resized_targets

    def _resize_image(self, image):
        resized_image, _, _ = self._resize_image_with_scale(image)
        return resized_image

    def _resize_image_with_scale(self, image):
        """Aspect-preserving scale to land the longest side on ``input_max_size``.

        Always scales — up as well as down. A crop smaller than
        ``input_max_size`` (e.g. a person_detect_first crop grown only to
        ``detector_min_box_size``) is bilinearly upscaled to fill more of the
        fixed canvas ``_fixed_canvas_size`` pads to, instead of being left at
        native resolution and mostly zero-padded: the backbone's fixed stride
        (32px of input per token) means a small subject spans more tokens
        after upscaling, giving the encoder/decoder actual spatial resolution
        to separate it (and anything on it) from its neighbors, at the cost of
        interpolation blur rather than any new captured detail.
        """
        if self.input_max_size is None or self.input_max_size <= 0:
            return image, 1.0, 1.0

        height, width = image.shape[-2:]
        longest_side = max(height, width)
        if longest_side == self.input_max_size:
            return image, 1.0, 1.0

        scale = self.input_max_size / float(longest_side)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return resized, resized_height / float(height), resized_width / float(width)

    def _prepare_labels(self, targets, images):
        device = next(self.model.parameters()).device
        labels = []
        for target, image in zip(targets, images):
            image_height, image_width = image.shape[-2:]
            boxes = target["boxes"].to(device).float()
            labels.append(
                {
                    "class_labels": target["labels"].to(device).long(),
                    "boxes": xyxy_to_xywhn(
                        boxes,
                        image_width=image_width,
                        image_height=image_height,
                    ),
                }
            )
        return labels


def build_rtdetr(
    num_classes: int,
    weights: Optional[str] = None,
    score_threshold: float = 0.5,
    nms_threshold: Optional[float] = None,
    image_mean: tuple = (0.485, 0.456, 0.406),
    image_std: tuple = (0.229, 0.224, 0.225),
    input_max_size: Optional[int] = 640,
    input_size_multiple: int = 32,
    ignore_mismatched_sizes: bool = True,
    trainable_backbone_layers: Optional[int] = None,
    **config_kwargs: Any,
) -> RTDETRAdapter:
    # RT-DETRv2 ships as its own model class (RTDetrV2ForObjectDetection): the
    # v2 checkpoints (PekingU/rtdetr_v2_*) won't load through the v1 class. We
    # route by repo id so a v2 selection in the weights dropdown "just works",
    # while the image processor is shared between the two.
    if _is_rtdetr_v2(weights):
        (
            RTDetrConfig,
            RTDetrForObjectDetection,
            RTDetrImageProcessor,
        ) = _load_transformers_rtdetr_v2()
    else:
        (
            RTDetrConfig,
            RTDetrForObjectDetection,
            RTDetrImageProcessor,
        ) = _load_transformers_rtdetr()

    input_max_size = config_kwargs.pop("input_max_size", input_max_size)
    input_size_multiple = config_kwargs.pop("input_size_multiple", input_size_multiple)

    id2label = config_kwargs.pop(
        "id2label",
        {class_id: str(class_id) for class_id in range(num_classes)},
    )
    label2id = config_kwargs.pop(
        "label2id",
        {class_name: class_id for class_id, class_name in id2label.items()},
    )

    if weights is True:
        weights = DEFAULT_RTDETR_WEIGHTS
    elif weights is False:
        weights = None

    def _from_scratch():
        config = RTDetrConfig(
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            **config_kwargs,
        )
        return RTDetrForObjectDetection(config)

    if weights is None:
        model = _from_scratch()
    else:
        try:
            config = RTDetrConfig.from_pretrained(weights, **config_kwargs)
            config.id2label = id2label
            config.label2id = label2id
            config.num_labels = num_classes
            model = RTDetrForObjectDetection.from_pretrained(
                weights,
                config=config,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"RT-DETR pretrained weights {weights!r} were requested but could "
                "not be loaded; refusing to silently train from random initialization."
            ) from exc

    if trainable_backbone_layers is not None:
        _freeze_rtdetr_backbone(model, trainable_backbone_layers)

    return RTDETRAdapter(
        model=model,
        image_processor=RTDetrImageProcessor(),
        num_classes=num_classes,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        image_mean=image_mean,
        image_std=image_std,
        input_max_size=input_max_size,
        input_size_multiple=input_size_multiple,
    )


def _freeze_rtdetr_backbone(model, trainable_backbone_layers: int) -> None:
    """Freeze the ResNet backbone's stem/stages, torchvision-style.

    ``trainable_backbone_layers`` follows torchvision's ``trainable_backbone_layers``
    convention exactly: 0 freezes the whole backbone (stem + all 4 stages), 5 leaves
    everything trainable. The backbone's BatchNorm is already frozen structurally by
    HF (``RTDetrFrozenBatchNorm2d``, ``freeze_backbone_batch_norms=True`` default), so
    no BatchNorm eval-mode bookkeeping is needed here.
    """
    trainable_backbone_layers = max(0, min(5, trainable_backbone_layers))
    resnet = model.model.backbone.model
    ordered = list(reversed(list(resnet.encoder.stages))) + [resnet.embedder]
    for module in ordered[trainable_backbone_layers:]:
        module.requires_grad_(False)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def rtdetr_prediction_to_friendy(
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


def _is_rtdetr_v2(weights) -> bool:
    """True when ``weights`` names an RT-DETRv2 checkpoint (repo id or path).

    v2 checkpoints must be built with ``RTDetrV2ForObjectDetection``; the v1
    class silently fails to load them. Matches both the HuggingFace repo-id
    spelling (``PekingU/rtdetr_v2_r50vd``) and the class-name spelling.
    """
    if not isinstance(weights, str):
        return False
    normalized = weights.lower()
    return "rtdetr_v2" in normalized or "rtdetrv2" in normalized


def _load_transformers_rtdetr():
    try:
        from transformers import (
            RTDetrConfig,
            RTDetrForObjectDetection,
            RTDetrImageProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "RT-DETR requires optional dependencies. "
            "Install it with `pip install -r requirements-rtdetr.txt`."
        ) from exc

    return RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor


def _load_transformers_rtdetr_v2():
    """RT-DETRv2 config/model classes (+ the shared v1 image processor).

    RT-DETRv2 landed in a later transformers release than v1, so a clear error
    beats an AttributeError if the installed transformers is too old.
    """
    try:
        from transformers import (
            RTDetrV2Config,
            RTDetrV2ForObjectDetection,
            RTDetrImageProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "RT-DETRv2 weights need a transformers new enough to ship "
            "RTDetrV2ForObjectDetection. Upgrade with `pip install -U transformers` "
            "(see requirements-rtdetr.txt)."
        ) from exc

    return RTDetrV2Config, RTDetrV2ForObjectDetection, RTDetrImageProcessor
