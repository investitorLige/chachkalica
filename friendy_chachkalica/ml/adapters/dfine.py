from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

try:
    from ...formats import clip_xyxy, xyxy_prediction_to_friendy, xyxy_to_xywhn
except ImportError:
    from formats import clip_xyxy, xyxy_prediction_to_friendy, xyxy_to_xywhn


# D-FINE (Peterande/D-FINE, Apache-2.0) is a fork of RT-DETRv2's codebase that
# replaces plain box regression with Fine-grained Distribution Refinement (FDR)
# and adds Global Optimal Localization Self-Distillation (GO-LSD) at train time.
# Unlike ECDet — which borrows D-FINE's decoder design but ships no installable
# package, hence vendored under vendor/edgecrafter — real D-FINE landed as a
# first-class architecture in transformers (DFineConfig/DFineForObjectDetection,
# with a full FDR+GO-LSD loss in transformers.loss.loss_d_fine), the same way
# RT-DETR did. So this adapter follows adapters/rtdetr.py's shape (an installed
# dependency, not a vendored subtree): weights IS the size selector (each
# ustc-community/dfine-{nano,small,medium,large,xlarge}-{coco,obj365,obj2coco}
# repo id carries its own backbone_config), there is no separate `variant` kwarg.
DEFAULT_DFINE_WEIGHTS = "ustc-community/dfine-medium-coco"


@dataclass
class DFineAdapter:
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
    name: str = "dfine"
    # Not yet measured on trained weights (unlike ecdet's UNTRUSTED_FP16 verdict,
    # which was — see trt_export/arch/__init__.py). Defaulting to False rather
    # than assuming it inherits RT-DETR's: HF RT-DETR overflows to NaN under fp16
    # autocast (decoder self-attn + GIoU loss), and D-FINE shares that same
    # deformable-decoder/GIoU lineage plus its own FDR head layers on top
    # (softmax over discretized bins, a KL-divergence GO-LSD term) — strictly
    # more fp16-fragile surface, not less. Revisit once measured; this is a
    # training-time AMP flag, a separate question from TRT export fp16 trust.
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
        labels = self._prepare_labels(targets, batch["pixel_values"].shape[-2:])
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
            labels = self._prepare_labels(targets, batch["pixel_values"].shape[-2:])
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
        resized_images = []
        scales = []
        for image in images:
            resized_image, scale_y, scale_x = self._resize_image_with_scale(image)
            resized_images.append(resized_image)
            scales.append((scale_y, scale_x))
        batch = self._prepare_batch(resized_images)
        outputs = self.model(**batch)
        # post_process_object_detection multiplies the normalized boxes by
        # target_sizes with no pad offset, so hand it the model input's own
        # HxW to get model-input pixels, then undo the per-axis resize to land
        # back in original-image pixels — same contract as RTDETRAdapter.predict,
        # and load-bearing for the same reason (box_coords: "input_normalized").
        input_height, input_width = batch["pixel_values"].shape[-2:]
        target_sizes = torch.tensor(
            [[input_height, input_width]] * len(images),
            dtype=torch.long,
            device=batch["pixel_values"].device,
        )
        predictions = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.score_threshold if score_threshold is None else score_threshold,
            target_sizes=target_sizes,
            use_focal_loss=getattr(self.model.config, "use_focal_loss", True),
        )
        results = []
        for prediction, image, (scale_y, scale_x) in zip(predictions, images, scales):
            boxes = prediction["boxes"].clone()
            boxes[:, [0, 2]] /= scale_x
            boxes[:, [1, 3]] /= scale_y
            image_height, image_width = image.shape[-2:]
            # Clip to bounds, matching RTDETRAdapter.predict — a box centred near
            # an edge decodes past the frame on this stretched, unpadded canvas.
            boxes = clip_xyxy(boxes, image_width=image_width, image_height=image_height)
            results.append(
                xyxy_prediction_to_friendy(
                    boxes,
                    prediction["scores"],
                    prediction["labels"],
                    image_width=image_width,
                    image_height=image_height,
                )
            )
        return results

    def _prepare_batch(self, images):
        """Normalize and batch already-resized images.

        With resizing enabled every image arrives at exactly the square canvas,
        so the pad below is a no-op and the model input is all real content. It
        only does work on the resize-disabled path (``input_max_size=None``),
        where the batch's own max HxW is rounded up to ``input_size_multiple``.
        """
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
        """Square input side every image is resized to, or ``None`` to fall back
        to the batch-derived padded size (only when resizing is disabled via
        ``input_max_size``).

        Confirmed against ``transformers.models.d_fine.modeling_d_fine``:
        ``DFineConvEncoder.forward`` downsamples ``pixel_mask`` per feature map
        and returns it alongside each feature map, but ``DFineModel.forward``
        immediately unpacks ``(source, mask)`` and discards ``mask`` when
        building ``proj_feats`` — never reaches the encoder or the decoder's
        two-stage anchor/topk selection. That selection's own ``valid_mask`` is
        purely geometric (anchor position vs. image bounds), same as RT-DETR.
        So padding is scored identically to real content — stretch to fill this
        canvas exactly (see ``_resize_image_with_scale``) rather than
        aspect-preserve and pad, exactly as RTDETRAdapter does and for the same
        reason.
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

    def _resize_image_with_scale(self, image):
        """Stretch the image onto the square ``_fixed_canvas_size`` canvas.

        Per-axis scaling, aspect ratio *not* preserved — upstream D-FINE's own
        preprocessing (inherited from RT-DETR: a plain ``Resize((640, 640))``),
        which keeps this adapter's geometry self-consistent the same way
        RTDETRAdapter's does (see that docstring for why letterboxing silently
        breaks training here).
        """
        canvas_size = self._fixed_canvas_size()
        if canvas_size is None:
            return image, 1.0, 1.0

        height, width = image.shape[-2:]
        if height == canvas_size and width == canvas_size:
            return image, 1.0, 1.0

        resized = F.interpolate(
            image.unsqueeze(0),
            size=(canvas_size, canvas_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return resized, canvas_size / float(height), canvas_size / float(width)

    def _prepare_labels(self, targets, input_size):
        """Normalize resized-pixel target boxes over the *model input* size.

        ``input_size`` is the batched ``pixel_values`` HxW, padding included —
        not the image's own extent. See RTDETRAdapter._prepare_labels for why.
        """
        device = next(self.model.parameters()).device
        image_height, image_width = int(input_size[0]), int(input_size[1])
        labels = []
        for target in targets:
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


def build_dfine(
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
) -> DFineAdapter:
    DFineConfig, DFineForObjectDetection, RTDetrImageProcessor = _load_transformers_dfine()

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
        weights = DEFAULT_DFINE_WEIGHTS
    elif weights is False:
        weights = None

    def _from_scratch():
        config = DFineConfig(
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            **config_kwargs,
        )
        return DFineForObjectDetection(config)

    if weights is None:
        model = _from_scratch()
    else:
        try:
            config = DFineConfig.from_pretrained(weights, **config_kwargs)
            config.id2label = id2label
            config.label2id = label2id
            config.num_labels = num_classes
            model = DFineForObjectDetection.from_pretrained(
                weights,
                config=config,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"D-FINE pretrained weights {weights!r} were requested but could "
                "not be loaded; refusing to silently train from random initialization."
            ) from exc

    if trainable_backbone_layers is not None:
        _freeze_dfine_backbone(model, trainable_backbone_layers)

    return DFineAdapter(
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


def _freeze_dfine_backbone(model, trainable_backbone_layers: int) -> None:
    """Freeze the HGNetV2 backbone's stem/stages, torchvision-style.

    ``trainable_backbone_layers`` follows torchvision's convention exactly: 0
    freezes the whole backbone (embedder + all 4 stages), 5 leaves everything
    trainable. Structurally identical to ``_freeze_rtdetr_backbone`` — D-FINE's
    HF backbone wrapper (``DFineConvEncoder``) has the same
    ``.model.{embedder,encoder.stages}`` shape as RT-DETR's ResNet one, just
    wrapping an ``HGNetV2Backbone`` instead. BatchNorm is already frozen
    structurally by HF (``freeze_backbone_batch_norms=True`` default), so no
    BatchNorm eval-mode bookkeeping is needed here.
    """
    trainable_backbone_layers = max(0, min(5, trainable_backbone_layers))
    hgnet = model.model.backbone.model
    ordered = list(reversed(list(hgnet.encoder.stages))) + [hgnet.embedder]
    for module in ordered[trainable_backbone_layers:]:
        module.requires_grad_(False)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _load_transformers_dfine():
    """D-FINE config/model classes + RT-DETR's shared image processor.

    D-FINE has no dedicated image processor in transformers (confirmed: no
    ``DFineImageProcessor``/``DFineFastImageProcessor`` class, and ``AutoImageProcessor``
    has no ``d_fine`` entry in its mapping) — it reuses ``RTDetrImageProcessor``,
    which is fine here since this adapter, like ``adapters/rtdetr.py``, only calls
    ``post_process_object_detection`` on it (a decode utility keyed on the output
    shape, not the encoder network) and does its own preprocessing by hand.
    """
    try:
        from transformers import (
            DFineConfig,
            DFineForObjectDetection,
            RTDetrImageProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "D-FINE requires optional dependencies. "
            "Install it with `pip install -r requirements-dfine.txt`."
        ) from exc

    return DFineConfig, DFineForObjectDetection, RTDetrImageProcessor
