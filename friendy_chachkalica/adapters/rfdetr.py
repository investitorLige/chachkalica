from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

try:
    from ..formats import xyxy_prediction_to_friendy, xyxy_to_xywhn
except ImportError:
    from formats import xyxy_prediction_to_friendy, xyxy_to_xywhn


# RF-DETR (Roboflow) detection variants that ship in the Apache-2.0 `rfdetr` package.
# These are free for commercial, closed-source use. Maps a short variant name to the
# corresponding constructor class exported by `rfdetr`.
APACHE_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}

# These variants live in the separate `rfdetr[plus]` package under the PML 1.0 license
# and are NOT free for commercial use. We refuse to build them so a commercial pipeline
# can never silently depend on a non-Apache checkpoint.
NON_FREE_VARIANTS = {"xlarge", "2xlarge"}

DEFAULT_RFDETR_VARIANT = "base"


@dataclass
class RFDETRAdapter:
    """Friendy Chachkalica adapter around RF-DETR's underlying LW-DETR network.

    ``model`` is the raw ``nn.Module`` so the shared training loop owns the optimizer,
    AMP, and checkpoint exactly as it does for the other adapters. The RF-DETR loss
    ``criterion`` and ``postprocess`` head are carried alongside it.
    """

    model: torch.nn.Module
    criterion: Any
    postprocess: Any
    num_classes: int
    resolution: int = 560
    score_threshold: float = 0.5
    # NOT applied in predict() — DETRs are set-based and run NMS-free. The
    # trainer reads this as the IoU for the operating-point val/test metrics
    # (precision/recall/F1/confusion); see train.resolve_operating_nms_threshold.
    nms_threshold: Optional[float] = None
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)
    name: str = "rfdetr"

    def to(self, device):
        self.model.to(device)
        self.criterion.to(device)
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
        self.model.eval()
        try:
            return self._loss_forward(images, targets)
        finally:
            self.model.train(was_training)

    def _loss_forward(self, images, targets):
        batch = self._prepare_batch(images)
        labels = self._prepare_labels(targets, images)
        outputs = self.model(batch, labels)
        loss_dict = self.criterion(outputs, labels)
        weight_dict = self.criterion.weight_dict
        loss = sum(
            loss_dict[key] * weight_dict[key]
            for key in loss_dict
            if key in weight_dict
        )
        return loss, loss_dict

    @torch.no_grad()
    def predict(self, images, score_threshold: Optional[float] = None):
        self.model.eval()
        threshold = self.score_threshold if score_threshold is None else score_threshold
        batch = self._prepare_batch(images)
        outputs = self.model(batch)
        # PostProcess scales normalized boxes back to each image's original pixel size.
        target_sizes = torch.tensor(
            [[image.shape[-2], image.shape[-1]] for image in images],
            dtype=torch.long,
            device=batch.device,
        )
        results = self.postprocess(outputs, target_sizes)
        predictions = []
        for result, image in zip(results, images):
            # RF-DETR's head has num_classes + 1 slots; the extra last slot is the
            # no-object/background class. Real classes are 0..num_classes-1, so drop
            # any background prediction along with sub-threshold ones.
            keep = (result["scores"] >= threshold) & (result["labels"] < self.num_classes)
            boxes = result["boxes"][keep]
            scores = result["scores"][keep]
            labels = result["labels"][keep]
            image_height, image_width = image.shape[-2:]
            predictions.append(
                xyxy_prediction_to_friendy(
                    boxes,
                    scores,
                    labels,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
        return predictions

    def _prepare_batch(self, images: List[torch.Tensor]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        image_mean = torch.tensor(self.image_mean, device=device).view(3, 1, 1)
        image_std = torch.tensor(self.image_std, device=device).view(3, 1, 1)

        prepared = []
        for image in images:
            # RF-DETR runs at a fixed square resolution. A square resize changes the
            # aspect ratio, but the targets are stored as normalized cxcywh relative to
            # each axis, so they stay correct without any box adjustment.
            resized = F.interpolate(
                image.to(device).float().unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            prepared.append((resized - image_mean) / image_std)
        return torch.stack(prepared)

    def _prepare_labels(self, targets, images) -> List[Dict[str, torch.Tensor]]:
        device = next(self.model.parameters()).device
        labels = []
        for target, image in zip(targets, images):
            image_height, image_width = image.shape[-2:]
            boxes = target["boxes"].to(device).float()
            labels.append(
                {
                    "labels": target["labels"].to(device).long(),
                    "boxes": xyxy_to_xywhn(
                        boxes,
                        image_width=image_width,
                        image_height=image_height,
                    ),
                }
            )
        return labels


def build_rfdetr(
    num_classes: int,
    variant: str = DEFAULT_RFDETR_VARIANT,
    weights: Any = True,
    score_threshold: float = 0.5,
    nms_threshold: Optional[float] = None,
    resolution: Optional[int] = None,
    **config_kwargs: Any,
) -> RFDETRAdapter:
    """Build an RF-DETR adapter.

    Args:
        num_classes: Number of foreground classes for the detection head.
        variant: One of ``nano``, ``small``, ``medium``, ``base``, ``large`` (all
            Apache-2.0). ``xlarge``/``2xlarge`` are rejected — they are part of
            ``rfdetr[plus]`` (PML 1.0) and are not free for commercial use.
        weights: ``True`` (default) loads the variant's published COCO-pretrained
            Apache weights for fine-tuning; a string is treated as an explicit
            checkpoint path; ``False``/``None`` trains from scratch.
        score_threshold: Default confidence cutoff used by :meth:`RFDETRAdapter.predict`.
        nms_threshold: IoU the trainer uses to NMS this model's predictions for
            the operating-point val/test metrics only. Never applied inside
            ``predict`` (DETR inference stays NMS-free) and never affects mAP.
        resolution: Optional square input resolution override; defaults to the
            variant's native resolution.
        **config_kwargs: Extra RF-DETR ModelConfig kwargs.
    """
    variant_key = str(variant).strip().lower()
    if variant_key in NON_FREE_VARIANTS:
        raise ValueError(
            f"RF-DETR variant '{variant}' belongs to rfdetr[plus] (PML 1.0 license) and is "
            f"NOT free for commercial use. Pick an Apache-2.0 variant: {sorted(APACHE_VARIANTS)}."
        )
    if variant_key not in APACHE_VARIANTS:
        raise ValueError(
            f"Unknown RF-DETR variant '{variant}'. Available: {sorted(APACHE_VARIANTS)}."
        )

    rfdetr_module, train_config_cls, build_criterion_from_config = _load_rfdetr()
    variant_cls = getattr(rfdetr_module, APACHE_VARIANTS[variant_key])

    constructor_kwargs = dict(config_kwargs)
    constructor_kwargs["num_classes"] = num_classes
    if resolution is not None:
        constructor_kwargs["resolution"] = resolution
    if weights is False or weights is None:
        # Explicit None tells RF-DETR to skip pretrained weights and train from scratch.
        constructor_kwargs["pretrain_weights"] = None
    elif isinstance(weights, str):
        constructor_kwargs["pretrain_weights"] = weights
    # weights is True -> leave pretrain_weights unset so the variant's published default applies.

    try:
        wrapper = variant_cls(**constructor_kwargs)
    except Exception as exc:  # noqa: BLE001
        if constructor_kwargs.get("pretrain_weights", "unset") is not None:
            raise RuntimeError(
                "RF-DETR pretrained weights were requested but could not be loaded; "
                "refusing to silently train from random initialization."
            ) from exc
        raise
    model_config = wrapper.model_config
    network = wrapper.model.model  # the underlying LW-DETR nn.Module

    # A minimal TrainConfig is enough: the criterion/postprocess builder only reads loss
    # coefficients and architectural fields, not the dataset paths.
    criterion, postprocess = build_criterion_from_config(
        model_config,
        train_config_cls(dataset_dir=".", output_dir="."),
    )

    return RFDETRAdapter(
        model=network,
        criterion=criterion,
        postprocess=postprocess,
        num_classes=num_classes,
        resolution=int(model_config.resolution),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        image_mean=tuple(wrapper.means),
        image_std=tuple(wrapper.stds),
    )


def _load_rfdetr():
    try:
        import rfdetr
        from rfdetr.config import TrainConfig
        from rfdetr.models import build_criterion_from_config
    except ImportError as exc:
        raise ImportError(
            "RF-DETR requires optional dependencies. Install them with "
            "`pip install -r requirements-rfdetr.txt` (the Apache-2.0 `rfdetr` package)."
        ) from exc

    return rfdetr, TrainConfig, build_criterion_from_config
