"""ECDet (EdgeCrafter) adapter.

ECDet is the detection instantiation of EdgeCrafter (Apache-2.0): a distilled
compact-ViT backbone (``ViTAdapter``/ECViT, patch 16, 2D RoPE) feeding an
RT-DETR-style ``HybridEncoder`` and a D-FINE-style ``ECTransformer`` decoder
(reg_max 32, 300 queries, contrastive denoising). Model code is vendored under
``vendor/edgecrafter`` — see that package's docstring for what was pruned.

Its geometry is *identical* to RT-DETR's, which is why this adapter reuses
``adapters/rtdetr.py``'s resize helpers verbatim in spirit: upstream's train and
val transform stacks are ``Resize [640, 640]`` → ``/255`` → ImageNet
``Normalize`` → boxes as normalized ``cxcywh`` (vendor/edgecrafter/configs/
ecdet.yml). That ``Resize`` takes a 2-element size, so it *stretches* — aspect
ratio is not preserved and there is no padding. Keeping that is not cosmetic:
the canvas is then entirely real content, so a box normalized over the image is
the same box normalized over the model input, and both the labels we hand the
criterion and the boxes we read back are correct by construction.

Two upstream conventions this adapter must respect, both verified against
``ecdetseg/engine/solver/ec_engine.py``:

1. **``ECCriterion`` pre-applies its own ``weight_dict``.** Every branch of its
   forward does ``l_dict = {k: l_dict[k] * self.weight_dict[k] ...}`` before
   suffixing the key with ``_aux_N``/``_dn_N``/``_enc_N``, so the returned dict
   is already weighted and the reduction is a plain ``sum(values())``. Weighting
   it again the way ``adapters/rfdetr.py`` does would double-count the five base
   terms and silently drop all ~35 auxiliary and denoising terms.
2. **The loss runs outside autocast.** ``ec_engine.py`` wraps the criterion call
   in ``torch.autocast(enabled=False)`` even in its AMP path. The matcher and the
   distribution-focal terms are the fp16-fragile part; the backbone/encoder
   forward is not. ``_loss_forward`` does the same, so ``supports_amp`` can stay
   True (unlike RT-DETR, which had to disable AMP wholesale).
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

try:
    from ...formats import clip_xyxy, xyxy_prediction_to_friendy, xyxy_to_xywhn
    from .retinanet import _set_batch_norm_eval
except ImportError:
    from formats import clip_xyxy, xyxy_prediction_to_friendy, xyxy_to_xywhn
    from ml.adapters.retinanet import _set_batch_norm_eval


DEFAULT_ECDET_VARIANT = "ecdet-l"

# Every published ECDet variant trains and evaluates at a square 640
# (``eval_spatial_size: [640, 640]`` in all four configs). Upstream documents
# 1280 as a supported alternative, so this is a default, not a constraint.
ECDET_NATIVE_SIZE = 640

# The ECViT patch embed is stride-16 and the encoder consumes feature strides
# [8, 16, 32], so the canvas side must stay a multiple of 32.
ECDET_SIZE_MULTIPLE = 32

# COCO-pretrained detector checkpoints, per variant (Apache-2.0, GitHub
# releases — plain HTTPS, so torch.hub can fetch them directly, unlike
# ByteTrack's Google Drive links).
ECDET_RELEASE_BASE = (
    "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1"
)
ECDET_PRETRAINED_URLS = {
    "ecdet-s": f"{ECDET_RELEASE_BASE}/ecdet_s.pth",
    "ecdet-m": f"{ECDET_RELEASE_BASE}/ecdet_m.pth",
    "ecdet-l": f"{ECDET_RELEASE_BASE}/ecdet_l.pth",
    "ecdet-x": f"{ECDET_RELEASE_BASE}/ecdet_x.pth",
}

# Sentinel ``weights`` value selecting "distilled ECViT backbone only": load the
# per-variant ecvit*.pth the config names and leave encoder/decoder freshly
# initialized. This is upstream's own default recipe (`train.py` with no `-t`),
# and it is the middle tier between COCO-pretrained and random init.
WEIGHTS_BACKBONE_ONLY = "backbone"

VARIANT_CONFIGS = {
    "ecdet-s": "ecdet_s.yml",
    "ecdet-m": "ecdet_m.yml",
    "ecdet-l": "ecdet_l.yml",
    "ecdet-x": "ecdet_x.yml",
}


@dataclass
class ECDetAdapter:
    model: torch.nn.Module
    criterion: torch.nn.Module
    num_classes: int
    num_top_queries: int = 300
    score_threshold: float = 0.5
    # NOT applied in predict() — ECDet is set-based and runs NMS-free, like
    # rtdetr/rfdetr. The trainer reads this as the IoU for the operating-point
    # val/test metrics (precision/recall/F1/confusion); see
    # train.resolve_operating_nms_threshold.
    nms_threshold: Optional[float] = None
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)
    input_max_size: Optional[int] = ECDET_NATIVE_SIZE
    input_size_multiple: int = ECDET_SIZE_MULTIPLE
    name: str = "ecdet"

    def to(self, device):
        self.model.to(device)
        # The criterion holds no parameters but its matcher allocates on the
        # inputs' device; moving it keeps parity with RFDETRAdapter.to.
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
        """Validation loss, computed in train mode with BatchNorm pinned to eval.

        Unlike RF-DETR, ECDet cannot produce a loss in eval mode at all: the
        decoder only emits ``aux_outputs`` while ``self.training`` is set, and
        ``ECCriterion.forward`` hard-asserts their presence
        (vendor/edgecrafter/edgecrafter/criterion.py:360) rather than degrading to
        the final layer alone. So the model has to be in train mode here.

        That would let validation batches update the encoder's BatchNorm running
        statistics — the exact leak ``YOLOXAdapter.validation_step`` guards, so
        this borrows its guard: flip the BN modules to eval, compute, restore the
        original mode. Note the loss still includes the denoising groups and
        dropout, so it is a train-mode loss on held-out data, not an eval-mode
        loss; it is comparable across epochs but not to a NMS-free eval metric.
        """
        was_training = self.model.training
        self.model.train()
        _set_batch_norm_eval(self.model)
        try:
            return self._loss_forward(images, targets)
        finally:
            self.model.train(was_training)

    def _loss_forward(self, images, targets):
        images, targets = self._resize_training_inputs(images, targets)
        batch = self._prepare_batch(images)
        labels = self._prepare_labels(targets, batch.shape[-2:])
        outputs = self.model(batch, targets=labels)
        # See the module docstring: the criterion applies weight_dict itself and
        # upstream reduces with a bare sum, outside autocast.
        with torch.autocast(device_type=batch.device.type, enabled=False):
            loss_dict = self.criterion(outputs, labels)
            loss = sum(loss_dict.values())
        return loss, loss_dict

    @torch.no_grad()
    def predict(self, images, score_threshold: Optional[float] = None):
        """Friendy predictions, replicating upstream ``PostProcessor``.

        ``PostProcessor.forward`` sigmoids the logits, takes the flattened
        ``[queries * classes]`` top-k down to ``num_top_queries``, derives
        ``label = idx % C`` / ``box = idx // C``, converts ``cxcywh -> xyxy`` and
        scales by the *original* size. The friendy step re-normalizes by that same
        size, so the scaling cancels and the friendy box is the model's normalized
        box — which is what keeps the exported graph's
        ``box_coords: "input_normalized"`` contract true.
        """
        self.model.eval()
        threshold = self.score_threshold if score_threshold is None else score_threshold

        resized_images = []
        scales = []
        for image in images:
            resized_image, scale_y, scale_x = self._resize_image_with_scale(image)
            resized_images.append(resized_image)
            scales.append((scale_y, scale_x))
        batch = self._prepare_batch(resized_images)
        outputs = self.model(batch)

        logits = outputs["pred_logits"]      # [B, Q, C]
        boxes_n = outputs["pred_boxes"]      # [B, Q, 4] normalized cxcywh
        input_height, input_width = batch.shape[-2:]

        num_classes = logits.shape[-1]
        top_k = min(self.num_top_queries, logits.shape[1] * num_classes)
        scores_all = torch.sigmoid(logits)   # focal-loss path (ECDet default)
        top_scores, top_index = torch.topk(scores_all.flatten(1), top_k, dim=-1)
        top_labels = top_index % num_classes
        box_index = top_index // num_classes

        cx, cy, w, h = boxes_n.unbind(dim=-1)
        xyxy_n = torch.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
        )
        gathered = xyxy_n.gather(
            dim=1, index=box_index.unsqueeze(-1).expand(-1, -1, 4)
        )

        results = []
        for index, (image, (scale_y, scale_x)) in enumerate(zip(images, scales)):
            keep = top_scores[index] >= threshold
            boxes = gathered[index][keep]
            # Normalized over the model input -> input pixels, then undo the
            # per-axis resize to land back in original-image pixels. On the
            # stretched path (the only one with a fixed canvas) that is exactly
            # `normalized * original size`.
            boxes = boxes * boxes.new_tensor(
                [input_width, input_height, input_width, input_height]
            )
            boxes[:, [0, 2]] /= scale_x
            boxes[:, [1, 3]] /= scale_y
            image_height, image_width = image.shape[-2:]
            # Clip to bounds, matching RTDETRAdapter.predict and the exported
            # graph's clip_boxes (onnx_export/arch/ecdet.py, which must keep
            # agreeing with this line — tests_clip_parity enforces it). The
            # decoder emits normalized cxcywh through a sigmoid, so a box centred
            # near an edge extends past the frame with no padding involved at all,
            # and would normalize past 1.0 against this image's own size.
            boxes = clip_xyxy(boxes, image_width=image_width, image_height=image_height)
            results.append(
                xyxy_prediction_to_friendy(
                    boxes,
                    top_scores[index][keep],
                    top_labels[index][keep],
                    image_width=image_width,
                    image_height=image_height,
                )
            )
        return results

    def _prepare_batch(self, images):
        """Normalize and stack already-resized images.

        ECDet takes a bare ``[B, 3, H, W]`` tensor — there is no pixel_mask to
        pass, and no padding to describe: with a fixed canvas every image arrives
        at exactly ``_fixed_canvas_size``. With resizing disabled
        (``input_max_size=None``) the batch's own max HxW is rounded up to
        ``input_size_multiple`` so the stride-32 encoder accepts it.
        """
        device = next(self.model.parameters()).device
        image_mean = torch.tensor(self.image_mean, device=device).view(3, 1, 1)
        image_std = torch.tensor(self.image_std, device=device).view(3, 1, 1)

        prepared = [
            ((image.to(device).float() - image_mean) / image_std) for image in images
        ]
        canvas_size = self._fixed_canvas_size()
        if canvas_size is not None:
            max_height = max_width = canvas_size
        else:
            max_height = _ceil_to_multiple(
                max(image.shape[-2] for image in prepared), self.input_size_multiple
            )
            max_width = _ceil_to_multiple(
                max(image.shape[-1] for image in prepared), self.input_size_multiple
            )

        return torch.stack(
            [
                F.pad(
                    image,
                    (0, max_width - image.shape[-1], 0, max_height - image.shape[-2]),
                )
                for image in prepared
            ]
        )

    def _fixed_canvas_size(self) -> Optional[int]:
        """Square side every image is stretched onto, or ``None`` when resizing is
        disabled (then the batch-derived padded size applies).

        ECDet's decoder pre-generates its anchors from ``eval_spatial_size`` at
        build time, so this is also the size the model is structurally built for
        — which is why ``input_max_size`` counts as a warm-start structural
        param and why the ONNX export is static in H/W.
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

        Per-axis scaling, aspect ratio *not* preserved — this is upstream ECDet's
        own preprocessing (``{type: Resize, size: [640, 640]}`` in both the train
        and val transform stacks), and it is what keeps the geometry
        self-consistent: the canvas is entirely real content, so a box normalized
        over the image is identical to the same box normalized over the model
        input. Always scales, up as well as down, so a crop smaller than the
        canvas (e.g. a people_detect_first crop grown only to
        ``detector_min_box_size``) is upscaled to span more of the backbone's
        16px-per-token stride.
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
        """Targets as ECDet's criterion wants them: normalized ``cxcywh``.

        ``input_size`` is the batched tensor's HxW, padding included — not the
        image's own extent. The decoder's reference points are normalized over
        the input tensor, so normalizing targets against anything else shifts
        every box it is asked to produce off the content it must classify.
        """
        device = next(self.model.parameters()).device
        image_height, image_width = int(input_size[0]), int(input_size[1])
        labels = []
        for target in targets:
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


def build_ecdet(
    num_classes: int,
    weights: Any = True,
    variant: str = DEFAULT_ECDET_VARIANT,
    score_threshold: float = 0.5,
    nms_threshold: Optional[float] = None,
    input_max_size: Optional[int] = ECDET_NATIVE_SIZE,
    input_size_multiple: int = ECDET_SIZE_MULTIPLE,
    trainable_backbone_layers: Optional[int] = None,
    **builder_options: Any,
) -> ECDetAdapter:
    """Build an ECDet adapter.

    ``weights`` is four-state:
        ``True``                    the variant's COCO checkpoint (ecdet_<size>.pth)
        ``"backbone"``              the distilled ECViT backbone only
        ``False`` / ``None``        random init throughout
        any other ``str``           that URL or path, as a detector checkpoint
    """
    if variant not in VARIANT_CONFIGS:
        available = ", ".join(sorted(VARIANT_CONFIGS))
        raise ValueError(f"Unknown ECDet variant {variant!r}. Available: {available}")

    core, configs_dir = _load_edgecrafter()

    if weights is True:
        try:
            weights = ECDET_PRETRAINED_URLS[variant]
        except KeyError as exc:  # unreachable given the check above, kept explicit
            raise ValueError(
                f"No published ECDet checkpoint for variant {variant!r}"
            ) from exc
    elif weights is False:
        weights = None

    # The distilled ViT weights are worth downloading only when nothing else will
    # overwrite them: a detector checkpoint already carries a trained backbone,
    # and "train from scratch" means scratch. So the backbone-only tier is the one
    # and only case that loads them.
    backbone_only = weights == WEIGHTS_BACKBONE_ONLY
    skip_load_backbone = not backbone_only

    canvas = _ceil_to_multiple(
        input_max_size if input_max_size else ECDET_NATIVE_SIZE, input_size_multiple
    )

    # NOTE load_config has a mutable default `cfg=dict()`, so a bare call would
    # accumulate state across builds — always pass a fresh dict.
    cfg = core.load_config(str(configs_dir / VARIANT_CONFIGS[variant]), cfg={})
    cfg["num_classes"] = int(num_classes)
    cfg["remap_mscoco_category"] = False
    # `eval_spatial_size` is __share__d into the decoder, which pre-generates its
    # anchors from it at build time — this is what makes the model structurally
    # sized for `canvas`.
    cfg["eval_spatial_size"] = [canvas, canvas]
    cfg.setdefault("ViTAdapter", {})["skip_load_backbone"] = skip_load_backbone
    if backbone_only:
        cfg["ViTAdapter"]["weights_path"] = _ensure_backbone_weights(
            cfg["ViTAdapter"]["name"]
        )
    for key, value in builder_options.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value

    global_cfg = core.merge_config(
        cfg, core.GLOBAL_CONFIG, inplace=False, overwrite=False
    )
    model = core.create(global_cfg["model"], global_cfg)
    criterion = core.create(global_cfg["criterion"], global_cfg)
    num_top_queries = int(cfg.get("PostProcessor", {}).get("num_top_queries", 300))

    if weights is not None and not backbone_only:
        try:
            _load_checkpoint(model, weights)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"ECDet pretrained weights {weights!r} were requested but could "
                "not be loaded; refusing to silently train from random initialization."
            ) from exc

    if trainable_backbone_layers is not None:
        _freeze_ecdet_backbone(model, trainable_backbone_layers)

    return ECDetAdapter(
        model=model,
        criterion=criterion,
        num_classes=int(num_classes),
        num_top_queries=num_top_queries,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        input_max_size=input_max_size,
        input_size_multiple=input_size_multiple,
    )


def _load_edgecrafter():
    """Import the vendored EdgeCrafter package and return ``(core, configs_dir)``.

    Imported lazily and under exactly one module spelling per process. Both
    matter: ``registry.py`` imports every builder eagerly, so a module-level
    import here would break all six architectures if anything in the subtree
    failed; and upstream's ``@register()`` asserts a class name is not already
    registered, so importing the subtree twice (once as
    ``friendy_chachkalica.vendor...``, once as bare ``vendor...`` — this package
    is run both ways) would raise "ECDet has been already registered".
    """
    from pathlib import Path

    try:
        from ...vendor.edgecrafter import core
        from ...vendor.edgecrafter import edgecrafter as _register_all  # noqa: F401
    except ImportError:
        from vendor.edgecrafter import core
        from vendor.edgecrafter import edgecrafter as _register_all  # noqa: F401

    configs_dir = Path(core.__file__).resolve().parent.parent / "configs"
    return core, configs_dir


def _ensure_backbone_weights(backbone_name: str) -> str:
    """Fetch the distilled ECViT weights into torch's hub cache; return the path.

    ``ViTAdapter._load_weights`` treats ``weights_path`` as an *existence check
    only*: when the file is there it loads it, and when it is not it downloads to
    a hard-coded ``Path(__file__).parents[2] / "ecvits"`` and ignores
    ``weights_path`` entirely. Under our vendored layout that directory is
    ``vendor/ecvits`` — inside the source tree, which means the download pollutes
    the checkout, is lost on every container image rebuild, and (because the
    existence check still points elsewhere) is re-fetched every single build.

    So we do the fetch ourselves, into the standard torch hub checkpoint cache
    that ``ECDET_PRETRAINED_URLS`` and ``adapters/yolox.py`` already use, and hand
    upstream a path that exists — which routes it down its clean
    load-from-file branch and leaves the vendored file byte-identical to upstream.
    """
    import os
    from pathlib import Path

    try:
        from ...vendor.edgecrafter.edgecrafter.ecvit import ViTAdapter
    except ImportError:
        from vendor.edgecrafter.edgecrafter.ecvit import ViTAdapter

    try:
        url = ViTAdapter.ecvit_url[backbone_name]
    except KeyError as exc:
        available = ", ".join(sorted(ViTAdapter.ecvit_url))
        raise ValueError(
            f"Unknown ECViT backbone {backbone_name!r}. Available: {available}"
        ) from exc

    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    target = cache_dir / os.path.basename(url)
    if not target.exists():
        # Downloads to <cache_dir>/<basename(url)> — the same path we then hand back.
        torch.hub.load_state_dict_from_url(
            url, model_dir=str(cache_dir), map_location="cpu", weights_only=True
        )
    return str(target)


def _load_checkpoint(model: torch.nn.Module, checkpoint_ref: str) -> None:
    """Load a published ECDet checkpoint from a URL or local path.

    Prefers the EMA weights when present — upstream trains with
    ``use_ema: True`` and its own export tool reads ``checkpoint['ema']['module']``
    first, because that is the state its reported AP was measured on.

    Tolerates a class-count mismatch: tensors whose shape differs from the model
    are skipped and left at their fresh init. For a new class count that is
    exactly the classification path — ``enc_score_head``, the per-layer
    ``dec_score_head.*`` and ``denoising_class_embed`` — while the backbone,
    encoder, and the class-agnostic box/corner heads still load.
    """
    if isinstance(checkpoint_ref, str) and checkpoint_ref.startswith(
        ("http://", "https://")
    ):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_ref, map_location="cpu"
        )
    else:
        checkpoint = torch.load(checkpoint_ref, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raise ValueError(
            "This is a Friendy training checkpoint. Select it as 'Your model' "
            "so the checked warm-start loader is used."
        )

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("ema"), dict) and "module" in checkpoint["ema"]:
            state_dict = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
    if not isinstance(state_dict, dict):
        raise ValueError("ECDet pretrained checkpoint does not contain a state dict")

    model_state = model.state_dict()
    compatible = {
        key: tensor
        for key, tensor in state_dict.items()
        if key in model_state
        and torch.is_tensor(tensor)
        and tensor.shape == model_state[key].shape
    }
    reinit = [key for key in state_dict if key not in compatible]
    if not compatible:
        raise ValueError(
            "ECDet pretrained checkpoint has no tensors compatible with this model"
        )
    model.load_state_dict(compatible, strict=False)
    if reinit:
        print(
            f"[ecdet] Loaded {len(compatible)}/{len(state_dict)} pretrained tensor(s); "
            f"re-initialized {len(reinit)} with a mismatched shape "
            f"(the classification head for a new class count): {reinit[:6]}"
            + (" …" if len(reinit) > 6 else "")
        )


def _freeze_ecdet_backbone(model, trainable_backbone_layers: int) -> None:
    """Freeze ECViT transformer blocks, torchvision-style.

    ``trainable_backbone_layers`` keeps torchvision's convention: 0 freezes the
    whole backbone (patch embed + every block), 5 leaves it fully trainable.
    ECViT has no 4-stage ResNet structure to map onto, so its blocks are split
    into 5 near-equal groups and the last ``trainable_backbone_layers`` of them —
    plus everything outside the block list — stay trainable, which preserves the
    "higher number = fine-tune deeper into the head" meaning operators expect.
    """
    trainable_backbone_layers = max(0, min(5, trainable_backbone_layers))
    backbone = model.backbone
    if trainable_backbone_layers == 5:
        return
    if trainable_backbone_layers == 0:
        backbone.requires_grad_(False)
        return

    blocks = list(getattr(getattr(backbone, "backbone", backbone), "blocks", []))
    if not blocks:
        # Unknown internal layout: fall back to all-or-nothing rather than
        # silently freezing the wrong half.
        backbone.requires_grad_(False)
        return

    group_size = max(1, len(blocks) // 5)
    frozen_upto = len(blocks) - trainable_backbone_layers * group_size
    for block in blocks[:frozen_upto]:
        block.requires_grad_(False)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple
