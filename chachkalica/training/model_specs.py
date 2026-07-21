"""Per-architecture builder-option specs — the config UI's mirror of the adapters.

The trainer (``friendy_chachkalica``) is model-specific only inside its ``adapters/``:
every ``build_<arch>(num_classes, **params)`` signature defines the knobs that
architecture accepts. This module re-declares those knobs so the admin can offer
a real form field per option (a size dropdown, thresholds, backbone weights …)
instead of hand-typed JSON, while the DB stays model-agnostic — every value here
is still stored in :attr:`ExperimentModel.params` and spread back into the model
YAML entry by :mod:`training.services.config_gen`.

Keep this in sync with ``/home/luka/workspace/chachkalica/friendy_chachkalica/adapters/*.py``:
one entry per selectable ``build_<arch>`` kwarg. ``key`` is the ``params`` key
(and the YAML kwarg name); ``kind`` picks the widget; ``default`` is the adapter
default shown as guidance (a blank field means "use the adapter default").
"""

import json
import os

# Each RF-DETR variant runs at a native square resolution, and its DINOv2 backbone
# requires the input side divisible by ``patch_size * num_windows``. Both are mirrored
# from the rfdetr package's per-variant ModelConfig so the admin form can pre-fill the
# resolution field on variant selection and validate a hand-typed override against the
# right multiple (56 for base's patch-14/4 backbone, 32 for the patch-16/2 variants).
RFDETR_VARIANT_RESOLUTION: dict[str, dict[str, int]] = {
    "nano":   {"native": 384, "multiple": 32},
    "small":  {"native": 512, "multiple": 32},
    "medium": {"native": 576, "multiple": 32},
    "base":   {"native": 560, "multiple": 56},
    "large":  {"native": 704, "multiple": 32},
}

# variant -> native resolution, handed to the form JS so selecting a variant fills
# the resolution field with its native value.
RFDETR_NATIVE_RESOLUTIONS = {
    variant: spec["native"] for variant, spec in RFDETR_VARIANT_RESOLUTION.items()
}

# Each spec: {key, label, kind, choices?, default?, help?}
#   kind in {"choice", "int", "float", "bool", "str"}
#   choices: list of "value" or ("value", "label")
ARCH_FIELD_SPECS: dict[str, list[dict]] = {
    "retinanet": [
        {
            "key": "variant", "label": "Size / variant", "kind": "choice",
            "choices": ["resnet50_fpn", "resnet50_fpn_v2"],
            "default": "resnet50_fpn_v2",
            "help": "RetinaNet backbone/FPN variant.",
        },
        {
            "key": "weights_backbone", "label": "Backbone weights", "kind": "str",
            "help": "ImageNet backbone weights, e.g. DEFAULT. Ignored when "
                    "'pretrained' loads the full COCO weights.",
        },
        {
            "key": "trainable_backbone_layers", "label": "Trainable backbone layers",
            "kind": "int",
            "help": "How many backbone stages to fine-tune (0–5). Blank = torchvision default.",
        },
    ],
    "rfdetr": [
        {
            "key": "variant", "label": "Size / variant", "kind": "choice",
            "choices": ["nano", "small", "medium", "base", "large"],
            "default": "base",
            "help": "RF-DETR size (all Apache-2.0). xlarge/2xlarge are non-free and rejected. "
                    "Selecting a variant fills the resolution below with its native value.",
            "attrs": {"data-native-resolutions": json.dumps(RFDETR_NATIVE_RESOLUTIONS)},
        },
        {
            "key": "resolution", "label": "Input resolution", "kind": "int",
            "help": "Square input size, pre-filled with the selected variant's native "
                    "resolution. Must stay divisible by the variant's patch stride (56 for "
                    "base, 32 for nano/small/medium/large) or it crashes on epoch 1. "
                    "Blank = the variant's native resolution.",
        },
        {
            "key": "score_threshold", "label": "Score threshold", "kind": "float",
            "default": 0.5,
            "help": "Default confidence cutoff used at prediction time.",
        },
        {
            "key": "nms_threshold", "label": "val_metrics_nms_iou_threshold", "kind": "float",
            "help": "IoU for deduplicating this model's boxes in val/test "
                    "precision/recall/F1 only — inference stays NMS-free and mAP is "
                    "unaffected. Blank = the experiment's operating NMS threshold.",
        },
        {
            "key": "freeze_backbone", "label": "Freeze backbone", "kind": "bool",
            "default": False,
            "help": "Freeze the DINOv2 encoder; the multi-scale projector on top of it "
                    "keeps training.",
        },
    ],
    "rtdetr": [
        # RT-DETR's backbone size *is* its pretrained checkpoint, so the size
        # selector lives in the unified "Pretrained weights" dropdown below
        # (see WEIGHTS_CATALOG["rtdetr"]) rather than as a separate variant kwarg.
        {
            "key": "score_threshold", "label": "Score threshold", "kind": "float",
            "default": 0.5, "help": "Default confidence cutoff used at prediction time.",
        },
        {
            "key": "nms_threshold", "label": "val_metrics_nms_iou_threshold", "kind": "float",
            "help": "IoU for deduplicating this model's boxes in val/test "
                    "precision/recall/F1 only — inference stays NMS-free and mAP is "
                    "unaffected. Blank = the experiment's operating NMS threshold.",
        },
        {
            "key": "input_max_size", "label": "Input max size", "kind": "int",
            "default": 640, "help": "Longest-side cap; larger inputs are downscaled.",
        },
        {
            "key": "input_size_multiple", "label": "Input size multiple", "kind": "int",
            "default": 32, "help": "Pad each side up to this multiple.",
        },
        {
            "key": "ignore_mismatched_sizes", "label": "Ignore mismatched sizes",
            "kind": "bool", "default": True,
            "help": "Re-init the head when the pretrained class count differs.",
        },
        {
            "key": "trainable_backbone_layers", "label": "Trainable backbone layers",
            "kind": "int",
            "help": "How many backbone stages to fine-tune (0–5). Blank = fully trainable.",
        },
    ],
    "fasterrcnn": [
        # Order follows the forward pass: input resize -> backbone -> RPN
        # (proposal filtering in the same order torchvision applies it:
        # pre-NMS top-N -> score filter -> NMS -> post-NMS top-N) -> RoI head/output.
        {
            "key": "min_size", "label": "Resize min side", "kind": "int",
            "help": "Shorter-side resize target before the backbone. Blank = torchvision "
                    "default (800).",
        },
        {
            "key": "max_size", "label": "Resize max side", "kind": "int",
            "help": "Longer-side resize cap. Blank = torchvision default (1333).",
        },
        {
            "key": "variant", "label": "Size / variant", "kind": "choice",
            "choices": [
                "resnet50_fpn", "resnet50_fpn_v2",
                "mobilenet_v3_large_fpn", "mobilenet_v3_large_320_fpn",
            ],
            "default": "resnet50_fpn_v2",
            "help": "Faster R-CNN backbone/FPN variant. resnet50_fpn_v2 has the best "
                    "accuracy; the mobilenet_v3 variants trade accuracy for speed. "
                    "torchvision's COCO-pretrained weights for every variant are "
                    "BSD-3-Clause (safe for commercial use) — tick 'pretrained' to load them.",
        },
        {
            "key": "weights_backbone", "label": "Backbone weights", "kind": "str",
            "help": "ImageNet backbone weights, e.g. DEFAULT. Ignored when "
                    "'pretrained' loads the full COCO weights.",
        },
        {
            "key": "trainable_backbone_layers", "label": "Trainable backbone layers",
            "kind": "int",
            "help": "How many backbone stages to fine-tune (0–5). Blank = torchvision default.",
        },
        {
            "key": "rpn_pre_nms_top_n_test", "label": "RPN pre-NMS top-N (eval)", "kind": "int",
            "help": "Proposals kept per FPN level before RPN NMS, at inference time. "
                    "Blank = torchvision default (1000).",
        },
        {
            "key": "rpn_score_thresh", "label": "RPN score threshold", "kind": "float",
            "help": "Objectness floor for dropping proposals early. Blank = torchvision "
                    "default (0.0).",
        },
        {
            "key": "rpn_nms_thresh", "label": "rpn_nms_iou_threshold", "kind": "float",
            "help": "IoU threshold for suppressing overlapping proposals. Blank = "
                    "torchvision default (0.7).",
        },
        {
            "key": "rpn_post_nms_top_n_test", "label": "RPN post-NMS top-N (eval)", "kind": "int",
            "help": "Proposals kept after RPN NMS, at inference time — this is how many "
                    "get fed to the RoI heads (the main proposal-count/speed knob). "
                    "Blank = torchvision default (1000).",
        },
        {
            "key": "box_score_thresh", "label": "Box score threshold", "kind": "float",
            "help": "Final-detection confidence floor applied by the RoI head. "
                    "Blank = torchvision default (0.05).",
        },
        {
            "key": "box_nms_thresh", "label": "box_nms_iou_threshold", "kind": "float",
            "help": "IoU threshold for the final class-aware NMS. Blank = torchvision "
                    "default (0.5).",
        },
        {
            "key": "box_detections_per_img", "label": "Detections per image", "kind": "int",
            "help": "Max final detections kept per image. Blank = torchvision default (100).",
        },
    ],
    "yolox": [
        {
            "key": "variant", "label": "Size / variant", "kind": "choice",
            "choices": ["yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"],
            "default": "yolox-s",
            "help": "YOLOX size (all Apache-2.0).",
        },
        {
            "key": "score_threshold", "label": "Score threshold", "kind": "float",
            "default": 0.3, "help": "Default confidence cutoff used at prediction time.",
        },
        {
            "key": "nms_threshold", "label": "nms_iou_threshold", "kind": "float",
            "default": 0.45, "help": "IoU threshold for non-maximum suppression.",
        },
        {
            "key": "trainable_backbone_layers", "label": "Trainable backbone layers",
            "kind": "int",
            "help": "How many backbone stages to fine-tune (0–5). Blank = fully trainable.",
        },
    ],
}

# Every params key any arch's form owns. On save we strip these from params
# before re-applying the selected arch's values, so switching arch never leaves
# a stale kwarg (e.g. yolox's nms_threshold) that a different adapter would reject.
ALL_SPEC_KEYS = {spec["key"] for specs in ARCH_FIELD_SPECS.values() for spec in specs}

FIELD_PREFIX = "xm_"  # form-field namespace: xm_<arch>_<key>


def field_name(arch: str, key: str) -> str:
    """Form-field name for an (arch, param-key) spec — unique across archs."""
    return f"{FIELD_PREFIX}{arch}_{key}"


def spec_field_names() -> list[str]:
    """Every builder-option form-field name, in arch/spec declaration order.

    Shared by the form (which declares the fields) and the admin inline (which
    lays them out) so the two never drift.
    """
    return [
        field_name(arch, spec["key"])
        for arch, specs in ARCH_FIELD_SPECS.items()
        for spec in specs
    ]


def normalized_choices(spec: dict) -> list[tuple[str, str]]:
    """Spec choices as (value, label) pairs (a bare string becomes (s, s))."""
    out = []
    for choice in spec.get("choices", []):
        if isinstance(choice, (list, tuple)):
            out.append((str(choice[0]), str(choice[1])))
        else:
            out.append((str(choice), str(choice)))
    return out


# ---------------------------------------------------------------------------
# Pretrained-weights selection (the ``weights`` params key)
# ---------------------------------------------------------------------------
# Unlike the builder options above, weights is ONE dropdown per model row whose
# options are (a) partly dynamic — the operator's own trained models are added
# by the form — and (b) variant-aware: an option can belong to a single variant.
# config_gen still only reads ``params["weights"]``; the form resolves the
# dropdown selection into that key (see ExperimentModelForm). See the
# ``pretrained-weights-catalog`` note for how these were verified.

WEIGHTS_KEY = "weights"
WEIGHTS_FIELD_PREFIX = "xm_weights_"  # form field name: xm_weights_<arch>

# Sentinel option values, resolved by ExperimentModelForm.save():
WEIGHTS_NONE = ""                # train from scratch (random init)
WEIGHTS_DEFAULT = "__default__"  # the arch/variant's published default (weights=True)
WEIGHTS_CUSTOM = "__custom__"    # use the free-text custom path/URL field
WEIGHTS_FRIENDY_PREFIX = "__friendy__:"
INIT_CHECKPOINT_KEY = "init_checkpoint"


def friendy_weights_value(checkpoint_path: str) -> str:
    """Encode a promoted Friendy checkpoint as a dropdown value."""
    return f"{WEIGHTS_FRIENDY_PREFIX}{checkpoint_path}"


def friendy_checkpoint_from_value(value: str) -> str | None:
    if isinstance(value, str) and value.startswith(WEIGHTS_FRIENDY_PREFIX):
        return value[len(WEIGHTS_FRIENDY_PREFIX):]
    return None

# Archs whose published default is variant-resolved by the adapter when it gets
# ``weights=True`` (torchvision COCO enum / YOLOX per-variant URL / RF-DETR
# per-variant default). RT-DETR is excluded: its size *is* its checkpoint, so it
# lists explicit repo ids instead of a single "default".
WEIGHTS_DEFAULT_ARCHS = {"retinanet", "fasterrcnn", "yolox", "rfdetr"}

# torchvision accepts weight-enum names, not arbitrary checkpoint paths/URLs.
# Its published defaults already have dedicated options.
WEIGHTS_CUSTOM_ARCHS = {"yolox", "rtdetr", "rfdetr"}

# Published, appropriately-licensed checkpoints offered per arch *beyond* the
# variant default. Each entry: {value, label, variant?}. ``value`` is written
# verbatim into params["weights"] (a URL, HF repo id, rfdetr registry filename,
# or local path). ``variant`` (optional) ties the option to a single variant of
# that arch — the form tags the <option> with data-variant so the JS shows it
# only while that variant is selected; omit it for options valid everywhere.
WEIGHTS_CATALOG: dict[str, list[dict]] = {
    "retinanet": [],   # torchvision exposes only one COCO enum per variant
    "fasterrcnn": [],  # (same) — default/none/custom is the whole story
    "yolox": [],       # ByteTrack person/crowd weights added once re-hosted locally
    "rfdetr": [
        {
            # Objects365-pretrained base backbone (matches base's dinov2 encoder).
            # Broader pretrain than COCO — often a better start for domain shift.
            "value": "rf-detr-base-o365.pth",
            "label": "Objects365 (base) — broader pretrain",
            "variant": "base",
            "train_res": "560",  # base's native square resolution
        },
    ],
    "rtdetr": [
        # Size == checkpoint for RT-DETR, so these double as the size selector.
        # RT-DETRv2 beats v1 at every size (largest gain on r18/r34); all Apache-2.0.
        # Every RT-DETR(v1/v2) COCO checkpoint is trained at a square 640.
        {"value": "PekingU/rtdetr_r18vd", "label": "r18vd — v1 (smallest)", "train_res": "640"},
        {"value": "PekingU/rtdetr_r34vd", "label": "r34vd — v1", "train_res": "640"},
        {"value": "PekingU/rtdetr_r50vd", "label": "r50vd — v1 (original default)", "train_res": "640"},
        {"value": "PekingU/rtdetr_r101vd", "label": "r101vd — v1 (largest)", "train_res": "640"},
        {"value": "PekingU/rtdetr_v2_r18vd", "label": "r18vd — v2 (+1.6 AP over v1)", "train_res": "640"},
        {"value": "PekingU/rtdetr_v2_r34vd", "label": "r34vd — v2 (+1.0 AP over v1)", "train_res": "640"},
        {"value": "PekingU/rtdetr_v2_r50vd", "label": "r50vd — v2", "train_res": "640"},
        {"value": "PekingU/rtdetr_v2_r101vd", "label": "r101vd — v2", "train_res": "640"},
    ],
}

# Input resolution each arch's variant-resolved "COCO pretrained (default)" option
# was trained at, surfaced next to the weights dropdown so an operator can choose a
# matching training resolution. It is *guidance, not a constraint* — detectors
# fine-tune fine off-resolution — so the UI only annotates the option and never
# auto-sets the training resolution.
#
# A dict value is a {variant: res} map the form/JS resolves against the row's
# selected variant (the default's resolution genuinely depends on the variant); a
# bare string is a single value used for every variant. RT-DETR is absent: it has
# no "default" option (its size IS its checkpoint — see WEIGHTS_CATALOG above).
_TORCHVISION_MULTISCALE = "800 shorter side (≤1333)"
WEIGHTS_DEFAULT_TRAIN_RES: dict[str, object] = {
    # YOLOX test/train size: 416 for nano/tiny, 640 for s/m/l/x.
    "yolox": {
        "yolox-nano": "416", "yolox-tiny": "416",
        "yolox-s": "640", "yolox-m": "640", "yolox-l": "640", "yolox-x": "640",
    },
    # RF-DETR's default is each variant's native square resolution.
    "rfdetr": {variant: str(res) for variant, res in RFDETR_NATIVE_RESOLUTIONS.items()},
    # torchvision COCO recipes: 800 shorter-side multi-scale, except the dedicated
    # low-res mobilenet_v3_large_320 variant (320 shorter side, ≤640).
    "retinanet": _TORCHVISION_MULTISCALE,
    "fasterrcnn": {
        "resnet50_fpn": _TORCHVISION_MULTISCALE,
        "resnet50_fpn_v2": _TORCHVISION_MULTISCALE,
        "mobilenet_v3_large_fpn": _TORCHVISION_MULTISCALE,
        "mobilenet_v3_large_320_fpn": "320 shorter side (≤640)",
    },
}


def weights_field_name(arch: str) -> str:
    """Form-field name for an arch's pretrained-weights dropdown."""
    return f"{WEIGHTS_FIELD_PREFIX}{arch}"


def weights_field_names() -> list[str]:
    """Every weights dropdown field name, in arch declaration order."""
    return [weights_field_name(arch) for arch in ARCH_FIELD_SPECS]


def weights_base_choices(arch: str) -> list[tuple[str, str]]:
    """Static (value, label) options for an arch's weights dropdown.

    The form appends the operator's own trained models (dynamic) on top of these.
    """
    out: list[tuple[str, str]] = [
        (WEIGHTS_NONE, "None — train from scratch (random init)")
    ]
    if arch in WEIGHTS_DEFAULT_ARCHS:
        out.append((WEIGHTS_DEFAULT, "COCO pretrained (default)"))
    for entry in WEIGHTS_CATALOG.get(arch, []):
        out.append((entry["value"], entry["label"]))
    if arch in WEIGHTS_CUSTOM_ARCHS:
        out.append((WEIGHTS_CUSTOM, "Custom native pretrained reference…"))
    return out


def weights_variant_map(arch: str) -> dict[str, str]:
    """{option value: variant} for options tied to a single variant.

    Used to tag <option>s so the JS hides mismatched ones; values absent from
    the map are valid for every variant.
    """
    return {
        entry["value"]: entry["variant"]
        for entry in WEIGHTS_CATALOG.get(arch, [])
        if entry.get("variant")
    }


def weights_res_map(arch: str) -> dict[str, str]:
    """{option value: pretrain resolution} for fixed-resolution catalog options.

    Fixed here means the checkpoint's training resolution doesn't depend on a
    variant selection (unlike the variant-resolved "default" option, whose
    resolution comes from :data:`WEIGHTS_DEFAULT_TRAIN_RES`). Used to tag <option>s
    with ``data-train-res`` so the JS can annotate the label.
    """
    return {
        entry["value"]: entry["train_res"]
        for entry in WEIGHTS_CATALOG.get(arch, [])
        if entry.get("train_res")
    }


def weights_default_res(arch: str):
    """Pretrain resolution of the arch's "COCO pretrained (default)" option.

    Returns a ``{variant: res}`` dict when it depends on the variant, a bare
    string when uniform, or ``None`` when the arch has no default option (RT-DETR).
    """
    return WEIGHTS_DEFAULT_TRAIN_RES.get(arch)


# ---------------------------------------------------------------------------
# Locally re-hosted ByteTrack YOLOX weights
# ---------------------------------------------------------------------------
# ByteTrack (MIT) publishes YOLOX-backbone detectors trained on
# CrowdHuman+MOT17+Cityperson+ETHZ — a person/crowd-detection starting point
# that transfers well to people/aerial data. They are single-class (person), so
# our YOLOX loader keeps the backbone/neck/box heads and re-inits the class head
# (verified: 636/642 tensors load, only head.cls_preds reinit). The checkpoints
# live on Google Drive, which torch.hub can't fetch, so they are re-hosted under
# the project's weights dir by ``manage.py fetch_pretrained_weights``. Keyed by
# our YOLOX variant. Google Drive file ids are from ifzhang/ByteTrack's model zoo.
BYTETRACK_YOLOX: dict[str, dict[str, str]] = {
    "yolox-nano": {"gdrive_id": "1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX",
                   "filename": "bytetrack_nano_mot17.pth.tar"},
    "yolox-tiny": {"gdrive_id": "1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju",
                   "filename": "bytetrack_tiny_mot17.pth.tar"},
    "yolox-s":    {"gdrive_id": "1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj",
                   "filename": "bytetrack_s_mot17.pth.tar"},
    "yolox-m":    {"gdrive_id": "11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun",
                   "filename": "bytetrack_m_mot17.pth.tar"},
    "yolox-l":    {"gdrive_id": "1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz",
                   "filename": "bytetrack_l_mot17.pth.tar"},
    "yolox-x":    {"gdrive_id": "1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
                   "filename": "bytetrack_x_mot17.pth.tar"},
}

# Where re-hosted weights live (relative to the project root, like configs/runs).
WEIGHTS_DIR_REL = "data/training/weights"


def weights_dir() -> str:
    """Absolute directory for re-hosted weights (under the project root)."""
    from django.conf import settings

    return os.path.join(str(settings.BASE_DIR), *WEIGHTS_DIR_REL.split("/"))


def bytetrack_yolox_path(variant: str) -> str:
    """Absolute path a ByteTrack checkpoint for ``variant`` would live at."""
    return os.path.join(weights_dir(), BYTETRACK_YOLOX[variant]["filename"])


def bytetrack_yolox_options() -> list[dict]:
    """ByteTrack YOLOX weights present on disk, as catalog-style entries.

    Existence-gated so the dropdown only offers what's actually been fetched;
    empty until ``manage.py fetch_pretrained_weights`` has run.
    """
    out: list[dict] = []
    for variant in BYTETRACK_YOLOX:
        path = bytetrack_yolox_path(variant)
        if os.path.exists(path):
            size = variant.split("-")[-1]
            out.append({
                "value": path,
                "label": f"ByteTrack person — CrowdHuman+MOT17 ({size})",
                "variant": variant,
                "train_res": "800×1440",  # ByteTrack MOT input size (HxW)
            })
    return out
