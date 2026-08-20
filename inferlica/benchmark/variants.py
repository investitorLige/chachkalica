"""Static per-arch benchmark registry: variant list, NMS status, TRT support.

Mirrors ``friendy_chachkalica.registry.MODEL_REGISTRY``'s 5 architectures, but
adds the "variant" axis this benchmark sweeps over (each arch's own
``build_*`` function already accepts a variant/weights kwarg -- this module
just enumerates the set worth benchmarking) plus static facts -- does this
arch's exported format already bake NMS in? does it have a TensorRT export
path at all? -- that hold regardless of which random-init weights get built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

ALL_FORMATS: Tuple[str, ...] = ("pt", "onnx", "engine")


@dataclass
class VariantSpec:
    name: str
    build_kwargs: Dict[str, Any] = field(default_factory=dict)
    formats: Tuple[str, ...] = ALL_FORMATS


# Archs the sweep should never attempt a TRT engine build for. Every registered
# arch now has a TRT path -- fasterrcnn builds via the EfficientNMS re-export in
# trt_export/arch/fasterrcnn.py (its passthrough graph can't be compiled). An
# engine that still fails to build surfaces as a per-cell status="error"
# (benchmark_cell catches it), not a crash, so there is nothing to hard-skip here.
# Kept as an (empty) extension point.
NO_TRT_ARCHS: set = set()

# Static per-arch NMS status: purely informational metadata written into every
# CSV, never branched on at benchmark time. fasterrcnn/retinanet/yolox always
# produce already-NMS'd output (torchvision-native ops or the vendored YOLOX
# postprocess() baked into .pt/.onnx; a fused EfficientNMS_TRT plugin node in
# .engine). rtdetr/rfdetr are NMS-free by architecture (DETR set-prediction)
# and must never have NMS applied to them -- their adapters say so explicitly.
NMS_STATUS: Dict[str, str] = {
    "fasterrcnn": "baked-in (torchvision NMS / EfficientNMS_TRT plugin for .engine)",
    "retinanet": "baked-in (torchvision NMS / EfficientNMS_TRT plugin for .engine)",
    "yolox": "baked-in (vendored YOLOX postprocess() / EfficientNMS_TRT plugin for .engine)",
    "rtdetr": "nms-free (DETR set-prediction)",
    "rfdetr": "nms-free (DETR set-prediction)",
    "ecdet": "nms-free (DETR set-prediction)",
    "dfine": "nms-free (DETR set-prediction)",
}

# Backbone block size (patch_size * num_windows) that each rfdetr variant's
# resolution must be an exact multiple of -- base is patch14/windows4 (56), the
# rest are patch16/windows2 (32). Used to snap an arbitrary requested image size
# to the nearest valid rfdetr resolution so the sweep can benchmark rfdetr
# across image sizes despite its resolution being architectural, not a runtime
# input size (see rfdetr_variant_specs).
RFDETR_BLOCK: Dict[str, int] = {"nano": 32, "small": 32, "medium": 32, "large": 32, "base": 56}


def _snap_to_block(image_size: int, block: int) -> int:
    """Nearest positive multiple of ``block`` to ``image_size``."""
    return max(block, int(round(image_size / block)) * block)


def rfdetr_variant_specs(image_size: int) -> List["VariantSpec"]:
    """rfdetr VariantSpecs with each variant's resolution snapped to the nearest
    valid multiple of its backbone block size for ``image_size``.

    rfdetr's ONNX/TRT export is static-shaped at whatever ``resolution`` the
    adapter reports, and that resolution must satisfy the windowed-attention
    block constraint -- so "benchmark rfdetr at 320/960" means picking the
    closest valid resolution per variant (e.g. at 960: 960 for the block-32
    variants, 952 for base). Used only when the sweep is asked to make rfdetr
    follow ``--image-size``; the default ARCH_VARIANTS list below keeps each
    variant at its native resolution instead.
    """
    order = ("nano", "small", "medium", "base", "large")
    return [
        VariantSpec(name, {"variant": name, "weights": False,
                           "resolution": _snap_to_block(image_size, RFDETR_BLOCK[name])})
        for name in order
    ]


def yolox_variant_specs(image_size: int) -> List["VariantSpec"]:
    """yolox VariantSpecs with the export canvas set to ``image_size``.

    yolox's ONNX/TRT input is a fixed square canvas
    (``YOLOXAdapter.input_max_size``, default 640, rounded to a multiple of 32),
    so it otherwise ignores ``--image-size`` -- a 320/960 synthetic input just
    gets letterboxed onto the 640 canvas. Passing ``input_max_size=image_size``
    through ``build_yolox`` builds the adapter (and therefore the traced
    ONNX/engine) at that size instead. ``image_size`` must be a multiple of 32.
    """
    specs = []
    for spec in ARCH_VARIANTS["yolox"]:
        kwargs = dict(spec.build_kwargs)
        kwargs["input_max_size"] = image_size
        specs.append(VariantSpec(spec.name, kwargs))
    return specs


def ecdet_variant_specs(image_size: int) -> List["VariantSpec"]:
    """ecdet VariantSpecs with the export canvas set to ``image_size``.

    Like yolox (and unlike rfdetr, whose resolution is constrained per variant),
    ecdet's canvas is a single ``input_max_size`` that every variant shares — all
    four are published at a square 640, and upstream documents 1280 as the
    high-resolution alternative. It is architectural rather than a runtime input
    size: ``ECTransformer`` generates its anchors from ``eval_spatial_size`` at
    build time and the ONNX export is static in H/W, so benchmarking ecdet at
    320/960 means *building* at that size, not feeding a different input.
    ``image_size`` must be a multiple of 32 (the encoder's stride).
    """
    specs = []
    for spec in ARCH_VARIANTS["ecdet"]:
        kwargs = dict(spec.build_kwargs)
        kwargs["input_max_size"] = image_size
        specs.append(VariantSpec(spec.name, kwargs))
    return specs


def dfine_variant_specs(image_size: int) -> List["VariantSpec"]:
    """dfine VariantSpecs with the export canvas set to ``image_size``.

    Same shape of override as ecdet/yolox: every published D-FINE variant trains
    and exports at a square 640 (``build_dfine``'s ``input_max_size`` default),
    and the sweep's synthetic input is letterboxed onto that canvas unless the
    adapter is *built* at the requested size. Unlike ecdet the graph itself is
    dynamic in H/W (the anchors trace symbolically -- see the dfine notes in
    trt_export/arch/__init__.py), but the exported ``.meta.json`` still records
    one square size, which is what the TRT profile and the letterbox both use.
    ``image_size`` must be a multiple of 32 (``input_size_multiple``).
    """
    specs = []
    for spec in ARCH_VARIANTS["dfine"]:
        kwargs = dict(spec.build_kwargs)
        kwargs["input_max_size"] = image_size
        specs.append(VariantSpec(spec.name, kwargs))
    return specs


ARCH_VARIANTS: Dict[str, List[VariantSpec]] = {
    "fasterrcnn": [
        # All formats incl. ".engine": fasterrcnn builds a TRT engine via the
        # EfficientNMS re-export (see trt_export/arch/fasterrcnn.py).
        VariantSpec("resnet50_fpn", {"variant": "resnet50_fpn"}),
        VariantSpec("resnet50_fpn_v2", {"variant": "resnet50_fpn_v2"}),
        VariantSpec("mobilenet_v3_large_fpn", {"variant": "mobilenet_v3_large_fpn"}),
        VariantSpec("mobilenet_v3_large_320_fpn", {"variant": "mobilenet_v3_large_320_fpn"}),
    ],
    "retinanet": [
        VariantSpec("resnet50_fpn", {"variant": "resnet50_fpn"}),
        VariantSpec("resnet50_fpn_v2", {"variant": "resnet50_fpn_v2"}),
    ],
    "rfdetr": [
        # weights=False is required: build_rfdetr defaults to weights=True,
        # which downloads the variant's published COCO checkpoint.
        #
        # resolution=640 pins most variants to the same input size as the other
        # 4 archs for a fair FPS/latency comparison (rfdetr's ONNX/TRT export is
        # always static-shaped at whatever `resolution` the adapter reports --
        # see friendy_chachkalica/ml/onnx_export/arch/rfdetr.py -- so this
        # overrides each variant's native resolution: nano 384->640,
        # small 512->640, medium 576->640). Verified they build + run a full
        # predict() at 640 with no shape errors.
        VariantSpec("nano", {"variant": "nano", "weights": False, "resolution": 640}),
        VariantSpec("small", {"variant": "small", "weights": False, "resolution": 640}),
        VariantSpec("medium", {"variant": "medium", "weights": False, "resolution": 640}),
        # base's windowed-attention backbone hard-requires the resolution be a
        # multiple of patch_size*num_windows=56 (640 isn't); 672 is the nearest
        # valid size at or above 640. base's .onnx/.engine run FAR slower than
        # its .pt (engine ~62 fps @ ~17% GPU util vs pt ~118 fps) -- a real
        # export cliff. It is NOT caused by base's num_windows=4 / patch_size=14
        # defaults, as first assumed: large below (num_windows=2, patch_size=16,
        # identical to medium) hits the SAME cliff at 704, so the driver is
        # resolution > 640, not the window count -- see the large note.
        VariantSpec("base", {"variant": "base", "weights": False, "resolution": 672}),
        # large stays at its NATIVE 704, deliberately NOT overridden to 640.
        # The 2026 RFDETRLargeConfig is architecturally identical to medium
        # (same dinov2_windowed_small encoder, hidden_dim 256, dec_layers 4,
        # num_windows 2, patch_size 16, [P4] projector) -- the ONLY difference
        # is native resolution (704 vs 576). That makes large-vs-medium a clean
        # controlled experiment on resolution, and it exposed a real finding:
        # rfdetr's EXPORTED formats fall off a cliff above 640. medium@640
        # engine ~266 fps @ 60% util; large@704 engine ~56 fps @ 16% util --
        # same graph, only the input size differs. The SAME large engine ran
        # ~255 fps when this was pinned to 640, and PT is unaffected at any size
        # (base/large pt ~110-118 fps), so it's specifically an ONNX/TRT
        # kernel-fusion cliff at the 672/704 token grids (42/44 patches/side vs
        # 40 at 640), reproduced on an idle GPU so it isn't a build-time
        # tactic-selection artifact. Overriding large->640 would hide this by
        # collapsing it onto medium@640 (verified identical). (The old
        # heavyweight ViT-B large is now RFDETRLargeDeprecatedConfig; the
        # shipping `large` is this ViT-S one.)
        VariantSpec("large", {"variant": "large", "weights": False, "resolution": 704}),
        # xlarge/2xlarge deliberately excluded: rfdetr[plus], PML-1.0
        # non-commercial license -- the adapter itself refuses to build them.
    ],
    "rtdetr": [
        # rtdetr has no `variant=` kwarg -- size is selected by an HF repo id
        # passed as `weights=`. `_rtdetr_repo` is a benchmark-only marker key
        # consumed by core._build_rtdetr_variant, never forwarded to build_model.
        VariantSpec("r18vd", {"_rtdetr_repo": "PekingU/rtdetr_r18vd"}),
        VariantSpec("r34vd", {"_rtdetr_repo": "PekingU/rtdetr_r34vd"}),
        VariantSpec("r50vd", {"_rtdetr_repo": "PekingU/rtdetr_r50vd"}),
        VariantSpec("r101vd", {"_rtdetr_repo": "PekingU/rtdetr_r101vd"}),
    ],
    "yolox": [
        VariantSpec("yolox-nano", {"variant": "yolox-nano", "weights": False}),
        VariantSpec("yolox-tiny", {"variant": "yolox-tiny", "weights": False}),
        VariantSpec("yolox-s", {"variant": "yolox-s", "weights": False}),
        VariantSpec("yolox-m", {"variant": "yolox-m", "weights": False}),
        VariantSpec("yolox-l", {"variant": "yolox-l", "weights": False}),
        VariantSpec("yolox-x", {"variant": "yolox-x", "weights": False}),
    ],
    "dfine": [
        # D-FINE has no `variant=` kwarg either -- like rtdetr, the size lives in
        # the HF repo id (`weights=`). Unlike rtdetr, these ids resolve WITHOUT a
        # download: all nine ustc-community repos are in this box's offline
        # HF_HOME cache (huggingface.co is unreachable here -- see the trainer's
        # HF_HUB_OFFLINE=1), so passing the real weights costs a local file read
        # and, unlike rtdetr's config-only fetch, actually yields a params count.
        # Weights change neither shape nor speed; the -coco line is the plain
        # COCO one (the obj2coco repos are the same five backbones).
        VariantSpec("nano", {"weights": "ustc-community/dfine-nano-coco"}),
        VariantSpec("small", {"weights": "ustc-community/dfine-small-coco"}),
        VariantSpec("medium", {"weights": "ustc-community/dfine-medium-coco"}),
        VariantSpec("large", {"weights": "ustc-community/dfine-large-coco"}),
        VariantSpec("xlarge", {"weights": "ustc-community/dfine-xlarge-coco"}),
    ],
    "ecdet": [
        # weights=False keeps the sweep offline: the distilled ECViT backbone and
        # the COCO detector checkpoints are both downloads, and neither changes
        # shape or speed. Every variant is published at a square 640, which is the
        # adapter default, so no explicit input_max_size here — ecdet_variant_specs
        # overrides it when the sweep follows --image-size.
        VariantSpec("ecdet-s", {"variant": "ecdet-s", "weights": False}),
        VariantSpec("ecdet-m", {"variant": "ecdet-m", "weights": False}),
        VariantSpec("ecdet-l", {"variant": "ecdet-l", "weights": False}),
        VariantSpec("ecdet-x", {"variant": "ecdet-x", "weights": False}),
    ],
}
