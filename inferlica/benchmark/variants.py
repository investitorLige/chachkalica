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
}

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
        VariantSpec("nano", {"variant": "nano", "weights": False}),
        VariantSpec("small", {"variant": "small", "weights": False}),
        VariantSpec("medium", {"variant": "medium", "weights": False}),
        VariantSpec("base", {"variant": "base", "weights": False}),
        VariantSpec("large", {"variant": "large", "weights": False}),
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
}
