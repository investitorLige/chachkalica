"""Arch-specific TensorRT ONNX preparation.

Most archs (rtdetr, rfdetr) compile straight from their standard ONNX graph — the
DETR-family top-k is fixed-size, so TensorRT is happy. Those have NO entry here
and ``build_engine`` compiles the standard ``.onnx`` directly.

Archs whose standard ONNX bakes data-dependent NMS/decode DO have an entry: a
``prep(adapter, meta, out_onnx_path)`` callable that re-exports a raw-output graph
and appends the ``EfficientNMS_TRT`` plugin (see ``efficientnms.py``). These need
the torch adapter (hence the ``.pt`` checkpoint), not just the standard ``.onnx``:

  * retinanet, yolox — single-stage; one decode point, chop before the final NMS.
  * fasterrcnn — two-stage; additionally replaces the RPN's variable proposal NMS
    with a fixed top-K selection (so RoIAlign sees a static count) and feeds
    per-class boxes to the plugin. This makes the graph compilable at the cost of
    an approximation of the torch/ONNX outputs (see ``fasterrcnn.py``).
"""

from __future__ import annotations

try:
    from .fasterrcnn import prep_fasterrcnn
    from .retinanet import prep_retinanet
    from .yolox import prep_yolox
except ImportError:  # run flat (cwd on sys.path)
    from trt_export.arch.fasterrcnn import prep_fasterrcnn  # type: ignore
    from trt_export.arch.retinanet import prep_retinanet  # type: ignore
    from trt_export.arch.yolox import prep_yolox  # type: ignore

# arch name -> prep callable. Archs absent here compile from their standard ONNX.
TRT_PREP_REGISTRY = {
    "yolox": prep_yolox,
    "retinanet": prep_retinanet,
    "fasterrcnn": prep_fasterrcnn,
}


def get_trt_prep(arch: str):
    """Return the arch's TRT ONNX-prep callable, or ``None`` for passthrough archs."""
    return TRT_PREP_REGISTRY.get(arch)


# --------------------------------------------------------------------------- FP16 policy
#
# On strongly-typed TRT (>= 11) fp16 comes from casting the ONNX graph, applying
# ONE op-block-list to every arch (see ``fp16_cast.py``). That blanket list is too
# coarse for some archs. Measured top-K L1 of a plain-fp16 engine vs its own fp32
# build (``fp16_diag.py``): yolox 9e-4 and retinanet 5e-3 are clean; but rtdetr
# blows up to ~1.9 (its anchor mask uses ``torch.finfo.max`` ≈ 3.4e38, which
# overflows fp16's 65504 ceiling and, once clamped, wrecks the box decode),
# fasterrcnn to ~0.74, and rfdetr drifts to ~2.5e-2.
#
# ARCH_FP16_OP_BLOCK keeps each arch's fp16-fragile op *types* in fp32 while the
# bulk (backbone/encoder) still runs fp16. UNTRUSTED_FP16 is the safety floor:
# archs still known-bad at fp16 build as fp32 when precision is left to auto, so a
# default build never silently ships a broken engine. An arch graduates out of the
# floor once its keep-list gets it under the parity gate.
ARCH_FP16_OP_BLOCK = {
    # Populated from the fp16_diag --sweep results (minimal set that gets each arch
    # under the parity gate). Empty = still on the fp32 floor below, sweep pending.
    #   rtdetr:     anchor Where(finfo.max) + reference-point / box-coord decode.
    #   fasterrcnn: two-stage box-delta decode + RPN selection math.
    #   rfdetr:     coordinate sigmoid / ms-deform-attn sampling drift (light).
    "rtdetr": [],
    "fasterrcnn": [],
    "rfdetr": [],
}

# arch -> node-name SUBSTRINGS whose nodes stay fp32 (region-level keep-list, robust
# to node re-indexing; op-type blocking is too coarse and the float16 converter
# breaks the graph for many types — see fp16_diag sweep results). Populated from the
# fp16_overflow_probe / node-sweep results.
#   fasterrcnn: box_roi_pool RoIAlign coord scaling overflows fp16 (~8.9e4).
#   rtdetr:     encoder-score + decoder head are precision-fragile (top-k selection).
ARCH_FP16_NODE_BLOCK = {
    "fasterrcnn": [],
    "rtdetr": [],
}

# Archs whose plain-fp16 output isn't trusted yet; an auto-precision build floors
# them to fp32. Empty this per arch once its keep-list recovers fp16 parity.
UNTRUSTED_FP16 = {"rtdetr", "fasterrcnn"}


def get_fp16_op_block(arch: str):
    """Extra ONNX op types to keep in fp32 when casting this arch's graph to fp16."""
    return list(ARCH_FP16_OP_BLOCK.get(arch, []))


def get_fp16_node_block(arch: str):
    """Node-name substrings whose nodes stay fp32 when casting this arch to fp16."""
    return list(ARCH_FP16_NODE_BLOCK.get(arch, []))


def is_fp16_trusted(arch: str) -> bool:
    """False for archs an auto-precision build should floor to fp32 (see above)."""
    return arch not in UNTRUSTED_FP16


__all__ = [
    "TRT_PREP_REGISTRY",
    "get_trt_prep",
    "ARCH_FP16_OP_BLOCK",
    "ARCH_FP16_NODE_BLOCK",
    "UNTRUSTED_FP16",
    "get_fp16_op_block",
    "get_fp16_node_block",
    "is_fp16_trusted",
]
