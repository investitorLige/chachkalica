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

import importlib

# arch name -> (module, callable). Archs absent here compile from their standard
# ONNX. The prep callables are imported ON DEMAND, not here: each one re-exports a
# graph from the torch model and so imports torch, while everything else in this
# module (the FP16 policy tables, and the question "does this arch need a prep at
# all?") is pure data. Importing them eagerly made the whole module torch-only,
# which put it out of reach of the slim build node (see buildnode/) — a node that
# compiles a ready-made graph still needs the policy tables.
_PREP_MODULES = {
    "yolox": ("yolox", "prep_yolox"),
    "retinanet": ("retinanet", "prep_retinanet"),
    "fasterrcnn": ("fasterrcnn", "prep_fasterrcnn"),
}


def has_trt_prep(arch: str) -> bool:
    """Whether ``arch`` needs an EfficientNMS re-export before it can be compiled.

    The torch-free half of :func:`get_trt_prep`. Callers that only need the yes/no
    — "can this arch compile straight from its standard ONNX?" — must use this;
    calling ``get_trt_prep`` for the answer drags torch in to get it.
    """
    return arch in _PREP_MODULES


def _load_prep(arch: str):
    module_name, attr = _PREP_MODULES[arch]
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ImportError:  # run flat (cwd on sys.path)
        module = importlib.import_module(f"trt_export.arch.{module_name}")
    return getattr(module, attr)


def get_trt_prep(arch: str):
    """Return the arch's TRT ONNX-prep callable, or ``None`` for passthrough archs.

    Importing the callable needs torch — see :func:`has_trt_prep` when all you want
    is whether one exists.
    """
    if not has_trt_prep(arch):
        return None
    return _load_prep(arch)


def __getattr__(name: str):
    # TRT_PREP_REGISTRY stays available for anything that wants the whole mapping,
    # but materializing it imports torch, so it is built only if actually touched.
    if name == "TRT_PREP_REGISTRY":
        registry = {arch: _load_prep(arch) for arch in _PREP_MODULES}
        globals()[name] = registry
        return registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- FP16 policy
#
# On strongly-typed TRT (>= 11) fp16 comes from casting the ONNX graph, applying
# ONE op-block-list to every arch (see ``fp16_cast.py``). ``fp16_diag.py`` builds each
# arch from a RANDOM-INIT fixture and once reported huge fp16 parity L1 for the
# selection-heavy archs (rtdetr ~1.9, fasterrcnn ~0.84). Those were a MEASUREMENT
# ARTIFACT, not a real fp16 defect: with untrained weights every score is ~tied, so the
# top-K / NMS selection is a coin-flip that any fp16 perturbation flips. On a TRAINED
# checkpoint (well-separated scores → stable selection) fp16 parity collapses to the
# gate — measured via ``trained_fp16_gate.py``: fasterrcnn 0.012-0.035, rtdetr ~1e-4.
# So do NOT trust fp16_diag's random-init L1 to floor an arch; gate on trained weights.
# (yolox 9e-4 / retinanet 5e-3 / rfdetr 2.5e-2 were fine even random-init.)
#
# ARCH_FP16_OP_BLOCK can still keep an arch's fp16-fragile op *types* in fp32 while the
# bulk runs fp16. It is empty for every arch — no arch needs a keep-list once judged on
# trained weights. UNTRUSTED_FP16 (below) is now empty too; see its note.
ARCH_FP16_OP_BLOCK = {
    # Empty: no arch needs an fp32 op keep-list once fp16 is judged on trained weights
    # (the random-init L1s that motivated these were selection-tie artifacts, above).
    "rtdetr": [],
    "fasterrcnn": [],
    "rfdetr": [],
}

# arch -> node-name SUBSTRINGS whose nodes stay fp32 (region-level keep-list, robust
# to node re-indexing). Empty for every arch: kept as a mechanism, but no arch needs it.
# (fasterrcnn's box_roi_pool RoIAlign coord Mul does overflow fp16 ~8.9e4, but that value
# only picks an integer FPN level via sqrt→log2→floor, so the overflow is numerically
# harmless — confirmed it does not affect parity.)
ARCH_FP16_NODE_BLOCK = {
    "fasterrcnn": [],
    "rtdetr": [],
}

# Archs an auto-precision build floors to fp32 because their plain-fp16 output isn't
# trusted. Now EMPTY: rtdetr and fasterrcnn were floored on fp16_diag's random-init L1
# (1.9 / 0.84), which turned out to be a selection-tie artifact — on trained weights
# their fp16 parity is at the gate (rtdetr ~1e-4, fasterrcnn 0.012-0.035, verified via
# ``trained_fp16_gate.py``). ``precision="auto"`` now requests fp16 for every arch. If a
# future arch is genuinely fp16-fragile on TRAINED weights (prove it with the gate, not
# random-init diag), add it back here.
#
# CAVEAT — fasterrcnn fp16 needs a STATIC profile. Its EfficientNMS graph's transform
# region computes output shapes from the input H/W; with a DYNAMIC profile (min!=max, the
# ``resize_mode:none`` default from profile.py, 64/640/1024) the fp16 build finds no
# tactic for those shape ops and build_engine falls back to fp32 (a log, not an error).
# fp16 builds cleanly at any FULLY-STATIC profile (min==opt==max, e.g. 320²/640² — both
# verified). So to actually ship fasterrcnn fp16, pin the size (min_hw==opt_hw==max_hw).
# rtdetr is unaffected: it exports ``resize_mode: square``, which profile.py already
# turns into a fully-static min==opt==max profile. TODO: consider a
# static/narrow fasterrcnn default in profile.py so auto-fp16 materializes unpinned.
#
# ecdet IS genuinely fp16-fragile on trained weights — measured the way this comment
# demands, with ``trained_fp16_gate.py`` on the upstream COCO ECDet-S checkpoint over 8
# real images at a static 640² profile (so the fasterrcnn shape-op caveat above does not
# apply; ecdet exports ``resize_mode: square``, already static):
#
#     worst confident L1(fp16 vs fp32) = inf   (gate 0.05)  -> FAIL
#
# ``inf`` means a confident fp32 detection had no same-label twin in the fp16 engine's
# output, on 5 of 8 images; the 3 images that did match scored 0.052/0.077/0.088, all
# past the gate. Detection counts diverge too (fp32=3 vs fp16=5 on one image). This is
# not the random-init selection-tie artifact that misled us on rtdetr/fasterrcnn — the
# checkpoint is trained and its scores are well separated. ``fp16_cast`` also reports
# "clamped 1 out-of-fp16-range constant(s) to ±65504" for this graph, so there is a
# concrete overflow to chase: ``_setup_ecdet`` is wired into ``fp16_diag.py`` and
# ``fp16_overflow_probe.py`` for whoever picks it up. Until then auto floors to fp32.
#
# NOTE this floor only binds ``precision="auto"``. The admin's TensorRT export form
# passes fp16/fp32 explicitly, so an operator who picks FP16 there still gets an fp16
# ecdet engine — and a degraded one.
UNTRUSTED_FP16: set[str] = {"ecdet"}


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
    "has_trt_prep",
    "ARCH_FP16_OP_BLOCK",
    "ARCH_FP16_NODE_BLOCK",
    "UNTRUSTED_FP16",
    "get_fp16_op_block",
    "get_fp16_node_block",
    "is_fp16_trusted",
]
