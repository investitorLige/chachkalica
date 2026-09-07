"""Arch-specific TensorRT ONNX preparation.

Most archs (ecdet, rtdetr, rfdetr, dfine) compile straight from their standard
ONNX graph — the DETR-family top-k is fixed-size, so TensorRT is happy. Those
have NO entry here and ``build_engine`` compiles the standard ``.onnx`` directly.

Archs whose standard ONNX bakes data-dependent NMS/decode DO have an entry: a
``prep(adapter, meta, out_onnx_path)`` callable that re-exports a raw-output graph
and appends the ``EfficientNMS_TRT`` plugin (see ``efficientnms.py``). These need
the torch adapter (hence the ``.pt`` checkpoint), not just the standard ``.onnx``:

  * retinanet, yolox — single-stage; one decode point, chop before the final NMS.
  * fasterrcnn — two-stage; additionally replaces the RPN's variable proposal NMS
    with a fixed top-K selection (so RoIAlign sees a static count) and feeds
    per-class boxes to the plugin. This makes the graph compilable at the cost of
    an approximation of the torch/ONNX outputs (see ``fasterrcnn.py``).

This is orthogonal to whether an arch can be built with a batch profile above 1
— see ``BATCH_AWARE_ARCHS``/``is_batch_aware`` below. ``has_trt_prep`` answers
"does compiling this arch's engine need surgery on the graph", not "does its
graph carry a real batch axis".
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
# output, on most of the 8 images; images that did match still scored 0.04-0.13, mostly
# past the gate. Detection counts diverge too. Not the random-init selection-tie artifact
# that misled us on rtdetr/fasterrcnn — the checkpoint is trained and its scores are well
# separated.
#
# ROOT CAUSE, confirmed empirically (not the constant-overflow theory this comment used
# to carry — that one is REFUTED, see below): a tensor-level fp32-vs-fp16 diff probe
# (cast the real trained ONNX graph with ``fp16_cast.cast_onnx_bytes_to_fp16``, expose
# every intermediate tensor, diff fp32 vs fp16-cast onnxruntime execution on a real
# image) shows the backbone and the RT-DETR-style encoder are fp16-clean (max abs diff
# 0.75 / 0.24 across ~1000 tensors each — ordinary rounding). The divergence originates
# and then compounds entirely inside ``TransformerDecoder``: max abs diff 246 on a
# tensor whose fp32 magnitude is only ~145 — i.e. genuine numeric blowup, not rounding —
# traced to the ``self_attn`` (``nn.MultiheadAttention``) ``MatMul`` nodes inside
# ``TransformerDecoderLayer`` (decoder.py), present in every one of the 4 decoder layers
# and dominating every other divergent op downstream of it (Softmax/Sigmoid outputs show
# up to ~1.0 diff too, but that is their *inputs* having already blown up, not the
# saturating ops themselves misbehaving).
#
# fp16_cast's "clamped 1 out-of-fp16-range constant(s) to ±65504" message (still logged
# for this graph) refers to ``decoder.anchors`` (``_generate_anchors``'s ``torch.inf``
# sentinel at 1264 invalid/near-edge grid positions, decoder.py ~line 639) and IS real,
# but is a red herring for this failure: swapping that whole buffer between ``inf`` and
# the fp16-cast's ``65504`` sentinel, in plain fp32 torch on 40 real images, produced
# bit-identical ``pred_logits``/``pred_boxes`` every time — those positions are never
# selected by the encoder's top-k (a trained model's real, feature-driven scores always
# beat the bias-only score an invalid, zeroed-memory position gets), so their anchor
# value, whatever it is, never reaches any output. (``_get_decoder_input`` in decoder.py
# now also masks the encoder score directly at those positions — see the comment there —
# which makes that non-selection a structural guarantee instead of a property of this one
# checkpoint's bias; verified bit-identical against the unmasked behavior on 40 real
# images. Kept as a robustness cleanup; it does not touch today's actual bug.)
#
# OPTION B ATTEMPTED, DID NOT CLEAR THE GATE (op-block-list route): the diff probe's own
# evidence says "keep MatMul fp32" is the natural candidate. Passing
# ``extra_fp32_ops=["MatMul"]`` through ``fp16_cast`` (the ``ARCH_FP16_OP_BLOCK``
# mechanism) does not produce a usable engine at all — onnxruntime's
# ``float16.convert_float_to_float16`` (the only converter installed; ``onnxconverter_
# common`` is not present in this image) emits a graph with duplicate ``*_cast_to_fp32``
# output names once MatMul (used pervasively, including in the fp16-clean encoder) is
# added to the op block list, and TensorRT's parser then fails topological sort: ``ERROR:
# Output name is not unique: .../self_attn/Transpose_output_0_cast_to_fp32`` / ``Failed to
# sort the model topologically``. Same class of converter bug already documented above as
# making node-level fp32 keep-lists "effectively unsupported" for rtdetr/fasterrcnn — here
# it hits op-*type* blocking instead, because the one op type that actually needs to stay
# fp32 (self-attention's MatMul) is also used everywhere else in the graph that is
# fp16-clean, and there is no op-type-only way to scope the keep-list to just the
# decoder's self-attention.
#
# TWO MORE ROUTES TRIED (a later pass), NEITHER SHIPPABLE:
#
# (1) A cheap check first, per a known TensorRT numeric-stability gotcha (NVIDIA/
#     TensorRT#4333): is the ``1/sqrt(head_dim)`` attention scale applied to Q *before* the
#     QK^T matmul, or to the product *after*? Inspected the exported graph directly —
#     ``.../self_attn/Mul_1`` (``* 0.204124...``, i.e. ``1/sqrt(24)``) multiplies Q's
#     tensor and its output feeds ``MatMul_3`` (the QK^T matmul) as an input. So the scale
#     is already applied PRE-matmul, the numerically-correct placement — nothing to move,
#     this was not the bug.
#
# (2) NVIDIA ModelOpt AutoCast (``nvidia-modelopt[all]``, ``python -m
#     modelopt.onnx.autocast``), calibrated on real images. Its default ``--data_max=512``
#     heuristic (flag activations whose magnitude is too large) does not catch this at
#     all — the fp32 magnitudes here are only ~145, well under 512; the failure is fp16
#     *accumulation* precision inside the matmul, not raw output magnitude. Adding an
#     explicit ``--nodes_to_exclude`` regex for the decoder self-attention region does get
#     it to skip exactly those nodes (converted 85.38% of the graph to fp16 either way) —
#     but the tool's OWN output graph fails ``onnx.checker.check_model(full_check=True)``:
#     a type inconsistency at ``/model/backbone/backbone/rope_embed/Mul`` (unrelated to our
#     exclude regex — the same failure appears with no excludes at all, and survives
#     ``--use_standalone_type_inference``). Under onnxruntime the malformed graph happens
#     to run anyway (lenient type handling); under TensorRT it builds "successfully" but
#     every output is NaN. Not shippable — this is a bug in ModelOpt's converter for this
#     architecture's RoPE embedding module, not something we can route around from here.
#
# (3) Hand-written ``onnx_graphsurgeon`` Cast patch, bypassing both auto-casters: start
#     from ``fp16_cast``'s own default (no exclusions) cast — which builds fine, it's just
#     wrong — and directly rewire the 2 matmuls-per-layer × 4 layers (``self_attn/MatMul_3``
#     = Q_scaled @ K^T, ``self_attn/MatMul_4`` = attn @ V; ``Softmax`` in between is already
#     fp32 via the default op block list, confirmed by inspecting the cast graph's node
#     dtypes — it was just being fed an already-blown-up fp16 input) with explicit
#     Cast-to-fp32 on their inputs and one Cast-back-to-fp16 after, using unique node names.
#     This graph is semantically correct — ``onnx.checker`` passes, and onnxruntime CPU
#     execution matches the fp32 reference exactly. Under TensorRT it is UNRELIABLE: the
#     very same patched ONNX, rebuilt fresh with nothing else changed, non-deterministically
#     produced either correct output or all-NaN output across repeated builds (TensorRT's
#     tactic autotuner evidently sometimes picks a numerically broken kernel for this exact
#     fp32-island-inside-an-otherwise-fp16-graph MatMul pattern). Setting
#     ``BuilderFlag.STRICT_NANS`` (forces honest NaN propagation instead of TensorRT's
#     default undefined-but-often-silent behavior) made the NaN appear on every single
#     rebuild (5/5) — i.e. the "sometimes fine" builds without that flag were not actually
#     correct, they were just not visibly propagating a NaN that was already there some of
#     the time via whatever tactic got picked. TRT 11 has no per-layer precision override
#     API to work around this more directly (``setPrecision``/``setOutputType`` were
#     removed; confirmed via TRT 11 migration docs), and ``TacticSource`` in TRT 11 no
#     longer exposes cuBLAS/cuBLASLt/cuDNN to restrict away from a suspect kernel (only
#     ``EDGE_MASK_CONVOLUTIONS``/``JIT_CONVOLUTIONS`` remain). Not shippable — an engine
#     that may silently build totally broken depending on autotuner luck is strictly worse
#     than staying on the fp32 floor.
#
# So: no clean Option A or Option B closes this gate today, across every route tried
# (op-block-list casting, ModelOpt AutoCast, hand-written graph surgery). ecdet stays on
# the fp16 safety floor. ``_setup_ecdet`` remains wired into ``fp16_diag.py`` and
# ``fp16_overflow_probe.py`` (though note the latter's random-init fixture does not
# reproduce this failure — the blowup needs trained-scale decoder activations) for
# whoever revisits this: candidates are a newer TensorRT release (if the strongly-typed
# mixed-precision MatMul bug above gets fixed), a fixed ``onnxconverter_common`` release
# (its node-block path is untested here — it was never installed, not just unused), a
# fixed ModelOpt AutoCast release for graphs using a RoPE embedding module like this one,
# or reworking upstream's decoder self-attention (e.g. to TensorRT's native fused
# attention op, which per NVIDIA's docs forces fp32 QK^T/softmax·V accumulation
# internally — but only when TensorRT recognizes the pattern as its fusable Attention op,
# which a generic ``nn.MultiheadAttention`` export's plain MatMul/Div/Softmax/MatMul does
# not trigger).
#
# dfine is deliberately NOT floored, and that is a judgement about the evidence rather
# than a gate run that passed — read this before "fixing" it either way.
#
# D-FINE compiles as a passthrough arch and its fp32 engine matches the torch adapter,
# so only fp16 was ever in question. It could not be settled here: the gate above
# demands TRAINED weights, huggingface.co is unreachable from this network (so the
# upstream ``ustc-community/dfine-*-coco`` checkpoints cannot be fetched), and no
# D-FINE has been trained in this repo yet. The stand-in was a 40-epoch from-scratch
# fit on 400 local single-class images — confident enough to look valid (top scores
# 0.9+, detections at 0.3) — and on it the gate reported:
#
#     worst confident L1(fp16 vs fp32) = inf   (gate 0.05)  -> FAIL
#
# fp16 emitted ZERO detections on 6 of 8 images where fp32 emitted 1-5.
#
# That number says nothing about D-FINE. The CONTROL is what settles it: rtdetr —
# which this comment block already records as fp16-clean at ~1e-4 on real trained
# weights — was trained by the SAME recipe on the SAME 400 images and scored
# identically, ``inf``, with fp16 emitting zero detections on all 8 images. An arch
# known to be fp16-good fails this fixture, so the fixture is measuring the fixture.
#
# The mechanism, for whoever revisits: fp16 UNDERFLOW — the opposite direction from the
# overflow ``fp16_overflow_probe.py`` hunts, and the opposite direction from ecdet's
# failure below. TensorRT flushes subnormals, so any tensor living entirely under fp16's
# smallest normal (6.1e-5) becomes exactly zero. Both fixtures hit that, in different
# places and with different symptoms — note the two are distinguished by whether a
# tensor is *nonzero but subnormal* (annihilated) or *exactly zero* (harmless in fp16;
# an easy thing to miscount when censusing a graph):
#
#   * dfine — 234 of 2040 consumed float activation tensors are annihilated, whole
#     Conv/BN/Relu chains inside the HGNetV2 backbone decaying 5.7e-5 -> 4.3e-5 ->
#     2.7e-5 -> 1.2e-5, i.e. near-dead branches. The signal never reaches the head:
#     scores collapse ~25x (0.886 -> 0.036) while boxes stay plausible and NOTHING is
#     NaN. An untrained backbone (with BN structurally frozen) has such branches; a
#     pretrained one should not.
#   * rtdetr — NOT the same thing, and this is why "both failed, so it's the fixture"
#     needed checking rather than asserting. Its graph has essentially no annihilated
#     activations (12 tensors, all ``Constant``s at exactly 1e-5: the
#     ``inverse_sigmoid`` epsilon). 1e-5 is subnormal in fp16, so it flushes to zero
#     and the clamp that exists precisely to stop ``log(0)`` stops clamping — its fp16
#     engine returns NaN, emitting 0 rows even at threshold 0.0. Trained reference
#     points evidently never sit at exactly 0/1 so the clamp never engages (hence
#     rtdetr's clean ~1e-4 on real weights); an undertrained model saturates them and
#     it does.
#
# Same root cause, different structures, and neither is an architecture defect — which
# is the whole point: an undertrained checkpoint breaks fp16 in ways that say nothing
# about the arch.
#
# NOT CLEARED, ONLY UNTESTED — the one risk this measurement could not look at. ecdet's
# failure (below) lives in the decoder's self-attention, and ecdet's decoder was
# explicitly modeled on D-FINE's: ``DFineSelfAttention`` is structurally the same block
# (separate q/k/v/o projections, ``scaling = head_dim**-0.5``, position embeddings on
# Q/K only). The ecdet note also says that blowup needs TRAINED-SCALE decoder
# activations and does not reproduce on a random-init fixture. Here the backbone
# underflow destroys the signal upstream of the decoder, so the decoder never saw
# trained-scale activations either way — every decoder-stage number measured on this
# fixture is downstream junk. So "dfine does not have ecdet's problem" is NOT
# established; only "the failure actually observed is not ecdet's" is.
#
# So dfine keeps ``precision="auto"`` -> fp16, on the same footing as rtdetr, its
# structural twin throughout this codebase (same adapter shape, same exporter, same
# passthrough TRT path, same square static profile). TO SETTLE IT PROPERLY — both the
# underflow question and the ecdet-decoder question at once — point the gate at the
# first real D-FINE checkpoint, one fine-tuned from ``ustc-community/dfine-*`` weights
# per model_specs' WEIGHTS_CATALOG, and add "dfine" here if it genuinely fails:
#
#     python -m friendy_chachkalica.ml.trt_export.trained_fp16_gate \
#         --checkpoint <run>/best.pt --images <val images> --hw 640x640
#
# NOTE this floor only binds ``precision="auto"``. The admin's TensorRT export form
# passes fp16/fp32 explicitly, so an operator who picks FP16 there still gets an fp16
# ecdet engine — and a degraded one.
#
# ─────────────────────────────────────────────────────────────────────────────
# SETTLED 2026-08-18 — dfine IS floored after all, measured exactly the way the
# block above demanded: real ``ustc-community`` COCO weights (now cached offline,
# see the HF cache note in docker-compose.yml), 24 real people photos, static 640².
# The from-scratch fixture that produced the earlier void verdict is not involved.
#
#   dfine-medium-coco          worst confident L1 = inf  (gate 0.05)  -> FAIL
#   dfine-large-obj2coco-e25   worst confident L1 = inf  (gate 0.05)  -> FAIL
#   rtdetr_r50vd (CONTROL)     worst confident L1 = 0.0081            -> PASS
#
# The control is the load-bearing part: rtdetr — structural twin, same exporter, same
# passthrough path, same profile — ran on the SAME images at the SAME threshold and was
# count-stable on every one (torch == fp32 == fp16), worst L1 0.008. So this is a real
# per-arch difference, not a fixture or threshold artifact.
#
# MECHANISM — it is the SCORES, not the boxes, and not underflow. Pulling every
# detection down to score 0.05 and pairing them label-matched (box L1 ~0.003-0.02 px, so
# the pairing is certain — these are the same detections, re-scored):
#   * boxes are essentially exact: median box L1 0.003 px, worst 0.01 px among
#     confident detections. Localization is untouched.
#   * scores swing wildly. dfine-medium: 33% of the 148 detections at fp32>=0.30 move by
#     more than 0.10, and 56 of them (38%) cross below 0.30 — e.g. 0.570 -> 0.077,
#     0.616 -> 0.169, and one going the other way, 0.324 -> 0.925. rtdetr on the same
#     images: 0 of 48, zero crossings.
#   * dfine-large-obj2coco-e25 is worse, and worst where it matters most: median
#     |Δscore| is 0.67 in the >=0.70 confident band (max 0.82), and the fp16 engine
#     returns ZERO detections at 0.3 on 5 of 8 gated images while fp32 finds 2-6.
# So an fp16 D-FINE engine keeps drawing boxes in the right places while its confidences
# become unreliable — detections silently fall under any operating threshold. That is a
# worse failure mode than a visible crash, which is why "auto" must not pick it.
#
# NOT diagnosed to a node yet. This is score-side instability, consistent with the
# ecdet-style decoder blowup this block flagged as the untested risk (``DFineSelfAttention``
# is the same block ecdet's decoder was modeled on) — but that is now a hypothesis with
# supporting evidence, not a measurement. The tensor-level fp32-vs-fp16 diff probe used
# on ecdet is the tool to confirm it. Note the earlier from-scratch UNDERFLOW finding is
# NOT what is happening here: these are trained weights with a healthy backbone.
#
# ─────────────────────────────────────────────────────────────────────────────
# rtmo — 2026-09-02. On the floor for a different reason than the two above: this
# arch has no trustworthy fp16 engine because it has no fp16 GRAPH. rtmo is not a
# friendy-trained arch — it is an mmpose/mmdeploy ``end2end.onnx`` export carried in
# as-is (see ``onnx_infer/arch/rtmo.py``), and both casters choke on it rather than
# producing something TensorRT will parse:
#
#   * ``fp16_cast``'s onnxruntime converter emits duplicate ``*_cast_to_fp32`` output
#     names, and the parser fails topological sort — the same converter bug this file
#     already documents for op-blocked rtdetr/ecdet graphs.
#   * ``onnxconverter_common``'s cast gets past that and then hits a genuine
#     ``Float``/``Half`` mismatch on an ElementWise ``Mul`` two nodes later, inside the
#     graph's own baked NMS/decode machinery.
#
# Neither is a tactic-search failure that a retry fixes, and a hand-patched fp16 graph
# was not worth chasing: the fp32 engine already runs a whole 640² frame in ~9ms, far
# inside any camera's frame rate, and this arch is only ever run at batch 1 on a whole
# frame. Without this entry ``auto`` would request fp16, waste a cast + a failed build
# on every rebuild, and land on fp32 anyway via ``builder.py``'s fallback.
# ─────────────────────────────────────────────────────────────────────────────
UNTRUSTED_FP16: set[str] = {"ecdet", "dfine", "rtmo"}

# ─────────────────────────────────────────────────────────────────────────────
# CAST BACKEND — 2026-08-18. The floor above says "this arch's fp16 engine is not
# trustworthy". That was measured with the only fp16 mechanism this repo had:
# ``fp16_cast.py``'s blanket cast of the whole graph (minus an op block-list).
# There is a second mechanism now — NVIDIA ModelOpt AutoCast (``modelopt_cast.py``),
# the route NVIDIA's own TensorRT 10.x→11.x migration guide points at for
# strongly-typed builds — and for **dfine it reproduces the torch model where the
# blanket cast destroys it**, measured on real ``ustc-community`` COCO weights over
# the same 24 people photos at a static 640² profile (torch fp32 as the reference,
# not the fp32 engine — see below for why that distinction turned out to matter):
#
#   runtime                dets>=0.3  crossed  |Δscore| med  box px  mAP50 vs torch
#   torch fp32 / ORT fp32    199/200      0/-       0.001      0.02   1.000
#   TRT fp16 graph-cast          109       98       0.068      2.30   0.720
#   TRT bf16 AutoCast            118      125       0.146      8.44   0.712
#   TRT fp16 AutoCast            194       13       0.004      0.48   0.996
#
# Mechanism, from AutoCast's own log: its ``data_max`` rule keeps exactly the
# nodes whose magnitudes make low precision unsafe in fp32 — ``decoder/Div`` and
# ``decoder/Log`` (the ``inverse_sigmoid`` 1e-5 epsilon, |x| up to 1e5), the
# ``encoder_attn`` level-index arithmetic (8000), the ``gateway`` counters (513)
# and the wrapper's flattened top-k index math (23979) — while the blanket cast
# pushes all of them through fp16. That is the whole difference: same weights,
# same profile, same TensorRT.
#
# ⚠ The bigger surprise, and the reason the reference above is torch and not the
# fp32 engine: a **TensorRT FP32 engine of this graph is itself lossy on D-FINE**
# — 155 detections vs torch's 199, |Δscore| median 0.030, mAP50 0.818, with the
# whole score distribution shifted DOWN (top score 0.906 -> 0.867). TF32 is not
# the cause (a TF32-disabled build scores the same, and is 15% slower, so the
# flag did take effect). onnxruntime on the same graph matches torch to ±0.002,
# so the graph is faithful and it is TensorRT's compilation of it that drifts.
# So the historical "dfine fp16 vs the fp32 engine" verdict was measured against
# a reference that had already lost most of the accuracy — judge low precision
# against torch or ORT for this arch, not against its own fp32 engine.
# ``trained_fp16_gate.py`` now measures its own reference every run and takes
# ``--reference torch``; run against the fp32 engine it FAILS the AutoCast engine
# on per-image lines like ``dets torch=4 fp32=2 fp16=5`` — i.e. it fails the
# candidate for detections the *reference* lost.
#
# BE HONEST ABOUT WHAT PASSED: the AutoCast engine does **not** pass that gate
# against torch either (worst L1 = inf on both dfine checkpoints) — but neither
# does the FP32 engine, on the same images, and the gate now says so. ``inf``
# means one confident torch detection had no same-label twin at threshold 0.3;
# on dfine-medium that is a single image where torch finds 2 and *every* TensorRT
# engine finds 1. Per image, the AutoCast engine sits at 0.0008-0.12 against
# torch (median ~0.01) where the blanket cast and bf16 are at inf on most.
# The decision above rests on the aggregate agreement (mAP50 0.996 / 0.983 vs
# torch, detection counts 194/199 and 410/410), not on a gate pass.
# The gate's own control is intact: rtdetr fp16 PASSES at 0.008 with its fp32
# reference 0.0026 from torch, and rtdetr **bf16 FAILS at 0.199** — same images,
# same gate, so the bf16 verdict below is not a threshold artifact.
#
# ARCH_CAST_BACKEND is that finding as policy: for these archs an fp16 build must
# go through AutoCast, and asking for graph_cast is an explicit opt-in to the
# known-bad engine. ecdet is deliberately NOT here — AutoCast was tried on it
# (see the notes above) and produced a graph that failed ``onnx.checker`` and an
# engine that returned NaN; nobody has re-tested that against modelopt 0.46.
ARCH_CAST_BACKEND = {"dfine": "autocast"}


def get_cast_backend(arch: str) -> str:
    """Which low-precision cast mechanism this arch's fp16 build must use."""
    return ARCH_CAST_BACKEND.get(arch, "graph_cast")


def resolve_auto_precision(arch: str, *, modelopt_available: bool) -> tuple[str, str]:
    """``precision="auto"`` -> ``(precision, cast_backend)`` for this arch.

    Three outcomes:

    * fp16-trusted arch -> ``("fp16", "graph_cast")``, unchanged behaviour.
    * an arch on the fp16 floor that AutoCast rescues (``ARCH_CAST_BACKEND``) ->
      ``("fp16", "autocast")`` when ModelOpt is installed, and the fp32 floor
      when it is not. "auto" means "the best precision this arch is *known* to
      survive", so falling back to fp32 here is the correct conservative answer,
      not a silent downgrade of a bf16/fp16 request (those raise — see
      ``builder.py``).
    * everything else on the floor -> ``("fp32", "auto")``.
    """
    if is_fp16_trusted(arch):
        return "fp16", get_cast_backend(arch)
    backend = ARCH_CAST_BACKEND.get(arch)
    if backend == "autocast" and modelopt_available:
        return "fp16", "autocast"
    return "fp32", "auto"


def get_fp16_op_block(arch: str):
    """Extra ONNX op types to keep in fp32 when casting this arch's graph to fp16."""
    return list(ARCH_FP16_OP_BLOCK.get(arch, []))


def get_fp16_node_block(arch: str):
    """Node-name substrings whose nodes stay fp32 when casting this arch to fp16."""
    return list(ARCH_FP16_NODE_BLOCK.get(arch, []))


def is_fp16_trusted(arch: str) -> bool:
    """False for archs an auto-precision build should floor to fp32 (see above)."""
    return arch not in UNTRUSTED_FP16


# --------------------------------------------------------------------------- batch policy
#
# Whether an arch's exported graph carries a real batch axis through to its
# detection outputs — the thing that makes a TensorRT optimization profile with
# max_batch > 1 actually do anything, rather than build an unused wider profile
# on top of a graph that still only ever processes one image.
#
# * yolox — ``arch/yolox.py``'s ``YOLOXRaw`` genuinely threads ``B`` through
#   before EfficientNMS, whose own plugin output is natively batched too.
# * ecdet, rtdetr, rfdetr, dfine — DETR-family, fixed top-k per image (see each
#   ``onnx_export/arch/*.py`` module docstring); the wrapper does the top-k
#   per row of the batch instead of indexing batch away.
#
# retinanet and fasterrcnn are DELIBERATELY NOT here despite having a
# ``trt_export/arch/`` prep module (``has_trt_prep`` is about EfficientNMS
# surgery, not batch-safety — see the module docstring). Their raw-export
# wrappers (``arch/retinanet.py``'s ``RetinaRaw``, ``arch/fasterrcnn.py``'s
# ``FasterRCNNRaw``) hardcode ``pixel_values[0]``/``.unsqueeze(0)`` — every
# batch row past 0 is silently ignored, and TensorRT would report the first
# image's detections for the whole batch (the exact failure mode
# ``chachak.bundle_export.cli._batchable_in_trt`` warns about for passthrough
# archs, just not actually averted here). Add them once that wrapper is
# rewritten to carry ``B`` through like ``YOLOXRaw`` does.
BATCH_AWARE_ARCHS: set[str] = {"yolox", "ecdet", "rtdetr", "rfdetr", "dfine"}


def is_batch_aware(arch: str) -> bool:
    """Whether ``arch``'s engine can be built with (and correctly decode) a
    TensorRT batch profile wider than 1. See ``BATCH_AWARE_ARCHS`` above."""
    return arch in BATCH_AWARE_ARCHS


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
    "BATCH_AWARE_ARCHS",
    "is_batch_aware",
]
