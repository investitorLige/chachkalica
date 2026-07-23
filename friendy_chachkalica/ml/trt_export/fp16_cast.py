"""Cast an ONNX graph's fp32 tensors to fp16, in-graph.

Needed on **TensorRT >= 11**, whose networks are "strongly typed": the engine's
precision comes from the ONNX graph's own tensor dtypes, not from a builder flag
(TRT 11 removed ``BuilderFlag.FP16`` — see ``builder.py``). To get a genuine FP16
engine there we must hand TRT an fp16 graph.

Design constraints that keep the rest of the system untouched:

* **Graph I/O stays fp32** (``keep_io_types=True``). The engine's input
  ``pixel_values`` and its ``boxes/scores/labels`` outputs remain fp32, so
  Contract A, the ``.meta.json`` sidecar (Contract B), and ``trt_infer`` need no
  changes — only the *interior* compute runs in fp16.
* **Numerically fragile ops stay fp32.** Score/box decode and selection ops
  overflow fp16's 65504 ceiling or lose ranking precision, which wrecks NMS
  output. We reuse the converter's ``DEFAULT_OP_BLOCK_LIST`` (already covers
  ``NonMaxSuppression`` / ``TopK`` / ``RoiAlign`` / ``Range`` / ``Min`` / ``Max``)
  and extend it with ``Exp`` (box-size decode overflows) and ``Softmax``.

The converter is ``onnxruntime.transformers.float16`` (ships with onnxruntime, no
new dependency) with a fallback to ``onnxconverter_common`` if present. If neither
is importable, ``cast_onnx_bytes_to_fp16`` returns ``(None, None)`` and the caller
honestly builds at the graph's native (fp32) precision.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Detection-decode ops the default block list misses but which still misbehave in
# fp16 (exp of raw box-size regression overflows; softmax over logits loses mass).
_EXTRA_FP32_OPS = ["Exp", "Softmax"]


FP16_MAX = 65504.0


def _clamp_large_constants(model, limit: float = FP16_MAX) -> int:
    """Clamp constant/initializer float values beyond fp16 range into [-limit, limit].

    Some archs bake sentinels far outside fp16 — e.g. RT-DETR's anchor mask uses
    ``torch.finfo(float32).max`` (3.4e38) for invalid positions. The float16
    converter clamps *initializers* but misses ``Constant`` *nodes*, so that 3.4e38
    survives the cast, becomes ``inf`` in fp16, and poisons the downstream decode
    (this is what op/node block-listing can't fix — the constant overflows before
    any blocked op sees it). Pre-clamping it to ±65504 keeps the sentinel "large"
    for masking while staying representable. Recurses into subgraphs (If/Loop).
    Returns the number of tensors clamped.
    """
    import numpy as np
    from onnx import numpy_helper

    n = 0

    def _clamp_tensor(t) -> bool:
        if t.data_type != 1:  # onnx.TensorProto.FLOAT
            return False
        arr = numpy_helper.to_array(t)
        if not np.any(np.abs(arr) > limit):
            return False
        clamped = np.clip(arr, -limit, limit).astype(arr.dtype)
        t.CopyFrom(numpy_helper.from_array(clamped, t.name))
        return True

    def _walk(graph):
        nonlocal n
        for init in graph.initializer:
            if _clamp_tensor(init):
                n += 1
        for node in graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.t.data_type == 1:
                        if _clamp_tensor(attr.t):
                            n += 1
            for attr in node.attribute:  # recurse subgraph bodies (If/Loop/Scan)
                if attr.g.ByteSize():
                    _walk(attr.g)
                for g in attr.graphs:
                    _walk(g)

    _walk(model.graph)
    return n


def _match_nodes_by_substring(model, substrings) -> List[str]:
    """Names of graph nodes whose name contains any substring (recurses subgraphs)."""
    subs = [s for s in (substrings or []) if s]
    if not subs:
        return []
    names: List[str] = []

    def _walk(graph):
        for node in graph.node:
            if node.name and any(s in node.name for s in subs):
                names.append(node.name)
            for attr in node.attribute:
                if attr.g.ByteSize():
                    _walk(attr.g)
                for g in attr.graphs:
                    _walk(g)

    _walk(model.graph)
    return names


def _pick_converter(need_node_block: bool):
    """Return the float16 converter module to use.

    NOTE (2026-07-17): per-node fp32 keep-lists don't currently work on rtdetr's and
    fasterrcnn's graphs with EITHER converter — onnxruntime's float16 fork emits an
    invalid graph (duplicate cast output names / toposort failure) and
    onnxconverter_common's crashes in ``remove_unnecessary_cast_node``. So node
    blocking is effectively unsupported today; those archs stay on the fp32 floor
    (see [[trt-fp16-per-arch]]). This selector still prefers onnxconverter_common
    when node blocking is requested and it's importable, in case a fixed release
    lands — otherwise it falls back to onnxruntime (the working no-node-block path).
    onnxconverter_common is NOT a dependency; the common path uses onnxruntime.
    """
    occ = ort = None
    try:
        from onnxconverter_common import float16 as occ  # type: ignore
    except ImportError:
        occ = None
    try:
        from onnxruntime.transformers import float16 as ort
    except ImportError:
        ort = None

    if need_node_block and occ is not None:
        return occ
    if ort is not None:
        return ort
    if occ is not None:
        return occ
    raise ImportError("no float16 converter (need onnxruntime or onnxconverter_common)")


def _convert(model, block_list: List[str], node_block_list=None):
    """Run whichever float16 converter is installed. Raises ImportError if none."""
    f16 = _pick_converter(need_node_block=bool(node_block_list))

    n_clamped = _clamp_large_constants(model)
    if n_clamped:
        print(f"[trt] fp16_cast: clamped {n_clamped} out-of-fp16-range constant(s) to ±{FP16_MAX:.0f}")

    default = list(getattr(f16, "DEFAULT_OP_BLOCK_LIST", []))
    op_block_list = default + [op for op in block_list if op not in default]
    kwargs = dict(keep_io_types=True, disable_shape_infer=False, op_block_list=op_block_list)
    if node_block_list:
        kwargs["node_block_list"] = list(node_block_list)
    casted = f16.convert_float_to_float16(model, **kwargs)
    method = "onnxruntime" if "onnxruntime" in f16.__name__ else "onnxconverter_common"
    return casted, method


def cast_onnx_bytes_to_fp16(
    onnx_bytes: bytes,
    *,
    extra_fp32_ops: Optional[List[str]] = None,
    node_block_list: Optional[List[str]] = None,
    node_block_substrings: Optional[List[str]] = None,
) -> Tuple[Optional[bytes], Optional[str]]:
    """Return ``(fp16_onnx_bytes, method)`` or ``(None, None)`` if uncastable.

    ``method`` names the converter used (for build provenance). ``(None, None)``
    means no fp16 converter was importable — the caller should fall back to an
    honest fp32 build rather than pretend the engine is fp16.

    ``node_block_substrings`` keeps every node whose *name* contains any of the
    given substrings in fp32 — a region-level keep-list robust to node re-indexing
    across re-exports (op-type blocking is too coarse and the converter breaks the
    graph for many types; exact node names are brittle). Used to protect an arch's
    fragile subgraph (e.g. fasterrcnn's ``box_roi_pool`` RoIAlign coord scaling).
    """
    try:
        import onnx
    except ImportError:
        return None, None

    block_list = list(_EXTRA_FP32_OPS)
    if extra_fp32_ops:
        block_list += [op for op in extra_fp32_ops if op not in block_list]

    try:
        model = onnx.load_model_from_string(onnx_bytes)
        nbl = list(node_block_list or [])
        if node_block_substrings:
            matched = _match_nodes_by_substring(model, node_block_substrings)
            print(f"[trt] fp16_cast: keeping {len(matched)} node(s) fp32 matching "
                  f"{list(node_block_substrings)}")
            nbl += [n for n in matched if n not in nbl]
        casted, method = _convert(model, block_list, node_block_list=nbl or None)
        return casted.SerializeToString(), method
    except ImportError:
        return None, None
