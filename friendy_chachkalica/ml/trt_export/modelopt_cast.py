"""Cast an ONNX graph to a mixed FP32/BF16 (or FP32/FP16) graph with NVIDIA
ModelOpt **AutoCast** — the precision route NVIDIA documents for TensorRT >= 11.

Why this exists next to ``fp16_cast.py``:

* TensorRT 11 networks are *strongly typed*. The builder no longer takes a
  precision flag (``--fp16``/``--bf16`` and ``BuilderFlag.FP16``/``BF16`` are
  gone — verified on the installed 11.1.0.106: ``trt.BuilderFlag`` has neither),
  so the engine's arithmetic precision is whatever the ONNX graph's tensor
  dtypes say it is. NVIDIA's own 10.x→11.x migration guide names the
  replacement: "Use ModelOpt AutoCast to produce a mixed-precision model before
  building."
* ``fp16_cast`` casts **everything** except a fixed op block-list. AutoCast
  instead *classifies* nodes — anything whose measured I/O magnitude exceeds
  ``data_max`` (or whose initializers exceed ``init_max``) is kept in FP32 —
  and inserts the Cast nodes around the boundary. That mixed result is the
  point: "correct mixed BF16 execution", not every tensor forced to BF16.
* There is no BF16 path in ``fp16_cast`` at all: the converters it uses
  (``onnxruntime.transformers.float16`` / ``onnxconverter_common.float16``) only
  emit FP16. ``onnx.TensorProto.BFLOAT16`` graphs come from AutoCast here.

**Opset**: AutoCast raises the graph to opset 22 for BF16 (13 for FP16) because
the older opsets simply do not admit ``bfloat16`` for most ops. Our exporters
emit opset 17 (``onnx_export/common.py``), so a BF16 conversion *always* carries
an opset upgrade — see ``cast_result.opset_before/opset_after``, which the build
provenance records rather than leaving implicit.

**I/O stays FP32** (``keep_io_types=True`` by default), exactly like
``fp16_cast``: ``pixel_values`` in and ``boxes/scores/labels`` out remain
float32, so Contract A, the ``.meta.json`` sidecar and ``trt_infer`` need no
changes and an FP32/FP16/BF16 engine trio is directly comparable.

ModelOpt is **not** a hard dependency of this repo (it pulls onnxruntime-gpu,
onnxscript and friends). When it is missing this module raises rather than
falling back to a lower-effort cast — a BF16 request that quietly produced an
FP16 or FP32 engine is the one outcome worse than a failed build.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
from typing import Dict, List, Optional

# onnx.TensorProto enum values, spelled out so this module can talk about dtypes
# without importing onnx at module scope (the build node imports this file for
# its policy only).
_ONNX_FLOAT = 1
_ONNX_FLOAT16 = 10
_ONNX_BFLOAT16 = 16

_LOW_PRECISION_ONNX_TYPE = {"fp16": _ONNX_FLOAT16, "bf16": _ONNX_BFLOAT16}

# AutoCast's own documented minimum opsets ("22 for BF16, 13 for FP16").
_MIN_OPSET = {"fp16": 13, "bf16": 22}


class _CapturingHandler(logging.Handler):
    """Keep ModelOpt's own "Converted N/M nodes (P%)" line for the provenance.

    ModelOpt counts every node in the graph; the local census below counts only
    nodes it can type. Recording both means the sidecar carries NVIDIA's own
    accounting next to ours instead of quietly disagreeing with it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # pragma: no cover - never break a build on logging
            pass

    def conversion_line(self) -> Optional[str]:
        for message in reversed(self.messages):
            if re.search(r"Converted \d+/\d+ nodes", message):
                return message.strip()
        return None


class ModelOptUnavailable(RuntimeError):
    """NVIDIA ModelOpt is not importable, so no AutoCast conversion can be done."""


class AutoCastError(RuntimeError):
    """AutoCast ran but did not produce a usable graph."""


@dataclass
class CastResult:
    """What the conversion produced, for the ``.engine.json`` provenance sidecar."""

    onnx_bytes: bytes
    precision: str
    method: str
    opset_before: int
    opset_after: int
    modelopt_version: str
    nodes_total: int
    nodes_low_precision: int
    keep_io_types: bool
    nodes_unclassified: int = 0
    nodes_to_exclude: List[str] = field(default_factory=list)
    op_types_to_exclude: List[str] = field(default_factory=list)
    data_max: Optional[float] = None
    calibration: str = "none"
    fp32_op_types: Dict[str, int] = field(default_factory=dict)
    modelopt_conversion_line: Optional[str] = None

    @property
    def nodes_graph_total(self) -> int:
        """Every non-Cast node, classified or not — ModelOpt's own denominator."""
        return self.nodes_total + self.nodes_unclassified

    @property
    def low_precision_fraction(self) -> float:
        """Share of the whole graph converted, matching ModelOpt's "converted N/M" log.

        The share of *classified* (float-typed) nodes is the more flattering number
        and both are in the provenance; this one is the comparable one.
        """
        total = self.nodes_graph_total
        return self.nodes_low_precision / total if total else 0.0

    def as_provenance(self) -> dict:
        return {
            "method": self.method,
            "modelopt_version": self.modelopt_version,
            "opset_before": self.opset_before,
            "opset_after": self.opset_after,
            "nodes_graph_total": self.nodes_graph_total,
            "nodes_classified": self.nodes_total,
            "nodes_low_precision": self.nodes_low_precision,
            "nodes_unclassified": self.nodes_unclassified,
            "low_precision_fraction": round(self.low_precision_fraction, 4),
            "keep_io_types": self.keep_io_types,
            "data_max": self.data_max,
            "calibration": self.calibration,
            "nodes_to_exclude": list(self.nodes_to_exclude),
            "op_types_to_exclude": list(self.op_types_to_exclude),
            "fp32_op_types": dict(self.fp32_op_types),
            "modelopt_reported": self.modelopt_conversion_line,
        }


def modelopt_version() -> Optional[str]:
    """Installed ``nvidia-modelopt`` version, or ``None`` if it is not importable."""
    try:
        import modelopt  # type: ignore
    except ImportError:
        return None
    return getattr(modelopt, "__version__", "unknown")


def require_modelopt() -> str:
    """Return the ModelOpt version or raise with the exact install command."""
    version = modelopt_version()
    if version is None:
        raise ModelOptUnavailable(
            "NVIDIA ModelOpt is not installed, so no BF16 (or AutoCast FP16) graph can "
            "be produced. TensorRT >= 11 is strongly typed — there is no builder flag "
            "to fall back on — so this build is refused rather than silently shipping "
            "an FP32 engine labelled bf16.\n"
            "    pip install 'nvidia-modelopt[onnx]'\n"
            "(see requirements-trt.txt; it is an optional extra because it pulls "
            "onnxruntime-gpu / onnxscript into the image)"
        )
    return version


def _default_opset(model) -> int:
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return int(entry.version)
    return 0


def _precision_census(model, low_type: int) -> tuple[int, int, Dict[str, int], int]:
    """``(nodes_classified, nodes_low_precision, fp32_op_type_counts, nodes_unknown)``.

    A node counts as low precision when any of its own value tensors (inputs or
    outputs, initializers included) carries the low-precision dtype, and as FP32
    when one carries ``float32`` and none carries the low precision. ``Cast``
    nodes are the conversion machinery itself and are excluded entirely, so the
    fraction reports compute rather than plumbing.

    ``nodes_unknown`` is the honest residue: ONNX only carries ``value_info`` for
    the tensors shape inference could type, and a graph is full of integer
    shape/index plumbing (Shape, Gather on int64, Range, ...) that has no float
    dtype to classify at all. Those are counted separately instead of being
    silently folded into either side — the fraction below is over *classified*
    nodes only, and reads differently from ModelOpt's own "converted N/M nodes"
    log line, which counts every node in the graph.
    """
    # ONNX only carries value_info for tensors something bothered to type. Without
    # running inference first this census undercounts badly (measured on D-FINE:
    # 1624 low-precision nodes counted where ModelOpt itself reported 3381), so
    # fill the types in first and fall back to the raw graph if inference refuses
    # (>2GB models, or an op inference does not know).
    try:
        import onnx.shape_inference

        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:  # pragma: no cover - inference is best effort
        pass

    dtypes: Dict[str, int] = {}
    for init in model.graph.initializer:
        dtypes[init.name] = init.data_type
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for info in collection:
            dtypes[info.name] = info.type.tensor_type.elem_type

    low = unknown = 0
    fp32_ops: Dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type == "Cast":
            continue
        names = [n for n in list(node.input) + list(node.output) if n]
        if any(dtypes.get(n) == low_type for n in names):
            low += 1
        elif any(dtypes.get(n) == _ONNX_FLOAT for n in names):
            fp32_ops[node.op_type] = fp32_ops.get(node.op_type, 0) + 1
        else:
            unknown += 1
    classified = low + sum(fp32_ops.values())
    return classified, low, dict(sorted(fp32_ops.items(), key=lambda kv: -kv[1])), unknown


def cast_onnx_bytes_with_autocast(
    onnx_bytes: bytes,
    *,
    precision: str = "bf16",
    keep_io_types: bool = True,
    data_max: float = 512.0,
    init_max: float = 65504.0,
    nodes_to_exclude: Optional[List[str]] = None,
    op_types_to_exclude: Optional[List[str]] = None,
    opset: Optional[int] = None,
    calibration_data: Optional[str] = None,
    providers: Optional[List[str]] = None,
    max_depth_of_reduction: Optional[int] = None,
    input_name: Optional[str] = None,
    input_shape: Optional[tuple] = None,
) -> CastResult:
    """Convert ``onnx_bytes`` to a mixed FP32/``precision`` graph via AutoCast.

    ``precision`` is ``"bf16"`` or ``"fp16"``. ``calibration_data`` is an ``.npz``
    of real inputs (AutoCast falls back to *random* data for its magnitude
    classification without it, which is exactly the fixture trap that has bitten
    this repo before — pass real images when the classification matters).

    ``input_name``/``input_shape`` matter when there is no calibration data: our
    graphs declare dynamic H/W, and AutoCast's own random-input fallback resolves
    every symbolic dimension to 1, which a detector graph cannot execute (D-FINE
    dies in onnxruntime with "Concat ... Axis 2 has mismatched dimensions of 1 and
    2" while AutoCast is collecting reference magnitudes). Given a shape, this
    writes the random batch itself at that shape so the reference run is at least
    valid. Real images still classify better — the shape only stops the crash.

    Raises ``ModelOptUnavailable`` if ModelOpt is missing and ``AutoCastError`` if
    the conversion produces an invalid graph. Never returns a graph at a
    precision other than the one asked for.
    """
    precision = str(precision).lower()
    if precision not in _LOW_PRECISION_ONNX_TYPE:
        raise ValueError(f"AutoCast precision must be fp16 or bf16, got {precision!r}")

    version = require_modelopt()
    import onnx
    from modelopt.onnx.autocast import convert_to_mixed_precision  # type: ignore

    before = onnx.load_model_from_string(onnx_bytes)
    opset_before = _default_opset(before)

    calibration_kind = "real" if calibration_data else "none"
    synthetic_dir = None
    if calibration_data is None and input_shape and input_name:
        import numpy as np

        synthetic_dir = tempfile.TemporaryDirectory()
        shape = tuple(int(d) for d in input_shape)
        npz = Path(synthetic_dir.name) / "random_input.npz"
        # Normalized-image-like: our graphs take ImageNet-standardized pixels, so
        # unit-normal noise sits in the right magnitude range for AutoCast's
        # data_max classification. It is still noise — see the docstring.
        np.savez(npz, **{input_name: np.random.randn(*shape).astype(np.float32)})
        calibration_data = str(npz)
        calibration_kind = "synthetic"
        print(f"[trt] AutoCast: no calibration data given — using random input at "
              f"{shape} (pass real images for a better fp32/{precision} split)")

    captured = _CapturingHandler()
    logging.getLogger("modelopt.onnx.autocast").addHandler(captured)
    try:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "model.onnx"
            src.write_bytes(onnx_bytes)
            converted = convert_to_mixed_precision(
                onnx_path=str(src),
                low_precision_type=precision,
                keep_io_types=keep_io_types,
                data_max=data_max,
                init_max=init_max,
                nodes_to_exclude=list(nodes_to_exclude) if nodes_to_exclude else None,
                op_types_to_exclude=list(op_types_to_exclude) if op_types_to_exclude else None,
                calibration_data=calibration_data,
                providers=list(providers) if providers else ["cpu"],
                max_depth_of_reduction=max_depth_of_reduction,
                opset=opset,
            )
    finally:
        logging.getLogger("modelopt.onnx.autocast").removeHandler(captured)
        if synthetic_dir is not None:
            synthetic_dir.cleanup()

    # AutoCast validates with onnx.checker internally, but its own docs record
    # architectures where the emitted graph still fails a full check (and TensorRT
    # then builds an engine that returns NaN). Check here too, loudly, rather than
    # discovering it as a silently wrong engine.
    try:
        onnx.checker.check_model(converted, full_check=True)
    except Exception as exc:  # onnx.checker raises several unrelated types
        raise AutoCastError(
            f"ModelOpt AutoCast produced a graph that fails onnx.checker "
            f"(full_check=True): {exc}. TensorRT may still 'build' this graph and "
            f"then emit NaN — refusing it here instead. Try --op-types-to-exclude "
            f"for the offending op, or a newer nvidia-modelopt."
        ) from exc

    low_type = _LOW_PRECISION_ONNX_TYPE[precision]
    total, low, fp32_ops, unknown = _precision_census(converted, low_type)
    if low == 0:
        raise AutoCastError(
            f"ModelOpt AutoCast converted 0 of {total} nodes to {precision}; the "
            f"resulting graph is effectively FP32. Refusing to ship it as "
            f"{precision} (try raising --data-max, or check that the graph's "
            f"op types are supported in {precision})."
        )

    opset_after = _default_opset(converted)
    expected_min = _MIN_OPSET[precision]
    if opset_after < expected_min:
        raise AutoCastError(
            f"converted graph is opset {opset_after}, below AutoCast's documented "
            f"minimum {expected_min} for {precision}"
        )

    return CastResult(
        onnx_bytes=converted.SerializeToString(),
        precision=precision,
        method=f"modelopt-autocast-{precision}",
        opset_before=opset_before,
        opset_after=opset_after,
        modelopt_version=version,
        nodes_total=total,
        nodes_low_precision=low,
        nodes_unclassified=unknown,
        keep_io_types=keep_io_types,
        nodes_to_exclude=list(nodes_to_exclude or []),
        op_types_to_exclude=list(op_types_to_exclude or []),
        data_max=data_max,
        calibration=calibration_kind,
        fp32_op_types=fp32_ops,
        modelopt_conversion_line=captured.conversion_line(),
    )


def cast_onnx_bytes_to_bf16(onnx_bytes: bytes, **kwargs) -> CastResult:
    """BF16 convenience wrapper — see :func:`cast_onnx_bytes_with_autocast`."""
    kwargs.pop("precision", None)
    return cast_onnx_bytes_with_autocast(onnx_bytes, precision="bf16", **kwargs)
