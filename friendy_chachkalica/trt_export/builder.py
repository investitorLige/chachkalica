"""The single TensorRT build call every arch funnels through: parse an ONNX
graph and serialize a TensorRT engine (the ``.engine`` plan).

Analogous to ``onnx_export/common.py::export_detection_wrapper`` — the one place
the low-level export mechanics live, so the per-checkpoint CLI stays thin. The
engine is built *from the ONNX graph* (decode / NMS / top-k already baked in —
Contract A), so this path is architecture-agnostic; only the optimization
profile (input H/W range, from ``profile.py``) is meta-derived.

FP16 is the default and falls back to FP32 if the platform lacks fast FP16 or the
FP16 build fails. Kept compatible across TensorRT 8.6 → 10 API differences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

HW = Tuple[int, int]


class TrtBuildError(RuntimeError):
    """The ONNX parser rejected the graph, or the builder produced no engine."""


def _make_network(builder, trt):
    """Create an explicit-batch network across TRT versions.

    TRT < 10 requires the ``EXPLICIT_BATCH`` flag; TRT >= 10 removed the enum
    member (networks are always explicit-batch) and ``create_network()`` takes no
    argument.
    """
    flag_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    if flag_enum is not None and hasattr(flag_enum, "EXPLICIT_BATCH"):
        return builder.create_network(1 << int(flag_enum.EXPLICIT_BATCH))
    return builder.create_network()


def _try_build(trt, logger, onnx_bytes, input_name, min_hw, opt_hw, max_hw, *, fp16, workspace_gb):
    """Build once at the requested precision. Returns ``(engine_bytes, used_fp16)``
    or ``(None, used_fp16)`` if the builder produced no engine (caller falls back).
    """
    builder = trt.Builder(logger)
    network = _make_network(builder, trt)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise TrtBuildError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):  # TRT >= 8.4
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:  # pragma: no cover - legacy TRT
        config.max_workspace_size = workspace_bytes

    # TensorRT >= 11 dropped Builder.platform_has_fast_fp16 (fast FP16 support is
    # no longer queryable this way); default to attempting FP16 there and let the
    # existing build-failed-fp16 -> retry-fp32 fallback above handle a platform
    # that can't actually build it.
    #
    # TensorRT 11 also removed BuilderFlag.FP16 (and INT8/BF16/...) entirely:
    # networks are now always "strongly typed" and run at whatever precision the
    # ONNX graph's own tensors declare, rather than a precision requested via a
    # builder flag (see NVIDIA's TensorRT 10->11 migration guide). Real FP16 on
    # TRT 11 needs the graph itself cast to fp16 (e.g. via ModelOpt AutoCast)
    # before this call; that's out of scope here, so we just build at the
    # graph's native precision (fp32, for a plain torch.onnx.export) instead of
    # raising on the missing flag.
    used_fp16 = False
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    if fp16 and fp16_flag is not None and getattr(builder, "platform_has_fast_fp16", True):
        config.set_flag(fp16_flag)
        used_fp16 = True
    elif fp16:
        print(
            f"[trt] FP16 requested but unavailable via BuilderFlag on TensorRT {trt.__version__} "
            "(TensorRT >= 11 uses strongly-typed networks; precision comes from the ONNX "
            "graph's own tensor dtypes). Building at the graph's native precision instead."
        )

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (1, 3, int(min_hw[0]), int(min_hw[1])),
        (1, 3, int(opt_hw[0]), int(opt_hw[1])),
        (1, 3, int(max_hw[0]), int(max_hw[1])),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return None, used_fp16
    return bytes(serialized), used_fp16


def build_engine_from_onnx(
    onnx_path,
    engine_path,
    *,
    min_hw: HW,
    opt_hw: HW,
    max_hw: HW,
    input_name: str = "pixel_values",
    precision: str = "fp16",
    workspace_gb: float = 4.0,
    logger=None,
) -> dict:
    """Parse ``onnx_path`` and serialize a TensorRT engine to ``engine_path``.

    Returns a provenance dict (precision actually used, TRT version, profile
    shapes) for the ``.engine.json`` sidecar. Falls back to FP32 when an FP16
    build is requested but unsupported or fails.
    """
    import tensorrt as trt

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    trt_logger = logger or trt.Logger(trt.Logger.WARNING)
    # Register the standard TensorRT plugins (EfficientNMS_TRT et al.) so the ONNX
    # parser resolves plugin nodes the arch prep may have appended. Harmless (and
    # idempotent) for graphs that use none.
    trt.init_libnvinfer_plugins(trt_logger, "")
    onnx_bytes = onnx_path.read_bytes()
    want_fp16 = str(precision).lower() == "fp16"

    serialized, used_fp16 = _try_build(
        trt, trt_logger, onnx_bytes, input_name, min_hw, opt_hw, max_hw,
        fp16=want_fp16, workspace_gb=workspace_gb,
    )
    if serialized is None and want_fp16:
        print("[trt] FP16 build produced no engine; retrying in FP32")
        serialized, used_fp16 = _try_build(
            trt, trt_logger, onnx_bytes, input_name, min_hw, opt_hw, max_hw,
            fp16=False, workspace_gb=workspace_gb,
        )
    if serialized is None:
        raise TrtBuildError(f"TensorRT failed to build an engine from {onnx_path}")

    engine_path.write_bytes(serialized)
    return {
        "precision": "fp16" if used_fp16 else "fp32",
        "tensorrt_version": trt.__version__,
        "input_name": input_name,
        "profile": {"min": list(min_hw), "opt": list(opt_hw), "max": list(max_hw)},
    }
