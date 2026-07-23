"""The single TensorRT build call every arch funnels through: parse an ONNX
graph and serialize a TensorRT engine (the ``.engine`` plan).

Analogous to ``onnx_export/common.py::export_detection_wrapper`` — the one place
the low-level export mechanics live, so the per-checkpoint CLI stays thin. The
engine is built *from the ONNX graph* (decode / NMS / top-k already baked in —
Contract A), so this path is architecture-agnostic; only the optimization
profile (input H/W range, from ``profile.py``) is meta-derived.

FP16 is the default and falls back to FP32 if the platform lacks fast FP16 or the
FP16 build fails. Kept compatible across TensorRT 8.6 → 11:

* TRT 8.6 → 10 request FP16 via ``BuilderFlag.FP16``.
* TRT >= 11 removed all precision flags (networks are "strongly typed"); FP16 is
  instead baked into the ONNX graph dtypes before the build — see ``fp16_cast``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

try:
    from .fp16_cast import cast_onnx_bytes_to_fp16
except ImportError:  # run flat (cwd on sys.path), mirroring cli.py
    from fp16_cast import cast_onnx_bytes_to_fp16  # type: ignore

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


def _try_build(
    trt, logger, onnx_bytes, input_name, min_hw, opt_hw, max_hw, *,
    request_fp16_flag, workspace_gb, min_batch=1, opt_batch=1, max_batch=1,
):
    """Build once. Returns ``(engine_bytes_or_None, flag_was_set)``.

    ``request_fp16_flag`` sets ``BuilderFlag.FP16`` when that flag exists (TRT
    8.6→10) and the platform reports fast FP16. On TRT >= 11 the flag is gone —
    FP16 there comes from the graph dtypes (the caller pre-casts the ONNX), so
    this stays a no-op and returns ``flag_was_set=False``.
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

    flag_was_set = False
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    if request_fp16_flag and fp16_flag is not None and getattr(builder, "platform_has_fast_fp16", True):
        config.set_flag(fp16_flag)
        flag_was_set = True

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (int(min_batch), 3, int(min_hw[0]), int(min_hw[1])),
        (int(opt_batch), 3, int(opt_hw[0]), int(opt_hw[1])),
        (int(max_batch), 3, int(max_hw[0]), int(max_hw[1])),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return None, flag_was_set
    return bytes(serialized), flag_was_set


def build_engine_from_onnx(
    onnx_path,
    engine_path,
    *,
    min_hw: HW,
    opt_hw: HW,
    max_hw: HW,
    input_name: str = "pixel_values",
    precision: str = "fp16",
    extra_fp32_ops=None,
    node_block_substrings=None,
    workspace_gb: float = 4.0,
    logger=None,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
) -> dict:
    """Parse ``onnx_path`` and serialize a TensorRT engine to ``engine_path``.

    Returns a provenance dict (precision actually used, TRT version, profile
    shapes) for the ``.engine.json`` sidecar. Falls back to FP32 when an FP16
    build is requested but unsupported or fails.

    ``min_batch``/``opt_batch``/``max_batch`` default to 1 (today's behavior,
    unchanged for every existing caller) — pass a wider range only for a graph
    whose wrapper actually carries a real batch dim through to its outputs
    (see ``arch/yolox.py``'s ``YOLOXRaw``); a graph that still collapses to
    batch=1 internally would just waste the wider profile, not break anything.
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

    # How FP16 is requested depends on the TRT major version:
    #   TRT 8.6 → 10: set BuilderFlag.FP16 (handled inside _try_build).
    #   TRT >= 11 ("strongly typed", flag removed): bake fp16 into the ONNX graph
    #     dtypes here, before parsing. Graph I/O is kept fp32 by the caster, so
    #     the engine's input/output contract (and .meta.json / trt_infer) is
    #     unchanged; only interior compute runs in fp16.
    fp16_via_flag = hasattr(trt.BuilderFlag, "FP16")
    build_bytes = onnx_bytes
    graph_cast_fp16 = False
    cast_method = None
    if want_fp16 and not fp16_via_flag:
        fp16_bytes, cast_method = cast_onnx_bytes_to_fp16(
            onnx_bytes, extra_fp32_ops=extra_fp32_ops, node_block_substrings=node_block_substrings
        )
        if fp16_bytes is not None:
            build_bytes = fp16_bytes
            graph_cast_fp16 = True
            print(
                f"[trt] TensorRT {trt.__version__}: cast ONNX graph to fp16 via "
                f"{cast_method} (strongly-typed network; graph I/O kept fp32)"
            )
        else:
            raise TrtBuildError(
                f"FP16 requested on TensorRT {trt.__version__} (strongly-typed, no "
                "BuilderFlag.FP16) but no ONNX fp16 converter is importable — install "
                "onnxruntime or onnxconverter_common, or pass precision='fp32' to build "
                "an explicit FP32 engine. Refusing to silently ship an FP32 engine as fp16."
            )

    serialized, flag_was_set = _try_build(
        trt, trt_logger, build_bytes, input_name, min_hw, opt_hw, max_hw,
        request_fp16_flag=want_fp16 and fp16_via_flag, workspace_gb=workspace_gb,
        min_batch=min_batch, opt_batch=opt_batch, max_batch=max_batch,
    )
    used_fp16 = flag_was_set or graph_cast_fp16

    if serialized is None and want_fp16:
        print("[trt] FP16 build produced no engine; retrying in FP32")
        serialized, _ = _try_build(
            trt, trt_logger, onnx_bytes, input_name, min_hw, opt_hw, max_hw,
            request_fp16_flag=False, workspace_gb=workspace_gb,
            min_batch=min_batch, opt_batch=opt_batch, max_batch=max_batch,
        )
        used_fp16 = False
        cast_method = None
    if serialized is None:
        raise TrtBuildError(f"TensorRT failed to build an engine from {onnx_path}")

    engine_path.write_bytes(serialized)
    return {
        "precision": "fp16" if used_fp16 else "fp32",
        "fp16_method": ("builder_flag" if flag_was_set else cast_method) if used_fp16 else None,
        "fp16_extra_fp32_ops": list(extra_fp32_ops or []) if graph_cast_fp16 else None,
        "fp16_node_block_substrings": list(node_block_substrings or []) if graph_cast_fp16 else None,
        "tensorrt_version": trt.__version__,
        "input_name": input_name,
        "profile": {"min": list(min_hw), "opt": list(opt_hw), "max": list(max_hw)},
        "batch": {"min": min_batch, "opt": opt_batch, "max": max_batch},
    }
