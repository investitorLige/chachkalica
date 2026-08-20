"""The single TensorRT build call every arch funnels through: parse an ONNX
graph and serialize a TensorRT engine (the ``.engine`` plan).

Analogous to ``onnx_export/common.py::export_detection_wrapper`` — the one place
the low-level export mechanics live, so the per-checkpoint CLI stays thin. The
engine is built *from the ONNX graph* (decode / NMS / top-k already baked in —
Contract A), so this path is architecture-agnostic; only the optimization
profile (input H/W range, from ``profile.py``) is meta-derived.

FP16 is the default and falls back to FP32 if the platform lacks fast FP16 or the
FP16 build fails. BF16 is also selectable, and never falls back (see below). Kept
compatible across TensorRT 8.6 → 11:

* TRT 8.6 → 10 request FP16/BF16 via ``BuilderFlag.FP16`` / ``BuilderFlag.BF16``.
* TRT >= 11 removed all precision flags (networks are "strongly typed"); the
  precision is instead baked into the ONNX graph dtypes before the build —
  ``fp16_cast`` (onnxruntime's float16 converter) for FP16, and
  ``modelopt_cast`` (NVIDIA ModelOpt AutoCast, the route NVIDIA's 10.x→11.x
  migration guide points at) for BF16 and for the AutoCast FP16 variant.

**BF16 never falls back.** An FP16 build that produces no engine retries at FP32
and says so in the provenance — long-standing behavior the benchmark relies on.
A BF16 build that cannot be cast or cannot be compiled raises instead: BF16 is
only ever chosen deliberately, so quietly handing back an FP16 or FP32 engine
would misreport what is being measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

try:
    from .fp16_cast import cast_onnx_bytes_to_fp16
    from .modelopt_cast import cast_onnx_bytes_with_autocast
except ImportError:  # run flat (cwd on sys.path), mirroring cli.py
    from fp16_cast import cast_onnx_bytes_to_fp16  # type: ignore
    from modelopt_cast import cast_onnx_bytes_with_autocast  # type: ignore

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
    request_flag_precision=None, workspace_gb, min_batch=1, opt_batch=1, max_batch=1,
    profiling_verbosity="default", tf32=True,
):
    """Build once. Returns ``(engine_bytes_or_None, flag_was_set)``.

    ``request_flag_precision`` is ``"fp16"``/``"bf16"``/``None`` and sets the
    matching ``BuilderFlag`` when that flag exists (TRT 8.6→10) — additionally
    requiring ``platform_has_fast_fp16`` for FP16. On TRT >= 11 the flags are
    gone; the precision there comes from the graph dtypes (the caller pre-casts
    the ONNX), so this stays a no-op and returns ``flag_was_set=False``.
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
    if request_flag_precision:
        flag = getattr(trt.BuilderFlag, str(request_flag_precision).upper(), None)
        fast_enough = (
            getattr(builder, "platform_has_fast_fp16", True)
            if request_flag_precision == "fp16"
            else True
        )
        if flag is not None and fast_enough:
            config.set_flag(flag)
            flag_was_set = True

    # TF32 is ON by default in every TensorRT: an "FP32" engine is really free to
    # run convolutions and matmuls through TF32 tensor cores, which keep FP32's
    # exponent but only 10 mantissa bits — the same mantissa as FP16. That makes a
    # default FP32 engine the wrong reference to judge FP16/BF16 against for a model
    # whose scores are mantissa-sensitive, so this can be turned off to get a true
    # FP32 baseline (at a real cost in latency).
    if not tf32:
        tf32_flag = getattr(trt.BuilderFlag, "TF32", None)
        if tf32_flag is not None:
            config.clear_flag(tf32_flag)

    # DETAILED keeps per-layer information in the plan so IEngineInspector can
    # report which layers/tensors ended up in which dtype (engine_inspect.py).
    # Off by default: it enlarges the plan a little, and every engine this repo
    # has already shipped was built without it.
    if str(profiling_verbosity).lower() == "detailed":
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

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
    cast_backend: str = "auto",
    autocast_options: Optional[dict] = None,
    profiling_verbosity: str = "default",
    tf32: bool = True,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
) -> dict:
    """Parse ``onnx_path`` and serialize a TensorRT engine to ``engine_path``.

    Returns a provenance dict (precision actually used, TRT version, profile
    shapes) for the ``.engine.json`` sidecar. Falls back to FP32 when an FP16
    build is requested but unsupported or fails; a BF16 request never falls back
    and raises instead (see the module docstring).

    ``precision`` is ``"fp32"``, ``"fp16"`` or ``"bf16"``. ``cast_backend`` picks
    how a low-precision graph is produced on strongly-typed TensorRT:
    ``"graph_cast"`` (onnxruntime's float16 converter, the long-standing FP16
    path) or ``"autocast"`` (NVIDIA ModelOpt AutoCast). ``"auto"`` means
    graph_cast for FP16 and autocast for BF16 — BF16 has no graph_cast
    implementation at all. ``autocast_options`` is forwarded to
    ``modelopt_cast.cast_onnx_bytes_with_autocast`` (``data_max``,
    ``calibration_data``, ``nodes_to_exclude``, ...).

    ``min_batch``/``opt_batch``/``max_batch`` default to 1 — pass a wider range
    only for a graph whose wrapper actually carries a real batch dim through to
    its outputs (see ``arch/__init__.py``'s ``BATCH_AWARE_ARCHS``); a graph that
    still collapses to batch=1 internally would silently return one image's
    detections for the whole batch rather than raise. This function itself does
    not check that — ``trt_export/cli.py``'s ``build_engine`` is the layer that
    enforces it, since only it knows the arch (from the meta sidecar).
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
    want = str(precision).lower()
    if want not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision must be fp32, fp16 or bf16 — got {precision!r}")
    low_precision = want in ("fp16", "bf16")

    # How a low precision is requested depends on the TRT major version:
    #   TRT 8.6 → 10: set BuilderFlag.FP16 / BF16 (handled inside _try_build).
    #   TRT >= 11 ("strongly typed", flags removed): bake the dtypes into the
    #     ONNX graph here, before parsing. Graph I/O is kept fp32 by both casters,
    #     so the engine's input/output contract (and .meta.json / trt_infer) is
    #     unchanged; only interior compute runs low-precision.
    flag_enum_has = hasattr(trt.BuilderFlag, want.upper())
    backend = str(cast_backend or "auto").lower()
    if backend == "auto":
        backend = "autocast" if want == "bf16" else "graph_cast"
    trt_major = int(str(trt.__version__).split(".")[0])
    if want == "bf16" and trt_major <= 10 and not flag_enum_has:
        # Weakly-typed TensorRT that predates BF16: the graph route is not an
        # option there either (its ONNX parser has no bfloat16 tensor type), so
        # there is nothing to fall back to and nothing to pretend.
        raise TrtBuildError(
            f"BF16 requested on TensorRT {trt.__version__}, which is weakly typed "
            f"and exposes no BuilderFlag.BF16 — this build cannot produce a BF16 "
            f"engine. Upgrade TensorRT (>= 10.x with BF16, or >= 11 for the "
            f"strongly-typed ModelOpt AutoCast route) or build fp16/fp32."
        )
    if want == "bf16" and backend != "autocast":
        raise ValueError(
            "BF16 has no graph_cast implementation (onnxruntime's/onnxconverter's "
            "float16 converters only emit fp16) — use cast_backend='autocast'."
        )

    build_bytes = onnx_bytes
    graph_cast = False
    cast_method = None
    cast_provenance = None
    if low_precision and not flag_enum_has:
        if backend == "autocast":
            options = dict(autocast_options or {})
            # Give AutoCast a valid input shape for its reference run when the
            # caller has no calibration data: it resolves dynamic dims to 1 on
            # its own, which our detector graphs cannot execute.
            options.setdefault("input_name", input_name)
            options.setdefault("input_shape", (int(opt_batch), 3, int(opt_hw[0]), int(opt_hw[1])))
            result = cast_onnx_bytes_with_autocast(onnx_bytes, precision=want, **options)
            build_bytes = result.onnx_bytes
            cast_method = result.method
            cast_provenance = result.as_provenance()
            graph_cast = True
            print(
                f"[trt] TensorRT {trt.__version__}: ModelOpt AutoCast -> {want}: "
                f"{result.nodes_low_precision} of {result.nodes_total} float-typed nodes "
                f"converted ({100 * result.low_precision_fraction:.1f}% of the "
                f"{result.nodes_graph_total}-node graph; the rest is integer "
                f"shape/index plumbing), opset {result.opset_before} -> "
                f"{result.opset_after} (strongly-typed; graph I/O kept fp32)"
            )
        else:
            cast_bytes, cast_method = cast_onnx_bytes_to_fp16(
                onnx_bytes, extra_fp32_ops=extra_fp32_ops,
                node_block_substrings=node_block_substrings,
            )
            if cast_bytes is None:
                raise TrtBuildError(
                    f"FP16 requested on TensorRT {trt.__version__} (strongly-typed, no "
                    "BuilderFlag.FP16) but no ONNX fp16 converter is importable — install "
                    "onnxruntime or onnxconverter_common, or pass precision='fp32' to build "
                    "an explicit FP32 engine. Refusing to silently ship an FP32 engine as fp16."
                )
            build_bytes = cast_bytes
            graph_cast = True
            print(
                f"[trt] TensorRT {trt.__version__}: cast ONNX graph to fp16 via "
                f"{cast_method} (strongly-typed network; graph I/O kept fp32)"
            )
    elif want == "bf16" and flag_enum_has:
        print(f"[trt] TensorRT {trt.__version__}: requesting BF16 via BuilderFlag.BF16")

    serialized, flag_was_set = _try_build(
        trt, trt_logger, build_bytes, input_name, min_hw, opt_hw, max_hw,
        request_flag_precision=want if (low_precision and flag_enum_has) else None,
        workspace_gb=workspace_gb,
        min_batch=min_batch, opt_batch=opt_batch, max_batch=max_batch,
        profiling_verbosity=profiling_verbosity, tf32=tf32,
    )
    used_low = flag_was_set or graph_cast

    if serialized is None and want == "bf16":
        # No fallback for BF16 — see the module docstring.
        raise TrtBuildError(
            f"BF16 build produced no engine for {onnx_path}. Not falling back to "
            f"fp16/fp32: a bf16 request is always deliberate, so a silently "
            f"downgraded engine would misreport what was built. TensorRT "
            f"{trt.__version__}, profile min={min_hw} opt={opt_hw} max={max_hw}."
        )
    if serialized is None and want == "fp16":
        print("[trt] FP16 build produced no engine; retrying in FP32")
        serialized, _ = _try_build(
            trt, trt_logger, onnx_bytes, input_name, min_hw, opt_hw, max_hw,
            request_flag_precision=None, workspace_gb=workspace_gb,
            min_batch=min_batch, opt_batch=opt_batch, max_batch=max_batch,
            profiling_verbosity=profiling_verbosity, tf32=tf32,
        )
        used_low = False
        cast_method = None
        cast_provenance = None
    if serialized is None:
        raise TrtBuildError(f"TensorRT failed to build an engine from {onnx_path}")

    engine_path.write_bytes(serialized)
    built = want if used_low else "fp32"
    return {
        "precision": built,
        # Historical key: what produced the low-precision graph. Kept spelled
        # "fp16_method" so existing readers (buildnode, benchmark CSVs, the
        # admin's engine sidecar view) keep working; `precision_method` is the
        # precision-neutral name for the same value.
        "fp16_method": ("builder_flag" if flag_was_set else cast_method) if used_low else None,
        "precision_method": ("builder_flag" if flag_was_set else cast_method) if used_low else None,
        "cast_backend": backend if used_low and graph_cast else None,
        "autocast": cast_provenance,
        "fp16_extra_fp32_ops": (
            list(extra_fp32_ops or []) if graph_cast and backend == "graph_cast" else None
        ),
        "fp16_node_block_substrings": (
            list(node_block_substrings or []) if graph_cast and backend == "graph_cast" else None
        ),
        "tf32": bool(tf32),
        "tensorrt_version": trt.__version__,
        "input_name": input_name,
        "profile": {"min": list(min_hw), "opt": list(opt_hw), "max": list(max_hw)},
        "batch": {"min": min_batch, "opt": opt_batch, "max": max_batch},
    }
