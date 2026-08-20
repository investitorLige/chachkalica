"""Build a TensorRT engine from a trained checkpoint or an exported ONNX graph.

Usage::

    python -m friendy_chachkalica.ml.trt_export.cli runs/foo/best.pt
    python -m friendy_chachkalica.ml.trt_export.cli runs/foo/best.onnx -o out/foo.engine --precision fp16

Given a ``.pt`` with no sibling ``.onnx`` yet, the ONNX export runs first
(reusing ``onnx_export.cli.export_checkpoint``) — so this is a one-shot
checkpoint → engine path. Given a ``.onnx``, it is compiled directly.

Writes ``<name>.engine`` + a verbatim ``<name>.meta.json`` copy (so ``trt_infer``
is self-contained) + a ``<name>.engine.json`` provenance sidecar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Union

try:
    from ..onnx_export.cli import export_checkpoint
    from ..onnx_export.common import INPUT_NAME
    from ...registry import build_model
    from .arch import (
        get_cast_backend,
        get_fp16_node_block,
        get_fp16_op_block,
        get_trt_prep,
        is_batch_aware,
        is_fp16_trusted,
        resolve_auto_precision,
    )
    from .modelopt_cast import modelopt_version
    from .builder import build_engine_from_onnx
    from .profile import profile_from_meta
except ImportError:  # run flat (cwd on sys.path), mirroring onnx_export/cli.py
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from registry import build_model  # type: ignore
    from ml.onnx_export.cli import export_checkpoint  # type: ignore
    from ml.onnx_export.common import INPUT_NAME  # type: ignore
    from ml.trt_export.arch import (  # type: ignore
        get_cast_backend,
        get_fp16_node_block,
        get_fp16_op_block,
        get_trt_prep,
        is_batch_aware,
        is_fp16_trusted,
        resolve_auto_precision,
    )
    from ml.trt_export.modelopt_cast import modelopt_version  # type: ignore
    from ml.trt_export.builder import build_engine_from_onnx  # type: ignore
    from ml.trt_export.profile import profile_from_meta  # type: ignore

HW = Tuple[int, int]


def _load_adapter(checkpoint_path: Path):
    """Rebuild a trained adapter from a ``.pt``, exactly like the ONNX exporter."""
    import torch

    state = torch.load(checkpoint_path, map_location="cpu")
    model_config = state.get("model_config", {}) or {}
    params = dict(model_config.get("params", {}) or {})
    adapter = build_model(
        state["model_name"], num_classes=model_config.get("num_classes"), **params
    )
    adapter.model.load_state_dict(state["model_state_dict"])
    adapter.eval()
    return adapter


def build_engine(
    source_path: Union[str, Path],
    engine_path: Union[str, Path, None] = None,
    *,
    precision: str = "auto",
    adapter=None,
    min_hw: Optional[HW] = None,
    opt_hw: Optional[HW] = None,
    max_hw: Optional[HW] = None,
    extra_fp32_ops=None,
    node_block_substrings=None,
    workspace_gb: float = 4.0,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    cast_backend: str = "auto",
    autocast_options: Optional[dict] = None,
) -> Path:
    """Compile ``source_path`` (a ``.pt`` or ``.onnx``) into a TensorRT engine.

    ``adapter`` (optional) is a pre-built torch adapter for the EfficientNMS archs
    (retinanet/yolox); pass it to build from a ``.onnx`` without the ``.pt`` (used
    by tests / callers that already hold the adapter). Ignored for passthrough archs.

    ``precision`` is ``"auto"`` (default), ``"fp16"``, ``"bf16"`` or ``"fp32"``.
    ``auto`` builds fp16 except for archs still on the fp16 safety floor
    (``UNTRUSTED_FP16``), which build fp32; the explicit precisions are honored
    verbatim (explicit wins over the floor). ``bf16`` is only available on
    strongly-typed TensorRT via ModelOpt AutoCast (see ``modelopt_cast.py``) or,
    on TRT <= 10, via ``BuilderFlag.BF16``; it never falls back to another
    precision — see ``build_trt.py`` for the standalone graph-level utility.
    On strongly-typed TRT the arch's per-arch fp32 keep-list (``ARCH_FP16_OP_BLOCK``)
    is applied to any fp16 build; ``extra_fp32_ops`` extends it (used by the sweep).

    ``min_batch``/``opt_batch``/``max_batch`` default to 1. Passing a wider
    range only ever helps an arch whose exported graph carries a real batch
    dim through to its outputs (see ``arch/__init__.py``'s ``BATCH_AWARE_ARCHS``
    — currently yolox, ecdet, rtdetr, rfdetr); requesting ``max_batch > 1`` for
    any other arch raises rather than silently building a profile the graph
    can't back correctly.

    Returns the written ``.engine`` path.
    """
    source_path = Path(source_path)

    checkpoint = None
    if source_path.suffix == ".pt":
        checkpoint = source_path
        onnx_path = source_path.with_suffix(".onnx")
        if not onnx_path.exists():
            print(f"[trt] No ONNX beside {source_path.name}; exporting it first")
            export_checkpoint(source_path, onnx_path)
    elif source_path.suffix == ".onnx":
        onnx_path = source_path
    else:
        raise ValueError(f"expected a .pt or .onnx path, got {source_path}")

    meta_path = onnx_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta sidecar not found next to ONNX: {meta_path}")
    meta = json.loads(meta_path.read_text())
    arch = meta.get("arch")

    if max_batch > 1 and not is_batch_aware(arch):
        raise ValueError(
            f"arch {arch!r} was requested with max_batch={max_batch}, but its "
            f"exported graph does not carry a real batch dimension through to "
            f"its outputs — TensorRT would silently repeat one image's "
            f"detections across the whole batch. Build with max_batch=1, or "
            f"see arch/__init__.py's BATCH_AWARE_ARCHS for which archs support "
            f"a wider profile."
        )

    # Resolve auto -> per-arch precision and cast backend (see arch/__init__.py:
    # untrusted archs floor to fp32, unless AutoCast is known to rescue them and
    # ModelOpt is installed); explicit fp16/bf16/fp32 pass through. The per-arch
    # fp32 keep-list applies to any graph_cast fp16 build.
    if precision == "auto":
        resolved, auto_backend = resolve_auto_precision(
            arch, modelopt_available=modelopt_version() is not None
        )
        if cast_backend == "auto":
            cast_backend = auto_backend
        if resolved == "fp32" and not is_fp16_trusted(arch):
            extra = ""
            if arch in ("dfine",) and modelopt_version() is None:
                extra = (" — install nvidia-modelopt[onnx] to get the AutoCast fp16 "
                         "engine this arch is known to survive")
            print(f"[trt] {arch}: on the fp16 safety floor -> building fp32 "
                  f"(pass precision='fp16' to override){extra}")
        elif resolved == "fp16" and auto_backend == "autocast" and not is_fp16_trusted(arch):
            print(f"[trt] {arch}: fp16 via ModelOpt AutoCast (its blanket-cast fp16 is "
                  f"on the safety floor; the AutoCast graph is not)")
        precision = resolved
    elif precision == "fp16" and cast_backend == "auto":
        # An explicit fp16 request still gets the arch's required backend. If that
        # is AutoCast and ModelOpt is missing, refuse: silently falling back to the
        # blanket cast would ship exactly the engine the floor exists to prevent.
        cast_backend = get_cast_backend(arch)
        if cast_backend == "autocast" and modelopt_version() is None:
            raise RuntimeError(
                f"arch {arch!r} only has a trustworthy fp16 engine via NVIDIA ModelOpt "
                f"AutoCast (its blanket-cast fp16 is on the safety floor — see "
                f"trt_export/arch/__init__.py), but nvidia-modelopt is not installed:\n"
                f"    pip install 'nvidia-modelopt[onnx]'\n"
                f"Pass cast_backend='graph_cast' to build the known-degraded engine "
                f"anyway, or precision='fp32'."
            )
    resolved_fp32_ops = get_fp16_op_block(arch) + [
        op for op in (extra_fp32_ops or []) if op not in get_fp16_op_block(arch)
    ]
    resolved_node_block = get_fp16_node_block(arch) + [
        s for s in (node_block_substrings or []) if s not in get_fp16_node_block(arch)
    ]

    engine_path = Path(engine_path) if engine_path else onnx_path.with_suffix(".engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    # Archs whose standard ONNX bakes data-dependent NMS (retinanet, yolox) get a
    # TRT-specific graph (raw boxes/scores + EfficientNMS_TRT). The rest compile
    # straight from the ONNX.
    prep = get_trt_prep(arch)
    if prep is None:
        onnx_to_build = onnx_path
    else:
        if adapter is None:
            if checkpoint is None:
                raise ValueError(
                    f"arch {arch!r} builds its TRT engine via an EfficientNMS re-export, "
                    f"which needs the .pt checkpoint (or an explicit adapter=) — pass "
                    f"the checkpoint, not the .onnx"
                )
            adapter = _load_adapter(checkpoint)
        onnx_to_build = engine_path.with_suffix(".trt.onnx")
        print(f"[trt] {arch}: re-exporting raw graph + EfficientNMS -> {onnx_to_build.name}")
        prep(adapter, meta, onnx_to_build)

    prof_min, prof_opt, prof_max = profile_from_meta(
        meta, min_hw=min_hw, opt_hw=opt_hw, max_hw=max_hw
    )

    print(
        f"[trt] Building engine: {onnx_to_build.name} arch={arch} "
        f"precision={precision} profile min={prof_min} opt={prof_opt} max={prof_max}"
    )
    provenance = build_engine_from_onnx(
        onnx_to_build,
        engine_path,
        min_hw=prof_min,
        opt_hw=prof_opt,
        max_hw=prof_max,
        input_name=INPUT_NAME,
        precision=precision,
        extra_fp32_ops=resolved_fp32_ops,
        node_block_substrings=resolved_node_block,
        workspace_gb=workspace_gb,
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        cast_backend=cast_backend,
        autocast_options=autocast_options,
    )

    # Self-contained artifact: a verbatim meta copy next to the engine (unless the
    # engine sits right beside the ONNX, where the sidecar already exists).
    engine_meta_path = engine_path.with_suffix(".meta.json")
    if engine_meta_path.resolve() != meta_path.resolve():
        engine_meta_path.write_text(json.dumps(meta, indent=2))

    provenance_path = Path(str(engine_path) + ".json")  # <name>.engine.json
    provenance_path.write_text(
        json.dumps(
            {
                **provenance,
                "arch": arch,
                "efficientnms": prep is not None,
                "source": str(source_path),
                # What was asked of build_engine_from_onnx (post "auto" resolution),
                # as opposed to `provenance["precision"]` which is what it actually
                # built (an fp16 request can silently fall back to fp32). Callers
                # that cache engines across differently-configured runs (e.g.
                # inferlica/benchmark) need this to tell "the same request, so the
                # cached engine is still valid" apart from "a different request
                # that happens to have produced the same on-disk filename".
                "requested_precision": precision,
            },
            indent=2,
        )
    )

    print(f"[trt] Wrote {engine_path} (precision={provenance['precision']})")
    print(f"[trt] Wrote {engine_meta_path}")
    print(f"[trt] Wrote {provenance_path}")
    return engine_path


def _parse_hw(value: Optional[str]) -> Optional[HW]:
    if not value:
        return None
    parts = value.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected HxW (e.g. 640x640), got {value!r}")
    return (int(parts[0]), int(parts[1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from a Friendy checkpoint or ONNX graph"
    )
    parser.add_argument("source", help="Path to a .pt checkpoint or an exported .onnx")
    parser.add_argument("--output", "-o", help="Engine output path (default: source with .engine)")
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument(
        "--cast-backend", choices=["auto", "graph_cast", "autocast"], default="auto",
        help="TRT>=11 only: how the low-precision graph is produced. auto = this "
             "repo's onnxruntime float16 cast for fp16, ModelOpt AutoCast for bf16.",
    )
    parser.add_argument("--min-hw", type=_parse_hw, help="Min input HxW (e.g. 64x64); overrides meta")
    parser.add_argument("--opt-hw", type=_parse_hw, help="Optimum input HxW; overrides meta")
    parser.add_argument("--max-hw", type=_parse_hw, help="Max input HxW; overrides meta")
    parser.add_argument("--min-batch", type=int, default=1, help="Min batch size (default 1)")
    parser.add_argument("--opt-batch", type=int, default=1, help="Optimum batch size (default 1)")
    parser.add_argument("--max-batch", type=int, default=1, help="Max batch size (default 1)")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    args = parser.parse_args()

    build_engine(
        args.source,
        args.output,
        precision=args.precision,
        min_hw=args.min_hw,
        opt_hw=args.opt_hw,
        max_hw=args.max_hw,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_gb=args.workspace_gb,
        cast_backend=args.cast_backend,
    )


if __name__ == "__main__":
    main()
