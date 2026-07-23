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
    from .arch import get_fp16_node_block, get_fp16_op_block, get_trt_prep, is_fp16_trusted
    from .builder import build_engine_from_onnx
    from .profile import profile_from_meta
except ImportError:  # run flat (cwd on sys.path), mirroring onnx_export/cli.py
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from registry import build_model  # type: ignore
    from ml.onnx_export.cli import export_checkpoint  # type: ignore
    from ml.onnx_export.common import INPUT_NAME  # type: ignore
    from ml.trt_export.arch import (  # type: ignore
        get_fp16_node_block,
        get_fp16_op_block,
        get_trt_prep,
        is_fp16_trusted,
    )
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
) -> Path:
    """Compile ``source_path`` (a ``.pt`` or ``.onnx``) into a TensorRT engine.

    ``adapter`` (optional) is a pre-built torch adapter for the EfficientNMS archs
    (retinanet/yolox); pass it to build from a ``.onnx`` without the ``.pt`` (used
    by tests / callers that already hold the adapter). Ignored for passthrough archs.

    ``precision`` is ``"auto"`` (default), ``"fp16"``, or ``"fp32"``. ``auto`` builds
    fp16 except for archs still on the fp16 safety floor (``UNTRUSTED_FP16``), which
    build fp32; ``fp16``/``fp32`` are honored verbatim (explicit wins over the floor).
    On strongly-typed TRT the arch's per-arch fp32 keep-list (``ARCH_FP16_OP_BLOCK``)
    is applied to any fp16 build; ``extra_fp32_ops`` extends it (used by the sweep).

    ``min_batch``/``opt_batch``/``max_batch`` default to 1 (today's behavior for
    every arch, unchanged). Passing a wider range only actually helps a wrapper
    that carries its real batch dim through end to end — right now that's
    yolox's EfficientNMS wrapper (``arch/yolox.py``); other archs would just
    build a wider, unused profile.

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

    # Resolve auto -> per-arch precision (untrusted archs floor to fp32); explicit
    # fp16/fp32 pass through. The per-arch fp32 keep-list applies to any fp16 build.
    if precision == "auto":
        precision = "fp16" if is_fp16_trusted(arch) else "fp32"
        if precision == "fp32":
            print(f"[trt] {arch}: on the fp16 safety floor -> building fp32 (pass precision='fp16' to override)")
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
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
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
    )


if __name__ == "__main__":
    main()
