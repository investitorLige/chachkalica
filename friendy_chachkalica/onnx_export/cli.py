"""Export a trained checkpoint to ``<name>.onnx`` + ``<name>.meta.json``.

Usage::

    .venv/bin/python -m friendy_chachkalica.onnx_export.cli runs/foo/best.pt
    .venv/bin/python friendy_chachkalica/onnx_export/cli.py runs/foo/best.pt -o out/foo.onnx

Rebuilds the adapter exactly like ``chachak/infer.py`` and ``eval_checkpoint.py``
(``build_model(model_name, num_classes, **params)`` + ``load_state_dict``), then
dispatches to the arch's exporter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import torch

try:
    from ..registry import build_model
    from ..trt_export.arch import get_fp16_node_block, get_fp16_op_block, is_fp16_trusted
    from ..trt_export.fp16_cast import cast_onnx_bytes_to_fp16
    from .registry import get_exporter
except ImportError:  # run as a flat script
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from registry import build_model  # type: ignore
    from onnx_export.registry import get_exporter  # type: ignore
    from trt_export.arch import (  # type: ignore
        get_fp16_node_block,
        get_fp16_op_block,
        is_fp16_trusted,
    )
    from trt_export.fp16_cast import cast_onnx_bytes_to_fp16  # type: ignore


def _as_class_map(classes: Union[Dict, List, None]) -> Dict[int, str]:
    if classes is None:
        return {}
    if isinstance(classes, dict):
        return {int(k): str(v) for k, v in classes.items()}
    return {index: str(name) for index, name in enumerate(classes)}


def _write_fp16_sidecar(
    fp32_onnx_path: Path,
    arch: str,
    *,
    force: bool = False,
) -> Union[Path, None]:
    """Cast the fp32 ONNX at ``fp32_onnx_path`` to a separate ``<name>.fp16.onnx``.

    A separate artifact — never a replacement. Graph I/O stays fp32
    (``keep_io_types``), so the fp32 ``.meta.json`` applies verbatim; we copy it to
    ``<name>.fp16.meta.json`` so the fp16 model loads standalone via onnx_infer's
    ``<stem>.meta.json`` convention.

    IMPORTANT: this file is for the onnxruntime(-GPU) consumer only. Do NOT feed it
    to the TRT builder: on strongly-typed TRT (>=11) engine precision comes from the
    graph dtypes, so an fp16 graph forces an fp16 engine regardless of the requested
    precision — bypassing the per-arch UNTRUSTED_FP16 fp32 floor. The TRT path builds
    fp16 itself from the canonical fp32 ``.onnx``.
    """
    if not is_fp16_trusted(arch) and not force:
        raise SystemExit(
            f"[export] Refusing fp16 export for {arch!r}: its plain-fp16 output is "
            f"not trusted (known parity break; see UNTRUSTED_FP16). Re-run with "
            f"--force-fp16 to write it anyway."
        )

    fp16_onnx_path = fp32_onnx_path.with_suffix(".fp16.onnx")
    fp16_meta_path = fp16_onnx_path.with_suffix(".meta.json")
    fp32_meta_path = fp32_onnx_path.with_suffix(".meta.json")

    fp16_bytes, method = cast_onnx_bytes_to_fp16(
        fp32_onnx_path.read_bytes(),
        extra_fp32_ops=get_fp16_op_block(arch),
        node_block_substrings=get_fp16_node_block(arch),
    )
    if fp16_bytes is None:
        raise SystemExit(
            "[export] --fp16 requested but no ONNX float16 converter is importable "
            "(need onnxruntime or onnxconverter_common)."
        )

    fp16_onnx_path.write_bytes(fp16_bytes)
    # Contract A/B unchanged (I/O kept fp32) -> the fp32 meta applies verbatim.
    fp16_meta_path.write_text(fp32_meta_path.read_text())
    print(f"[export] Wrote {fp16_onnx_path} (fp16 via {method}; I/O kept fp32)")
    print(f"[export] Wrote {fp16_meta_path}")
    return fp16_onnx_path


def export_checkpoint(
    checkpoint_path: Union[str, Path],
    onnx_path: Union[str, Path, None] = None,
    *,
    fp16: bool = False,
    force_fp16: bool = False,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    onnx_path = Path(onnx_path) if onnx_path else checkpoint_path.with_suffix(".onnx")
    meta_path = onnx_path.with_suffix(".meta.json")

    print(f"[export] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model_name = state["model_name"]
    model_config = state.get("model_config", {}) or {}
    num_classes = model_config.get("num_classes")
    params = dict(model_config.get("params", {}) or {})
    class_map = _as_class_map((state.get("train_dataset") or {}).get("classes"))

    adapter = build_model(model_name, num_classes=num_classes, **params)
    adapter.model.load_state_dict(state["model_state_dict"])
    adapter.eval()
    print(f"[export] Rebuilt adapter: {model_name} num_classes={num_classes} classes={len(class_map)}")

    exporter = get_exporter(model_name)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    meta = exporter(
        adapter,
        num_classes=num_classes,
        params=params,
        class_map=class_map,
        onnx_path=onnx_path,
    )

    with open(meta_path, "w") as file:
        json.dump(meta, file, indent=2)
    print(f"[export] Wrote {onnx_path}")
    print(f"[export] Wrote {meta_path}")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Friendy checkpoint to ONNX + meta.json")
    parser.add_argument("checkpoint", help="Path to a checkpoint, e.g. runs/foo/best.pt")
    parser.add_argument("--output", "-o", help="ONNX output path (default: checkpoint with .onnx)")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
