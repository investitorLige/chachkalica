"""Convert a Friendy checkpoint (.pt/.pth) to ONNX + a TensorRT engine, both
written into one output directory, then smoke-test both through onnx_infer /
trt_infer to confirm the exported artifacts actually run.

    python inferlica/export.py runs/foo/best.pt --output-dir out/foo

Writes into ``--output-dir`` (basename taken from the checkpoint's stem):

- ``<stem>.onnx`` + ``<stem>.meta.json``               (friendy_chachkalica.ml.onnx_export)
- ``<stem>.engine`` + ``<stem>.meta.json`` + ``<stem>.engine.json``  (friendy_chachkalica.ml.trt_export)
- ``<stem>.trt.onnx``  (only for archs whose TRT engine needs a raw-output +
  EfficientNMS re-export, e.g. retinanet/yolox)

See ``inferlica/commands.md`` for the full flag reference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from friendy_chachkalica.ml.onnx_export.cli import export_checkpoint
from friendy_chachkalica.registry import build_model
from friendy_chachkalica.ml.trt_export.cli import build_engine

HW = Tuple[int, int]


def _build_adapter(checkpoint_path: Path):
    """Rebuild a trained adapter from a ``.pt``, mirroring onnx_export/trt_export's
    own checkpoint loading so the engine build (which may need the adapter for an
    EfficientNMS re-export) doesn't require re-reading the checkpoint itself."""
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


def _smoke_test(onnx_path: Path, engine_path: Path, device: str) -> None:
    import torch

    dummy = torch.rand(3, 640, 640)  # CHW, [0,1] float — preprocess() resizes to whatever the arch needs

    from onnx_infer import load_onnx_adapter

    onnx_adapter, _ = load_onnx_adapter(onnx_path, device="cpu")
    preds = onnx_adapter.predict([dummy])
    print(f"[verify] onnx_infer ran ok: {preds[0].shape[0]} boxes on a dummy image")

    try:
        from trt_infer import load_trt_adapter

        trt_adapter, _ = load_trt_adapter(engine_path, device=device)
        preds = trt_adapter.predict([dummy])
        print(f"[verify] trt_infer ran ok: {preds[0].shape[0]} boxes on a dummy image")
    except Exception as exc:
        print(f"[verify] trt_infer smoke test failed: {exc}")
        raise


def export_and_build(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    precision: str = "fp16",
    min_hw: Optional[HW] = None,
    opt_hw: Optional[HW] = None,
    max_hw: Optional[HW] = None,
    workspace_gb: float = 4.0,
    verify: bool = True,
    device: str = "cuda",
) -> Tuple[Path, Path]:
    """Export ``checkpoint_path`` to ``<output_dir>/<stem>.onnx`` and
    ``<output_dir>/<stem>.engine``. Returns ``(onnx_path, engine_path)``."""
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_path.stem

    onnx_path = export_checkpoint(checkpoint_path, output_dir / f"{stem}.onnx")

    adapter = _build_adapter(checkpoint_path)
    engine_path = build_engine(
        onnx_path,
        output_dir / f"{stem}.engine",
        precision=precision,
        adapter=adapter,
        min_hw=min_hw,
        opt_hw=opt_hw,
        max_hw=max_hw,
        workspace_gb=workspace_gb,
    )

    if verify:
        _smoke_test(onnx_path, engine_path, device)

    return onnx_path, engine_path


def _parse_hw(value: Optional[str]) -> Optional[HW]:
    if not value:
        return None
    parts = value.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected HxW (e.g. 640x640), got {value!r}")
    return (int(parts[0]), int(parts[1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Friendy checkpoint to ONNX + a TensorRT engine in one output dir"
    )
    parser.add_argument("checkpoint", help="Path to a .pt/.pth checkpoint, e.g. runs/foo/best.pt")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to write <stem>.onnx / <stem>.engine into")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--min-hw", type=_parse_hw, help="Min input HxW (e.g. 64x64); overrides meta")
    parser.add_argument("--opt-hw", type=_parse_hw, help="Optimum input HxW; overrides meta")
    parser.add_argument("--max-hw", type=_parse_hw, help="Max input HxW; overrides meta")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--device", default="cuda", help="Device for the trt_infer smoke test (default: cuda)")
    parser.add_argument("--no-verify", action="store_true", help="Skip the onnx_infer/trt_infer smoke test")
    args = parser.parse_args()

    export_and_build(
        args.checkpoint,
        args.output_dir,
        precision=args.precision,
        min_hw=args.min_hw,
        opt_hw=args.opt_hw,
        max_hw=args.max_hw,
        workspace_gb=args.workspace_gb,
        verify=not args.no_verify,
        device=args.device,
    )


if __name__ == "__main__":
    main()
