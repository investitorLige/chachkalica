"""Build a TensorRT engine from an ONNX graph at a named precision — FP32, FP16
or BF16 — picking the mechanism that the *installed* TensorRT version actually
supports, and reporting what was really built.

    python -m friendy_chachkalica.ml.trt_export.build_trt \\
        --onnx model.onnx --output model.engine --precision fp16

    python -m friendy_chachkalica.ml.trt_export.build_trt \\
        --onnx model.onnx --output model.engine --precision bf16

Two eras of TensorRT, two mechanisms — detected at runtime, never assumed:

* **TensorRT <= 10 (weakly typed).** Precision is a builder flag. This is the
  path D-FINE's own README documents for FP16
  (``trtexec --onnx=model.onnx --saveEngine=model.engine --fp16``), and the one
  its published T4 latency numbers were measured on (TensorRT 10.4). The ONNX
  graph stays FP32 and TensorRT picks FP16 tactics where it likes. BF16 uses the
  analogous ``--bf16`` / ``BuilderFlag.BF16`` when the installed build exposes it.
* **TensorRT >= 11 (strongly typed).** NVIDIA removed every precision flag
  (``--fp16``, ``--bf16``, ``--int8``, ``--best``) — the network's types *are*
  the ONNX graph's types. NVIDIA's 10.x→11.x migration guide gives the
  replacement: convert the graph offline with ModelOpt AutoCast, then build.
  Equivalent CLI::

      python -m modelopt.onnx.autocast --onnx_path model.onnx \\
          --output_path model_bf16.onnx --low_precision_type bf16 --keep_io_types
      trtexec --onnx=model_bf16.onnx --saveEngine=model.engine

  (This repo builds through the TensorRT Python API rather than ``trtexec``,
  because the pip ``tensorrt`` wheel ships no ``trtexec`` binary — the API calls
  are the same builder underneath.)

BF16 **never** silently degrades: if ModelOpt is missing, if AutoCast emits an
invalid graph, or if the builder returns no plan, this fails with the reason.
FP16 keeps the repo's long-standing "retry at FP32 and record it" behavior.

The default FP16 backend on TRT >= 11 stays this repo's ``fp16_cast`` graph cast
(``onnxruntime.transformers.float16``), which is what every FP16 number recorded
in ``arch/__init__.py`` was measured with. ``--fp16-backend autocast`` builds the
NVIDIA-documented mixed FP32/FP16 variant instead, which keeps high-magnitude
nodes in FP32 — worth trying for an arch whose plain FP16 cast fails parity.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    from .builder import build_engine_from_onnx
    from .profile import profile_from_meta
    from . import engine_inspect
    from ..onnx_export.common import INPUT_NAME
except ImportError:  # run flat (cwd on sys.path), mirroring cli.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.trt_export.builder import build_engine_from_onnx  # type: ignore
    from ml.trt_export.profile import profile_from_meta  # type: ignore
    from ml.trt_export import engine_inspect  # type: ignore
    from ml.onnx_export.common import INPUT_NAME  # type: ignore

HW = Tuple[int, int]


def trt_environment() -> dict:
    """Versions / hardware this build is happening on — recorded in the report."""
    env = {"python": platform.python_version(), "trtexec": shutil.which("trtexec")}
    try:
        import tensorrt as trt

        env["tensorrt"] = trt.__version__
        env["tensorrt_major"] = int(str(trt.__version__).split(".")[0])
        env["strongly_typed"] = not hasattr(trt.BuilderFlag, "FP16")
        env["builder_flags"] = sorted(
            f for f in dir(trt.BuilderFlag) if f.isupper()
        )
        env["datatypes"] = sorted(f for f in dir(trt.DataType) if f.isupper())
    except ImportError:
        env["tensorrt"] = None
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name()
            env["compute_capability"] = list(torch.cuda.get_device_capability())
    except ImportError:
        pass
    try:
        from .modelopt_cast import modelopt_version
    except ImportError:  # flat run
        from ml.trt_export.modelopt_cast import modelopt_version  # type: ignore
    env["modelopt"] = modelopt_version()
    return env


def _graph_input_hw(onnx_path: Path, input_name: str) -> Optional[HW]:
    """Static H/W baked into the graph's input, or ``None`` if it is dynamic."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    for info in model.graph.input:
        if info.name != input_name and len(model.graph.input) > 1:
            continue
        dims = info.type.tensor_type.shape.dim
        if len(dims) != 4:
            return None
        h, w = dims[2], dims[3]
        if h.HasField("dim_value") and w.HasField("dim_value"):
            return (int(h.dim_value), int(w.dim_value))
        return None
    return None


def resolve_profile(
    onnx_path: Path,
    *,
    input_name: str,
    min_hw: Optional[HW],
    opt_hw: Optional[HW],
    max_hw: Optional[HW],
) -> Tuple[HW, HW, HW, str]:
    """Pick the optimization profile: explicit flags > meta sidecar > graph shape.

    Returns ``(min, opt, max, source)``. A fully-dynamic graph with no meta and no
    flags is an error rather than a guess — the profile decides both what shapes
    the engine accepts and (historically, for some archs) whether a low-precision
    build finds tactics at all.
    """
    meta_path = onnx_path.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        prof = profile_from_meta(meta, min_hw=min_hw, opt_hw=opt_hw, max_hw=max_hw)
        return (*prof, f"meta sidecar ({meta_path.name})")

    if min_hw and opt_hw and max_hw:
        return (min_hw, opt_hw, max_hw, "explicit --min-hw/--opt-hw/--max-hw")

    static = _graph_input_hw(onnx_path, input_name)
    if static:
        return (min_hw or static, opt_hw or static, max_hw or static, "static ONNX input shape")

    raise SystemExit(
        f"cannot derive an optimization profile for {onnx_path.name}: no "
        f"{meta_path.name} beside it and the graph's input H/W is dynamic. Pass "
        f"--min-hw/--opt-hw/--max-hw explicitly."
    )


def build(
    onnx_path,
    engine_path,
    *,
    precision: str = "fp16",
    input_name: str = INPUT_NAME,
    min_hw: Optional[HW] = None,
    opt_hw: Optional[HW] = None,
    max_hw: Optional[HW] = None,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    workspace_gb: float = 4.0,
    fp16_backend: str = "graph_cast",
    autocast_options: Optional[dict] = None,
    inspect: bool = True,
    profiling_verbosity: str = "detailed",
    tf32: bool = True,
) -> dict:
    """Build one engine and return a report dict (also written as ``<engine>.json``)."""
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    precision = precision.lower()

    prof_min, prof_opt, prof_max, profile_source = resolve_profile(
        onnx_path, input_name=input_name, min_hw=min_hw, opt_hw=opt_hw, max_hw=max_hw
    )
    env = trt_environment()
    backend = "autocast" if precision == "bf16" else fp16_backend

    print(f"[build_trt] TensorRT {env.get('tensorrt')} "
          f"({'strongly typed' if env.get('strongly_typed') else 'weakly typed'}), "
          f"GPU {env.get('gpu')} sm_{''.join(str(c) for c in env.get('compute_capability', []))}")
    print(f"[build_trt] {onnx_path.name} -> {engine_path.name}  precision={precision} "
          f"backend={backend if precision != 'fp32' else '-'} tf32={'on' if tf32 else 'off'}")
    print(f"[build_trt] profile from {profile_source}: min={prof_min} opt={prof_opt} max={prof_max} "
          f"batch {min_batch}/{opt_batch}/{max_batch}")

    provenance = build_engine_from_onnx(
        onnx_path,
        engine_path,
        min_hw=prof_min,
        opt_hw=prof_opt,
        max_hw=prof_max,
        input_name=input_name,
        precision=precision,
        workspace_gb=workspace_gb,
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        cast_backend="auto" if precision == "fp32" else backend,
        autocast_options=autocast_options,
        profiling_verbosity=profiling_verbosity,
        tf32=tf32,
    )

    if provenance["precision"] != precision:
        # Only reachable for fp16 (bf16 raises inside the builder) — say it out
        # loud rather than leaving it in the sidecar for someone to notice later.
        print(f"[build_trt] WARNING: requested {precision}, engine actually built "
              f"{provenance['precision']}")

    report = {
        "requested_precision": precision,
        "environment": env,
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "profile_source": profile_source,
        **provenance,
    }
    if inspect:
        try:
            report["inspection"] = engine_inspect.describe(engine_path)
        except Exception as exc:  # inspection must never fail a good build
            report["inspection_error"] = str(exc)

    # Self-contained artifact, same convention as cli.py: the meta sidecar rides
    # along so trt_infer can load the engine on its own.
    meta_path = onnx_path.with_suffix(".meta.json")
    if meta_path.exists():
        target = engine_path.with_suffix(".meta.json")
        if target.resolve() != meta_path.resolve():
            target.write_text(meta_path.read_text())

    report_path = Path(str(engine_path) + ".json")
    report_path.write_text(json.dumps(report, indent=2))

    size_mb = engine_path.stat().st_size / 1e6
    print(f"[build_trt] wrote {engine_path} ({size_mb:.1f} MB), report -> {report_path.name}")
    insp = report.get("inspection")
    if insp:
        print(f"[build_trt] engine tensors by dtype: {insp['tensors']}")
        print(f"[build_trt] engine IO: " + ", ".join(
            f"{n}={s['dtype']}" for n, s in insp["io"].items()))
    return report


def _parse_hw(value: Optional[str]) -> Optional[HW]:
    if not value:
        return None
    parts = value.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected HxW (e.g. 640x640), got {value!r}")
    return (int(parts[0]), int(parts[1]))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a TensorRT engine at fp32 / fp16 / bf16 (version-aware)",
    )
    ap.add_argument("--onnx", required=True, help="Input ONNX graph")
    ap.add_argument("--output", "-o", help="Engine output path (default: <onnx>.<precision>.engine)")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    ap.add_argument("--input-name", default=INPUT_NAME, help=f"Graph input name (default {INPUT_NAME})")
    ap.add_argument("--min-hw", type=_parse_hw)
    ap.add_argument("--opt-hw", type=_parse_hw)
    ap.add_argument("--max-hw", type=_parse_hw)
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--opt-batch", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--workspace-gb", type=float, default=4.0)
    ap.add_argument("--fp16-backend", choices=["graph_cast", "autocast"], default="graph_cast",
                    help="TRT>=11 only: how to produce the fp16 graph (default: this repo's "
                         "onnxruntime float16 cast; 'autocast' uses ModelOpt's mixed FP32/FP16)")
    ap.add_argument("--data-max", type=float, default=512.0,
                    help="AutoCast: keep nodes whose I/O magnitude exceeds this in FP32")
    ap.add_argument("--init-max", type=float, default=65504.0,
                    help="AutoCast: keep nodes with initializers above this in FP32")
    ap.add_argument("--calibration-data",
                    help="AutoCast: .npz of real inputs for magnitude classification "
                         "(random data is used without it)")
    ap.add_argument("--nodes-to-exclude", nargs="*", default=None,
                    help="AutoCast: regex patterns of node names to keep in FP32")
    ap.add_argument("--op-types-to-exclude", nargs="*", default=None,
                    help="AutoCast: op types to keep in FP32")
    ap.add_argument("--max-depth-of-reduction", type=int, default=None,
                    help="AutoCast: keep deep-reduction nodes (big matmuls/convs) in FP32")
    ap.add_argument("--io-low-precision", action="store_true",
                    help="Let the graph's public inputs/outputs take the low precision too "
                         "(default keeps them FP32, so trt_infer is unchanged)")
    ap.add_argument("--no-tf32", action="store_true",
                    help="Disable TensorRT's TF32 tactics (on by default in every TRT). "
                         "A default 'fp32' engine already runs convs/matmuls at TF32's "
                         "10-bit mantissa — pass this for a true FP32 reference.")
    ap.add_argument("--no-inspect", action="store_true", help="Skip the engine precision census")
    ap.add_argument("--profiling-verbosity", choices=["default", "detailed"], default="detailed")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    engine_path = Path(args.output) if args.output else onnx_path.with_suffix(
        f".{args.precision}.engine"
    )

    autocast_options = {
        "keep_io_types": not args.io_low_precision,
        "data_max": args.data_max,
        "init_max": args.init_max,
        "calibration_data": args.calibration_data,
        "nodes_to_exclude": args.nodes_to_exclude,
        "op_types_to_exclude": args.op_types_to_exclude,
        "max_depth_of_reduction": args.max_depth_of_reduction,
    }

    build(
        onnx_path,
        engine_path,
        precision=args.precision,
        input_name=args.input_name,
        min_hw=args.min_hw,
        opt_hw=args.opt_hw,
        max_hw=args.max_hw,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_gb=args.workspace_gb,
        fp16_backend=args.fp16_backend,
        autocast_options=autocast_options,
        inspect=not args.no_inspect,
        profiling_verbosity=args.profiling_verbosity,
        tf32=not args.no_tf32,
    )


if __name__ == "__main__":
    main()
