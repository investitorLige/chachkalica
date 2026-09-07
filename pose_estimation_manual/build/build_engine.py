#!/usr/bin/env python3
"""Compile the RTMO pose ONNX into a TensorRT ``.engine`` + its provenance sidecar.

    python build/build_engine.py bundle/models/rtmo.trt.onnx -o bundle/models/rtmo.engine

Run this **inside the same TensorRT container the serving worker uses** — an
engine only deserializes under the TensorRT version that built it:

    docker run --rm --gpus all -v "$PWD:/work" -w /work \
        nvcr.io/nvidia/tensorrt:26.07-py3 \
        python build/build_engine.py bundle/models/rtmo.trt.onnx \
            -o bundle/models/rtmo.engine

Nothing in here is RTMO-specific except the defaults: a static ``[1,3,640,640]``
profile and the ``dets``/``keypoints`` output check at the end. The RTMO export
needs **no graph surgery** — TensorRT's own ONNX parser turns the exporter's
``NonMaxSuppression`` node into an internal NMS layer, so the unmodified
``end2end.onnx`` builds as-is (see ../README.md, "Why no graph surgery").

Writes ``<engine>.json`` next to the engine: precision, TensorRT version, GPU,
the profile it was optimized for. ``app_glue/smoke_engine.py`` and the admin both
read that sidecar to diagnose a version mismatch, so don't skip it.

**Precision.** TensorRT 11 networks are *strongly typed*: ``trt.BuilderFlag`` no
longer has an ``FP16`` member at all, so precision is a property of the ONNX
graph's own tensor dtypes, not a builder switch. ``--fp16`` therefore casts the
graph first (via ``onnxconverter_common``) and is **known to fail on this
graph** — see ../README.md, "Why this engine is fp32". It is kept here so the
failure is reproducible rather than folklore, and so the sidecar records
``requested_precision`` honestly when it falls back.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# 4 GiB of tactic workspace. The RTMO graph is ~176MB of weights and its NMS
# machinery is small; this is generous headroom, not a tuned value. Lower it if
# the build box has to share the card.
DEFAULT_WORKSPACE_GIB = 4


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("onnx", help="Path to the ONNX graph to compile.")
    parser.add_argument(
        "-o", "--out", default=None,
        help="Engine output path (default: the ONNX path with a .engine suffix, "
             "with a trailing '.trt' stripped from the stem).",
    )
    parser.add_argument("--arch", default="rtmo", help="Value recorded in the sidecar's 'arch'.")
    parser.add_argument(
        "--hw", type=int, default=640,
        help="Static input side. RTMO's export fixes H=W=640 in the graph itself; "
             "changing this only makes sense for a graph exported at another size.",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Static batch. The graph's batch axis is dynamic ('batch'), but every "
             "whole-frame bundle here runs one image per forward pass, and a wider "
             "profile costs build time and device memory for nothing.",
    )
    parser.add_argument("--workspace-gib", type=int, default=DEFAULT_WORKSPACE_GIB)
    parser.add_argument(
        "--fp16", action="store_true",
        help="Cast the graph to fp16 before building. Known to fail on this graph.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="TensorRT VERBOSE logging. Worth it the first time a build fails.",
    )
    return parser.parse_args(argv)


def _default_out(onnx_path: Path) -> Path:
    stem = onnx_path.stem
    if stem.endswith(".trt"):  # rtmo.trt.onnx -> rtmo.engine
        stem = stem[: -len(".trt")]
    return onnx_path.with_name(stem + ".engine")


def _cast_fp16(onnx_path: Path, out_dir: Path) -> Path:
    """Cast a graph to fp16 with ``onnxconverter_common`` and return the new path.

    Two casters were tried on the RTMO graph and both produced something
    TensorRT's parser rejects — ``onnxruntime.transformers.float16`` emits
    duplicate cast-node output names, and this one gets two nodes further before
    hitting a genuine Float/Half mismatch on an ElementWise Mul inside the
    decode/NMS block. Reproduced here rather than hidden.
    """
    from onnxconverter_common import float16  # build-only dep
    import onnx

    print(f"[build] casting {onnx_path.name} to fp16 (onnxconverter_common)")
    model = onnx.load(str(onnx_path))
    casted = float16.convert_float_to_float16(model, keep_io_types=True)
    out_path = out_dir / (onnx_path.stem + ".fp16.onnx")
    onnx.save(casted, str(out_path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=out_path.name + ".data")
    print(f"[build] wrote {out_path}")
    return out_path


def main(argv=None) -> int:
    args = _parse_args(argv)

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        print(f"FAIL: no such ONNX: {onnx_path}", file=sys.stderr)
        return 2
    engine_path = Path(args.out).resolve() if args.out else _default_out(onnx_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    import tensorrt as trt

    print(f"[build] tensorrt {trt.__version__}")
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    # The engines for the fused-NMS archs embed EfficientNMS_TRT; RTMO does not,
    # but registering the standard plugins is free and keeps this script usable
    # for the other archs in vendor/onnx_infer/arch/.
    trt.init_libnvinfer_plugins(logger, "")

    requested_precision = "fp16" if args.fp16 else "fp32"
    source_onnx = onnx_path
    precision_method = None
    if args.fp16:
        try:
            source_onnx = _cast_fp16(onnx_path, engine_path.parent)
            precision_method = "onnxconverter_common.float16"
        except Exception as exc:  # noqa: BLE001 - the fallback is the point
            print(f"[build] fp16 cast failed ({type(exc).__name__}: {exc}); building fp32")
            source_onnx, precision_method = onnx_path, None

    builder = trt.Builder(logger)
    # No flags: a TensorRT 11 network is strongly typed, and its dtypes come from
    # the ONNX graph. There is no BuilderFlag.FP16 to set here.
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    print(f"[build] parsing {source_onnx}")
    if not parser.parse_from_file(str(source_onnx)):
        for i in range(parser.num_errors):
            print(f"  parser error {i}: {parser.get_error(i)}", file=sys.stderr)
        print("\nFAIL: the ONNX did not parse. If this mentions a Float/Half mismatch "
              "you are on the --fp16 path; drop it and build fp32.", file=sys.stderr)
        return 1

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"[build] input  {input_name} {tuple(input_tensor.shape)}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"[build] output {out.name} {tuple(out.shape)}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib << 30)

    # One static shape: min == opt == max. A dynamic profile is what makes
    # TensorRT allocate device memory for the widest case, and it has bitten this
    # project before (see TrtModel's execution-context error message).
    shape = (args.batch, 3, args.hw, args.hw)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min=shape, opt=shape, max=shape)
    config.add_optimization_profile(profile)
    print(f"[build] profile {input_name} min=opt=max={shape}")

    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("\nFAIL: build_serialized_network returned None — TensorRT logged the "
              "reason above. Re-run with --verbose.", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started
    engine_path.write_bytes(bytes(serialized))
    print(f"[build] wrote {engine_path} ({engine_path.stat().st_size / 1e6:.0f} MB) "
          f"in {elapsed:.0f}s")

    # ── the provenance sidecar ──
    gpu_name = None
    try:
        import torch

        gpu_name = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - purely informational
        pass

    precision = "fp16" if precision_method else "fp32"
    sidecar = {
        "precision": precision,
        "precision_method": precision_method,
        "requested_precision": requested_precision,
        "tensorrt_version": trt.__version__,
        "input_name": input_name,
        "gpu": gpu_name,
        "builder": "build/build_engine.py",
        "arch": args.arch,
        "nms": "native_onnx_nms_layer",
        "profile": {"min": [args.hw, args.hw], "opt": [args.hw, args.hw], "max": [args.hw, args.hw]},
        "batch": {"min": args.batch, "opt": args.batch, "max": args.batch},
        "source": source_onnx.name,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    sidecar_path = Path(str(engine_path) + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"[build] wrote {sidecar_path}")
    if precision != requested_precision:
        print(f"[build] NOTE: requested {requested_precision}, built {precision} — "
              f"recorded in the sidecar so it is never mistaken for the other.")

    # ── the one check worth doing here: did the outputs survive the parse? ──
    output_names = {network.get_output(i).name for i in range(network.num_outputs)}
    if args.arch == "rtmo" and output_names != {"dets", "keypoints"}:
        print(f"\nWARN: expected outputs {{dets, keypoints}}, got {output_names}. "
              f"vendor/trt_infer/session.py detects this arch by exactly those two "
              f"names — anything else falls through to the passthrough path and will "
              f"be mis-read as (boxes, scores, labels).", file=sys.stderr)

    print("\nOK. Next: python build/verify_engine.py "
          f"{engine_path} --onnx {onnx_path} --images <dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
