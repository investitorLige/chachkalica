"""Read back what precision a built TensorRT engine actually runs in.

"The BF16 engine built successfully" is not evidence that anything ran in BF16 —
TensorRT is free to keep any layer in a higher precision, and on a strongly-typed
network (TRT >= 11) it keeps exactly the dtypes the ONNX graph asked for, which
is only ever a subset of the graph after AutoCast leaves the numerically
sensitive nodes in FP32. This module reads the engine's own layer information and
counts it.

Two caveats worth knowing before reading the numbers:

* **Layer info needs a DETAILED plan.** ``IEngineInspector`` can only report
  layer detail that the builder chose to retain, so the engine must have been
  built with ``config.profiling_verbosity = ProfilingVerbosity.DETAILED``
  (``build_engine_from_onnx(..., profiling_verbosity="detailed")``; ``build_trt.py``
  does this by default). Otherwise you get layer names and nothing else.
* **TRT 11 fuses aggressively.** Most of a DETR graph collapses into a handful of
  ``kgen``/myelin super-layers, so "how many layers are BF16" is a coarse
  question. The honest measure is over *tensors*: every layer's inputs and
  outputs carry a concrete ``Datatype``, and those are the values the kernels
  actually move. Both censuses are reported; the tensor one is the one to quote.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional


def load_engine_info(engine_path, *, runtime=None) -> dict:
    """Deserialize ``engine_path`` and return its EngineInspector JSON as a dict."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = runtime or trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(f"could not deserialize {engine_path}")
    inspector = engine.create_engine_inspector()
    raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    info = json.loads(raw)
    if isinstance(info, list):  # some versions return a bare layer list
        info = {"Layers": info}
    return info


def precision_census(info: dict) -> dict:
    """Count layers and tensors by datatype in an EngineInspector JSON dict.

    Returns ``{"layers": {dtype: n}, "tensors": {dtype: n}, "layer_types":
    {kind: n}, "num_layers": n}``. Datatype strings are TensorRT's own spelling
    (``Float``, ``Half``, ``BFloat16``, ``Int8``, ``Int32``, ...).
    """
    layers = info.get("Layers") or []
    layer_dtypes: Counter = Counter()
    tensor_dtypes: Counter = Counter()
    layer_types: Counter = Counter()
    seen_tensors: set = set()

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_types[str(layer.get("LayerType", "?"))] += 1
        outputs = layer.get("Outputs") or []
        # A layer's precision, as far as the plan reports it: an explicit
        # "Precision" field when present, else the dtype of its first output.
        precision = layer.get("Precision")
        if not precision and outputs and isinstance(outputs[0], dict):
            precision = outputs[0].get("Datatype")
        layer_dtypes[str(precision or "Unknown")] += 1

        for tensor in list(layer.get("Inputs") or []) + list(outputs):
            if not isinstance(tensor, dict):
                continue
            key = tensor.get("Name")
            if key in seen_tensors:
                continue
            seen_tensors.add(key)
            tensor_dtypes[str(tensor.get("Datatype", "Unknown"))] += 1

    return {
        "num_layers": len(layers),
        "layers": dict(layer_dtypes.most_common()),
        "tensors": dict(tensor_dtypes.most_common()),
        "layer_types": dict(layer_types.most_common()),
    }


def io_dtypes(engine_path) -> dict:
    """``{tensor_name: (mode, dtype)}`` for the engine's public I/O bindings."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    engine = trt.Runtime(logger).deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(f"could not deserialize {engine_path}")
    out = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        out[name] = (
            "input" if mode == trt.TensorIOMode.INPUT else "output",
            str(engine.get_tensor_dtype(name)).replace("DataType.", ""),
        )
    return out


def describe(engine_path, *, layer_limit: int = 0) -> dict:
    """Everything this module knows about one engine, as a plain dict."""
    info = load_engine_info(engine_path)
    census = precision_census(info)
    described = {
        "engine": str(engine_path),
        "size_bytes": Path(engine_path).stat().st_size,
        "io": {k: {"mode": v[0], "dtype": v[1]} for k, v in io_dtypes(engine_path).items()},
        **census,
    }
    if layer_limit:
        described["layer_detail"] = [
            {
                "name": layer.get("Name"),
                "type": layer.get("LayerType"),
                "outputs": [
                    {"name": t.get("Name"), "dtype": t.get("Datatype"), "format": t.get("Format")}
                    for t in (layer.get("Outputs") or [])
                    if isinstance(t, dict)
                ],
            }
            for layer in (info.get("Layers") or [])[:layer_limit]
            if isinstance(layer, dict)
        ]
    return described


def main() -> None:
    ap = argparse.ArgumentParser(description="Report what precision a TensorRT engine runs in")
    ap.add_argument("engine", help="Path to a .engine plan")
    ap.add_argument("--layers", type=int, default=0, help="Also dump the first N layers' detail")
    ap.add_argument("--json", dest="json_out", help="Write the full report to this path")
    args = ap.parse_args()

    report = describe(args.engine, layer_limit=args.layers)
    print(f"engine     : {report['engine']}  ({report['size_bytes'] / 1e6:.1f} MB)")
    print(f"IO         : " + ", ".join(
        f"{name}[{spec['mode']}]={spec['dtype']}" for name, spec in report["io"].items()
    ))
    print(f"layers     : {report['num_layers']}  by output dtype: {report['layers']}")
    print(f"tensors    : {report['tensors']}")
    print(f"layer types: {report['layer_types']}")
    for layer in report.get("layer_detail", []):
        dtypes = ",".join(str(o["dtype"]) for o in layer["outputs"])
        print(f"  {layer['type']:<10} {str(layer['name'])[:70]:<70} -> {dtypes}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
