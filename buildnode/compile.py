"""Compile a ready-to-build ONNX graph into a TensorRT engine, without torch.

This is deliberately NOT ``friendy_chachkalica.ml.trt_export.cli.build_engine``.
That function is the full checkpoint→engine path: it can export a ``.pt`` to ONNX
and it can re-export an EfficientNMS graph from a torch model, so it imports torch
at module scope and cannot be loaded here at all.

What a build node does is the tail of that function — take a graph that is already
in its final form, derive the optimization profile from the meta, and compile —
and every piece of that is torch-free (``builder.py``, ``profile.py``,
``fp16_cast.py``, and the arch FP16 policy tables). This module is that tail, and
it writes the same three artifacts ``build_engine`` does so the output is
indistinguishable: ``<name>.engine``, a verbatim ``<name>.meta.json``, and the
``<name>.engine.json`` provenance sidecar that ``trt_infer.session`` checks.

**Which graphs are "ready to build".** rtdetr and rfdetr compile straight from
their standard ONNX. yolox, retinanet and fasterrcnn do not — their standard graph
bakes a data-dependent NMS that TensorRT cannot compile, so they must first be
re-exported as raw outputs plus an ``EfficientNMS_TRT`` node, and that re-export
needs the torch model. A node therefore cannot build those archs from a plain
``.onnx``; it needs the already-prepared graph, which is what the caller sends
(``prepared=True``) and what the baked person detector is.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

HW = Tuple[int, int]


class CompileError(RuntimeError):
    """A graph this node cannot compile, with a reason an operator can act on."""


def _trt_export():
    """The torch-free subset of the trainer's trt_export package."""
    from friendy_chachkalica.ml.trt_export.arch import (
        get_fp16_node_block,
        get_fp16_op_block,
        has_trt_prep,
        is_fp16_trusted,
    )
    from friendy_chachkalica.ml.trt_export.builder import build_engine_from_onnx
    from friendy_chachkalica.ml.trt_export.profile import profile_from_meta

    return {
        "build_engine_from_onnx": build_engine_from_onnx,
        "profile_from_meta": profile_from_meta,
        "get_fp16_op_block": get_fp16_op_block,
        "get_fp16_node_block": get_fp16_node_block,
        "is_fp16_trusted": is_fp16_trusted,
        "has_trt_prep": has_trt_prep,
    }


def read_meta(onnx_path: Path) -> Dict[str, Any]:
    meta_path = onnx_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise CompileError(
            f"No meta sidecar beside {onnx_path.name} (expected {meta_path.name}). "
            f"An ONNX without its Contract-B meta cannot be compiled or loaded."
        )
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CompileError(f"Unreadable meta sidecar {meta_path.name}: {exc}") from exc


def check_buildable(meta: Dict[str, Any], *, prepared: bool) -> None:
    """Raise if this node cannot compile ``meta``'s arch from the graph it was given.

    Called before any GPU work so the caller gets a clear 400 rather than a
    confusing failure several minutes into a build.
    """
    api = _trt_export()
    arch = meta.get("arch")
    if api["has_trt_prep"](arch) and not prepared:
        raise CompileError(
            f"arch {arch!r} cannot be compiled from a standard ONNX graph: its baked "
            f"NMS has to be replaced with an EfficientNMS_TRT node first, and that "
            f"re-export needs the .pt checkpoint and torch, which this node does not "
            f"have. Send the prepared graph instead (the trainer's /export_trt_onnx), "
            f"or build this model on the trainer."
        )


def compile_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    meta: Optional[Dict[str, Any]] = None,
    precision: str = "fp16",
    prepared: bool = False,
    input_hw: Optional[HW] = None,
    min_hw: Optional[HW] = None,
    opt_hw: Optional[HW] = None,
    max_hw: Optional[HW] = None,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    workspace_gb: float = 4.0,
) -> Path:
    """Compile ``onnx_path`` into ``engine_path``. Returns the engine path.

    ``input_hw`` pins a fully-static profile (min==opt==max) — the same knob the
    trainer's ``/export_trt`` exposes, and the only way Faster R-CNN builds FP16.
    ``min_hw``/``opt_hw``/``max_hw`` set the three profile corners individually
    (for a detector that must accept a range of frame sizes); they are ignored when
    ``input_hw`` is given. Any corner left ``None`` is derived from the meta by
    ``profile_from_meta``. ``prepared`` says the graph already carries its
    EfficientNMS node.
    """
    api = _trt_export()
    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    meta = meta if meta is not None else read_meta(onnx_path)
    arch = meta.get("arch")
    check_buildable(meta, prepared=prepared)

    if precision == "auto":
        precision = "fp16" if api["is_fp16_trusted"](arch) else "fp32"

    def _hw(value: Optional[HW]) -> Optional[HW]:
        return (int(value[0]), int(value[1])) if value else None

    if input_hw:
        static = _hw(input_hw)
        want_min = want_opt = want_max = static
    else:
        want_min, want_opt, want_max = _hw(min_hw), _hw(opt_hw), _hw(max_hw)

    prof_min, prof_opt, prof_max = api["profile_from_meta"](
        meta, min_hw=want_min, opt_hw=want_opt, max_hw=want_max
    )

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = api["build_engine_from_onnx"](
        onnx_path,
        engine_path,
        min_hw=prof_min,
        opt_hw=prof_opt,
        max_hw=prof_max,
        # Contract A's fixed input name; hardcoded rather than imported because
        # onnx_export.common (where INPUT_NAME lives) imports torch.
        input_name="pixel_values",
        precision=precision,
        extra_fp32_ops=api["get_fp16_op_block"](arch),
        node_block_substrings=api["get_fp16_node_block"](arch),
        workspace_gb=workspace_gb,
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
    )

    # The same self-contained trio build_engine writes, so a bundle assembled
    # around this engine is byte-for-byte the shape the runtime expects.
    meta_path = engine_path.with_suffix(".meta.json")
    if not meta_path.exists() or meta_path.resolve() != onnx_path.with_suffix(".meta.json").resolve():
        meta_path.write_text(json.dumps(meta, indent=2))

    # `built_by` + the GPU identity are what let the Django side recognise a
    # foreign engine and skip its load test with a useful message instead of
    # reporting the bundle as broken (training/services/bundles.py::_foreign_build).
    from . import gpu as gpu_info

    identity = gpu_info.describe()
    Path(str(engine_path) + ".json").write_text(
        json.dumps(
            {
                **provenance,
                "arch": arch,
                "efficientnms": bool(api["has_trt_prep"](arch)),
                "source": str(onnx_path),
                "requested_precision": precision,
                "built_by": "buildnode",
                "gpu_name": identity["gpu_name"],
                "compute_capability": identity["compute_capability"],
                "driver_version": identity["driver_version"],
            },
            indent=2,
        )
    )
    return engine_path


def built_precision(engine_path: Path, fallback: str) -> str:
    """What the engine ACTUALLY built at — an fp16 request can fall back to fp32."""
    try:
        return json.loads(Path(str(engine_path) + ".json").read_text()).get("precision", fallback)
    except (OSError, json.JSONDecodeError):
        return fallback
