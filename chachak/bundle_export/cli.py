"""Export a chachak pipeline as a self-contained, runnable bundle.

Usage::

    python -m chachak.bundle_export.cli chachak/configs/people_detect_first.yaml
    python -m chachak.bundle_export.cli request.yaml -o dist/ppe --conf 0.4
    python -m chachak.bundle_export.cli request.yaml --format engine

``onnx_export`` makes one *model* portable. This makes the whole *pipeline*
portable: every checkpoint the request names (the model, any extra models, and
the person detector) is exported to ONNX — optionally also compiled to a
TensorRT engine — and dropped next to a ``pipeline.json`` carrying every
non-path field of the request, the inference runtime, and an ``infer.py`` whose
only routine argument is ``--conf``. The recipient needs torch, numpy, pillow,
pyyaml and onnxruntime; nothing from this repo.

    python infer.py path/to/images --conf 0.4
"""

from __future__ import annotations

import argparse
import json
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import PipelineConfig, load_pipeline_config
from ..pipeline import _inference_score_threshold
from ..run import _needs_detector
from .manifest import build_manifest
from .readme import render_readme
from .vendor import vendor_runtime

_TEMPLATES = Path(__file__).resolve().parent / "templates"

# Below this, a threshold is a metric-sweep setting (chachak defaults to 0.001 so
# mAP sees the full precision/recall curve), not something to deploy with.
_SWEEP_THRESHOLD = 0.01


def _artifact_meta(path: Path) -> Dict[str, Any]:
    """The Contract-B sidecar next to an exported artifact."""
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise SystemExit(f"[bundle] Expected a meta sidecar next to {path.name}: {meta_path}")
    return json.loads(meta_path.read_text())


def _copy_artifact(source: Path, destination: Path) -> None:
    """Copy an already-exported artifact plus its ``.meta.json`` sidecar."""
    meta = source.with_suffix(".meta.json")
    if not meta.exists():
        raise SystemExit(
            f"[bundle] {source} has no {meta.name} beside it, so it cannot be bundled. "
            f"Re-export it with friendy_chachkalica.ml.onnx_export."
        )
    shutil.copy2(source, destination)
    shutil.copy2(meta, destination.with_suffix(".meta.json"))


def _export_role(
    source: Path,
    models_dir: Path,
    role: str,
    *,
    want_engine: bool,
    precision: str,
    batch: int,
) -> Dict[str, Any]:
    """Put one checkpoint's artifacts in the bundle. Returns per-format info.

    ``source`` may be a ``.pt`` (exported here), or an already-exported ``.onnx`` /
    ``.engine`` (copied with its sidecar) — chachak requests legitimately point at
    any of the three.
    """
    from friendy_chachkalica.ml.onnx_export.cli import export_checkpoint

    suffix = source.suffix.lower()
    onnx_path = models_dir / f"{role}.onnx"
    engine_path = models_dir / f"{role}.engine"
    produced: Dict[str, Path] = {}

    if suffix == ".pt":
        print(f"[bundle] {role}: exporting {source.name} -> {onnx_path.name}")
        export_checkpoint(source, onnx_path)
        produced["onnx"] = onnx_path
    elif suffix == ".onnx":
        print(f"[bundle] {role}: copying exported {source.name}")
        _copy_artifact(source, onnx_path)
        produced["onnx"] = onnx_path
    elif suffix == ".engine":
        # An engine can't be decompiled back to ONNX, so such a bundle is
        # engine-only and inherits the engine's hardware lock-in.
        print(f"[bundle] {role}: copying prebuilt engine {source.name}")
        _copy_artifact(source, engine_path)
        produced["engine"] = engine_path
    else:
        raise SystemExit(
            f"[bundle] {role}: expected a .pt checkpoint or an exported .onnx/.engine, "
            f"got {source}"
        )

    arch = _artifact_meta(next(iter(produced.values())))["arch"]
    max_batch: Dict[str, Optional[int]] = {"onnx": None}

    if want_engine and "engine" not in produced:
        max_batch["engine"] = _build_engine(
            produced["onnx"], engine_path, source, arch, precision=precision, batch=batch
        )
        produced["engine"] = engine_path
    elif "engine" in produced:
        max_batch["engine"] = _engine_profile_batch(engine_path)

    meta = _artifact_meta(next(iter(produced.values())))
    return {
        "paths": produced,
        "max_batch": max_batch,
        "arch": arch,
        "classes": meta.get("class_map") or {},
        "source": str(source),
    }


def _batchable_in_trt(arch: str) -> bool:
    """Whether this arch's engine decodes a ``B > 1`` submission correctly.

    ``TrtAdapter`` stacks same-shaped images into one engine call. The EfficientNMS
    archs emit fixed-size per-image outputs plus a ``num_detections`` count, which
    ``trt_infer.session._unpack_efficientnms`` splits per image. The passthrough
    archs compile the standard ONNX graph, whose outputs carry a single detection
    axis and no batch dimension — there is nothing to split them by. So only the
    former get a batch profile wider than 1.
    """
    from friendy_chachkalica.ml.trt_export.arch import get_trt_prep

    return get_trt_prep(arch) is not None


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    source: Path,
    arch: str,
    *,
    precision: str,
    batch: int,
) -> int:
    """Compile the bundled ONNX into ``engine_path``; returns its max batch."""
    from friendy_chachkalica.ml.trt_export.cli import _load_adapter, build_engine

    max_batch = batch if _batchable_in_trt(arch) else 1
    if max_batch == 1 and batch > 1:
        print(
            f"[bundle] {arch}: engine built with a batch-1 profile (its graph has no "
            f"batch axis on the detection outputs); the bundle will cap this model's "
            f"batch at 1 when running the engine"
        )
    # The EfficientNMS archs rebuild their TRT graph from the torch model, so they
    # need the adapter when the source ONNX isn't beside its checkpoint (ours isn't
    # — it lives in the bundle).
    adapter = None
    if _batchable_in_trt(arch):
        if source.suffix.lower() != ".pt":
            raise SystemExit(
                f"[bundle] {arch} rebuilds its TensorRT graph with a fused-NMS plugin, "
                f"which needs the .pt checkpoint. Point the request at the checkpoint "
                f"instead of {source.name}, or export with --format onnx."
            )
        adapter = _load_adapter(source)
    build_engine(
        onnx_path,
        engine_path,
        precision=precision,
        adapter=adapter,
        min_batch=1,
        opt_batch=max_batch,
        max_batch=max_batch,
    )
    return max_batch


def _engine_profile_batch(engine_path: Path) -> int:
    """Max batch of a prebuilt engine, read off its optimization profile."""
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        name = next(
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        )
        return int(engine.get_tensor_profile_shape(name, 0)[-1][0])
    except Exception as error:  # no TensorRT here, or an unreadable plan
        print(
            f"[bundle] Could not read {engine_path.name}'s batch profile ({error}); "
            f"assuming 1"
        )
        return 1


def export_bundle(
    request: str | Path,
    output_dir: str | Path | None = None,
    *,
    fmt: str = "onnx",
    conf: Optional[float] = None,
    precision: str = "auto",
    overwrite: bool = False,
) -> Path:
    """Export the pipeline described by ``request`` (a chachak YAML) as a bundle.

    ``fmt`` is ``"onnx"`` (portable, the default) or ``"engine"``. An engine bundle's
    ``models/`` directory still gets the portable ONNX files copied in (in case
    someone wants to rebuild an engine for different hardware), but ``pipeline.json``
    only points ``infer.py`` at the ``.engine`` paths — engines only run on the GPU
    and TensorRT version they were built on, and there is no automatic ONNX
    fallback at run time.

    ``conf`` overrides the confidence the bundle ships as its ``--conf`` default;
    it defaults to the request's own inference threshold.
    """
    config: PipelineConfig = load_pipeline_config(request)
    return export_bundle_for_config(
        config, output_dir, fmt=fmt, conf=conf, precision=precision, overwrite=overwrite,
        source=str(Path(request).resolve()),
    )


def export_bundle_for_config(
    config: PipelineConfig,
    output_dir: str | Path | None = None,
    *,
    fmt: str = "onnx",
    conf: Optional[float] = None,
    precision: str = "auto",
    overwrite: bool = False,
    source: Optional[str] = None,
) -> Path:
    """Export the pipeline described by an in-memory ``config`` as a bundle.

    Same as :func:`export_bundle`, minus the YAML load — for callers that already
    hold a :class:`~chachak.config.PipelineConfig` (e.g. a caller building one from
    a database record instead of a request file). See :func:`export_bundle` for
    what ``fmt`` and ``conf`` do. ``source`` is recorded under the manifest's
    ``provenance.source_request`` — a request YAML path, or ``None`` when there
    isn't one (e.g. a config built from a database record).
    """
    bundle_dir = Path(output_dir) if output_dir else config.output_dir / "bundle"
    bundle_dir = bundle_dir.resolve()

    if bundle_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"[bundle] {bundle_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(bundle_dir)
    models_dir = bundle_dir / "models"
    models_dir.mkdir(parents=True)

    want_engine = fmt == "engine"
    roles: List[Tuple[str, Path, int]] = [
        ("model", config.model_checkpoint, config.infer_batch_size)
    ]
    for index, extra in enumerate(config.extra_checkpoints, start=1):
        roles.append((f"extra_{index}", extra, config.infer_batch_size))
    if _needs_detector(config):
        roles.append(("detector", config.detector.checkpoint, config.detector.batch_size))

    exported = {
        role: _export_role(
            Path(source), models_dir, role,
            want_engine=want_engine, precision=precision, batch=batch,
        )
        for role, source, batch in roles
    }

    # Only offer a format every role has: a pipeline can't run half on ONNX.
    artifacts: Dict[str, Dict[str, Any]] = {}
    for candidate in ("onnx", "engine"):
        if not all(candidate in info["paths"] for info in exported.values()):
            continue
        artifacts[candidate] = {
            "model": exported["model"]["paths"][candidate].relative_to(bundle_dir).as_posix(),
            "extra_models": [
                exported[role]["paths"][candidate].relative_to(bundle_dir).as_posix()
                for role in exported
                if role.startswith("extra_")
            ],
            "detector": (
                exported["detector"]["paths"][candidate].relative_to(bundle_dir).as_posix()
                if "detector" in exported
                else None
            ),
            "max_batch": {
                "model": min(
                    [
                        info["max_batch"].get(candidate)
                        for role, info in exported.items()
                        if not role.startswith("detector")
                        and info["max_batch"].get(candidate) is not None
                    ]
                    or [None]
                ),
                "detector": (
                    exported["detector"]["max_batch"].get(candidate)
                    if "detector" in exported
                    else None
                ),
            },
        }
    if not artifacts:
        raise SystemExit(
            "[bundle] No single format is available for every role, so the pipeline "
            "could not be bundled. Point the request at .pt checkpoints (or matching "
            "exported artifacts) for all of: "
            + ", ".join(f"{role} ({'/'.join(info['paths'])})" for role, info in exported.items())
        )

    default_format = "engine" if want_engine and "engine" in artifacts else "onnx"
    if default_format not in artifacts:
        default_format = next(iter(artifacts))
        engine_only = [
            role for role, info in exported.items() if "onnx" not in info["paths"]
        ]
        print(
            f"[bundle] WARNING: this bundle is {default_format}-only, not portable. "
            f"{', '.join(engine_only)} was given as a prebuilt .engine, which cannot be "
            f"decompiled back to ONNX, so no ONNX bundle is possible. The result runs "
            f"only on the GPU model + TensorRT version those engines were built for. "
            f"Point the request at the .pt checkpoint instead to get a portable bundle."
        )

    default_conf = float(conf) if conf is not None else _inference_score_threshold(config)
    if default_conf < _SWEEP_THRESHOLD:
        print(
            f"[bundle] WARNING: shipping --conf default {default_conf} — that is a "
            f"metric-sweep threshold (it keeps nearly every raw detection so mAP can "
            f"see the full curve), not a deployment one. Pass --conf to set the value "
            f"the bundle should default to."
        )

    manifest = build_manifest(
        config,
        artifacts=artifacts,
        default_format=default_format,
        default_conf=default_conf,
        provenance={
            "source_request": source,
            "models": {
                role: {
                    "checkpoint": info["source"],
                    "arch": info["arch"],
                    "classes": info["classes"],
                }
                for role, info in exported.items()
            },
        },
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    (bundle_dir / "pipeline.json").write_text(json.dumps(manifest, indent=2))

    runtime_entries = vendor_runtime(bundle_dir / "runtime", include_trt="engine" in artifacts)

    entrypoint = bundle_dir / "infer.py"
    shutil.copy2(_TEMPLATES / "infer.py", entrypoint)
    entrypoint.chmod(entrypoint.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    shutil.copy2(_TEMPLATES / "requirements.txt", bundle_dir / "requirements.txt")
    (bundle_dir / "README.md").write_text(render_readme(manifest, exported))

    print(f"[bundle] Wrote {bundle_dir}")
    print(f"[bundle]   pipeline.json     {manifest['pipeline']}, "
          f"{len(manifest['classes'])} classes, conf default {default_conf}")
    print(f"[bundle]   models/           {', '.join(sorted(p.name for info in exported.values() for p in info['paths'].values()))}")
    print(f"[bundle]   runtime/          {', '.join(runtime_entries)}")
    print(f"[bundle]   formats           {', '.join(sorted(artifacts))} (default {default_format})")
    print(f"[bundle] Run it with: cd {bundle_dir} && python infer.py IMAGES --conf {default_conf}")
    return bundle_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a chachak pipeline as a self-contained runnable bundle"
    )
    parser.add_argument("request", help="Path to a chachak pipeline request YAML")
    parser.add_argument(
        "--output", "-o", help="Bundle directory (default: <request output_dir>/bundle)"
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "engine"],
        default="onnx",
        help="Artifact format to prefer. 'engine' also bundles the portable ONNX, "
             "and requires a GPU with TensorRT (default: onnx)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        help="Confidence threshold the bundle defaults to (default: the request's)",
    )
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto",
                        help="TensorRT precision, for --format engine")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing bundle")
    args = parser.parse_args()

    export_bundle(
        args.request,
        args.output,
        fmt=args.format,
        conf=args.conf,
        precision=args.precision,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
