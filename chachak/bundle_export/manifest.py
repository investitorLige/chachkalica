"""``pipeline.json`` — Contract C: the whole pipeline, portable.

``onnx_export`` writes Contract B (``meta.json``): everything needed to run *one
graph* — preprocessing, class map, output layout. That is enough for a single
model but says nothing about the pipeline wrapped around it: tiling geometry, the
person detector and its crop expansion, the NMS used to merge overlapping
sub-region predictions. Those live only in a chachak request YAML, whose paths
point at the exporter's own filesystem.

Contract C is that request, made portable, and flat: consumers (e.g.
mercury_watchroom's ``Bundle.clean()``) validate a top-level string ``pipeline``
and a top-level ``artifacts.model`` path, not a nested object.

* every non-path field of :class:`chachak.config.PipelineConfig` except
  ``pipeline`` itself, spread as top-level manifest keys, carried verbatim so
  the consumer runs the geometry the pipeline was tuned with;
* ``pipeline`` — the plain pipeline name (e.g. ``"people_detect_first"``), not a
  nested object;
* ``artifacts`` — bundle-relative paths to the exported model / detector /
  extra models, for whichever single format (``onnx`` or ``engine``) was
  primary at export time — replacing the host checkpoint paths;
* ``defaults`` — the confidence thresholds the bundle ships with, so a consumer
  who passes no ``--conf`` reproduces the exported pipeline exactly;
* ``provenance`` — the host paths and architectures the bundle came from, for
  traceability only; nothing reads it back.

The portable fields are derived from ``asdict(config)`` minus the path keys
rather than an allow-list, so a field added to ``PipelineConfig`` flows into new
bundles automatically instead of being silently dropped.

This module is vendored into the bundle as ``runtime/bundle_manifest.py`` and is
the *only* definition of the contract: the exporter calls
:func:`build_manifest`, the bundle's ``infer.py`` calls
:func:`pipeline_request_from_manifest`. ``SCHEMA_VERSION`` is guarded on read so
an older ``infer.py`` refuses a newer manifest loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

# Keys of PipelineConfig that name a file on the *exporter's* machine. They are
# replaced by `artifacts` (the bundled models) or supplied at run time (the
# images to run on, where to write results), and are recorded under `provenance`.
_PATH_KEYS = ("model_checkpoint", "extra_checkpoints", "images", "labels", "output_dir")
_DETECTOR_PATH_KEYS = ("checkpoint",)

# Formats a bundle can carry, most-preferred first.
FORMATS = ("engine", "onnx")


class ManifestError(RuntimeError):
    """The manifest is malformed, or a newer schema than this reader supports."""


def _plain(value: Any) -> Any:
    """JSON-safe deep copy: ``Path`` -> str, tuples -> lists, dict keys -> str."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def build_manifest(
    config,
    *,
    artifacts: Dict[str, Dict[str, Any]],
    default_format: str,
    default_conf: float,
    provenance: Dict[str, Any],
    created_utc: str,
    gpu_infer: bool = False,
) -> Dict[str, Any]:
    """Assemble ``pipeline.json`` from a :class:`chachak.config.PipelineConfig`.

    ``artifacts`` (the parameter) maps a format name to ``{"model": rel,
    "detector": rel|None, "extra_models": [rel, ...], "max_batch": {...}}`` with
    bundle-relative POSIX paths — one entry per format the exporter actually
    produced. Only ``default_format``-or-``"engine"``-preferred entry survives
    into the manifest's own flat ``artifacts`` key (see ``primary_format``
    below); the manifest never nests artifacts by format.
    ``default_conf`` is the model-stage confidence the bundle ships with — what
    ``infer.py`` uses when the consumer passes no ``--conf``.

    ``gpu_infer`` records that this bundle also carries the GPU-only ``infer_gpu.py``
    entrypoint. It is written under ``bundle`` rather than at the top level because
    ``test_bundle_export.py`` asserts the top-level key set exactly (against ``PipelineConfig``'s
    fields) while nothing pins ``bundle``'s keys — and it is **omitted entirely when false**, so
    an ordinary bundle's ``pipeline.json`` is byte-for-byte what it was before this option
    existed.
    """
    from dataclasses import asdict

    if default_format not in artifacts:
        raise ManifestError(
            f"default_format {default_format!r} has no entry in artifacts "
            f"({', '.join(sorted(artifacts)) or 'none'})"
        )

    pipeline = _plain(asdict(config))
    for key in _PATH_KEYS:
        pipeline.pop(key, None)
    for key in _DETECTOR_PATH_KEYS:
        (pipeline.get("detector") or {}).pop(key, None)

    # The thresholds are promoted into `defaults` (where infer.py's --conf lands),
    # so drop them from the carried config to keep one source of truth. Everything
    # else — tiling, crop expansion, merge NMS, batch sizes — stays verbatim.
    pipeline.pop("score_threshold", None)
    pipeline.pop("map_score_threshold", None)
    detector_conf = (pipeline.get("detector") or {}).pop("score_threshold", None)

    # `config.pipeline` (e.g. "people_detect_first") names which pipeline this
    # is, not a config field — it becomes the top-level `pipeline` string, and
    # everything else in `pipeline` is spread flat into the manifest instead of
    # living under a nested "pipeline" object.
    pipeline_name = pipeline.pop("pipeline")

    # A TensorRT engine implies the exporter also carried the portable ONNX
    # (see FORMATS / resolve_format), so "engine" is the format whose paths
    # best represent the bundle when only one flat pair is kept.
    primary_format = "engine" if "engine" in artifacts else "onnx"
    primary_artifacts = _plain(artifacts[primary_format])

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bundle": {
            "name": config.name,
            "created_utc": created_utc,
            "default_format": default_format,
            "exporter": "chachak.bundle_export",
        },
        "pipeline": pipeline_name,
        "artifacts": {
            "model": primary_artifacts["model"],
            "detector": primary_artifacts.get("detector"),
            # Not read by mercury_watchroom's validator (it only requires a flat
            # `model` string), but load-bearing for infer.py: a chain/multi-model
            # bundle needs `extra_models` to rebuild `extra_checkpoints`, and a
            # TensorRT bundle needs `max_batch` to avoid handing an engine built
            # with a batch-1 profile a stacked batch (see `batch_caps`).
            "extra_models": primary_artifacts.get("extra_models") or [],
            "max_batch": primary_artifacts.get("max_batch") or {},
        },
        "defaults": {
            "conf": float(default_conf),
            "detector_conf": None if detector_conf is None else float(detector_conf),
        },
        # Duplicated from defaults.conf: mercury_watchroom's pipeline_config.py
        # reads a top-level `score_threshold` (falling back to 0.5 if absent)
        # and does not look under `defaults`. Keep both in sync — this is
        # additive, not a replacement for `defaults.conf`.
        "score_threshold": float(default_conf),
        "provenance": _plain(provenance),
    }
    if gpu_infer:
        manifest["bundle"]["gpu_infer"] = True
    manifest.update(pipeline)
    return manifest


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """Read and schema-guard a ``pipeline.json``."""
    import json

    path = Path(path)
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ManifestError(f"pipeline manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"pipeline manifest is not valid JSON: {path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ManifestError(f"pipeline manifest must be a JSON object: {path}")

    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ManifestError(
            f"{path} has schema_version {version!r}, but this runner speaks "
            f"{SCHEMA_VERSION}. Re-export the bundle, or use the infer.py that "
            f"shipped with it."
        )
    if not isinstance(manifest.get("pipeline"), str) or not manifest["pipeline"]:
        raise ManifestError(f"{path} is missing the top-level 'pipeline' string")
    for key in ("artifacts", "defaults"):
        if not isinstance(manifest.get(key), dict):
            raise ManifestError(f"{path} is missing the {key!r} object")
    return manifest


def resolve_format(manifest: Dict[str, Any], requested: Optional[str] = None) -> str:
    """Which artifact format this bundle carries.

    The old per-format ``artifacts`` layout let one bundle carry both an ONNX and
    an engine artifact set and fall back between them. The flat manifest carries
    only one — the format that was primary at export time (see ``build_manifest``:
    an engine bundle's ``models/`` dir still has a portable ONNX file, but
    ``pipeline.json`` only points at the engine paths) — so there is nothing to
    fall back to. This just reports which one it is, read off the model artifact's
    extension, and raises if ``requested`` asks for the other.
    """
    model_path = str((manifest.get("artifacts") or {}).get("model") or "")
    if not model_path:
        raise ManifestError("bundle carries no model artifact")
    fmt = "engine" if model_path.endswith(".engine") else "onnx"
    if requested and requested != fmt:
        raise ManifestError(
            f"requested format {requested!r}, but this bundle only carries {fmt!r} artifacts"
        )
    return fmt


def artifact_paths(
    manifest: Dict[str, Any], bundle_dir: str | Path, fmt: Optional[str] = None
) -> Dict[str, Any]:
    """Absolute paths for the bundle's artifacts, checked for existence.

    ``fmt`` is accepted (and ignored) for call-site compatibility with the old
    per-format API: the flat manifest carries exactly one artifact set, whatever
    :func:`resolve_format` reports it to be.
    """
    bundle_dir = Path(bundle_dir)
    entry = manifest.get("artifacts") or {}
    if not entry.get("model"):
        raise ManifestError("bundle has no model artifact")

    def resolve(relative: str) -> Path:
        path = bundle_dir / relative
        if not path.exists():
            raise ManifestError(f"artifact listed in pipeline.json is missing: {path}")
        return path

    return {
        "model": resolve(entry["model"]),
        "detector": resolve(entry["detector"]) if entry.get("detector") else None,
        "extra_models": [resolve(item) for item in (entry.get("extra_models") or [])],
    }


def batch_caps(manifest: Dict[str, Any], fmt: Optional[str] = None) -> Dict[str, Optional[int]]:
    """Largest batch each role's artifact may be given.

    ONNX artifacts are run one image per session call by ``OnnxAdapter``, so they
    have no cap. TensorRT is different: ``TrtAdapter`` submits same-shaped images
    as one real ``[B,3,H,W]`` batch, decoded correctly only for an arch this
    engine was actually built with a wider profile for (see
    ``trt_export.arch.BATCH_AWARE_ARCHS`` for which archs that can even be asked
    for — retinanet/fasterrcnn never; the DETR-family passthrough archs can, but
    still build batch-1 by default). The exporter records the resulting ceiling
    per role under ``artifacts.max_batch``; ``None`` means uncapped. ``fmt`` is
    accepted (and ignored) for call-site compatibility — see :func:`artifact_paths`.
    """
    caps = (manifest.get("artifacts") or {}).get("max_batch") or {}
    return {
        "model": caps.get("model"),
        "detector": caps.get("detector"),
    }


def _clamp(value: int, cap: Optional[int]) -> int:
    return value if cap is None else max(1, min(int(value), int(cap)))


# Wrapper keys around the flat, portable ``PipelineConfig`` fields — everything
# else in the manifest is a config field spread in by `build_manifest` and belongs
# in the rebuilt request as-is.
_WRAPPER_KEYS = ("schema_version", "bundle", "artifacts", "defaults", "provenance")


def pipeline_request_from_manifest(
    manifest: Dict[str, Any],
    paths: Dict[str, Any],
    *,
    fmt: Optional[str] = None,
    images: str | Path,
    output_dir: str | Path,
    conf: Optional[float] = None,
    detector_conf: Optional[float] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Rebuild a chachak request dict from the manifest plus run-time arguments.

    The result is fed straight to :func:`chachak.config.pipeline_config_from_dict`,
    so the bundle validates its pipeline through the same loader the training repo
    uses — no second parser to keep in step.

    ``conf`` becomes ``score_threshold`` with ``map_score_threshold`` left unset:
    chachak's ``_inference_score_threshold`` prefers ``map_score_threshold`` when
    present, so writing only ``score_threshold`` makes ``--conf`` the single
    threshold every model stage (tile or person crop) filters on.

    Batch sizes are clamped to ``artifacts.max_batch`` (see :func:`batch_caps`).
    They only set how many tiles/crops go through the model per forward pass, so
    clamping costs throughput and changes nothing about the detections. ``fmt`` is
    accepted (and ignored) for call-site compatibility — see :func:`artifact_paths`.
    """
    defaults = manifest["defaults"]
    request: Dict[str, Any] = {
        key: value for key, value in manifest.items() if key not in _WRAPPER_KEYS
    }
    caps = batch_caps(manifest)

    request["model_checkpoint"] = str(paths["model"])
    request["extra_checkpoints"] = [str(item) for item in paths["extra_models"]]
    request["images"] = str(images)
    request["output_dir"] = str(output_dir)
    request["labels"] = None

    request["score_threshold"] = float(defaults["conf"] if conf is None else conf)
    request["map_score_threshold"] = None

    detector = dict(request.get("detector") or {})
    if paths["detector"] is not None:
        detector["checkpoint"] = str(paths["detector"])
    resolved_detector_conf = (
        detector_conf if detector_conf is not None else defaults.get("detector_conf")
    )
    if resolved_detector_conf is not None:
        detector["score_threshold"] = float(resolved_detector_conf)
    # The detector's own batch size is a separate knob (frames per detector forward
    # pass), so a single --batch-size moves both.
    detector["batch_size"] = _clamp(
        batch_size if batch_size is not None else detector.get("batch_size", 4),
        caps["detector"],
    )
    request["detector"] = detector

    if device is not None:
        request["device"] = device
    request["infer_batch_size"] = _clamp(
        batch_size if batch_size is not None else request.get("infer_batch_size", 4),
        caps["model"],
    )
    return request


def class_map(manifest: Dict[str, Any]) -> Dict[int, str]:
    """The pipeline's declared class space as ``{id: name}``."""
    return {int(key): str(value) for key, value in (manifest.get("classes") or {}).items()}


def describe(manifest: Dict[str, Any]) -> List[str]:
    """Human-readable one-line-per-item summary, printed by ``infer.py``."""
    bundle = manifest.get("bundle") or {}
    pipeline_name = manifest.get("pipeline")
    chain = manifest.get("chain") or []
    lines = [
        f"bundle           {bundle.get('name')} (exported {bundle.get('created_utc')})",
        f"pipeline         {pipeline_name}" + (f" -> {'+'.join(chain)}" if chain else ""),
        f"classes          {', '.join(class_map(manifest).values())}",
        f"merge_nms_iou    {manifest.get('merge_nms_iou')}",
    ]
    if pipeline_name in {"batch_detect", "batch_people"} or chain:
        tiling = manifest.get("tiling") or {}
        size = (
            f"{tiling.get('tile_size_px')}px"
            if tiling.get("tile_size_px")
            else f"{tiling.get('tile_width_pct')}%x{tiling.get('tile_height_pct')}%"
        )
        lines.append(
            f"tiling           {size} overlap={tiling.get('overlap')} nms_iou={tiling.get('nms_iou')}"
        )
    detector = manifest.get("detector") or {}
    if (manifest.get("artifacts") or {}).get("detector"):
        # A pinned person_class_id wins over the name lookup (chachak.detector
        # .load_detector), so report whichever actually selects the class.
        person = (
            f"class_id={detector['person_class_id']}"
            if detector.get("person_class_id") is not None
            else f"person='{detector.get('person_class_name')}'"
        )
        lines.append(
            f"detector         {person} "
            f"expand_ratio={detector.get('expand_ratio')} "
            f"nms_iou={detector.get('nms_iou')} min_box_size={detector.get('min_box_size')}"
        )
    return lines
