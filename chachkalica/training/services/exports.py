"""Discovery of exported inference artifacts under the export output root.

The ONNX/TensorRT export actions (see ``training.admin.TrainedModelAdmin``) write
``<name>-best.onnx`` / ``<name>-last.engine`` (plus sibling ``.meta.json``) into
:attr:`TrainingSettings.exports_root`. Nothing about them is stored in the DB —
the files *are* the record — so anything that wants to offer "run this through an
exported model" has to scan the directory. That scan lives here so the export
actions and the consumers (video inference) agree on one root.

An artifact is addressed by its path *relative* to the root; :func:`resolve` maps
that back to an absolute path and refuses anything that escapes the root, so a
crafted form value can't point inference at an arbitrary file.

Every export also carries the pipeline it must be served with, so an ``.onnx`` or
``.engine`` answers the same questions a ``.pt`` does — see
:func:`export_pipeline_sidecar` / :func:`read_pipeline_defaults`, and
:mod:`training.services.pipeline_meta` for the schema they share.
"""

import json
import shutil
from pathlib import Path

from training import pipelines
from training.models import DEFAULT_PERSON_DETECTOR_CHECKPOINT, TrainingSettings
from training.services import config_gen

# Sidecar carrying the chachak pipeline config a trained model's export was
# produced from — see :func:`pipeline_defaults_for` / :func:`write_pipeline_sidecar`.
PIPELINE_SIDECAR_SUFFIX = ".pipeline.json"

# Suffixes chachak's ``load_checkpoint_adapter`` can load directly as a complete
# inference artifact (ONNX via onnxruntime, .engine via TensorRT).
ARTIFACT_SUFFIXES = (".onnx", ".engine")

_KIND_LABELS = {".onnx": "ONNX", ".engine": "TensorRT"}


def exports_root(ts: TrainingSettings | None = None) -> Path:
    """Absolute export output directory (relative settings resolve to the project root)."""
    ts = ts or TrainingSettings.load()
    return config_gen._resolve(ts.exports_root)


def list_artifacts(ts: TrainingSettings | None = None) -> list[dict]:
    """Exported artifacts under :func:`exports_root`, newest first.

    Recursive, since operators commonly export one subdirectory per model. Each
    entry is ``{relpath, name, kind, size_mb, absolute}``; ``relpath`` is the
    form value and ``kind`` is the human label ("ONNX" / "TensorRT").
    """
    root = exports_root(ts)
    if not root.is_dir():
        return []

    entries = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ARTIFACT_SUFFIXES:
            continue
        relpath = path.relative_to(root).as_posix()
        entries.append({
            "relpath": relpath,
            "name": relpath,
            "kind": _KIND_LABELS[path.suffix.lower()],
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            "mtime": path.stat().st_mtime,
            "absolute": str(path),
        })
    entries.sort(key=lambda e: (-e["mtime"], e["relpath"]))
    return entries


def resolve(relpath: str, ts: TrainingSettings | None = None) -> Path:
    """Absolute path for an artifact addressed relative to :func:`exports_root`.

    Raises ``ValueError`` for a blank value, a path that escapes the root, a
    non-file, or a suffix chachak cannot load.
    """
    relpath = (relpath or "").strip()
    if not relpath:
        raise ValueError("No exported artifact selected.")

    root = exports_root(ts)
    candidate = (root / relpath).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise ValueError(
            f"{relpath!r} is outside the export output directory ({root})."
        ) from None
    if candidate.suffix.lower() not in ARTIFACT_SUFFIXES:
        raise ValueError(
            f"{relpath!r} is not an exported artifact "
            f"({' / '.join(ARTIFACT_SUFFIXES)})."
        )
    if not candidate.is_file():
        raise ValueError(f"Exported artifact not found: {candidate}")
    return candidate


def _needs_detector(pipeline: str, chain: list) -> bool:
    return pipeline in pipelines.DETECTOR_PIPELINES or (
        pipeline == pipelines.CHAIN
        and any(c in pipelines.DETECTOR_PIPELINES for c in (chain or []))
    )


def export_pipeline_sidecar(trained_model, artifact_path: Path) -> dict:
    """Copy ``trained_model``'s frozen pipeline metadata beside ``artifact_path``,
    so the exported artifact carries the same record the ``.pt`` does.

    The blob is :func:`training.services.pipeline_meta.for_trained_model`
    verbatim — one schema for every format, rather than each destination form's
    own subset. It is written *unconditionally*, including for a model trained on
    plain full frames: "raw" on record is a statement a consumer can prefill
    from, whereas a missing sidecar is indistinguishable from an export made
    before this existed, and sends the operator back to filling the form by hand.

    Also copies the person-detector checkpoint the pipeline needs (if any) next
    to ``artifact_path`` and points the metadata at the copy, so the detector
    travels with the export instead of depending on a path that may not exist
    wherever this artifact eventually runs — see
    ``cameras.services.live_inference`` / ``videos.services.inference``, which
    both just need a checkpoint path, trained or exported alike.
    """
    from training.services import pipeline_meta

    defaults = pipeline_meta.for_trained_model(trained_model)

    if _needs_detector(defaults["pipeline"], defaults["chain"]):
        from videos.services.frame_extraction import _resolve_checkpoint

        checkpoint = defaults["detector_checkpoint"] or DEFAULT_PERSON_DETECTOR_CHECKPOINT
        detector_path = _resolve_checkpoint(checkpoint)
        if detector_path.is_file():
            dest = artifact_path.with_suffix(f".detector{detector_path.suffix}")
            shutil.copy2(detector_path, dest)
            defaults["detector_checkpoint"] = str(dest)

    sidecar = artifact_path.with_suffix(PIPELINE_SIDECAR_SUFFIX)
    sidecar.write_text(json.dumps(defaults, indent=2))
    return defaults


def _classes_from_meta(artifact_path: Path) -> list[str] | None:
    """Class names off ``artifact_path``'s ``.meta.json`` sidecar, ordered by id.

    Fallback for :func:`build_bundle_request` when the catalogued
    :class:`~training.models.TrainedModel` has no ``classes`` of its own.
    """
    meta_path = artifact_path.with_suffix(".meta.json")
    try:
        class_map = json.loads(meta_path.read_text()).get("class_map") or {}
    except (OSError, ValueError):
        return None
    if not class_map:
        return None
    return [name for _, name in sorted(class_map.items(), key=lambda item: int(item[0]))]


def build_bundle_request(trained_model, artifact_path: Path) -> dict | None:
    """A chachak pipeline request (dict form) for bundling ``artifact_path``.

    ``artifact_path`` is an already-exported ``.onnx``/``.engine`` for
    ``trained_model`` (what :func:`export_pipeline_sidecar` was just called on).
    The result is handed to ``runner.export_bundle`` almost verbatim; the
    trainer service resolves it through the same
    ``chachak.config.pipeline_config_from_dict`` the training repo uses.

    Built from the model's frozen pipeline metadata (see
    :mod:`training.services.pipeline_meta`) rather than by walking back to the
    experiment, so a bundle carries the same geometry the ``.pipeline.json``
    sidecar and the admin prefills do — and can still be built after the source
    experiment is gone.

    Returns ``None`` for a model trained on plain full frames (there's no
    pipeline geometry to bundle) or when no class names are on record at all (a
    bundle's manifest can't carry an empty class space).
    """
    from training.services import pipeline_meta

    meta = pipeline_meta.for_trained_model(trained_model)
    if meta["pipeline"] == pipeline_meta.RAW:
        return None

    classes = list(trained_model.classes) or _classes_from_meta(artifact_path)
    if not classes:
        return None

    request: dict = {
        "name": trained_model.name,
        "pipeline": meta["pipeline"],
        "model_checkpoint": str(artifact_path),
        # Bundling never touches a dataset; chachak's parser requires these
        # fields to be present, not that they resolve to anything real.
        "images": ".",
        "labels": None,
        "classes": classes,
        "score_threshold": meta["score_threshold"],
        "chain": list(meta["chain"]),
    }
    if meta["merge_nms_iou"] is not None:
        request["merge_nms_iou"] = meta["merge_nms_iou"]
    # chachak's tiling/detector parsers do `float(raw.get(key, default))`
    # unconditionally for these, so a key present with value None would crash
    # rather than fall through to the default — only carry the ones actually set.
    tiling = {
        key: value for key, value in {
            "tile_size_px": meta["tile_size_px"],
            "tile_width_pct": meta["tile_width_pct"],
            "tile_height_pct": meta["tile_height_pct"],
            "overlap": meta["overlap"],
        }.items() if value is not None
    }
    if tiling:
        request["tiling"] = tiling

    if _needs_detector(meta["pipeline"], meta["chain"]):
        from videos.services.frame_extraction import _resolve_checkpoint

        checkpoint = meta["detector_checkpoint"] or DEFAULT_PERSON_DETECTOR_CHECKPOINT
        request["detector"] = {
            "checkpoint": str(_resolve_checkpoint(checkpoint)),
            "expand_ratio": meta["detector_expand_ratio"],
            "min_box_size": meta["detector_min_box_size"],
        }

    return request


def read_pipeline_defaults(relpath: str, ts: TrainingSettings | None = None) -> dict | None:
    """Pipeline metadata for an artifact addressed relative to :func:`exports_root`.

    Reads the ``.pipeline.json`` sidecar and normalizes it through
    :func:`training.services.pipeline_meta.normalize`, so a sidecar written
    before a field existed still answers with the full schema (missing keys mean
    "chachak's default", which is what the older writer meant by omitting them).

    Falls back to the catalogued :class:`~training.models.TrainedModel` whose
    export this artifact is, matched by filename stem — that covers every
    artifact exported before sidecars were written, which is most of what is
    sitting in the export root today. Returns ``None`` only when neither source
    has anything to say: a bad relpath, or an artifact with no sidecar and no
    model in the catalogue.
    """
    from training.services import pipeline_meta

    try:
        artifact = resolve(relpath, ts)
    except ValueError:
        return None

    sidecar = artifact.with_suffix(PIPELINE_SIDECAR_SUFFIX)
    if sidecar.is_file():
        try:
            return pipeline_meta.normalize(json.loads(sidecar.read_text()))
        except (OSError, ValueError):
            pass  # fall through to the catalogue

    return _pipeline_defaults_from_catalogue(artifact)


# The export actions name artifacts "<model name>-best.onnx" / "<model name>-last
# .engine" (see training.admin.TrainedModelAdmin), so the model name is the stem
# minus that suffix.
_EXPORT_STEM_SUFFIXES = ("-best", "-last")


def _pipeline_defaults_from_catalogue(artifact: Path) -> dict | None:
    """Pipeline metadata for a sidecar-less artifact, via the model it was exported
    from — matched on the filename the export actions produce.

    Best-effort by design: an artifact whose name no longer matches any
    catalogued model just gets ``None``, same as before.
    """
    from training.models import TrainedModel
    from training.services import pipeline_meta

    stem = artifact.stem
    candidates = [stem] + [
        stem[: -len(suffix)] for suffix in _EXPORT_STEM_SUFFIXES if stem.endswith(suffix)
    ]
    model = (
        TrainedModel.objects
        .filter(name__in=candidates)
        .select_related("source_run_result__run__experiment")
        .first()
    )
    return pipeline_meta.for_trained_model(model) if model is not None else None
