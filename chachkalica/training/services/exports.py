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

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

from django.utils import timezone

from training import pipelines
from training.models import DEFAULT_PERSON_DETECTOR_CHECKPOINT, TrainingSettings
from training.services import config_gen

logger = logging.getLogger(__name__)

# Sidecar carrying the chachak pipeline config a trained model's export was
# produced from — see :func:`pipeline_defaults_for` / :func:`write_pipeline_sidecar`.
PIPELINE_SIDECAR_SUFFIX = ".pipeline.json"

# Sidecar an ``.onnx``/``.engine`` carries its input geometry, normalization and
# class map in. Its name is derived from the artifact's, not recorded anywhere:
# both loaders resolve it as ``artifact.with_suffix(".meta.json")``
# (``onnx_infer/adapter.py``, ``trt_infer/adapter.py``) and open it directly, so an
# artifact without it cannot be loaded at all — which is why it has to travel with
# every copy made here (see :func:`_copy_artifact_set`).
META_SIDECAR_SUFFIX = ".meta.json"

# Suffixes chachak's ``load_checkpoint_adapter`` can load directly as a complete
# inference artifact (ONNX via onnxruntime, .engine via TensorRT).
ARTIFACT_SUFFIXES = (".onnx", ".engine")

_KIND_LABELS = {".onnx": "ONNX", ".engine": "TensorRT"}

# Person-detector copies :func:`export_pipeline_sidecar` leaves beside an export are
# named ``<artifact stem>.detector<suffix>``. By suffix they are indistinguishable
# from an exported model, but they are a *dependency* of the artifact next to them,
# addressed only through that artifact's ``.pipeline.json`` — so neither the scan
# nor the resolver may offer one as a model (see :func:`_is_detector_copy`).
DETECTOR_COPY_MARKER = ".detector"


def exports_root(ts: TrainingSettings | None = None) -> Path:
    """Absolute export output directory (relative settings resolve to the project root)."""
    ts = ts or TrainingSettings.load()
    return config_gen._resolve(ts.exports_root)


def _is_detector_copy(path: Path) -> bool:
    """Is this one of the person-detector copies left beside an export?

    Keyed on the ``.detector`` marker :func:`export_pipeline_sidecar` puts in the
    name (``m-best.detector.engine`` -> stem ``m-best.detector``), since the
    filesystem is the only record these files have.
    """
    return Path(path.stem).suffix == DETECTOR_COPY_MARKER


def list_artifacts(ts: TrainingSettings | None = None) -> list[dict]:
    """Exported artifacts under :func:`exports_root`, newest first.

    Recursive, since operators commonly export one subdirectory per model. Each
    entry is ``{relpath, name, kind, size_mb, absolute}``; ``relpath`` is the
    form value and ``kind`` is the human label ("ONNX" / "TensorRT").

    Person-detector copies are skipped: they carry a model suffix but are a
    dependency of the export beside them, and offering one as a selectable model
    means running a single-class person detector where the operator picked their
    own model (see :func:`_is_detector_copy`).
    """
    root = exports_root(ts)
    if not root.is_dir():
        return []

    entries = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ARTIFACT_SUFFIXES:
            continue
        if _is_detector_copy(path):
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
    non-file, a suffix chachak cannot load, or a person-detector copy. The last
    one is what :func:`list_artifacts` already leaves out of the dropdowns;
    refusing it here too makes that an invariant of the addressing scheme rather
    than a filter one code path happens to apply.
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
    if _is_detector_copy(candidate):
        raise ValueError(
            f"{relpath!r} is the person detector belonging to the export beside it, "
            f"not a model of its own. Select that export instead — it already runs "
            f"this detector as part of its pipeline."
        )
    if not candidate.is_file():
        raise ValueError(f"Exported artifact not found: {candidate}")
    return candidate


_needs_detector = pipelines.needs_detector


def _copy_artifact_set(source: Path, dest: Path) -> None:
    """Copy one inference artifact *and* the ``.meta.json`` its loader resolves by name.

    An ``.onnx``/``.engine`` is a pair, not a file (see ``META_SIDECAR_SUFFIX``):
    copying only the graph produces something that looks complete and raises
    ``FileNotFoundError`` on a path nobody wrote the moment it is loaded.
    """
    shutil.copy2(source, dest)
    meta = source.with_suffix(META_SIDECAR_SUFFIX)
    if meta.is_file():
        shutil.copy2(meta, dest.with_suffix(META_SIDECAR_SUFFIX))


def _copy_detector_beside(detector_path: Path, artifact_path: Path) -> Path | None:
    """Copy a person-detector checkpoint next to ``artifact_path``; return the copy.

    Returns ``None`` — meaning "keep pointing at ``detector_path``" — when the
    source is missing, or when it is an ``.onnx``/``.engine`` with no
    ``.meta.json`` beside it. That second case can't be made into a loadable
    copy, and a record pointing at the original is strictly better than one
    pointing at a copy that raises: the original at least loads wherever this
    export was produced.
    """
    if not detector_path.is_file():
        return None

    if (
        detector_path.suffix.lower() in ARTIFACT_SUFFIXES
        and not detector_path.with_suffix(META_SIDECAR_SUFFIX).is_file()
    ):
        logger.warning(
            "Person detector %s has no %s sidecar, so it cannot be copied next to "
            "%s as a loadable artifact; the exported pipeline metadata keeps "
            "pointing at the original path.",
            detector_path, META_SIDECAR_SUFFIX, artifact_path.name,
        )
        return None

    dest = artifact_path.with_suffix(f"{DETECTOR_COPY_MARKER}{detector_path.suffix}")
    _copy_artifact_set(detector_path, dest)

    # chachak prefers an ONNX artifact exported beside a .pt over the checkpoint
    # itself (chachak/infer.py's load_checkpoint_adapter looks for
    # `checkpoint.with_suffix(".onnx")`), so a lone .pt copy would silently switch
    # the detector onto the torch path — needing the training arch packages
    # installed wherever this runs. Carrying the sibling keeps the same preference
    # resolvable from the copy: `m-best.detector.pt` -> `m-best.detector.onnx`.
    sibling = detector_path.with_suffix(".onnx")
    if (
        detector_path.suffix.lower() not in ARTIFACT_SUFFIXES
        and sibling.is_file()
        and sibling.with_suffix(META_SIDECAR_SUFFIX).is_file()
    ):
        _copy_artifact_set(sibling, dest.with_suffix(".onnx"))

    return dest


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
    to ``artifact_path`` — as a whole artifact set, sidecar included — and points
    the metadata at the copy, so the detector travels with the export instead of
    depending on a path that may not exist wherever this artifact eventually runs
    — see ``cameras.services.live_inference`` / ``videos.services.inference``,
    which both just need a checkpoint path, trained or exported alike. The
    metadata is only repointed when a loadable copy was actually made
    (:func:`_copy_detector_beside`).
    """
    from training.services import pipeline_meta

    defaults = pipeline_meta.for_trained_model(trained_model)

    if _needs_detector(defaults["pipeline"], defaults["chain"]):
        from videos.services.frame_extraction import _resolve_checkpoint

        checkpoint = defaults["detector_checkpoint"] or DEFAULT_PERSON_DETECTOR_CHECKPOINT
        copied = _copy_detector_beside(_resolve_checkpoint(checkpoint), artifact_path)
        if copied is not None:
            defaults["detector_checkpoint"] = str(copied)

    sidecar = artifact_path.with_suffix(PIPELINE_SIDECAR_SUFFIX)
    sidecar.write_text(json.dumps(defaults, indent=2))
    return defaults


def _classes_from_meta(artifact_path: Path) -> list[str] | None:
    """Class names off ``artifact_path``'s ``.meta.json`` sidecar, ordered by id.

    Fallback for :func:`build_bundle_request` when the catalogued
    :class:`~training.models.TrainedModel` has no ``classes`` of its own.
    """
    meta_path = artifact_path.with_suffix(META_SIDECAR_SUFFIX)
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
        # Same None-dropping rule as `tiling` above, for the same reason:
        # chachak's detector parser does `float(raw.get(key, default))`, so a key
        # present with value None raises TypeError instead of falling through to
        # the default. A partial record (a sidecar or manifest written before a
        # field existed) normalizes those two knobs to None.
        request["detector"] = {
            "checkpoint": str(_resolve_checkpoint(checkpoint)),
            **{
                key: value for key, value in {
                    "expand_ratio": meta["detector_expand_ratio"],
                    "min_box_size": meta["detector_min_box_size"],
                }.items() if value is not None
            },
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


# ------------------------------------------------------------------- .pt bundle
#
# A ".pt bundle" is a different thing from the chachak bundles above: those
# convert to a torch-free ONNX/TensorRT runtime for *serving*; this one carries
# the raw checkpoint plus every catalog/pipeline/training fact needed to
# re-create the `TrainedModel` row itself on another machine (see
# `docs/pt-bundles.md`, `TrainedModelAdmin.export_pt_bundle`). No import side is
# built for it on purpose — a human or agent reads the manifest and recreates
# the row by hand elsewhere.

PT_BUNDLE_SCHEMA_VERSION = 1

_PT_BUNDLE_README = """\
# {name} — .pt bundle

A self-contained export of the trained model **{name}** ({arch}), for cataloguing
on another machine as a `TrainedModel` row.

## Contents

- `checkpoint.pt` — the promoted "best" checkpoint, verbatim. It is a pickled
  dict (`torch.load`), not a full pickled model: `model_name` + `model_config`
  (`num_classes`, `params`) pick the architecture out of
  `friendy_chachkalica.registry.build_model`, and `model_state_dict` loads onto
  it. See `chachak/infer.py::load_checkpoint_adapter` for the exact recipe:

  ```python
  state = torch.load("checkpoint.pt", map_location="cpu")
  adapter = build_model(state["model_name"], num_classes=state["model_config"]["num_classes"],
                        **state["model_config"].get("params", {{}}))
  adapter.model.load_state_dict(state["model_state_dict"])
  ```

- `manifest.json` — everything else: the catalog record (name/stage/classes/
  metrics), the serving pipeline this model must run through
  (`pipeline_metadata` — tiling/detector/NMS geometry), and training provenance
  (hyperparameters, dataset roster, metrics) where available. One schema,
  documented in `docs/pt-bundles.md` in the source repo.
- `checkpoint.pipeline.json` — the same `pipeline_metadata` blob, in the
  standalone sidecar form every chachak-produced artifact carries (see
  `training.services.pipeline_meta`).
- `checkpoint.detector.*` (only if the pipeline needs a person detector) — the
  detector checkpoint this model was trained/must be served through, copied
  alongside so the pipeline is runnable without depending on a path that may
  not exist on the new machine.
- `training_config.yaml` (if present) — the exact friendy_chachkalica YAML this
  checkpoint was trained from.

## Recreating the catalog row

On the destination, copy `checkpoint.pt` (and `checkpoint.detector.*`, if
present) somewhere on that machine's shared filesystem, then create a
`TrainedModel` with `manifest.json`'s `trained_model` fields, `checkpoint_path`
pointing at the copied checkpoint, and `pipeline_metadata` set to the
manifest's `pipeline_metadata` blob verbatim.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_snapshot(experiment) -> dict | None:
    """Every :class:`~training.models.Experiment` hyperparameter, plus its
    child model/dataset rosters — the training provenance a `.pt` alone can't
    answer (see :mod:`training.services.pipeline_meta`'s docstring: it only
    scopes itself to serving geometry, not how the weights were trained)."""
    if experiment is None:
        return None
    fields = [
        "epochs", "batch_size", "num_workers", "device", "seed", "amp",
        "gradient_clip_norm", "early_stopping_patience", "best_metric", "val_interval",
        "optimizer_name", "lr", "weight_decay", "optimizer_params",
        "scheduler_name", "scheduler_params",
        "eval_batch_size", "eval_num_workers", "eval_score_threshold",
        "eval_operating_nms_threshold", "iou_thresholds",
        "pipeline", "detector_checkpoint", "detector_expand_ratio", "detector_min_box_size",
        "tile_size_px", "tile_width_pct", "tile_height_pct", "overlap", "merge_nms_iou", "chain",
    ]
    return {"name": experiment.name, **{f: getattr(experiment, f) for f in fields}}


def _experiment_models_snapshot(experiment) -> list:
    if experiment is None:
        return []
    return [
        {"arch": m.arch, "num_classes": m.num_classes, "pretrained": m.pretrained, "params": m.params}
        for m in experiment.models.all()
    ]


def _experiment_datasets_snapshot(experiment) -> list:
    if experiment is None:
        return []
    return [
        {
            "dataset_name": d.dataset.name,
            "storage_type": d.dataset.storage_type,
            "storage_root": d.dataset.storage_root,
            "role": d.role,
            "label_source": d.label_source,
            "annotator": d.annotator.username if d.annotator_id else None,
            "explicit_labels_path": d.explicit_labels_path,
            "aug_hflip": d.aug_hflip,
            "aug_hflip_fraction": d.aug_hflip_fraction,
            "aug_scale_crop": d.aug_scale_crop,
            "aug_scale_crop_fraction": d.aug_scale_crop_fraction,
        }
        for d in experiment.datasets.select_related("dataset", "annotator").all()
    ]


def _run_result_snapshot(run_result) -> dict | None:
    if run_result is None:
        return None
    return {
        "run_name": run_result.run_name,
        "run_index": run_result.run_index,
        "model_arch": run_result.model_arch,
        "train_dataset_name": run_result.train_dataset_name,
        "best_epoch": run_result.best_epoch,
        "best_loss": run_result.best_loss,
        "run_dir": run_result.run_dir,
        "val_metrics": run_result.val_metrics,
        "test_metrics": run_result.test_metrics,
    }


def _training_run_snapshot(training_run) -> dict | None:
    if training_run is None:
        return None
    return {
        "status": training_run.status,
        "config_yaml_path": training_run.config_yaml_path,
        "output_dir": training_run.output_dir,
        "started_at": training_run.started_at.isoformat() if training_run.started_at else None,
        "finished_at": training_run.finished_at.isoformat() if training_run.finished_at else None,
    }


def _copy_training_config(training_run, dest: Path) -> bool:
    """Best-effort verbatim copy of the YAML the checkpoint was actually trained
    from. ``False`` (not an error) when there is no source run, or its config
    file is no longer on disk — the manifest's ``experiment`` snapshot already
    carries the same hyperparameters, so this is a nice-to-have, not load-bearing.
    """
    config_path = (getattr(training_run, "config_yaml_path", "") or "").strip()
    if not config_path:
        return False
    source = config_gen._resolve(config_path)
    if not source.is_file():
        return False
    shutil.copy2(source, dest)
    return True


def _build_pt_bundle_manifest(
    trained_model, checkpoint_src: Path, checkpoint_dst: Path, pipeline_blob: dict,
    *, training_config_included: bool,
) -> dict:
    from training.services import runner

    run_result = trained_model.source_run_result
    training_run = getattr(run_result, "run", None)
    experiment = getattr(training_run, "experiment", None)

    try:
        inspected = runner.inspect_checkpoint(checkpoint_src)
    except Exception:  # noqa: BLE001 - best-effort hint only, mirrors _trained_size_hint
        inspected = None

    checkpoint_entry = {
        "filename": checkpoint_dst.name,
        "original_path": str(checkpoint_src),
        "sha256": _sha256(checkpoint_dst),
        "size_bytes": checkpoint_dst.stat().st_size,
    }
    if inspected:
        checkpoint_entry["inspected"] = inspected

    return {
        "schema_version": PT_BUNDLE_SCHEMA_VERSION,
        "bundle_kind": "pt_bundle",
        "exported_at": timezone.now().isoformat(),
        "trained_model": {
            "name": trained_model.name,
            "description": trained_model.description,
            "stage": trained_model.stage,
            "arch": trained_model.arch,
            "num_classes": trained_model.num_classes,
            "classes": list(trained_model.classes),
            "metrics": trained_model.metrics,
        },
        "checkpoint": checkpoint_entry,
        "pipeline_metadata": pipeline_blob,
        "provenance": {
            "run_result": _run_result_snapshot(run_result),
            "training_run": _training_run_snapshot(training_run),
            "experiment": _experiment_snapshot(experiment),
            "experiment_models": _experiment_models_snapshot(experiment),
            "experiment_datasets": _experiment_datasets_snapshot(experiment),
            "training_config_included": training_config_included,
        },
    }


def build_pt_bundle(trained_model, checkpoint_path: str, archive_path: Path) -> dict:
    """Package ``trained_model``'s ``checkpoint_path`` plus every catalog/pipeline/
    training fact on record into one self-contained ``.tar.gz`` at ``archive_path``
    — see :mod:`training.admin`'s ``TrainedModelAdmin.export_pt_bundle`` and
    ``docs/pt-bundles.md`` for the layout/schema this produces.

    Builds in a temp staging dir first so a failure partway through never leaves
    a half-written bundle at ``archive_path``.
    """
    checkpoint_src = Path(checkpoint_path)
    if not checkpoint_src.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_src}")

    stem = archive_path.name
    if stem.endswith(".tar.gz"):
        stem = stem[: -len(".tar.gz")]

    with tempfile.TemporaryDirectory(prefix="ptbundle-") as tmp:
        stage = Path(tmp) / stem
        stage.mkdir(parents=True)

        checkpoint_dst = stage / "checkpoint.pt"
        shutil.copy2(checkpoint_src, checkpoint_dst)

        pipeline_blob = export_pipeline_sidecar(trained_model, checkpoint_dst)

        training_run = getattr(trained_model.source_run_result, "run", None)
        training_config_included = _copy_training_config(training_run, stage / "training_config.yaml")

        manifest = _build_pt_bundle_manifest(
            trained_model, checkpoint_src, checkpoint_dst, pipeline_blob,
            training_config_included=training_config_included,
        )
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        (stage / "README.md").write_text(
            _PT_BUNDLE_README.format(name=trained_model.name, arch=trained_model.arch)
        )

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(stage, arcname=stage.name)

    return {"bundle_path": str(archive_path), "manifest": manifest}
