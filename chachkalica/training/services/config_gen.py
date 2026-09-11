"""Turn an :class:`~training.models.Experiment` into friendy_chachkalica YAML.

friendy_chachkalica (``/home/luka/workspace/chachkalica/friendy_chachkalica``) is config-driven: it
reads an experiment YAML whose ``datasets.{train(list),val,test}`` entries each
point at an ``images`` dir and a ``labels`` dir with a ``classes`` list, plus
``models``/``training``/``evaluation`` blocks. We generate that YAML with
**absolute** paths so it resolves regardless of where the trainer runs, reusing
the same on-disk resolvers the fleet annotation side already uses.
"""

import math
from pathlib import Path

import yaml
from django.conf import settings

from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services.paths import source_root, target_root
from training import pipelines
from training.models import (
    Experiment,
    ExperimentDataset,
    ExperimentModel,
    TrainingSettings,
    default_iou_thresholds,
    DEFAULT_PERSON_DETECTOR_CHECKPOINT,
)


def _resolve(path: str) -> Path:
    """Absolute path: as-is if absolute, else relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else Path(settings.BASE_DIR) / p


def resolve_label_dir(dataset, label_source: str, annotator=None, explicit_path: str = "") -> Path:
    """Resolve a dataset's labels directory for a given label-source choice.

    Shared by training (:class:`ExperimentDataset`) and standalone eval
    (:class:`EvalRun`) so both pick labels the same way:
    ``source`` -> data/source/<name>/labels, ``annotator`` ->
    data/target/<name>/<username>, ``explicit`` -> a given path.
    """
    if label_source == ExperimentDataset.SOURCE:
        return datasets_svc.labels_source_dir(dataset)
    if label_source == ExperimentDataset.ANNOTATOR:
        if annotator is None:
            raise ValueError(f"{dataset.name}: annotator output selected but no annotator set.")
        return target_root() / dataset.name / annotator.username
    if label_source == ExperimentDataset.EXPLICIT:
        if not (explicit_path or "").strip():
            raise ValueError(f"{dataset.name}: explicit label path selected but empty.")
        return _resolve(explicit_path.strip())
    if label_source == ExperimentDataset.NONE:
        return None
    raise ValueError(f"Unknown label source {label_source!r}")


def label_dir(exp_dataset: ExperimentDataset) -> Path:
    """Resolve the labels directory feeding this dataset, per its label source."""
    return resolve_label_dir(
        exp_dataset.dataset, exp_dataset.label_source,
        exp_dataset.annotator, exp_dataset.explicit_labels_path,
    )


def dataset_classes(dataset) -> list[str]:
    """Class names for a dataset, from its on-disk classes.txt."""
    classes, _tools = lsapi.parse_classes_file(source_root() / dataset.name / "classes.txt")
    return classes


def images_dir(dataset) -> Path:
    return lsapi.image_source_dir(source_root() / dataset.name)


def dataset_entry(exp_dataset: ExperimentDataset) -> dict:
    """Build one YAML dataset entry: {name, images, labels, classes[, augmentation]}."""
    entry = {
        "name": exp_dataset.dataset.name,
        "images": str(images_dir(exp_dataset.dataset)),
        "labels": str(label_dir(exp_dataset)),
        "classes": dataset_classes(exp_dataset.dataset),
    }
    augmentation = augmentation_entry(exp_dataset)
    if augmentation:
        entry["augmentation"] = augmentation
    return entry


def augmentation_entry(exp_dataset: ExperimentDataset) -> dict:
    """The dataset's `augmentation` block: enabled checkboxes -> fractions.

    The trainer only augments train datasets (and rejects the key elsewhere),
    so flags on val/test rows are never emitted — model.clean() already blocks
    saving them, but rows predating that validation shouldn't break a run.
    """
    if exp_dataset.role != ExperimentDataset.TRAIN:
        return {}
    augmentation = {}
    if exp_dataset.aug_hflip and exp_dataset.aug_hflip_fraction:
        augmentation["hflip"] = exp_dataset.aug_hflip_fraction
    if exp_dataset.aug_scale_crop and exp_dataset.aug_scale_crop_fraction:
        augmentation["scale_crop"] = exp_dataset.aug_scale_crop_fraction
    return augmentation


# rtdetr's (and dfine's — same HF hybrid encoder/two-stage topk, confirmed
# against modeling_d_fine.py) encoder runs topk(num_queries) over its
# feature-pyramid tokens, so num_queries must not exceed that token count or the
# forward pass crashes ("selected index k out of range"). The adapter stretches
# every crop onto a square canvas fixed by input_max_size (see
# RTDETRAdapter._resize_image_with_scale / DFineAdapter's twin), so the crash
# floor is a function of input_max_size, NOT the raw crop size: model_entry
# below enforces input_max_size >= input_size_multiple * ceil(sqrt(num_queries)),
# which makes the coarsest (stride input_size_multiple) level alone clear
# num_queries with margin. 25 is kept low anyway — HF's own default (300)
# assumes near-full-frame subjects with many objects, whereas 25 comfortably
# covers the handful of PPE items on one person crop. This replaces the old
# coupling to detector_min_box_size, which floored the *crop* size back when
# crops were fed at native resolution. Only injected for people_detect_first
# (see model_entry) — other pipelines don't hit this mismatch (batch_people's
# tiles are already sized generously; full-frame training was never undersized
# to begin with). Name kept rtdetr-specific for history; the value and mechanism
# are shared with dfine, not rtdetr-only.
PEOPLE_DETECT_FIRST_RTDETR_NUM_QUERIES_DEFAULT = 25

# Archs whose HF-style hybrid encoder does the topk(num_queries) two-stage
# selection above, and so need the people_detect_first guard below.
_TOPK_ENCODER_ARCHS = (ExperimentModel.RTDETR, ExperimentModel.DFINE)


def model_entry(exp_model: ExperimentModel, pipeline_name: str | None = None) -> dict:
    """Build one YAML model entry; our name/num_classes win over params.

    The ``pretrained`` checkbox maps to ``weights: true`` — every adapter reads
    ``weights=True`` as "load the published COCO-pretrained weights" (retinanet,
    rtdetr, yolox, rfdetr, fasterrcnn, ecdet; dfine reads it as its own default
    checkpoint too). An explicit ``weights`` in ``params`` (e.g. a path, URL, or
    ecdet's ``backbone`` sentinel) is left untouched and wins over the checkbox.

    ``pipeline_name`` is the owning experiment's pipeline, passed by
    :func:`build_experiment_dict` (``None`` for standalone/test callers, which
    skips the injection below). For an rtdetr or dfine model on
    people_detect_first, an explicit ``params["num_queries"]`` always wins;
    otherwise :data:`PEOPLE_DETECT_FIRST_RTDETR_NUM_QUERIES_DEFAULT` is injected
    so the run doesn't crash on their shared un-cropped-frame default of 300.
    """
    params = dict(exp_model.params or {})
    entry = {
        **params,
        "name": exp_model.arch,
        "num_classes": exp_model.num_classes if exp_model.num_classes is not None else "auto",
    }
    if exp_model.pretrained and "weights" not in params:
        entry["weights"] = True
    if (
        exp_model.arch in _TOPK_ENCODER_ARCHS
        and pipeline_name == pipelines.PEOPLE_DETECT_FIRST
    ):
        if "num_queries" not in params:
            entry["num_queries"] = PEOPLE_DETECT_FIRST_RTDETR_NUM_QUERIES_DEFAULT
        # Guard rtdetr's topk crash floor here (see the comment on
        # PEOPLE_DETECT_FIRST_RTDETR_NUM_QUERIES_DEFAULT) rather than letting the
        # trainer die at the first forward pass. input_max_size is now the
        # working resolution each crop is upscaled to; require the coarsest
        # (stride input_size_multiple) feature level to hold >= num_queries tokens
        # on its own. Absent input_max_size -> the adapter's 640 default, safe.
        input_max_size = params.get("input_max_size")
        if input_max_size is not None:
            num_queries = int(entry["num_queries"])
            multiple = int(params.get("input_size_multiple", 32) or 32)
            min_side = multiple * math.ceil(math.sqrt(num_queries))
            if int(input_max_size) < min_side:
                raise ValueError(
                    f"{exp_model.arch}: Input max size {int(input_max_size)} is too "
                    f"small for {pipelines.PEOPLE_DETECT_FIRST} with "
                    f"num_queries={num_queries}. Person crops are upscaled to this "
                    f"resolution and topk({num_queries}) runs over the resulting "
                    f"feature tokens, so anything below {min_side} crashes the "
                    f"forward pass and lower values under-resolve small PPE items. "
                    f"Raise Input max size to at least {min_side} (leave it blank "
                    f"for the 640 default), or lower num_queries."
                )
    return entry


def _scheduler(experiment: Experiment):
    if experiment.scheduler_name == "none":
        return None
    return {"name": experiment.scheduler_name, **(experiment.scheduler_params or {})}


def pipeline_block(experiment: Experiment) -> dict | None:
    """The ``pipeline`` block for the experiment YAML, or ``None`` when unset.

    Emits only non-blank knobs so chachak's own defaults apply where the operator
    left a field empty — except the detector checkpoint on a detector-requiring
    pipeline, which falls back to ``DEFAULT_PERSON_DETECTOR_CHECKPOINT`` instead
    of being left unset (mirrors ``build_pipeline_request``).

    ``detector.min_box_size`` (from ``experiment.detector_min_box_size``) is
    emitted for every detector pipeline — see
    ``Experiment.detector_min_box_size``'s help text for why the floor exists
    (paired with the ``num_queries`` default :func:`model_entry` injects for
    rtdetr on people_detect_first).
    """
    name = experiment.pipeline
    if not name:
        return None

    data: dict = {"name": name}

    if name == pipelines.CHAIN and experiment.chain:
        data["chain"] = list(experiment.chain)

    needs_detector = pipelines.needs_detector(name, experiment.chain or [])
    # Gated on needs_detector, not just "is the field non-blank": the field
    # carries a non-blank default (the bundled person engine) so it's never
    # actually empty, which would otherwise leak a detector block into
    # ordinary tiling/raw pipelines that don't use one.
    checkpoint = (
        (experiment.detector_checkpoint or DEFAULT_PERSON_DETECTOR_CHECKPOINT)
        if needs_detector else ""
    )
    if checkpoint:
        # Experiment paths are relative to the Django project root, while the
        # generated YAML lives under ``configs_root``.
        detector: dict = {"checkpoint": str(_resolve(checkpoint))}
        if experiment.detector_expand_ratio is not None:
            detector["expand_ratio"] = experiment.detector_expand_ratio
        # Emitted for batch_people too, not just people_detect_first. It used to
        # be scoped to the latter on the theory that batch_people crops come from
        # fixed-size tiles and so can't shrink to degenerate sizes — but
        # BatchPeoplePipeline only *finds* people in tiles and then crops the
        # original frame (chachak.pipeline.BatchPeoplePipeline.process_batch), so
        # its crops are exactly as small as people_detect_first's. Meanwhile
        # chachak applies the floor for both (crop_regions has no pipeline gate),
        # so scoping it here meant a batch_people model trained with no floor and
        # was then served with one.
        if experiment.detector_min_box_size:
            detector["min_box_size"] = experiment.detector_min_box_size
        data["detector"] = detector

    tiling: dict = {}
    if experiment.tile_size_px:
        tiling["tile_size_px"] = experiment.tile_size_px
    if experiment.tile_width_pct:
        tiling["tile_width_pct"] = experiment.tile_width_pct
    if experiment.tile_height_pct:
        tiling["tile_height_pct"] = experiment.tile_height_pct
    if experiment.overlap is not None:
        tiling["overlap"] = experiment.overlap
    if tiling:
        data["tiling"] = tiling

    if experiment.merge_nms_iou is not None:
        data["merge_nms_iou"] = experiment.merge_nms_iou

    return data


def build_experiment_dict(experiment: Experiment, output_dir: Path | str) -> dict:
    """Assemble the full friendy_chachkalica experiment dict.

    Raises ``ValueError`` if the roster is invalid (no train dataset, more than
    one val/test, or no models) — surfaced by the admin action.
    """
    rows = list(experiment.datasets.all())
    train = [dataset_entry(r) for r in rows if r.role == ExperimentDataset.TRAIN]
    vals = [r for r in rows if r.role == ExperimentDataset.VAL]
    tests = [r for r in rows if r.role == ExperimentDataset.TEST]

    if not train:
        raise ValueError("Add at least one train dataset.")
    if len(vals) > 1:
        raise ValueError("At most one val dataset is allowed.")
    if len(tests) > 1:
        raise ValueError("At most one test dataset is allowed.")

    models = list(experiment.models.all())
    if not models:
        raise ValueError("Add at least one model architecture.")

    datasets: dict = {"train": train}
    if vals:
        datasets["val"] = dataset_entry(vals[0])
    if tests:
        datasets["test"] = dataset_entry(tests[0])

    pipeline = pipeline_block(experiment)

    result = {
        "name": experiment.name,
        "output_dir": str(output_dir),
        "datasets": datasets,
        "models": [model_entry(m, pipeline_name=experiment.pipeline) for m in models],
        "training": {
            "epochs": experiment.epochs,
            "batch_size": experiment.batch_size,
            "num_workers": experiment.num_workers,
            "device": experiment.device,
            "seed": experiment.seed,
            "amp": experiment.amp,
            "gradient_clip_norm": experiment.gradient_clip_norm,
            "early_stopping_patience": experiment.early_stopping_patience,
            # "f1+map50" is one choice value meaning "average of both"; the
            # trainer takes a list and tracks the mean.
            "best_metric": experiment.best_metric.split("+"),
            "val_interval": experiment.val_interval,
            "optimizer": {
                "name": experiment.optimizer_name,
                "lr": experiment.lr,
                "weight_decay": experiment.weight_decay,
                "params": experiment.optimizer_params or {},
            },
            "scheduler": _scheduler(experiment),
        },
        "evaluation": {
            "batch_size": experiment.eval_batch_size,
            "num_workers": experiment.eval_num_workers,
            # Fixed low so map50/map50_95 sweep the full precision-recall curve;
            # eval_score_threshold is the separate, user-set operating point for
            # precision/recall/f1 (see Experiment.eval_score_threshold help text).
            "map_score_threshold": 0.001,
            "score_threshold": experiment.eval_score_threshold,
            # NMS for the operating-point metrics only (mAP stays NMS-free); a
            # model entry's own nms_threshold param overrides it per run.
            "operating_nms_threshold": experiment.eval_operating_nms_threshold,
            "iou_thresholds": experiment.iou_thresholds,
        },
    }
    if pipeline is not None:
        result["pipeline"] = pipeline
    return result


def build_yaml(experiment: Experiment, output_dir: Path | str) -> str:
    """Render the experiment dict to YAML text (no files written)."""
    data = build_experiment_dict(experiment, output_dir)
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def run_paths(experiment: Experiment, run_id: int, ts: TrainingSettings | None = None):
    """Return (config_yaml_path, output_dir) for a run."""
    ts = ts or TrainingSettings.load()
    stem = f"{experiment.name}-{run_id}"
    yaml_path = _resolve(ts.configs_root) / f"{stem}.yaml"
    output_dir = _resolve(ts.runs_root) / stem
    return yaml_path, output_dir


def classes_for_name(dataset_name: str) -> list[str]:
    """Class names for a dataset directory, by name (no DB row required)."""
    classes, _tools = lsapi.parse_classes_file(source_root() / dataset_name / "classes.txt")
    return classes


def eval_request_paths(eval_run, ts: TrainingSettings | None = None):
    ts = ts or TrainingSettings.load()
    stem = f"eval-{eval_run.pk}"
    return _resolve(ts.configs_root) / f"{stem}.yaml", _resolve(ts.runs_root) / stem


def combined_checkpoints(run) -> list[str]:
    """Checkpoint paths for a combined eval's *extra* models (beyond the primary).

    Empty for an ordinary single-model run. Raises ``ValueError`` if any
    combined model lacks a checkpoint, mirroring the primary model's own check.
    """
    paths = []
    for m in run.combined_models.all():
        if not m.checkpoint_path:
            raise ValueError(f"{m.name}: no checkpoint path to evaluate.")
        paths.append(m.checkpoint_path)
    return paths


def build_eval_request(eval_run, output_dir: Path | str, ts: TrainingSettings | None = None) -> dict:
    """Assemble the eval request consumed by friendy_chachkalica's ml/eval_checkpoint.py.

    ``classes`` is the *eval dataset's* class space (the target labels); the
    model's own train-class space is read from the checkpoint by the trainer.
    When ``eval_run`` combines 2+ models, ``extra_checkpoints`` carries the
    others' checkpoint paths and the trainer merges all models' predictions
    into one result (see ``eval_checkpoint.eval_combined_checkpoints``).
    """
    ts = ts or TrainingSettings.load()
    tm = eval_run.trained_model
    ds = eval_run.dataset
    if not tm.checkpoint_path:
        raise ValueError(f"{tm.name}: no checkpoint path to evaluate.")
    data = {
        "name": f"eval-{eval_run.pk}",
        "checkpoint_path": tm.checkpoint_path,
        "images": str(images_dir(ds)),
        "classes": dataset_classes(ds),
        "output_dir": str(output_dir),
        "map_score_threshold": eval_run.map_score_threshold,
        "score_threshold": eval_run.score_threshold,
        "iou_thresholds": default_iou_thresholds(),
        "device": ts.default_device,
    }
    labels = resolve_label_dir(
        ds, eval_run.label_source, eval_run.annotator, eval_run.explicit_labels_path)
    if labels is not None:
        data["labels"] = str(labels)

    extra_checkpoints = combined_checkpoints(eval_run)
    if extra_checkpoints:
        data["extra_checkpoints"] = extra_checkpoints
    return data


def write_eval_request(eval_run, ts: TrainingSettings | None = None) -> tuple[Path, str]:
    """Generate the eval request YAML for ``eval_run`` and persist its paths."""
    ts = ts or TrainingSettings.load()
    request_path, output_dir = eval_request_paths(eval_run, ts)
    data = build_eval_request(eval_run, output_dir, ts)
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(text, encoding="utf-8")
    eval_run.request_yaml_path = str(request_path)
    eval_run.output_dir = str(output_dir)
    eval_run.save(update_fields=["request_yaml_path", "output_dir"])
    return request_path, text


def pipeline_request_paths(pe, ts: TrainingSettings | None = None):
    ts = ts or TrainingSettings.load()
    stem = f"pipeline-{pe.pk}"
    return _resolve(ts.configs_root) / f"{stem}.yaml", _resolve(ts.runs_root) / stem


def build_pipeline_request(pe, output_dir: Path | str, ts: TrainingSettings | None = None) -> dict:
    """Assemble the chachak request consumed by ``chachak/run.py``.

    ``classes`` is the *eval dataset's* class space (the target labels); the
    model's own train-class space is read from the checkpoint by chachak. Only
    non-default detector/tiling knobs are emitted so chachak's own defaults apply
    when the operator left a field blank — except the detector checkpoint on a
    detector-requiring pipeline, which falls back to
    ``DEFAULT_PERSON_DETECTOR_CHECKPOINT`` rather than being left unset (mirrors
    ``pipeline_block``). When ``pe`` combines 2+ models, ``extra_checkpoints``
    carries the others' paths and chachak merges every model's predictions into
    one result.

    Carries the same pipeline vocabulary as the training YAML
    (:func:`pipeline_block`) and the single-image predict payload
    (:func:`build_predict_request`), so a model can be evaluated through exactly
    the geometry it was trained with — see
    ``chachkalica/docs/pipeline-metadata.md``.
    """
    ts = ts or TrainingSettings.load()
    tm = pe.trained_model
    ds = pe.dataset
    if not tm.checkpoint_path:
        raise ValueError(f"{tm.name}: no checkpoint path to evaluate.")

    from eval_pipelines.models import PipelineEvalRun

    data = {
        "name": f"pipeline-{pe.pk}",
        "pipeline": pe.pipeline,
        "model_checkpoint": tm.checkpoint_path,
        "images": str(images_dir(ds)),
        "classes": dataset_classes(ds),
        "output_dir": str(output_dir),
        "map_score_threshold": pe.map_score_threshold,
        "score_threshold": pe.score_threshold,
        "iou_thresholds": default_iou_thresholds(),
        "device": ts.default_device,
    }

    labels = resolve_label_dir(ds, pe.label_source, pe.annotator, pe.explicit_labels_path)
    if labels is not None:
        data["labels"] = str(labels)

    if pe.pipeline == PipelineEvalRun.CHAIN and pe.chain:
        data["chain"] = list(pe.chain)

    needs_detector = pipelines.needs_detector(pe.pipeline, pe.chain or [])
    # Mirrors pipeline_block's fallback and its needs_detector gating — see
    # that function for why checking "is the field non-blank" alone isn't safe.
    checkpoint = (
        (pe.detector_checkpoint or DEFAULT_PERSON_DETECTOR_CHECKPOINT)
        if needs_detector else ""
    )
    if checkpoint:
        detector: dict = {"checkpoint": str(_resolve(checkpoint))}
        if pe.detector_expand_ratio is not None:
            detector["expand_ratio"] = pe.detector_expand_ratio
        # Emitted for every detector pipeline, matching pipeline_block — see there
        # for why scoping this to people_detect_first was wrong.
        if pe.detector_min_box_size:
            detector["min_box_size"] = pe.detector_min_box_size
        data["detector"] = detector

    tiling = {}
    if pe.tile_size_px:
        tiling["tile_size_px"] = pe.tile_size_px
    if pe.tile_width_pct:
        tiling["tile_width_pct"] = pe.tile_width_pct
    if pe.tile_height_pct:
        tiling["tile_height_pct"] = pe.tile_height_pct
    if pe.overlap is not None:
        tiling["overlap"] = pe.overlap
    if tiling:
        data["tiling"] = tiling
    if pe.merge_nms_iou is not None:
        data["merge_nms_iou"] = pe.merge_nms_iou

    extra_checkpoints = combined_checkpoints(pe)
    if extra_checkpoints:
        data["extra_checkpoints"] = extra_checkpoints

    return data


def write_pipeline_request(pe, ts: TrainingSettings | None = None) -> tuple[Path, str]:
    """Generate the chachak request YAML for ``pe`` and persist its paths."""
    ts = ts or TrainingSettings.load()
    request_path, output_dir = pipeline_request_paths(pe, ts)
    data = build_pipeline_request(pe, output_dir, ts)
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(text, encoding="utf-8")
    pe.request_yaml_path = str(request_path)
    pe.output_dir = str(output_dir)
    pe.save(update_fields=["request_yaml_path", "output_dir"])
    return request_path, text


# The predictions file each eval kind leaves in its output_dir (a torch pickle of
# per-image {image_path, predictions} records) — the source for "promote to labels".
PREDICTIONS_FILE = {"pipeline": "predictions.pt", "base": "eval_predictions.pt"}

# Backups of a dataset's prior source labels land here (one subdir per promoting
# run) so a promote never silently destroys hand-checked labels.
BACKUP_LABELS_SUBDIR = "backup_labels"


def build_promote_payload(eval_obj, kind: str, score_threshold: float) -> dict:
    """Assemble the trainer ``/promote_labels`` payload for one eval run.

    ``kind`` is ``"pipeline"`` (a :class:`PipelineEvalRun`) or ``"base"`` (an
    :class:`EvalRun`); it selects the predictions filename the run wrote. The
    predictions are promoted into the dataset's *source* ``labels/`` folder
    regardless of which ``label_source`` the eval scored against — promoting is
    always about becoming the source of truth. Raises ``ValueError`` when the run
    has no output dir or checkpoint yet.

    A combined run (2+ models) has no single owning checkpoint — but its saved
    predictions are already indexed in the *eval dataset's* class space (every
    model was remapped into it before merging, see ``eval_combined_checkpoints``
    / ``chachak/run.py``), so promotion there is an identity name-remap: we send
    ``prediction_classes`` instead of ``checkpoint_path`` and the trainer skips
    loading a checkpoint (see ``promote_labels.promote_labels``).
    """
    tm = eval_obj.trained_model
    ds = eval_obj.dataset
    if not eval_obj.output_dir:
        raise ValueError(f"{eval_obj}: no output dir — run the eval before promoting.")

    is_combined = getattr(eval_obj, "is_combined", False)
    if not is_combined and not tm.checkpoint_path:
        raise ValueError(f"{tm.name}: no checkpoint path.")
    try:
        predictions_file = PREDICTIONS_FILE[kind]
    except KeyError:
        raise ValueError(f"Unknown eval kind {kind!r}") from None

    dataset_root = source_root() / ds.name
    classes = dataset_classes(ds)
    payload = {
        "predictions_path": str(Path(eval_obj.output_dir) / predictions_file),
        "images_dir": str(images_dir(ds)),
        "dataset_classes": classes,
        "labels_dir": str(datasets_svc.labels_source_dir(ds)),
        "backup_dir": str(dataset_root / BACKUP_LABELS_SUBDIR / Path(eval_obj.output_dir).name),
        "score_threshold": float(score_threshold),
    }
    if is_combined:
        payload["prediction_classes"] = classes
    else:
        payload["checkpoint_path"] = tm.checkpoint_path
    return payload


#: The match-table artifact each eval kind writes, next to its predictions.
MATCH_TABLE_FILE = {"pipeline": "predictions_matches.json", "base": "eval_matches.json"}


def build_match_table_payload(eval_obj, kind: str) -> dict:
    """Assemble the trainer ``/match_table`` payload to backfill one eval's table.

    The match table (``metrics.match_table``) is written at eval time, so an
    eval that finished before that existed has none and cannot be sliced by
    annotation tag. Rebuilding it needs no model and no GPU — only the
    predictions the eval already saved and the labels on disk.

    Every threshold is taken from the eval's **stored metrics** where it
    recorded one, falling back to the row's own fields. The metrics are what
    the eval actually ran with (``operating_nms_threshold`` in particular is
    derived inside the trainer from the model's params and never appears on the
    request), and a table built with different thresholds would describe a
    different evaluation than the metrics printed beside it.
    """
    ds = eval_obj.dataset
    if not eval_obj.output_dir:
        raise ValueError(f"{eval_obj}: no output dir — run the eval first.")
    try:
        predictions_file = PREDICTIONS_FILE[kind]
    except KeyError:
        raise ValueError(f"Unknown eval kind {kind!r}") from None

    predictions_path = Path(eval_obj.output_dir) / predictions_file
    if not predictions_path.exists():
        raise ValueError(
            f"{eval_obj}: {predictions_path.name} is missing — its predictions were "
            "cleaned up or the run never finished, so there is nothing to rebuild from."
        )

    metrics = eval_obj.metrics if isinstance(eval_obj.metrics, dict) else {}
    classes = dataset_classes(ds)
    payload = {
        "predictions_path": str(predictions_path),
        "classes": classes,
        "iou_thresholds": metrics.get("iou_thresholds") or default_iou_thresholds(),
        "score_threshold": float(eval_obj.score_threshold),
        "map_score_threshold": float(eval_obj.map_score_threshold),
        "operating_nms_threshold": metrics.get("operating_nms_threshold"),
    }

    labels = resolve_label_dir(
        ds, eval_obj.label_source,
        getattr(eval_obj, "annotator", None),
        getattr(eval_obj, "explicit_labels_path", "") or "",
    )
    if labels is not None:
        payload["labels_dir"] = str(labels)

    if getattr(eval_obj, "is_combined", False):
        payload["prediction_classes"] = classes
    else:
        tm = eval_obj.trained_model
        if not tm.checkpoint_path:
            raise ValueError(f"{tm.name}: no checkpoint path to read train classes from.")
        payload["checkpoint_path"] = tm.checkpoint_path
    return payload


def build_predict_request(
    model_checkpoint: str,
    pipeline: str,
    image_path: str,
    *,
    detector_checkpoint: str = "",
    detector_expand_ratio: float | None = None,
    detector_min_box_size: float | None = None,
    tile_size_px: int | None = None,
    tile_width_pct: float | None = None,
    tile_height_pct: float | None = None,
    overlap: float | None = None,
    merge_nms_iou: float | None = None,
    chain: list[str] | None = None,
    score_threshold: float = 0.05,
    ts: TrainingSettings | None = None,
) -> dict:
    """Assemble the ``POST /predict_image`` payload for one image.

    Slim sibling of :func:`build_pipeline_request`: no labels/output_dir (the
    single-image path persists nothing) and no dataset ``classes`` (prediction
    class names come from the checkpoint). ``pipeline`` may be ``"raw"`` (run the
    model directly) or any chachak pipeline name; ``model_checkpoint`` may be a
    ``.pt`` or an exported ``.onnx`` / ``.engine`` (chachak's
    ``load_checkpoint_adapter`` dispatches on the suffix).

    Only non-default detector/tiling knobs are emitted, so chachak's own defaults
    apply wherever the caller passed nothing — same contract as
    :func:`build_pipeline_request`.
    """
    ts = ts or TrainingSettings.load()
    model_checkpoint = (model_checkpoint or "").strip()
    if not model_checkpoint:
        raise ValueError("No model checkpoint to run.")
    from eval_pipelines.models import PipelineEvalRun

    valid_pipelines = {"raw", *(value for value, _label in PipelineEvalRun.PIPELINE_CHOICES)}
    if pipeline not in valid_pipelines:
        raise ValueError(f"Unknown pipeline: {pipeline!r}.")

    chain = list(chain or [])
    if pipeline == PipelineEvalRun.CHAIN and not chain:
        raise ValueError("pipeline 'chain' requires at least one chain member.")
    unknown_chain = [c for c in chain if c not in valid_pipelines or c == "raw"]
    if unknown_chain:
        raise ValueError(f"Unknown chain member(s): {', '.join(unknown_chain)}.")

    needs_detector = pipelines.needs_detector(pipeline, chain)
    if needs_detector and not detector_checkpoint:
        raise ValueError(f"pipeline '{pipeline}' requires a detector checkpoint.")

    payload = {
        "model_checkpoint": model_checkpoint,
        "image_path": image_path,
        "pipeline": pipeline,
        "score_threshold": score_threshold,
        "device": ts.default_device,
    }
    # Gated on needs_detector (not "is detector_checkpoint non-blank"): a caller
    # may pass a non-blank checkpoint for a pipeline that doesn't use one (e.g.
    # a CameraInference row whose field carries its non-blank default while set
    # to "raw" or "batch_detect") — needs_detector already raised above if a
    # detector-requiring pipeline got here without one, so this is always safe.
    if needs_detector:
        payload["detector_checkpoint"] = detector_checkpoint
        if detector_expand_ratio is not None:
            payload["detector_expand_ratio"] = detector_expand_ratio
        if detector_min_box_size:
            payload["detector_min_box_size"] = detector_min_box_size
    if tile_size_px:
        payload["tile_size_px"] = tile_size_px
    if tile_width_pct:
        payload["tile_width_pct"] = tile_width_pct
    if tile_height_pct:
        payload["tile_height_pct"] = tile_height_pct
    if overlap is not None:
        payload["overlap"] = overlap
    # Only when set: the trainer service forwards this into chachak's
    # `merge_nms_iou`, which is parsed with an unconditional float() and would
    # crash on an explicit null rather than falling through to the default.
    if merge_nms_iou is not None:
        payload["merge_nms_iou"] = merge_nms_iou
    if chain:
        payload["chain"] = chain
    return payload


def build_preview_request(
    tm,
    pipeline: str,
    image_path: str,
    **kwargs,
) -> dict:
    """:func:`build_predict_request` for a :class:`TrainedModel`'s checkpoint."""
    if not tm.checkpoint_path:
        raise ValueError(f"{tm.name}: no checkpoint path to preview.")
    return build_predict_request(tm.checkpoint_path, pipeline, image_path, **kwargs)


def write_config(experiment: Experiment, run) -> tuple[Path, str]:
    """Generate the YAML for ``run`` and persist its paths on the run.

    Returns (yaml_path, yaml_text).
    """
    yaml_path, output_dir = run_paths(experiment, run.pk)
    text = build_yaml(experiment, output_dir)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(text, encoding="utf-8")
    run.config_yaml_path = str(yaml_path)
    run.output_dir = str(output_dir)
    run.save(update_fields=["config_yaml_path", "output_dir"])
    return yaml_path, text
