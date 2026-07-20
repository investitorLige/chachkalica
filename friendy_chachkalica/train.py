import argparse
import json
import os
import random
import tempfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader

try:
    from .config import DatasetConfig, ExperimentConfig, ExperimentRun, ModelConfig, build_experiment_runs, load_config
    from .data import build_eval_dataloader, build_train_dataloader
    from .device import resolve_device
    from .metrics import (
        HARD_IMAGE_METRIC,
        HARD_IMAGE_METRIC_DESCRIPTION,
        evaluate_detection,
        select_hard_images,
    )
    from .postprocess import apply_class_aware_nms
    from .registry import build_model
except ImportError:
    from config import DatasetConfig, ExperimentConfig, ExperimentRun, ModelConfig, build_experiment_runs, load_config
    from data import build_eval_dataloader, build_train_dataloader
    from device import resolve_device
    from metrics import (
        HARD_IMAGE_METRIC,
        HARD_IMAGE_METRIC_DESCRIPTION,
        evaluate_detection,
        select_hard_images,
    )
    from postprocess import apply_class_aware_nms
    from registry import build_model

try:
    from .cropping import crop_batch
    from .tiling import tile_batch
except ImportError:
    from cropping import crop_batch
    from tiling import tile_batch

# Pipelines the model can be *trained through* — each has a training-time
# analogue that turns a full frame into the same sub-frames the model is
# validated/served on:
#   * batch_detect            -> tile the frame                 (tiling.tile_batch)
#   * people_detect_first     -> crop around detected people    (cropping.crop_batch)
#   * batch_people            -> tile, detect people, then crop (cropping.crop_batch)
# The crop pipelines run the person detector at train time too, so training sees
# the exact person crops inference produces. Keep this in sync with
# chachkalica.training.pipelines.TRAINABLE_PIPELINES.
_TRAINABLE_PIPELINES = {"batch_detect", "people_detect_first", "batch_people"}


class ExperimentTrainingError(RuntimeError):
    """Raised after all requested runs finish when one or more failed."""

    def __init__(
        self,
        failures: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        results_path: Path,
    ) -> None:
        self.failures = failures
        self.results = results
        self.results_path = results_path
        names = ", ".join(str(item.get("run_name", item.get("run_index"))) for item in failures)
        super().__init__(
            f"{len(failures)} training run(s) failed ({names}); details: {results_path}"
        )


def _ensure_chachak_importable() -> None:
    """Put the repo root on sys.path so ``import chachak`` resolves in-process.

    chachak lives at ``<repo_root>/chachak`` and shares this torch/CUDA env.
    Mirrors ``service._ensure_chachak_importable``.
    """
    import sys

    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def _pipeline_needs_detector(spec: Any) -> bool:
    if spec.name in {"people_detect_first", "batch_people"}:
        return True
    if spec.name == "chain":
        return any(c in {"people_detect_first", "batch_people"} for c in spec.chain)
    return False


def _transform_train_batch(
    images: List[torch.Tensor],
    targets: List[Dict[str, Any]],
    config: ExperimentConfig,
    pipeline: Any,
):
    """Apply a trainable pipeline's train-time transform to one loader batch.

    Tiling pipelines tile each frame; person-crop pipelines crop around detected
    people (running the frozen detector held by ``pipeline``). Both expand one
    frame into several sub-frame samples with re-mapped targets. Assumes
    ``config.pipeline.name`` is in :data:`_TRAINABLE_PIPELINES`.
    """
    if _pipeline_needs_detector(config.pipeline):
        if pipeline is None:
            raise ValueError(
                f"pipeline '{config.pipeline.name}' crops around detected people "
                "at train time but no detector pipeline was built"
            )
        return crop_batch(images, targets, pipeline)
    return tile_batch(images, targets, config.pipeline.tiling)


def build_run_pipeline(adapter: Any, config: ExperimentConfig, device: torch.device):
    """Build a chachak pipeline wrapping the *live* in-memory ``adapter``, or None.

    Returns ``None`` when the experiment has no pipeline configured. Used for
    val/test inference (``process_batch``) so metrics reflect the same pipeline
    the model will be served through — no checkpoint reload, the pipeline holds
    the live adapter by reference so it always predicts with current weights.
    Detector pipelines load their person detector from disk once here.
    """
    spec = config.pipeline
    if spec is None:
        return None

    _ensure_chachak_importable()
    from chachak.config import pipeline_config_from_dict
    from chachak.detector import load_detector
    from chachak.registry import build_pipeline

    # chachak's PipelineConfig requires model_checkpoint/images/labels/classes/
    # output_dir, but process_batch touches none of them (no dataloader; classes
    # come from the live adapter). Placeholders mirror the predict endpoint.
    raw: Dict[str, Any] = {
        "pipeline": spec.name,
        "model_checkpoint": ".",
        "images": ".",
        "labels": ".",
        "output_dir": ".",
        "classes": ["_"],
        "device": str(device),
        "infer_batch_size": config.evaluation.batch_size or config.training.batch_size,
        # Keep the low-confidence tail so mAP can sweep the PR curve, matching
        # _predict_with_config's score_threshold handling.
        "score_threshold": config.evaluation.score_threshold,
    }
    if config.evaluation.map_score_threshold is not None:
        raw["map_score_threshold"] = config.evaluation.map_score_threshold
    if spec.detector_checkpoint is not None:
        detector_raw: Dict[str, Any] = {"checkpoint": str(spec.detector_checkpoint)}
        if spec.detector_expand_ratio is not None:
            detector_raw["expand_ratio"] = spec.detector_expand_ratio
        raw["detector"] = detector_raw
    tiling: Dict[str, Any] = {}
    if spec.tiling.tile_size_px is not None:
        tiling["tile_size_px"] = spec.tiling.tile_size_px
    if spec.tiling.tile_width_pct is not None:
        tiling["tile_width_pct"] = spec.tiling.tile_width_pct
    if spec.tiling.tile_height_pct is not None:
        tiling["tile_height_pct"] = spec.tiling.tile_height_pct
    if spec.tiling.overlap is not None:
        tiling["overlap"] = spec.tiling.overlap
    if tiling:
        raw["tiling"] = tiling
    if spec.merge_nms_iou is not None:
        raw["merge_nms_iou"] = spec.merge_nms_iou
    if spec.chain:
        raw["chain"] = list(spec.chain)

    pipeline_config = pipeline_config_from_dict(raw, Path(__file__).resolve().parent)

    detector = None
    if _pipeline_needs_detector(spec):
        detector = load_detector(
            pipeline_config.detector.checkpoint,
            device,
            person_class_name=pipeline_config.detector.person_class_name,
            person_class_id=pipeline_config.detector.person_class_id,
            score_threshold=pipeline_config.detector.score_threshold,
            batch_size=pipeline_config.infer_batch_size,
        )
    return build_pipeline(pipeline_config, adapter, device, detector)


def resolve_operating_nms_threshold(
    config: ExperimentConfig,
    model_config: Any,
) -> Optional[float]:
    """NMS IoU for the operating-point val/test metrics of one run.

    A model entry's own ``nms_threshold`` param wins (for yolox that is also its
    internal predict-time NMS, so re-applying it is a no-op; for the DETRs it
    exists purely for this), falling back to the experiment-wide
    ``evaluation.operating_nms_threshold``. None disables it — mAP is never
    affected either way.
    """
    value = (getattr(model_config, "params", None) or {}).get("nms_threshold")
    if value is None:
        value = config.evaluation.operating_nms_threshold
    return None if value is None else float(value)


def _best_metric_name(config: ExperimentConfig) -> str:
    """Identifier of what best-checkpoint selection tracks, e.g. ``val_map50``
    or ``val_f1+map50`` (an average of both).

    Stamped into checkpoints so a resume can tell whether a stored best_score
    is comparable (older checkpoints tracked val loss, where lower was better,
    or a different metric selection).
    """
    return "val_" + "+".join(config.training.best_metric)


def _best_metric_score(
    config: ExperimentConfig,
    val_map_summary: Optional[Dict[str, Any]],
) -> Optional[float]:
    """The configured selection score for one epoch: the named val metric, or
    the mean when several are configured. All choices are higher-is-better."""
    if val_map_summary is None:
        return None
    values = [val_map_summary.get(metric) for metric in config.training.best_metric]
    if any(value is None for value in values):
        return None
    return float(sum(values)) / len(values)


def train_from_config(
    config_path: str | Path,
    evaluate_after_train: bool = True,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    """Train every model declared in one Friendy Chachkalica YAML config."""
    print(f"[train] Starting from config: {config_path}")
    config = load_config(config_path)
    return train_experiment(
        config,
        evaluate_after_train=evaluate_after_train,
        resume=resume,
    )


def train_experiment(
    config: ExperimentConfig,
    evaluate_after_train: bool = True,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    if config.training.seed is not None:
        print(f"[train] Setting random seed: {config.training.seed}")
        _set_seed(config.training.seed)

    device = resolve_device(config.training.device)
    print(f"[train] Using device: {device}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Output directory: {config.output_dir}")
    _write_yaml(config.output_dir / "config.resolved.yaml", _to_builtin(config))
    print(f"[train] Wrote resolved config: {config.output_dir / 'config.resolved.yaml'}")

    train_loaders: Dict[tuple, DataLoader] = {}
    eval_loaders: Dict[tuple, DataLoader] = {}
    results = []
    runs = build_experiment_runs(config)
    results_path = config.output_dir / "results.yaml"
    print(f"[train] Training {len(runs)} run(s) (resume={resume})")
    for run in runs:
        result_path = config.output_dir / run.name / "result.yaml"
        if resume and result_path.exists():
            print(f"[train] Run {run.name} already complete, skipping (found {result_path})")
            results.append(_read_yaml(result_path))
            _write_yaml(results_path, _to_builtin(results))
            continue

        train_loader = _get_train_loader(config, run.train_dataset, train_loaders)
        val_loader = _get_eval_loader(config, run.val_dataset, eval_loaders)
        test_loader = _get_eval_loader(config, run.test_dataset, eval_loaders)

        try:
            result = train_model(
                config=config,
                run=run,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                evaluate_after_train=False,
                resume=resume,
            )
        except Exception as exc:
            print(f"[train] Run {run.name} FAILED: {exc}")
            traceback.print_exc()
            result = {
                "run_index": run.index,
                "run_name": run.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        _write_yaml(results_path, _to_builtin(results))

    _raise_if_training_failed(results, results_path)

    if evaluate_after_train:
        _evaluate_all_runs_after_train(config)

    return results


def _raise_if_training_failed(
    results: List[Dict[str, Any]],
    results_path: Path,
) -> None:
    failures = [result for result in results if result.get("error") is not None]
    if failures:
        print(
            f"[train] Experiment FAILED: {len(failures)}/{len(results)} run(s) failed; "
            f"details written to {results_path}"
        )
        raise ExperimentTrainingError(failures, results, results_path)


def _evaluate_all_runs_after_train(config: ExperimentConfig) -> None:
    """Run the val/test evaluation phase across every run from its saved checkpoints.

    This is the consolidated "testing phase": it covers every run uniformly,
    including runs that were skipped by --resume because they were already
    trained. Evaluation reads each run's best/last checkpoint, so no retraining
    happens here.
    """
    if config.val_dataset is None and config.test_dataset is None:
        print("[train] No val/test dataset configured; skipping post-train evaluation")
        return

    # Imported lazily: val.py imports predict_dataset from this module, so a
    # top-level import here would be circular.
    try:
        from .val import val_experiment
    except ImportError:
        from val import val_experiment

    checkpoint = "best" if config.val_dataset is not None else "last"
    print(f"[train] Post-train evaluation phase start (checkpoint={checkpoint})")
    if config.val_dataset is not None:
        print("[train] Evaluating val split for all runs")
        val_experiment(config, split="val", checkpoint=checkpoint)
    if config.test_dataset is not None:
        print("[train] Evaluating test split for all runs")
        val_experiment(config, split="test", checkpoint=checkpoint)
    print("[train] Post-train evaluation phase done")


def _get_train_loader(
    config: ExperimentConfig,
    dataset_config: DatasetConfig,
    cache: Dict[tuple, DataLoader],
) -> DataLoader:
    cache_key = _dataset_cache_key(dataset_config)
    loader = cache.get(cache_key)
    if loader is None:
        print(f"[train] Creating train loader for dataset={dataset_config.name}")
        loader = build_train_dataloader(config, dataset_config)
        cache[cache_key] = loader
    else:
        print(f"[train] Reusing train loader for dataset={dataset_config.name}")
    return loader


def _get_eval_loader(
    config: ExperimentConfig,
    dataset_config: Optional[DatasetConfig],
    cache: Dict[tuple, DataLoader],
) -> Optional[DataLoader]:
    if dataset_config is None:
        return None

    cache_key = _dataset_cache_key(dataset_config)
    loader = cache.get(cache_key)
    if loader is None:
        print(f"[train] Creating eval loader for dataset={dataset_config.name} role={dataset_config.role}")
        loader = build_eval_dataloader(dataset_config, config)
        cache[cache_key] = loader
    else:
        print(f"[train] Reusing eval loader for dataset={dataset_config.name} role={dataset_config.role}")
    return loader


_WARM_START_STRUCTURAL_PARAMS = {
    "retinanet": ("variant",),
    "fasterrcnn": ("variant",),
    "yolox": ("variant",),
    "rfdetr": ("variant", "resolution"),
    # RT-DETR's repository id selects the model topology (r18/r50/v1/v2).
    "rtdetr": ("weights",),
}


def _read_initial_checkpoint(
    checkpoint_path: Path,
    expected_model_name: str,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, Mapping) or not isinstance(
        state.get("model_state_dict"), Mapping
    ):
        raise ValueError(
            f"{checkpoint_path} is not a Friendy training checkpoint "
            "(missing model_state_dict)."
        )

    checkpoint_model_name = state.get("model_name")
    if checkpoint_model_name != expected_model_name:
        raise ValueError(
            f"Warm-start architecture mismatch: checkpoint is "
            f"{checkpoint_model_name!r}, requested model is {expected_model_name!r}."
        )
    return dict(state)


def _warm_start_build_params(
    model_config: ModelConfig,
    checkpoint_state: Dict[str, Any],
) -> Dict[str, Any]:
    source_config = checkpoint_state.get("model_config") or {}
    source_params = dict(source_config.get("params") or {})
    current_params = dict(model_config.params)

    for key in _WARM_START_STRUCTURAL_PARAMS.get(model_config.name, ()):
        source_value = source_params.get(key)
        current_value = current_params.get(key)
        if (
            source_value is not None
            and current_value is not None
            and source_value != current_value
        ):
            raise ValueError(
                f"Warm-start {model_config.name} parameter mismatch for {key!r}: "
                f"checkpoint uses {source_value!r}, current model requests "
                f"{current_value!r}."
            )

    build_params = {**source_params, **current_params}
    if model_config.name != "rtdetr":
        # These architectures select topology independently of weights. Build
        # without another download, then restore the Friendy checkpoint below.
        build_params["weights"] = False
    return build_params


def _load_warm_start_state(
    model: torch.nn.Module,
    checkpoint_state: Dict[str, Any],
    checkpoint_path: Path,
) -> None:
    source_state = checkpoint_state["model_state_dict"]
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in model_state
        and torch.is_tensor(value)
        and value.shape == model_state[key].shape
    }
    loaded_parameters = sum(model_state[key].numel() for key in compatible)
    total_parameters = sum(value.numel() for value in model_state.values())
    coverage = loaded_parameters / max(1, total_parameters)
    if not compatible or coverage < 0.5:
        raise ValueError(
            f"Warm-start checkpoint {checkpoint_path} is not sufficiently compatible: "
            f"{len(compatible)}/{len(model_state)} tensors, "
            f"{coverage:.1%} of parameters."
        )

    model.load_state_dict(compatible, strict=False)
    reinitialized = len(model_state) - len(compatible)
    print(
        f"[train] Warm-start loaded {len(compatible)}/{len(model_state)} tensors "
        f"({coverage:.1%} of parameters) from {checkpoint_path}; "
        f"reinitialized {reinitialized} task-specific/incompatible tensor(s)."
    )


def train_model(
    config: ExperimentConfig,
    run: ExperimentRun,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
    device: torch.device,
    evaluate_after_train: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    model_config = run.model
    train_dataset_config = run.train_dataset
    run_name = run.name
    run_dir = config.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[train] Run {run.index} start: name={run_name} model={model_config.name} "
        f"num_classes={model_config.num_classes} train_dataset={train_dataset_config.name}"
    )
    print(f"[train] Run directory: {run_dir}")

    initial_state = None
    build_params = dict(model_config.params)
    if model_config.init_checkpoint is not None:
        print(f"[train] Reading warm-start checkpoint: {model_config.init_checkpoint}")
        initial_state = _read_initial_checkpoint(
            model_config.init_checkpoint,
            expected_model_name=model_config.name,
        )
        build_params = _warm_start_build_params(model_config, initial_state)

    effective_model_config = ModelConfig(
        name=model_config.name,
        num_classes=model_config.num_classes,
        params=build_params,
    )
    print(f"[train] Building model adapter: {model_config.name}")
    adapter = build_model(
        model_config.name,
        num_classes=model_config.num_classes,
        **build_params,
    )
    if initial_state is not None:
        _load_warm_start_state(
            adapter.model,
            initial_state,
            model_config.init_checkpoint,
        )
    adapter.to(device)
    print(f"[train] Model moved to device: {device}")

    optimizer = build_optimizer(adapter.model.parameters(), config)
    scheduler = build_scheduler(optimizer, config)
    scaler = _build_grad_scaler(config, device, adapter)
    print(
        f"[train] Optimizer={config.training.optimizer.name} lr={config.training.optimizer.lr} "
        f"scheduler={config.training.scheduler.name} amp={scaler is not None}"
    )

    history = []
    best_score = None
    best_epoch = None
    epochs_without_improvement = 0
    start_epoch = 1
    best_metric_name = _best_metric_name(config)
    val_interval = config.training.val_interval
    operating_nms_threshold = resolve_operating_nms_threshold(config, model_config)
    print(
        f"[train] Best-checkpoint selection metric: {best_metric_name} "
        f"(val_interval={val_interval} operating_nms_threshold={operating_nms_threshold})"
    )

    # When the experiment has a pipeline, val (and post-train test) inference is
    # routed through it. The pipeline wraps the live adapter by reference, so it
    # always predicts with the current epoch's weights.
    eval_pipeline = build_run_pipeline(adapter, config, device)
    if eval_pipeline is not None:
        print(f"[train] Val/test inference routed through chachak pipeline: {config.pipeline.name}")
        if config.pipeline.name in _TRAINABLE_PIPELINES:
            if _pipeline_needs_detector(config.pipeline):
                print(
                    "[train] Training on person crops: the person detector runs "
                    "each batch and ground-truth objects outside every detected "
                    "person crop are not seen (detector recall bounds train too)."
                )
            else:
                print("[train] Training on tiled frames.")

    last_checkpoint = run_dir / "last.pt"
    if resume and last_checkpoint.exists():
        print(f"[train] Resuming run {run_name} from checkpoint: {last_checkpoint}")
        state = torch.load(last_checkpoint, map_location=device)
        adapter.model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        history = list(state.get("history") or [])
        best_score = state.get("best_score")
        best_epoch = _best_epoch_from_history(history)
        if best_score is not None and state.get("best_metric") != best_metric_name:
            # Scores on different metrics aren't comparable (older checkpoints
            # even tracked val loss, where lower was better), so a stale best
            # would never — or wrongly — be beaten.
            print(
                f"[train] Resume: checkpoint tracked best_score on a different "
                f"metric ({state.get('best_metric')!r}), now selecting on "
                f"{best_metric_name}; restarting best-model tracking"
            )
            best_score = None
            best_epoch = None
        # Restore the early-stopping counter, otherwise a resumed run gets up
        # to `patience` extra epochs before stopping.
        epochs_without_improvement = _epochs_since_best(history, best_epoch)
        start_epoch = int(state.get("epoch", 0)) + 1
        print(
            f"[train] Resumed run {run_name} at epoch {start_epoch}/{config.training.epochs} "
            f"(best_score={best_score} best_epoch={best_epoch} "
            f"epochs_without_improvement={epochs_without_improvement})"
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        print(f"[train] Run {run_name} epoch {epoch}/{config.training.epochs} start")
        train_summary = train_one_epoch(
            adapter=adapter,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            device=device,
            pipeline=eval_pipeline,
        )

        val_summary = None
        val_map_summary = None
        # Validation runs every `val_interval` epochs; the final epoch always
        # validates so the run ends with a scored epoch (and thus a best.pt).
        run_val = val_loader is not None and (
            epoch % val_interval == 0 or epoch == config.training.epochs
        )
        if val_loader is not None and not run_val:
            print(
                f"[train] Run {run_name} epoch {epoch}: skipping validation "
                f"(val_interval={val_interval})"
            )
        if run_val:
            print(f"[train] Run {run_name} epoch {epoch}: evaluating validation loss")
            val_summary = evaluate_loss(
                adapter,
                val_loader,
                device,
                config=config,
                source_classes=run.val_dataset.classes if run.val_dataset is not None else None,
                model_classes=train_dataset_config.classes,
                pipeline=eval_pipeline,
            )
            print(f"[train] Run {run_name} epoch {epoch}: evaluating validation mAP")
            val_map_summary = evaluate_map(
                adapter,
                val_loader,
                device,
                config=config,
                hard_images_path=run_dir / "val_predictions.pt",
                prediction_classes=train_dataset_config.classes,
                target_classes=(
                    run.val_dataset.classes
                    if run.val_dataset is not None
                    else train_dataset_config.classes
                ),
                operating_nms_threshold=operating_nms_threshold,
                pipeline=eval_pipeline,
            )

        if scheduler is not None:
            scheduler.step()

        # The best checkpoint is selected on the configured val metric(s)
        # (training.best_metric, higher is better; several are averaged), not
        # val loss: the summed loss mixes objectness/cls/box terms and
        # routinely diverges from detection quality.
        score = _best_metric_score(config, val_map_summary)
        is_best = score is not None and (best_score is None or score > best_score)
        if is_best:
            best_score = score
            best_epoch = epoch
        if score is not None:
            epochs_without_improvement = 0 if is_best else epochs_without_improvement + 1

        epoch_summary = {
            "epoch": epoch,
            "train": train_summary,
            "val": val_summary,
            # Compact subset only: the full metrics dict (per_class etc.) would
            # bloat history.yaml and every checkpoint.
            "val_map": _compact_map_summary(val_map_summary),
            # What checkpoint selection tracked this epoch, so progress
            # displays can label the score without knowing the config.
            "best_metric": best_metric_name,
            "best_metric_score": score,
            "lr": _current_lr(optimizer),
            "is_best": is_best,
        }
        history.append(epoch_summary)

        checkpoint = {
            "epoch": epoch,
            "model_name": model_config.name,
            "model_config": _to_builtin(effective_model_config),
            "init_checkpoint": str(model_config.init_checkpoint) if model_config.init_checkpoint else None,
            "train_dataset": _to_builtin(train_dataset_config),
            "model_state_dict": adapter.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_score": best_score,
            "best_metric": best_metric_name,
            "history": history,
        }
        save_checkpoint(checkpoint, run_dir / "last.pt")
        print(f"[train] Saved checkpoint: {run_dir / 'last.pt'}")
        if is_best:
            save_checkpoint(checkpoint, run_dir / "best.pt")
            print(f"[train] Saved new best checkpoint: {run_dir / 'best.pt'}")

        _write_yaml(run_dir / "history.yaml", _to_builtin(history))
        print(
            f"[train] Run {run_name} epoch {epoch} done: "
            f"train_loss={train_summary.get('loss')} "
            f"val_loss={val_summary.get('loss') if val_summary else None} "
            f"val_map50={val_map_summary.get('map50') if val_map_summary else None} "
            f"val_map50_95={val_map_summary.get('map50_95') if val_map_summary else None} "
            f"{best_metric_name}={score} "
            f"lr={_current_lr(optimizer)} best={is_best}"
        )

        # Early stopping: once the best metric hasn't improved for `patience`
        # *scored* epochs (val_interval > 1 skips epochs without scoring them),
        # stop — the best.pt checkpoint already holds the best epoch, so
        # continuing just overfits. Only active when a val set produced a score.
        patience = config.training.early_stopping_patience
        if patience is not None and score is not None and epochs_without_improvement >= patience:
            print(
                f"[train] Run {run_name} early stopping at epoch {epoch}: no val "
                f"improvement for {epochs_without_improvement} scored epoch(s) "
                f"(best epoch {best_epoch}, best {best_metric_name}={best_score})"
            )
            break

    if evaluate_after_train and test_loader is not None:
        print(f"[train] Run {run_name}: running post-train test prediction")
        best_checkpoint = run_dir / "best.pt"
        if val_loader is not None and best_checkpoint.exists():
            print(f"[train] Loading best checkpoint for test: {best_checkpoint}")
            state = torch.load(best_checkpoint, map_location=device)
            adapter.model.load_state_dict(state["model_state_dict"])
        prediction_path = run_dir / "test_predictions.pt"
        test_metrics = predict_dataset(
            adapter,
            test_loader,
            device,
            prediction_path,
            config,
            num_classes=model_config.num_classes,
            prediction_classes=train_dataset_config.classes,
            target_classes=run.test_dataset.classes if run.test_dataset is not None else None,
            eval_classes=run.test_dataset.classes if run.test_dataset is not None else None,
            operating_nms_threshold=operating_nms_threshold,
            pipeline=eval_pipeline,
        )
    else:
        prediction_path = None
        test_metrics = None

    result = {
        "run_index": run.index,
        "model": model_config.name,
        "model_num_classes": model_config.num_classes,
        "train_dataset": train_dataset_config.name,
        "train_dataset_images": str(train_dataset_config.images),
        "train_dataset_labels": str(train_dataset_config.labels),
        "train_dataset_role": train_dataset_config.role,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "init_checkpoint": str(model_config.init_checkpoint) if model_config.init_checkpoint else None,
        "best_epoch": best_epoch,
        "best_metric": best_metric_name,
        "best_score": best_score,
        "best_loss": _val_loss_at_epoch(history, best_epoch),
        "last_epoch": history[-1]["epoch"] if history else config.training.epochs,
        "last_train_loss": history[-1]["train"]["loss"] if history else None,
        "last_val_loss": history[-1]["val"]["loss"] if history and history[-1]["val"] else None,
        "best_checkpoint": str(run_dir / "best.pt") if best_epoch is not None else None,
        "last_checkpoint": str(run_dir / "last.pt"),
        "test_predictions": str(prediction_path) if prediction_path is not None else None,
        "test_metrics": test_metrics,
    }
    _write_yaml(run_dir / "result.yaml", _to_builtin(result))
    print(f"[train] Run {run_name} complete: result={run_dir / 'result.yaml'}")
    return result


def train_one_epoch(
    adapter: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    config: ExperimentConfig,
    device: torch.device,
    pipeline: Any = None,
) -> Dict[str, Any]:
    adapter.train()
    total_loss = 0.0
    total_images = 0
    loss_totals: Dict[str, float] = {}

    # A trainable pipeline turns each frame into several sub-frame samples
    # (tiles, or person crops), so a loader batch expands into more (smaller)
    # samples than batch_size. We re-chunk the expanded samples back into
    # batch_size micro-batches, each its own optimizer step, to keep per-step
    # memory in line with untransformed training.
    transform = config.pipeline is not None and config.pipeline.name in _TRAINABLE_PIPELINES
    micro_bs = max(1, config.training.batch_size)

    for images, targets in loader:
        images, targets = _move_batch_to_device(images, targets, device)

        if transform:
            images, targets = _transform_train_batch(images, targets, config, pipeline)
            if not images:
                continue  # nothing to train on (all background / no detections)

        for start in range(0, len(images), micro_bs):
            chunk_images = images[start : start + micro_bs]
            chunk_targets = targets[start : start + micro_bs]
            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(config, device, adapter):
                loss, loss_items = adapter.training_step(chunk_images, chunk_targets)

            _require_finite_loss(
                loss,
                loss_items,
                phase="training",
            )

            if scaler is not None:
                scaler.scale(loss).backward()
                if config.training.gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        adapter.model.parameters(),
                        config.training.gradient_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.training.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        adapter.model.parameters(),
                        config.training.gradient_clip_norm,
                    )
                optimizer.step()

            batch_size = len(chunk_images)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_images += batch_size
            _accumulate_losses(loss_totals, loss_items, batch_size)

    return _summarize_losses(total_loss, total_images, loss_totals)


@torch.no_grad()
def evaluate_loss(
    adapter: Any,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    source_classes: Optional[Dict[int, str]] = None,
    model_classes: Optional[Dict[int, str]] = None,
    pipeline: Any = None,
) -> Dict[str, Any]:
    total_loss = 0.0
    total_images = 0
    loss_totals: Dict[str, float] = {}

    transform = config.pipeline is not None and config.pipeline.name in _TRAINABLE_PIPELINES
    micro_bs = max(1, config.evaluation.batch_size or config.training.batch_size)

    for images, targets in loader:
        images, targets = _move_batch_to_device(images, targets, device)
        targets = _remap_targets_to_model_classes(targets, source_classes, model_classes)
        if transform:
            images, targets = _transform_train_batch(images, targets, config, pipeline)
            if not images:
                continue

        for start in range(0, len(images), micro_bs):
            chunk_images = images[start : start + micro_bs]
            chunk_targets = targets[start : start + micro_bs]
            loss_step = getattr(adapter, "validation_step", adapter.training_step)
            with _autocast_context(config, device, adapter):
                loss, loss_items = loss_step(chunk_images, chunk_targets)
            _require_finite_loss(
                loss,
                loss_items,
                phase="validation",
            )
            batch_size = len(chunk_images)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_images += batch_size
            _accumulate_losses(loss_totals, loss_items, batch_size)

    return _summarize_losses(total_loss, total_images, loss_totals)


@torch.no_grad()
def evaluate_map(
    adapter: Any,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    hard_images_path: Optional[str | Path] = None,
    prediction_classes: Optional[Dict[int, str]] = None,
    target_classes: Optional[Dict[int, str]] = None,
    operating_nms_threshold: Optional[float] = None,
    pipeline: Any = None,
) -> Dict[str, Any]:
    """Predict over a loader and compute detection metrics.

    Used for per-epoch validation so best-checkpoint selection and early
    stopping can track val mAP instead of val loss. When ``hard_images_path`` is
    supplied, it also refreshes the live ``val_hard_images.json`` viewer artifact.
    Predictions are remapped by class name onto ``target_classes``, mirroring
    ``predict_dataset``.
    """
    was_training = adapter.model.training
    adapter.eval()
    all_predictions = []
    all_targets = []
    records = []
    try:
        for images, targets in loader:
            images, targets = _move_batch_to_device(images, targets, device)
            predictions = _predict_with_config(
                adapter, images, config, pipeline=pipeline, targets=targets
            )
            _require_prediction_batch(images, targets, predictions, phase="validation")
            predictions = _apply_eval_nms(predictions, config)
            for target, prediction in zip(targets, predictions):
                prediction = prediction.detach().cpu()
                target_cpu = _target_to_cpu(target)
                all_predictions.append(prediction)
                all_targets.append(target_cpu)
                records.append(
                    {
                        "image_path": target.get("image_path"),
                        "label_path": target.get("label_path"),
                        "orig_size": _cpu_value(target.get("orig_size")),
                    }
                )
    finally:
        adapter.train(was_training)

    if not compute_metrics:
        return {
            "prediction_only": True,
            "evaluated_at": started_at.isoformat(timespec="seconds"),
            "eval_seconds": round(time.perf_counter() - start_perf, 3),
        }

    metrics = evaluate_detection(
        all_predictions,
        all_targets,
        iou_thresholds=config.evaluation.iou_thresholds,
        score_threshold=config.evaluation.score_threshold,
        map_score_threshold=config.evaluation.map_score_threshold,
        prediction_classes=prediction_classes,
        target_classes=target_classes,
        eval_classes=target_classes,
        operating_nms_threshold=operating_nms_threshold,
    )
    _print_eval_map_debug(
        metrics,
        all_predictions,
        all_targets,
        prediction_classes=prediction_classes,
        target_classes=target_classes,
    )
    if hard_images_path is not None:
        _write_hard_images(
            hard_images_path,
            all_predictions,
            all_targets,
            records,
            config=config,
            prediction_classes=prediction_classes,
            target_classes=target_classes,
            eval_classes=target_classes,
            operating_nms_threshold=operating_nms_threshold,
        )
    return metrics


@torch.no_grad()
def predict_dataset(
    adapter: Any,
    loader: DataLoader,
    device: torch.device,
    output_path: str | Path,
    config: Optional[ExperimentConfig] = None,
    num_classes: Optional[int] = None,
    prediction_classes: Optional[Dict[int, str]] = None,
    target_classes: Optional[Dict[int, str]] = None,
    eval_classes: Optional[Dict[int, str]] = None,
    operating_nms_threshold: Optional[float] = None,
    pipeline: Any = None,
    compute_metrics: bool = True,
) -> Dict[str, Any]:
    adapter.eval()
    records = []
    all_predictions = []
    all_targets = []
    started_at = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    print(f"[train] Predicting dataset to: {output_path} (started {started_at.isoformat(timespec='seconds')})")
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images, targets = _move_batch_to_device(images, targets, device)
        predictions = _predict_with_config(
            adapter, images, config, pipeline=pipeline, targets=targets
        )
        _require_prediction_batch(images, targets, predictions, phase="evaluation")
        predictions = _apply_eval_nms(predictions, config)
        print(f"[train] Predicted batch {batch_index}: images={len(images)}")
        for target, prediction in zip(targets, predictions):
            prediction = prediction.detach().cpu()
            target_cpu = _target_to_cpu(target)
            all_predictions.append(prediction)
            all_targets.append(target_cpu)
            records.append(
                {
                    "image_path": target.get("image_path"),
                    "label_path": target.get("label_path"),
                    "orig_size": _cpu_value(target.get("orig_size")),
                    "predictions": prediction,
                }
            )
    torch.save(records, output_path)
    print(f"[train] Saved predictions: {output_path} records={len(records)}")

    if not compute_metrics:
        return {
            "prediction_only": True,
            "evaluated_at": started_at.isoformat(timespec="seconds"),
            "eval_seconds": round(time.perf_counter() - start_perf, 3),
        }

    metrics = evaluate_detection(
        all_predictions,
        all_targets,
        iou_thresholds=config.evaluation.iou_thresholds if config is not None else None,
        score_threshold=config.evaluation.score_threshold if config is not None else 0.001,
        map_score_threshold=config.evaluation.map_score_threshold if config is not None else None,
        num_classes=num_classes,
        prediction_classes=prediction_classes,
        target_classes=target_classes,
        eval_classes=eval_classes,
        operating_nms_threshold=operating_nms_threshold,
    )
    # Stamp the run with wall-clock timing so downstream tooling can show when the
    # eval ran and how long it took, alongside the quality metrics.
    metrics["evaluated_at"] = started_at.isoformat(timespec="seconds")
    metrics["eval_seconds"] = round(time.perf_counter() - start_perf, 3)
    print(
        f"[train] Metrics: map50={metrics.get('map50')} "
        f"map50_95={metrics.get('map50_95')} precision={metrics.get('precision')} "
        f"recall={metrics.get('recall')} eval_seconds={metrics.get('eval_seconds')}"
    )

    _write_hard_images(
        output_path,
        all_predictions,
        all_targets,
        records,
        config=config,
        prediction_classes=prediction_classes,
        target_classes=target_classes,
        eval_classes=eval_classes,
        operating_nms_threshold=operating_nms_threshold,
    )
    return metrics


def _write_hard_images(
    predictions_path: str | Path,
    all_predictions: List[torch.Tensor],
    all_targets: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    config: Optional[ExperimentConfig],
    prediction_classes: Optional[Dict[int, str]],
    target_classes: Optional[Dict[int, str]],
    eval_classes: Optional[Dict[int, str]],
    operating_nms_threshold: Optional[float] = None,
    top_k: int = 50,
    iou_threshold: float = 0.5,
    score_threshold: Optional[float] = None,
    max_display_predictions: int = 20,
) -> None:
    """Persist the ``top_k`` hardest images alongside the predictions file.

    Writes ``<split>_hard_images.json`` next to ``<split>_predictions.pt`` (self-contained:
    image paths + normalized boxes + class names), which the admin viewer renders. Guarded so
    a split with no ground truth is skipped and any failure never sinks the eval that already
    produced its metrics. Ranking uses the deployed operating confidence and NMS;
    AP's low-confidence collection floor remains exclusive to AP integration.
    """
    if not all_targets or not any(int(target['labels'].numel()) for target in all_targets):
        print("[train] Skipping hard-images artifact: no ground-truth labels in split")
        return

    if score_threshold is None:
        score_threshold = (
            config.evaluation.score_threshold
            if config is not None
            else 0.25
        )

    predictions_path = Path(predictions_path)
    if predictions_path.name.endswith("_predictions.pt"):
        out_name = predictions_path.name[: -len("_predictions.pt")] + "_hard_images.json"
    else:
        out_name = predictions_path.stem + "_hard_images.json"
    output_path = predictions_path.with_name(out_name)

    try:
        ranking_predictions = [
            apply_class_aware_nms(prediction, operating_nms_threshold)
            for prediction in all_predictions
        ]
        images = select_hard_images(
            ranking_predictions,
            all_targets,
            records,
            top_k=top_k,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            prediction_classes=prediction_classes,
            target_classes=target_classes,
            eval_classes=eval_classes,
            max_display_predictions=max_display_predictions,
        )
        payload = {
            "metric": HARD_IMAGE_METRIC,
            "metric_description": HARD_IMAGE_METRIC_DESCRIPTION,
            "iou_threshold": float(iou_threshold),
            "score_threshold": float(score_threshold),
            "operating_nms_threshold": (
                None if operating_nms_threshold is None else float(operating_nms_threshold)
            ),
            "max_display_predictions": int(max_display_predictions),
            "top_k": int(top_k),
            "num_images_ranked": len(all_targets),
            "images": images,
        }
        _atomic_write_json(output_path, payload)
        print(f"[train] Saved hard images: {output_path} count={len(images)}")
    except Exception as exc:  # noqa: BLE001 - artifact is best-effort; never break the eval
        print(f"[train] WARNING: failed to write hard images ({output_path}): {exc}")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Publish JSON atomically so live readers never observe a partial epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write(chr(10))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _print_eval_map_debug(
    metrics: Dict[str, Any],
    predictions: List[torch.Tensor],
    targets: List[Dict[str, Any]],
    *,
    prediction_classes: Optional[Dict[int, str]],
    target_classes: Optional[Dict[int, str]],
) -> None:
    raw_predictions = int(sum(prediction.shape[0] for prediction in predictions if prediction is not None))
    max_score = None
    score_parts = [prediction[:, 4].detach().cpu() for prediction in predictions if prediction is not None and prediction.numel()]
    if score_parts:
        max_score = round(float(torch.cat(score_parts).max().item()), 4)
    target_count = int(sum(int(target["labels"].numel()) for target in targets))
    pred_names = {str(name) for name in (prediction_classes or {}).values()}
    target_names = {str(name) for name in (target_classes or {}).values()}
    common_names = sorted(pred_names & target_names)
    print(
        "[train] Val mAP debug: "
        f"raw_predictions={raw_predictions} "
        f"operating_predictions={metrics.get('num_predictions')} "
        f"targets={target_count} max_score={max_score} "
        f"eval_classes_with_gt={metrics.get('num_eval_classes_with_gt')}/"
        f"{metrics.get('num_eval_classes')} "
        f"class_name_overlap={len(common_names)}"
    )
    if prediction_classes is not None and target_classes is not None and len(common_names) < len(target_names):
        missing = sorted(target_names - pred_names)
        if missing:
            print(f"[train] Val mAP debug: target classes not predicted by this model: {missing}")


def _require_prediction_batch(images, targets, predictions, *, phase: str) -> None:
    image_count = len(images)
    target_count = len(targets)
    prediction_count = -1 if predictions is None else len(predictions)
    if image_count != target_count or prediction_count != image_count:
        raise RuntimeError(
            f"{phase} batch cardinality mismatch: images={image_count} "
            f"targets={target_count} predictions={prediction_count}"
        )


def _apply_eval_nms(
    predictions: List[torch.Tensor],
    config: Optional[ExperimentConfig],
) -> List[torch.Tensor]:
    threshold = None if config is None else config.evaluation.nms_threshold
    return [apply_class_aware_nms(prediction, threshold) for prediction in predictions]


def _predict_with_config(
    adapter: Any,
    images: List[torch.Tensor],
    config: Optional[ExperimentConfig],
    pipeline: Any = None,
    targets: Optional[List[Dict[str, Any]]] = None,
) -> List[torch.Tensor]:
    # When a chachak pipeline is attached, inference runs through it (tiling /
    # crop-around-people etc.) instead of a plain full-frame forward. It returns
    # one full-frame-normalized (N, 6) Friendy tensor per image — the same shape
    # adapter.predict yields — and already merges tile/crop duplicates.
    if pipeline is not None:
        return pipeline.process_batch(images, targets)

    if config is None:
        return adapter.predict(images)

    threshold = config.evaluation.map_score_threshold
    if threshold is None:
        threshold = config.evaluation.score_threshold
    return adapter.predict(images, score_threshold=threshold)


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    optimizer_config = config.training.optimizer
    name = optimizer_config.name.lower()
    params = list(parameters)
    kwargs = dict(optimizer_config.params)

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            **kwargs,
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            **kwargs,
        )
    if name == "sgd":
        kwargs.setdefault("momentum", 0.9)
        return torch.optim.SGD(
            params,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            **kwargs,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    scheduler_config = config.training.scheduler
    if scheduler_config.name is None:
        return None

    name = scheduler_config.name.lower()
    kwargs = dict(scheduler_config.params)
    if name in {"step", "step_lr", "steplr"}:
        kwargs.setdefault("step_size", 30)
        kwargs.setdefault("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name in {"multistep", "multi_step", "multi_step_lr", "multisteplr"}:
        kwargs.setdefault("milestones", [60, 80])
        kwargs.setdefault("gamma", 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    if name in {"cosine", "cosine_annealing", "cosine_annealing_lr"}:
        kwargs.setdefault("T_max", config.training.epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name in {"exponential", "exponential_lr"}:
        kwargs.setdefault("gamma", 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)

    raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")


def save_checkpoint(checkpoint: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _move_batch_to_device(
    images: List[torch.Tensor],
    targets: List[Dict[str, Any]],
    device: torch.device,
) -> tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    moved_images = [image.to(device, non_blocking=True) for image in images]
    moved_targets = []
    for target in targets:
        moved_targets.append(
            {
                key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
                for key, value in target.items()
            }
        )
    return moved_images, moved_targets


def _remap_targets_to_model_classes(
    targets: List[Dict[str, Any]],
    source_classes: Optional[Dict[int, str]],
    model_classes: Optional[Dict[int, str]],
) -> List[Dict[str, Any]]:
    if source_classes is None or model_classes is None:
        return targets

    model_name_to_id = {str(name): int(class_id) for class_id, name in model_classes.items()}
    source_to_model_id = {
        int(source_id): model_name_to_id[str(source_name)]
        for source_id, source_name in source_classes.items()
        if str(source_name) in model_name_to_id
    }

    return [
        _remap_target_to_model_classes(target, source_to_model_id)
        for target in targets
    ]


def _remap_target_to_model_classes(
    target: Dict[str, Any],
    source_to_model_id: Dict[int, int],
) -> Dict[str, Any]:
    labels = target["labels"].long()
    if labels.numel() == 0:
        return target

    remapped_labels = torch.full_like(labels, fill_value=-1)
    for source_id, model_id in source_to_model_id.items():
        remapped_labels[labels == source_id] = int(model_id)

    keep = remapped_labels >= 0
    remapped_target = dict(target)
    remapped_target["labels"] = remapped_labels[keep]
    remapped_target["boxes"] = target["boxes"][keep]

    if "area" in target and torch.is_tensor(target["area"]):
        remapped_target["area"] = target["area"][keep]
    if "iscrowd" in target and torch.is_tensor(target["iscrowd"]):
        remapped_target["iscrowd"] = target["iscrowd"][keep]

    return remapped_target


def _dataset_cache_key(dataset_config: DatasetConfig) -> tuple:
    return (
        dataset_config.name,
        str(dataset_config.images),
        str(dataset_config.labels),
        dataset_config.role,
        tuple(sorted((dataset_config.augmentation or {}).items())),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_grad_scaler(
    config: ExperimentConfig,
    device: torch.device,
    adapter: Any,
) -> Optional[torch.amp.GradScaler]:
    if not _amp_enabled(config, device, adapter):
        return None
    return torch.amp.GradScaler("cuda")


def _autocast_context(config: ExperimentConfig, device: torch.device, adapter: Any):
    enabled = _amp_enabled(config, device, adapter)
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def _amp_enabled(config: ExperimentConfig, device: torch.device, adapter: Any) -> bool:
    return (
        config.training.amp
        and device.type == "cuda"
        and getattr(adapter, "supports_amp", True)
    )


def _require_finite_loss(
    loss: torch.Tensor,
    loss_items: Dict[str, torch.Tensor],
    phase: str,
) -> None:
    if not torch.is_tensor(loss) or loss.numel() != 1:
        raise TypeError(
            f"{phase} loss must be a scalar tensor, got {type(loss).__name__} "
            f"with shape {getattr(loss, 'shape', None)}"
        )
    if bool(torch.isfinite(loss.detach()).all()):
        return

    nonfinite_items = [
        name
        for name, value in loss_items.items()
        if torch.is_tensor(value) and not bool(torch.isfinite(value.detach()).all())
    ]
    detail = f"; non-finite components: {', '.join(nonfinite_items)}" if nonfinite_items else ""
    raise FloatingPointError(f"Non-finite {phase} loss: {loss.detach().cpu().item()}{detail}")


def _accumulate_losses(
    loss_totals: Dict[str, float],
    loss_items: Dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for name, value in loss_items.items():
        if not torch.is_tensor(value):
            continue
        loss_totals[name] = loss_totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size


def _summarize_losses(
    total_loss: float,
    total_images: int,
    loss_totals: Dict[str, float],
) -> Dict[str, Any]:
    if total_images == 0:
        return {"loss": None, "num_images": 0, "loss_items": {}}
    return {
        "loss": total_loss / total_images,
        "num_images": total_images,
        "loss_items": {
            name: value / total_images
            for name, value in sorted(loss_totals.items())
        },
    }


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _write_yaml(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.safe_dump(value, file, sort_keys=False)


def _read_yaml(path: str | Path) -> Any:
    with open(path) as file:
        return yaml.safe_load(file)


def _best_epoch_from_history(history: List[Dict[str, Any]]) -> Optional[int]:
    best_epoch = None
    for entry in history:
        if entry.get("is_best"):
            best_epoch = entry.get("epoch", best_epoch)
    return best_epoch


def _epochs_since_best(history: List[Dict[str, Any]], best_epoch: Optional[int]) -> int:
    """Scored epochs after ``best_epoch`` — the resumed early-stopping counter."""
    if best_epoch is None:
        return 0
    return sum(
        1
        for entry in history
        if (entry.get("val_map") or entry.get("val")) is not None
        and entry.get("epoch", 0) > best_epoch
    )


def _val_loss_at_epoch(history: List[Dict[str, Any]], epoch: Optional[int]) -> Optional[float]:
    if epoch is None:
        return None
    for entry in history:
        if entry.get("epoch") == epoch and entry.get("val"):
            return entry["val"].get("loss")
    return None


def _compact_map_summary(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None
    keys = ("map50", "map50_95", "precision", "recall", "f1", "f1_confidence", "num_targets")
    return {key: metrics.get(key) for key in keys}


def _to_builtin(value: Any) -> Any:
    if is_dataclass(value):
        return _to_builtin(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _target_to_cpu(target: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: _cpu_value(value)
        for key, value in target.items()
    }


def _cpu_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Friendy Chachkalica models from YAML config")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs with a complete result.yaml and continue any run with a last.pt checkpoint from its next epoch",
    )
    args = parser.parse_args()
    results = train_from_config(args.config, resume=args.resume)
    print(yaml.safe_dump(_to_builtin(results), sort_keys=False))


if __name__ == "__main__":
    main()
