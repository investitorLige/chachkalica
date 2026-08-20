"""Standalone evaluation of a single trained checkpoint on an arbitrary dataset.

``val.py`` evaluates checkpoints *within* an experiment's run-directory layout.
This module instead evaluates one catalogued checkpoint (a registered model)
against a dataset chosen after the fact — the "test a trained model" path. The
checkpoint already carries everything needed to rebuild the model
(``model_name``, ``model_config``, and the training ``classes``), so the caller
only supplies the eval dataset (images, labels, classes) and where to write.

Driven by a small request YAML so the trainer service can launch it as a
subprocess, mirroring ``run.py``:

    .venv/bin/python eval_checkpoint.py request.yaml

Request YAML fields: checkpoint_path, images, labels, classes (list/mapping),
output_dir, and optional name, score_threshold, map_score_threshold, nms_threshold,
iou_thresholds, batch_size, num_workers, device.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml

try:
    from ..config import (
        DatasetConfig,
        EvaluationConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
    )
    from ..data import build_eval_dataloader
    from ..device import resolve_device
    from ..metrics import (
        EVAL_HARD_IMAGES_FRACTION,
        evaluate_detection,
        remap_raw_predictions_to_eval_classes,
    )
    from ..registry import build_model
    from .train import (
        _apply_eval_nms,
        _cpu_value,
        _move_batch_to_device,
        _predict_with_config,
        _require_prediction_batch,
        _target_to_cpu,
        _write_hard_images,
        predict_dataset,
        resolve_operating_nms_threshold,
    )
    from .val import _to_builtin, _write_yaml
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import (
        DatasetConfig,
        EvaluationConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
    )
    from data import build_eval_dataloader
    from device import resolve_device
    from metrics import (
        EVAL_HARD_IMAGES_FRACTION,
        evaluate_detection,
        remap_raw_predictions_to_eval_classes,
    )
    from registry import build_model
    from ml.train import (
        _apply_eval_nms,
        _cpu_value,
        _move_batch_to_device,
        _predict_with_config,
        _require_prediction_batch,
        _target_to_cpu,
        _write_hard_images,
        predict_dataset,
        resolve_operating_nms_threshold,
    )
    from ml.val import _to_builtin, _write_yaml


def _as_class_map(classes: Union[Dict, List]) -> Dict[int, str]:
    if isinstance(classes, dict):
        return {int(k): str(v) for k, v in classes.items()}
    return {index: str(name) for index, name in enumerate(classes)}


def eval_checkpoint(
    checkpoint_path: Union[str, Path],
    images: Union[str, Path],
    labels: Optional[Union[str, Path]],
    classes: Union[Dict, List],
    output_dir: Union[str, Path],
    *,
    name: str = "eval",
    score_threshold: float = 0.001,
    map_score_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    operating_nms_threshold: Optional[float] = None,
    iou_thresholds: Optional[List[float]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    device: str = "auto",
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    print(f"[eval] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model_name = state["model_name"]
    model_config = state.get("model_config", {}) or {}
    num_classes = model_config.get("num_classes")
    params = dict(model_config.get("params", {}) or {})
    train_classes_raw = (state.get("train_dataset") or {}).get("classes")
    if not train_classes_raw:
        # Falling back to the eval class list would silently mislabel every
        # prediction whenever the model's training class order differs from it.
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not record its training class names "
            "(train_dataset.classes), so predictions cannot be remapped by name onto "
            "the eval classes. Re-train (or re-save the checkpoint) with class names."
        )
    train_classes = _as_class_map(train_classes_raw)
    eval_classes = _as_class_map(classes)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = resolve_device(device)
    print(f"[eval] Building model adapter: {model_name} num_classes={num_classes} device={dev}")
    adapter = build_model(model_name, num_classes=num_classes, **params)
    adapter.to(dev)
    adapter.model.load_state_dict(state["model_state_dict"])

    dataset_config = DatasetConfig(
        name=f"{name}-data", images=Path(images), labels=Path(labels) if labels else None,
        classes=eval_classes, role="test",
    )
    evaluation = EvaluationConfig(
        batch_size=batch_size, num_workers=num_workers,
        score_threshold=score_threshold,
        map_score_threshold=map_score_threshold,
        nms_threshold=nms_threshold,
        operating_nms_threshold=operating_nms_threshold,
        iou_thresholds=iou_thresholds or EvaluationConfig().iou_thresholds,
    )
    training = TrainingConfig(batch_size=batch_size, num_workers=num_workers, device=device)
    config = ExperimentConfig(
        name=name, train_datasets=[dataset_config],
        models=[ModelConfig(name=model_name, num_classes=num_classes, params=params)],
        output_dir=output_dir, test_dataset=dataset_config,
        training=training, evaluation=evaluation,
    )

    loader = build_eval_dataloader(dataset_config, config)
    prediction_path = output_dir / "eval_predictions.pt"
    metrics = predict_dataset(
        adapter, loader, dev, prediction_path, config,
        num_classes=num_classes,
        prediction_classes=train_classes,
        target_classes=eval_classes,
        eval_classes=eval_classes,
        operating_nms_threshold=resolve_operating_nms_threshold(config, config.models[0]),
        compute_metrics=labels is not None,
        hard_images_top_k_fraction=EVAL_HARD_IMAGES_FRACTION,
    )

    result = {
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "num_classes": num_classes,
        "eval_dataset": dataset_config.name,
        "images": str(images),
        "labels": str(labels) if labels is not None else None,
        "metrics": metrics,
    }
    result_path = output_dir / "eval_result.yaml"
    _write_yaml(result_path, _to_builtin(result))
    print(f"[eval] Wrote eval result: {result_path}")
    return result


def _load_checkpoint_adapter_for_eval(checkpoint_path: Union[str, Path], device):
    """Load one checkpoint's model adapter + its training class map.

    Shared setup step for :func:`eval_combined_checkpoints`; mirrors the single-
    checkpoint loading done inline at the top of :func:`eval_checkpoint`.
    """
    checkpoint_path = Path(checkpoint_path)
    print(f"[eval] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model_name = state["model_name"]
    model_config = state.get("model_config", {}) or {}
    num_classes = model_config.get("num_classes")
    params = dict(model_config.get("params", {}) or {})
    train_classes_raw = (state.get("train_dataset") or {}).get("classes")
    if not train_classes_raw:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not record its training class names "
            "(train_dataset.classes), so predictions cannot be remapped by name onto "
            "the eval classes. Re-train (or re-save the checkpoint) with class names."
        )
    adapter = build_model(model_name, num_classes=num_classes, **params)
    adapter.to(device)
    adapter.model.load_state_dict(state["model_state_dict"])
    adapter.eval()
    return adapter, _as_class_map(train_classes_raw)


def eval_combined_checkpoints(
    checkpoint_paths: List[Union[str, Path]],
    images: Union[str, Path],
    labels: Optional[Union[str, Path]],
    classes: Union[Dict, List],
    output_dir: Union[str, Path],
    *,
    name: str = "eval",
    score_threshold: float = 0.001,
    map_score_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    operating_nms_threshold: Optional[float] = None,
    iou_thresholds: Optional[List[float]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    device: str = "auto",
) -> Dict[str, Any]:
    """Evaluate 2+ checkpoints combined: merge their predictions into one result.

    Each checkpoint's raw predictions are remapped by class name onto
    ``classes`` (the eval dataset's own class space) before merging, so
    combining is only correct for checkpoints whose class lists are disjoint by
    name — the Django admin enforces that before enqueuing (see
    ``training.services.combine.overlapping_class_names``). Iterates the
    dataloader once, running every checkpoint per batch, so the merge never
    needs a second pass or relies on loader ordering staying stable across
    passes. Writes ``eval_predictions.pt`` + ``eval_result.yaml`` shaped
    identically to :func:`eval_checkpoint`, with the merged predictions already
    in the eval class space (there is no single owning train class space for a
    combined run — see ``promote_labels.promote_labels``'s ``prediction_classes``).
    """
    if len(checkpoint_paths) < 2:
        raise ValueError("eval_combined_checkpoints needs at least 2 checkpoints.")

    eval_classes = _as_class_map(classes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)

    adapters = []
    train_classes_list = []
    for checkpoint_path in checkpoint_paths:
        adapter, train_classes = _load_checkpoint_adapter_for_eval(checkpoint_path, dev)
        adapters.append(adapter)
        train_classes_list.append(train_classes)

    dataset_config = DatasetConfig(
        name=f"{name}-data", images=Path(images), labels=Path(labels) if labels else None,
        classes=eval_classes, role="test",
    )
    evaluation = EvaluationConfig(
        batch_size=batch_size, num_workers=num_workers,
        score_threshold=score_threshold,
        map_score_threshold=map_score_threshold,
        nms_threshold=nms_threshold,
        operating_nms_threshold=operating_nms_threshold,
        iou_thresholds=iou_thresholds or EvaluationConfig().iou_thresholds,
    )
    training = TrainingConfig(batch_size=batch_size, num_workers=num_workers, device=device)
    config = ExperimentConfig(
        name=name, train_datasets=[dataset_config],
        models=[ModelConfig(name="combined", num_classes=len(eval_classes))],
        output_dir=output_dir, test_dataset=dataset_config,
        training=training, evaluation=evaluation,
    )
    loader = build_eval_dataloader(dataset_config, config)

    records = []
    all_predictions = []
    all_targets = []
    for batch_index, (batch_images, targets) in enumerate(loader, start=1):
        batch_images, targets = _move_batch_to_device(batch_images, targets, dev)

        per_model_predictions = []
        for adapter, train_classes in zip(adapters, train_classes_list):
            predictions = _predict_with_config(adapter, batch_images, config, targets=targets)
            _require_prediction_batch(batch_images, targets, predictions, phase="evaluation")
            predictions = _apply_eval_nms(predictions, config)
            per_model_predictions.append([
                remap_raw_predictions_to_eval_classes(
                    prediction.detach().cpu(), train_classes, eval_classes
                )
                for prediction in predictions
            ])
        print(
            f"[eval] Predicted batch {batch_index}: images={len(batch_images)} "
            f"models={len(adapters)}"
        )

        for image_index, target in enumerate(targets):
            merged = torch.cat(
                [model_predictions[image_index] for model_predictions in per_model_predictions],
                dim=0,
            )
            target_cpu = _target_to_cpu(target)
            all_predictions.append(merged)
            all_targets.append(target_cpu)
            records.append(
                {
                    "image_path": target.get("image_path"),
                    "label_path": target.get("label_path"),
                    "orig_size": _cpu_value(target.get("orig_size")),
                    "predictions": merged,
                }
            )

    prediction_path = output_dir / "eval_predictions.pt"
    torch.save(records, prediction_path)
    print(f"[eval] Saved predictions: {prediction_path} records={len(records)}")

    metrics = (evaluate_detection(
        all_predictions,
        all_targets,
        iou_thresholds=config.evaluation.iou_thresholds,
        score_threshold=config.evaluation.score_threshold,
        map_score_threshold=config.evaluation.map_score_threshold,
        num_classes=len(eval_classes),
        # Merged predictions are already remapped into eval_classes, so this
        # remap step inside evaluate_detection is an identity mapping.
        prediction_classes=eval_classes,
        target_classes=eval_classes,
        eval_classes=eval_classes,
        operating_nms_threshold=operating_nms_threshold,
    ) if labels is not None else {"prediction_only": True})

    if labels is not None:
        _write_hard_images(
            prediction_path,
            all_predictions,
            all_targets,
            records,
            config=config,
            prediction_classes=eval_classes,
            target_classes=eval_classes,
            eval_classes=eval_classes,
            operating_nms_threshold=operating_nms_threshold,
            top_k_fraction=EVAL_HARD_IMAGES_FRACTION,
        )

    result = {
        "checkpoint": [str(p) for p in checkpoint_paths],
        "model": "combined",
        "num_classes": len(eval_classes),
        "eval_dataset": dataset_config.name,
        "images": str(images),
        "labels": str(labels) if labels is not None else None,
        "metrics": metrics,
    }
    result_path = output_dir / "eval_result.yaml"
    _write_yaml(result_path, _to_builtin(result))
    print(f"[eval] Wrote eval result: {result_path}")
    return result


def eval_from_request(request_path: Union[str, Path]) -> Dict[str, Any]:
    request = yaml.safe_load(Path(request_path).read_text())
    extra_checkpoints = request.get("extra_checkpoints")
    if extra_checkpoints:
        return eval_combined_checkpoints(
            checkpoint_paths=[request["checkpoint_path"], *extra_checkpoints],
            images=request["images"],
            labels=request.get("labels"),
            classes=request["classes"],
            output_dir=request["output_dir"],
            name=request.get("name", "eval"),
            score_threshold=request.get("score_threshold", 0.001),
            map_score_threshold=request.get("map_score_threshold"),
            nms_threshold=request.get("nms_threshold"),
            operating_nms_threshold=request.get("operating_nms_threshold"),
            iou_thresholds=request.get("iou_thresholds"),
            batch_size=request.get("batch_size", 4),
            num_workers=request.get("num_workers", 4),
            device=request.get("device", "auto"),
        )
    return eval_checkpoint(
        checkpoint_path=request["checkpoint_path"],
        images=request["images"],
        labels=request.get("labels"),
        classes=request["classes"],
        output_dir=request["output_dir"],
        name=request.get("name", "eval"),
        score_threshold=request.get("score_threshold", 0.001),
        map_score_threshold=request.get("map_score_threshold"),
        nms_threshold=request.get("nms_threshold"),
        operating_nms_threshold=request.get("operating_nms_threshold"),
        iou_thresholds=request.get("iou_thresholds"),
        batch_size=request.get("batch_size", 4),
        num_workers=request.get("num_workers", 4),
        device=request.get("device", "auto"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a single checkpoint on a dataset")
    parser.add_argument("request", help="Path to an eval request YAML")
    args = parser.parse_args()
    result = eval_from_request(args.request)
    print(yaml.safe_dump(_to_builtin(result), sort_keys=False))


if __name__ == "__main__":
    main()
