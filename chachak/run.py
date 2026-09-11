"""CLI entrypoint for chachak pipelines.

    python chachak/run.py request.yaml

Loads a pipeline config, loads the trained model (and a detector when the
pipeline needs one), builds the eval dataloader with Friendy's loader, runs the
pipeline, and writes ``predictions.pt`` + ``result.yaml`` in the same shape as
``friendy_chachkalica/ml/eval_checkpoint.py`` so results drop into the eval flow.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

try:
    from ._friendy import (
        EVAL_HARD_IMAGES_FRACTION,
        DatasetConfig,
        EvaluationConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
        _to_builtin,
        _write_hard_images,
        _write_match_table,
        _write_yaml,
        build_eval_dataloader,
        evaluate_detection,
        remap_raw_predictions_to_eval_classes,
        resolve_device,
    )
    from .boxes import merge_predictions
    # _needs_detector is re-exported (not defined here) so bundle_export can
    # reach it without importing this module's torch — see chachak/config.py.
    from .config import _needs_detector, load_pipeline_config  # noqa: F401
    from .detector import load_detector
    from .infer import load_checkpoint_adapter
    from .pipeline import _frame_size
    from .registry import build_pipeline
except ImportError:  # run as a flat script
    from _friendy import (
        EVAL_HARD_IMAGES_FRACTION,
        DatasetConfig,
        EvaluationConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
        _to_builtin,
        _write_hard_images,
        _write_match_table,
        _write_yaml,
        build_eval_dataloader,
        evaluate_detection,
        remap_raw_predictions_to_eval_classes,
        resolve_device,
    )
    from boxes import merge_predictions
    from config import _needs_detector, load_pipeline_config  # noqa: F401
    from detector import load_detector
    from infer import load_checkpoint_adapter
    from pipeline import _frame_size
    from registry import build_pipeline


def build_pipeline_runtime(config, device):
    """Load the model (and detector when needed) and build the pipeline.

    Returns ``(pipeline, info)``. Shared by :func:`run_pipeline` (the batch eval
    path) and the trainer service's synchronous single-image predict endpoint,
    which reuses the same adapter/detector/pipeline construction but skips the
    dataloader.
    """
    model_adapter, info = load_checkpoint_adapter(config.model_checkpoint, device)

    detector = None
    if _needs_detector(config):
        detector = load_detector(
            config.detector.checkpoint,
            device,
            person_class_name=config.detector.person_class_name,
            person_class_id=config.detector.person_class_id,
            score_threshold=config.detector.score_threshold,
            batch_size=config.detector.batch_size,
        )

    pipeline = build_pipeline(config, model_adapter, device, detector)
    return pipeline, info


def run_pipeline(config) -> Dict[str, Any]:
    if config.extra_checkpoints:
        return run_combined_pipeline(config)

    device = resolve_device(config.device)

    pipeline, info = build_pipeline_runtime(config, device)
    num_classes = info["num_classes"]
    detector = pipeline.detector

    # Build the eval dataloader through Friendy so datasets/labels load identically
    # to eval_checkpoint.py. Frames-per-batch is infer_batch_size; the model only
    # ever sees infer_batch_size tiles/crops at a time inside the pipeline.
    dataset_config = DatasetConfig(
        name=f"{config.name}-data",
        images=config.images,
        labels=config.labels,
        classes=config.classes,
        role="test",
    )
    experiment = ExperimentConfig(
        name=config.name,
        train_datasets=[dataset_config],
        models=[ModelConfig(name=info["model_name"], num_classes=num_classes)],
        output_dir=config.output_dir,
        test_dataset=dataset_config,
        training=TrainingConfig(
            batch_size=config.infer_batch_size,
            num_workers=config.num_workers,
            device=config.device,
        ),
        evaluation=EvaluationConfig(
            batch_size=config.infer_batch_size,
            num_workers=config.num_workers,
            score_threshold=config.score_threshold,
            map_score_threshold=config.map_score_threshold,
            iou_thresholds=config.iou_thresholds,
        ),
    )
    loader = build_eval_dataloader(dataset_config, experiment)

    # Evaluate exactly like training/eval_checkpoint: predictions are remapped by
    # NAME onto the eval class space, keeping only the classes both the model and
    # the dataset share. Falling back to the eval class list when the checkpoint
    # doesn't record its own training classes would silently mislabel every
    # prediction whenever the model's class order differs — so we refuse instead.
    train_classes = info["train_classes"]
    if not train_classes:
        raise ValueError(
            f"Checkpoint {config.model_checkpoint} does not record its training class "
            "names (train_dataset.classes), so predictions cannot be remapped by name "
            "onto the eval classes. Re-train (or re-save the checkpoint) with class names."
        )
    result = pipeline.run(
        loader,
        config.output_dir,
        num_classes=num_classes,
        prediction_classes=train_classes,
        target_classes=config.classes,
        eval_classes=config.classes,
        compute_metrics=config.labels is not None,
    )

    output = {
        "pipeline": config.pipeline,
        "name": config.name,
        "model_checkpoint": str(config.model_checkpoint),
        "detector_checkpoint": (
            str(config.detector.checkpoint) if detector is not None else None
        ),
        "images": str(config.images),
        "labels": str(config.labels) if config.labels is not None else None,
        "predictions": str(result["prediction_path"]),
        "metrics": result["metrics"],
    }
    result_path = Path(config.output_dir) / "result.yaml"
    _write_yaml(result_path, _to_builtin(output))
    print(f"[chachak] Wrote result: {result_path}")
    return output


def run_combined_pipeline(config) -> Dict[str, Any]:
    """Run 2+ checkpoints through the same pipeline, merging predictions per image.

    Mirrors :func:`run_pipeline`'s single-model flow, but builds one pipeline
    per checkpoint (``config.model_checkpoint`` plus ``config.extra_checkpoints``),
    sharing a single detector across all of them when the pipeline needs one.
    Each model's raw predictions are remapped onto ``config.classes`` by name
    (:func:`remap_raw_predictions_to_eval_classes`) — combining is only valid for
    checkpoints whose class lists are disjoint by name, enforced upstream by the
    Django admin (``training.services.combine.overlapping_class_names``) — then
    unioned per frame with :func:`merge_predictions` (the same primitive
    :class:`pipeline.ChainedPipeline` uses) before scoring once. Writes
    ``predictions.pt`` + ``result.yaml`` shaped identically to
    :func:`run_pipeline`, with predictions already in the eval class space.
    """
    device = resolve_device(config.device)
    checkpoints = [config.model_checkpoint, *config.extra_checkpoints]

    detector = None
    runtimes = []  # (pipeline, train_classes) per checkpoint
    for checkpoint in checkpoints:
        model_adapter, info = load_checkpoint_adapter(checkpoint, device)
        if _needs_detector(config) and detector is None:
            detector = load_detector(
                config.detector.checkpoint,
                device,
                person_class_name=config.detector.person_class_name,
                person_class_id=config.detector.person_class_id,
                score_threshold=config.detector.score_threshold,
                batch_size=config.detector.batch_size,
            )
        pipeline = build_pipeline(config, model_adapter, device, detector)
        train_classes = info["train_classes"]
        if not train_classes:
            raise ValueError(
                f"Checkpoint {checkpoint} does not record its training class "
                "names (train_dataset.classes), so predictions cannot be remapped by name "
                "onto the eval classes. Re-train (or re-save the checkpoint) with class names."
            )
        runtimes.append((pipeline, train_classes))

    dataset_config = DatasetConfig(
        name=f"{config.name}-data",
        images=config.images,
        labels=config.labels,
        classes=config.classes,
        role="test",
    )
    experiment = ExperimentConfig(
        name=config.name,
        train_datasets=[dataset_config],
        models=[ModelConfig(name="combined", num_classes=len(config.classes))],
        output_dir=config.output_dir,
        test_dataset=dataset_config,
        training=TrainingConfig(
            batch_size=config.infer_batch_size,
            num_workers=config.num_workers,
            device=config.device,
        ),
        evaluation=EvaluationConfig(
            batch_size=config.infer_batch_size,
            num_workers=config.num_workers,
            score_threshold=config.score_threshold,
            map_score_threshold=config.map_score_threshold,
            iou_thresholds=config.iou_thresholds,
        ),
    )
    loader = build_eval_dataloader(dataset_config, experiment)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    all_predictions = []
    all_targets = []
    started = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        frame_sizes = [_frame_size(image) for image in images]

        per_model_frame_preds = []
        for pipe, train_classes in runtimes:
            raw_preds = pipe.process_batch(images, targets)
            per_model_frame_preds.append([
                remap_raw_predictions_to_eval_classes(
                    prediction.detach().cpu(), train_classes, config.classes
                )
                for prediction in raw_preds
            ])

        merged_preds = [
            merge_predictions(
                [per_model_frame_preds[m][i] for m in range(len(runtimes))],
                *frame_sizes[i],
                config.merge_nms_iou,
            )
            for i in range(len(images))
        ]

        for target, prediction in zip(targets, merged_preds):
            all_predictions.append(prediction)
            all_targets.append(target)
            orig_size = target.get("orig_size")
            if torch.is_tensor(orig_size):
                orig_size = orig_size.detach().cpu().tolist()
            records.append(
                {
                    "image_path": target.get("image_path"),
                    "label_path": target.get("label_path"),
                    "orig_size": orig_size,
                    "predictions": prediction,
                }
            )
        print(
            f"[chachak] combined: batch {batch_index} frames={len(images)} "
            f"models={len(runtimes)} total={len(records)}"
        )

    prediction_path = output_dir / "predictions.pt"
    torch.save(records, prediction_path)
    print(f"[chachak] Saved predictions: {prediction_path} records={len(records)}")

    if config.labels is None:
        metrics = {"prediction_only": True}
    else:
        metrics = evaluate_detection(
        all_predictions,
        all_targets,
        iou_thresholds=config.iou_thresholds,
        score_threshold=config.score_threshold,
        map_score_threshold=config.map_score_threshold,
        num_classes=len(config.classes),
        # Merged predictions are already remapped into config.classes, so this
        # remap step inside evaluate_detection is an identity mapping.
        prediction_classes=config.classes,
        target_classes=config.classes,
        eval_classes=config.classes,
        )
    metrics["eval_seconds"] = round(time.perf_counter() - started, 3)
    print(
        f"[chachak] combined metrics: map50={metrics.get('map50')} "
        f"map50_95={metrics.get('map50_95')} precision={metrics.get('precision')} "
        f"recall={metrics.get('recall')}"
    )

    if config.labels is not None:
        _write_hard_images(
            prediction_path,
            all_predictions,
            all_targets,
            records,
            config=None,
            prediction_classes=config.classes,
            target_classes=config.classes,
            eval_classes=config.classes,
            score_threshold=config.score_threshold,
            top_k_fraction=EVAL_HARD_IMAGES_FRACTION,
        )
        _write_match_table(
            prediction_path,
            all_predictions,
            all_targets,
            records,
            config=None,
            num_classes=len(config.classes),
            prediction_classes=config.classes,
            target_classes=config.classes,
            eval_classes=config.classes,
            iou_thresholds=config.iou_thresholds,
            score_threshold=config.score_threshold,
            map_score_threshold=config.map_score_threshold,
        )

    output = {
        "pipeline": config.pipeline,
        "name": config.name,
        "model_checkpoint": [str(p) for p in checkpoints],
        "detector_checkpoint": str(config.detector.checkpoint) if detector is not None else None,
        "images": str(config.images),
        "labels": str(config.labels) if config.labels is not None else None,
        "predictions": str(prediction_path),
        "metrics": metrics,
    }
    result_path = output_dir / "result.yaml"
    _write_yaml(result_path, _to_builtin(output))
    print(f"[chachak] Wrote result: {result_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a chachak inference/eval pipeline")
    parser.add_argument("request", help="Path to a pipeline request YAML")
    args = parser.parse_args()
    config = load_pipeline_config(args.request)
    output = run_pipeline(config)
    print(yaml.safe_dump(_to_builtin(output), sort_keys=False))


if __name__ == "__main__":
    main()
