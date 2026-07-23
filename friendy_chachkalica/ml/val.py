import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

try:
    from ..config import DatasetConfig, ExperimentConfig, ExperimentRun, build_experiment_runs, load_config
    from ..data import build_eval_dataloader
    from ..device import resolve_device
    from ..registry import build_model
    from .train import build_run_pipeline, predict_dataset, resolve_operating_nms_threshold
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import DatasetConfig, ExperimentConfig, ExperimentRun, build_experiment_runs, load_config
    from data import build_eval_dataloader
    from device import resolve_device
    from registry import build_model
    from ml.train import build_run_pipeline, predict_dataset, resolve_operating_nms_threshold


def val_from_config(
    config_path: str | Path,
    split: str = "test",
    checkpoint: str = "best",
) -> List[Dict[str, Any]]:
    print(f"[val] Starting from config: {config_path} split={split} checkpoint={checkpoint}")
    config = load_config(config_path)
    return val_experiment(config, split=split, checkpoint=checkpoint)


def val_experiment(
    config: ExperimentConfig,
    split: str = "test",
    checkpoint: str = "best",
) -> List[Dict[str, Any]]:
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    if checkpoint not in {"best", "last"}:
        raise ValueError("checkpoint must be 'best' or 'last'")

    device = resolve_device(config.training.device)
    print(f"[val] Using device: {device}")
    loaders: Dict[DatasetConfig, Any] = {}
    results = []

    runs = build_experiment_runs(config)
    print(f"[val] Evaluating {len(runs)} run(s) split={split} checkpoint={checkpoint}")
    for run in runs:
        dataset_config = _run_eval_dataset(run, split)
        if dataset_config is None:
            print(f"[val] Skipping run {run.name}: no {split} dataset configured")
            continue

        loader = _get_eval_loader(config, dataset_config, loaders)
        try:
            result = evaluate_run(
                config=config,
                run=run,
                dataset_config=dataset_config,
                loader=loader,
                device=device,
                split=split,
                checkpoint=checkpoint,
            )
        except Exception as exc:  # noqa: BLE001 - one run's failure must not sink the rest
            # Mirrors train.py's per-run guard: a model that failed to train (no
            # checkpoint) or errors during eval is recorded and skipped so the
            # runs that DID succeed still get their metrics and the run completes.
            print(f"[val] Run {run.name} FAILED: {exc}")
            results.append({"run_index": run.index, "run_name": run.name, "error": str(exc)})
            continue
        results.append(result)

    output_path = config.output_dir / f"{split}_results.yaml"
    _write_yaml(output_path, _to_builtin(results))
    print(f"[val] Wrote {split} results: {output_path}")
    return results


@torch.no_grad()
def evaluate_run(
    config: ExperimentConfig,
    run: ExperimentRun,
    dataset_config: DatasetConfig,
    loader: Any,
    device: torch.device,
    split: str,
    checkpoint: str,
) -> Dict[str, Any]:
    run_dir = config.output_dir / run.name
    checkpoint_path = _checkpoint_path(run_dir, checkpoint)
    print(
        f"[val] Evaluating run={run.name} model={run.model.name} "
        f"dataset={dataset_config.name} checkpoint={checkpoint_path}"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for run {run.name}: {checkpoint_path}")

    print(f"[val] Building model adapter: {run.model.name}")
    adapter = build_model(
        run.model.name,
        num_classes=run.model.num_classes,
        **run.model.params,
    )
    adapter.to(device)

    print(f"[val] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    adapter.model.load_state_dict(state["model_state_dict"])

    # Route inference through the experiment's pipeline (tiling / crop) when set,
    # so val/test metrics match how the model will be served. None otherwise.
    eval_pipeline = build_run_pipeline(adapter, config, device)
    if eval_pipeline is not None:
        print(f"[val] Inference routed through chachak pipeline: {config.pipeline.name}")

    prediction_path = run_dir / f"{split}_predictions.pt"
    metrics = predict_dataset(
        adapter,
        loader,
        device,
        prediction_path,
        config,
        num_classes=run.model.num_classes,
        prediction_classes=run.train_dataset.classes,
        target_classes=dataset_config.classes,
        eval_classes=dataset_config.classes,
        operating_nms_threshold=resolve_operating_nms_threshold(config, run.model),
        pipeline=eval_pipeline,
    )

    result = {
        "run_index": run.index,
        "run_name": run.name,
        "model": run.model.name,
        "model_num_classes": run.model.num_classes,
        "train_dataset": run.train_dataset.name,
        "eval_dataset": dataset_config.name,
        "eval_dataset_images": str(dataset_config.images),
        "eval_dataset_labels": str(dataset_config.labels),
        "eval_dataset_role": dataset_config.role,
        "checkpoint": str(checkpoint_path),
        "predictions": str(prediction_path),
        "metrics": metrics,
    }
    _write_yaml(run_dir / f"{split}_result.yaml", _to_builtin(result))
    print(f"[val] Run {run.name} complete: result={run_dir / f'{split}_result.yaml'}")
    return result


def _run_eval_dataset(run: ExperimentRun, split: str) -> Optional[DatasetConfig]:
    if split == "val":
        return run.val_dataset
    return run.test_dataset


def _get_eval_loader(
    config: ExperimentConfig,
    dataset_config: DatasetConfig,
    cache: Dict[tuple, Any],
) -> Any:
    cache_key = _dataset_cache_key(dataset_config)
    loader = cache.get(cache_key)
    if loader is None:
        print(f"[val] Creating eval loader for dataset={dataset_config.name} role={dataset_config.role}")
        loader = build_eval_dataloader(dataset_config, config)
        cache[cache_key] = loader
    else:
        print(f"[val] Reusing eval loader for dataset={dataset_config.name} role={dataset_config.role}")
    return loader


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    return run_dir / f"{checkpoint}.pt"


def _dataset_cache_key(dataset_config: DatasetConfig) -> tuple:
    return (
        dataset_config.name,
        str(dataset_config.images),
        str(dataset_config.labels),
        dataset_config.role,
    )


def _write_yaml(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.safe_dump(value, file, sort_keys=False)


def _to_builtin(value: Any) -> Any:
    if is_dataclass(value):
        return _to_builtin(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Friendy Chachkalica runs")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    args = parser.parse_args()
    results = val_from_config(args.config, split=args.split, checkpoint=args.checkpoint)
    print(yaml.safe_dump(_to_builtin(results), sort_keys=False))


if __name__ == "__main__":
    main()
