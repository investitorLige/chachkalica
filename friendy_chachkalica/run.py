import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

try:
    from .config import ExperimentConfig, load_config
    from .export import export_universal_csv
    from .ml.train import train_experiment
    from .ml.val import val_experiment
except ImportError:
    from config import ExperimentConfig, load_config
    from export import export_universal_csv
    from ml.train import train_experiment
    from ml.val import val_experiment


def run_from_config(config_path: str | Path, resume: bool = False) -> Dict[str, Any]:
    print(f"[run] Starting full pipeline from config: {config_path}")
    config = load_config(config_path)
    return run_experiment(config, resume=resume)


def run_experiment(config: ExperimentConfig, resume: bool = False) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] Output directory: {config.output_dir}")

    print(f"[run] Training phase start (resume={resume})")
    train_results = train_experiment(config, evaluate_after_train=False, resume=resume)
    print(f"[run] Training phase done: runs={len(train_results)}")
    train_results_path = config.output_dir / "results.yaml"

    summary = {
        "output_dir": str(config.output_dir),
        "train_results": str(train_results_path),
        "metrics_csv": None,
        "num_train_runs": len(train_results),
        "val_results": None,
        "test_results": None,
    }

    checkpoint = _default_eval_checkpoint(config)
    print(f"[run] Evaluation checkpoint policy: {checkpoint}")

    if config.val_dataset is not None:
        print("[run] Validation phase start")
        val_results = val_experiment(config, split="val", checkpoint=checkpoint)
        val_results_path = config.output_dir / "val_results.yaml"
        summary.update(
            {
                "val_results": str(val_results_path),
                "num_val_runs": len(val_results),
            }
        )

    if config.test_dataset is not None:
        print("[run] Test phase start")
        test_results = val_experiment(config, split="test", checkpoint=checkpoint)
        test_results_path = config.output_dir / "test_results.yaml"
        summary.update(
            {
                "test_results": str(test_results_path),
                "num_test_runs": len(test_results),
            }
        )

    print("[run] Export phase start")
    metrics_csv_path = export_universal_csv(
        train_results_path,
        config.output_dir / "metrics.csv",
        val_results_path=config.output_dir / "val_results.yaml" if config.val_dataset is not None else None,
        test_results_path=config.output_dir / "test_results.yaml" if config.test_dataset is not None else None,
    )

    summary["metrics_csv"] = str(metrics_csv_path)
    summary["checkpoint"] = checkpoint
    summary_path = config.output_dir / "run_summary.yaml"
    _write_yaml(summary_path, _to_builtin(summary))
    summary["summary"] = str(summary_path)
    print(f"[run] Pipeline complete: summary={summary_path} metrics_csv={metrics_csv_path}")
    return summary


def _default_eval_checkpoint(config: ExperimentConfig) -> str:
    return "best" if config.val_dataset is not None else "last"


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
    parser = argparse.ArgumentParser(description="Run a full Friendy Chachkalica experiment")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs and continue interrupted runs from their last checkpoint",
    )
    args = parser.parse_args()
    summary = run_from_config(args.config, resume=args.resume)
    print(yaml.safe_dump(_to_builtin(summary), sort_keys=False))


if __name__ == "__main__":
    main()
