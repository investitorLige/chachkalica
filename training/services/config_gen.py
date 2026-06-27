"""Turn an :class:`~training.models.Experiment` into friendy_mercury YAML.

friendy_mercury (``/home/luka/workspace/friendy_mercury``) is config-driven: it
reads an experiment YAML whose ``datasets.{train(list),val,test}`` entries each
point at an ``images`` dir and a ``labels`` dir with a ``classes`` list, plus
``models``/``training``/``evaluation`` blocks. We generate that YAML with
**absolute** paths so it resolves regardless of where the trainer runs, reusing
the same on-disk resolvers the fleet annotation side already uses.
"""

from pathlib import Path

import yaml
from django.conf import settings

from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services.paths import source_root, target_root
from training.models import Experiment, ExperimentDataset, ExperimentModel, TrainingSettings


def _resolve(path: str) -> Path:
    """Absolute path: as-is if absolute, else relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else Path(settings.BASE_DIR) / p


def label_dir(exp_dataset: ExperimentDataset) -> Path:
    """Resolve the labels directory feeding this dataset, per its label source."""
    if exp_dataset.label_source == ExperimentDataset.SOURCE:
        return datasets_svc.labels_source_dir(exp_dataset.dataset)
    if exp_dataset.label_source == ExperimentDataset.ANNOTATOR:
        if exp_dataset.annotator is None:
            raise ValueError(
                f"{exp_dataset.dataset.name}: annotator output selected but no annotator set."
            )
        return target_root() / exp_dataset.dataset.name / exp_dataset.annotator.username
    if exp_dataset.label_source == ExperimentDataset.EXPLICIT:
        if not exp_dataset.explicit_labels_path.strip():
            raise ValueError(
                f"{exp_dataset.dataset.name}: explicit label path selected but empty."
            )
        return _resolve(exp_dataset.explicit_labels_path.strip())
    raise ValueError(f"Unknown label source {exp_dataset.label_source!r}")


def dataset_entry(exp_dataset: ExperimentDataset) -> dict:
    """Build one YAML dataset entry: {name, images, labels, classes}."""
    name = exp_dataset.dataset.name
    dataset_dir = source_root() / name
    images_dir = lsapi.image_source_dir(dataset_dir)
    classes, _tools = lsapi.parse_classes_file(dataset_dir / "classes.txt")
    return {
        "name": name,
        "images": str(images_dir),
        "labels": str(label_dir(exp_dataset)),
        "classes": classes,
    }


def model_entry(exp_model: ExperimentModel) -> dict:
    """Build one YAML model entry; our name/num_classes win over params."""
    return {
        **(exp_model.params or {}),
        "name": exp_model.arch,
        "num_classes": exp_model.num_classes if exp_model.num_classes is not None else "auto",
    }


def _scheduler(experiment: Experiment):
    if experiment.scheduler_name == "none":
        return None
    return {"name": experiment.scheduler_name, **(experiment.scheduler_params or {})}


def build_experiment_dict(experiment: Experiment, output_dir: Path | str) -> dict:
    """Assemble the full friendy_mercury experiment dict.

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

    return {
        "name": experiment.name,
        "output_dir": str(output_dir),
        "datasets": datasets,
        "models": [model_entry(m) for m in models],
        "training": {
            "epochs": experiment.epochs,
            "batch_size": experiment.batch_size,
            "num_workers": experiment.num_workers,
            "device": experiment.device,
            "seed": experiment.seed,
            "amp": experiment.amp,
            "gradient_clip_norm": experiment.gradient_clip_norm,
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
            "score_threshold": experiment.eval_score_threshold,
            "iou_thresholds": experiment.iou_thresholds,
        },
    }


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
