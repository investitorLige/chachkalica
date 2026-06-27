"""Parse a finished friendy_mercury run's output into the database.

After ``run_experiment`` finishes it leaves, under the run's ``output_dir``:

* ``results.yaml`` — one entry per train run (train-dataset × model), keyed by a
  unique ``run_name`` (also: model, train_dataset, checkpoints, best epoch/loss).
* ``val_results.yaml`` / ``test_results.yaml`` — per-run evaluation ``metrics``,
  matched back by ``run_name``.
* ``run_summary.yaml`` — written last, so its presence means "done".

We join these by ``run_name`` into :class:`~training.models.RunResult` rows and
stash the high-level summary on ``TrainingRun.results``.
"""

from pathlib import Path

import yaml

from training.models import RunResult


def _load_yaml(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def summary_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "run_summary.yaml"


def is_complete(output_dir: str | Path) -> bool:
    """A run is done once the trainer has written its final summary."""
    return summary_path(output_dir).exists()


def _metrics_by_run_name(entries) -> dict:
    out = {}
    for entry in entries or []:
        name = entry.get("run_name")
        if name:
            out[name] = entry.get("metrics")
    return out


def ingest_run(run) -> dict:
    """Populate RunResult rows for ``run`` from its output_dir. Returns a summary.

    Idempotent: existing RunResult rows for the run are replaced.
    """
    output_dir = Path(run.output_dir)
    train_results = _load_yaml(output_dir / "results.yaml") or []
    val_metrics = _metrics_by_run_name(_load_yaml(output_dir / "val_results.yaml"))
    test_metrics = _metrics_by_run_name(_load_yaml(output_dir / "test_results.yaml"))
    summary = _load_yaml(summary_path(output_dir)) or {}

    run.run_results.all().delete()
    created = 0
    for entry in train_results:
        run_name = entry.get("run_name") or f"run-{entry.get('run_index')}"
        RunResult.objects.create(
            run=run,
            run_name=run_name,
            run_index=entry.get("run_index"),
            model_arch=entry.get("model") or "",
            train_dataset_name=entry.get("train_dataset") or "",
            best_epoch=entry.get("best_epoch"),
            best_loss=entry.get("best_loss"),
            run_dir=entry.get("run_dir") or "",
            best_checkpoint=entry.get("best_checkpoint") or "",
            last_checkpoint=entry.get("last_checkpoint") or "",
            val_metrics=val_metrics.get(run_name),
            test_metrics=test_metrics.get(run_name) or entry.get("test_metrics"),
        )
        created += 1

    run.results = summary
    run.save(update_fields=["results"])
    return {"run_results": created, "summary": summary}
