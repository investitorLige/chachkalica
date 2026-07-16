"""Read friendy_chachkalica's per-epoch training history for live display.

Each internal run writes ``<output_dir>/<run_name>/history.yaml`` — a list of
per-epoch dicts (``epoch``, ``train``, ``val``, ``lr``, ``is_best``) rewritten
after *every* epoch. So unlike the final ``results.yaml`` (which only the
ingest step reads, after the run finishes), this reflects progress live. The
TrainingRun admin surfaces it as one compact line per epoch.
"""

from pathlib import Path

import yaml


def _load_history(history_path: Path) -> list[dict]:
    try:
        with open(history_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return []
    return data if isinstance(data, list) else []


def run_histories(output_dir: str | Path | None) -> list[dict]:
    """Return ``[{run_name, epochs:[...]}]`` for each run dir under output_dir.

    Sorted by run name. Empty when output_dir is missing or no run has written
    a history yet (e.g. a run still spinning up, or one that errored early).
    """
    root = Path(output_dir) if output_dir else None
    if not root or not root.is_dir():
        return []
    histories = []
    for history_path in sorted(root.glob("*/history.yaml")):
        epochs = _load_history(history_path)
        if epochs:
            histories.append({"run_name": history_path.parent.name, "epochs": epochs})
    return histories


def best_epoch_entry(run_dir: str | Path | None, best_epoch: int | None = None) -> dict | None:
    """The ``history.yaml`` entry for the epoch whose checkpoint was saved.

    ``run_dir`` is a single internal run's directory (``RunResult.run_dir``).
    Matches ``best_epoch`` when given, else falls back to the last epoch flagged
    ``is_best``. Returns None when the run wrote no history (e.g. it errored
    before finishing its first epoch).
    """
    if not run_dir:
        return None
    epochs = _load_history(Path(run_dir) / "history.yaml")
    if not epochs:
        return None
    if best_epoch is not None:
        for epoch in epochs:
            if epoch.get("epoch") == best_epoch:
                return epoch
    flagged = [e for e in epochs if e.get("is_best")]
    return flagged[-1] if flagged else None


def flatten_epoch(epoch: dict) -> dict:
    """Flat, JSON-serialisable metrics for one epoch (charts + metrics table).

    Reshapes a raw ``history.yaml`` entry (nested ``train``/``val``/``val_map``
    dicts) into a single flat row. Shared by the live-report page's initial
    context and its polling endpoint so the metric keys are defined in one place.
    """
    train = epoch.get("train") or {}
    val = epoch.get("val") or {}
    val_map = epoch.get("val_map") or {}
    return {
        "epoch": epoch.get("epoch"),
        "train_loss": train.get("loss"),
        "val_loss": val.get("loss"),
        "map50": val_map.get("map50"),
        "map50_95": val_map.get("map50_95"),
        "precision": val_map.get("precision"),
        "recall": val_map.get("recall"),
        "f1": val_map.get("f1"),
        "lr": epoch.get("lr"),
        "best_metric": epoch.get("best_metric"),
        "best_metric_score": epoch.get("best_metric_score"),
        "is_best": bool(epoch.get("is_best")),
    }


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def epoch_line(epoch: dict) -> str:
    """One compact human-readable line summarising an epoch."""
    train = epoch.get("train") or {}
    val = epoch.get("val") or {}
    # val_map holds the compact per-epoch mAP summary (map50, map50_95, ...);
    # map50 is the metric best-checkpoint selection tracks.
    val_map = epoch.get("val_map") or {}
    lr = epoch.get("lr")
    parts = [
        f"epoch {epoch.get('epoch')}",
        f"train_loss={_fmt(train.get('loss'))}",
        f"val_loss={_fmt(val.get('loss'))}",
        f"val_map50={_fmt(val_map.get('map50'))}",
    ]
    # Runs configured with training.best_metric stamp each epoch with the
    # metric checkpoint selection tracked (e.g. val_f1+map50); show its score
    # unless it's plain val_map50, which the line already carries.
    best_metric = epoch.get("best_metric")
    if best_metric and best_metric != "val_map50":
        parts.append(f"{best_metric}={_fmt(epoch.get('best_metric_score'))}")
    parts.append(f"lr={lr:.2e}" if isinstance(lr, (int, float)) else "lr=—")
    if epoch.get("is_best"):
        parts.append("★ best")
    return "   ".join(parts)
