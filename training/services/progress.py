"""Read friendy_mercury's per-epoch training history for live display.

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


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def epoch_line(epoch: dict) -> str:
    """One compact human-readable line summarising an epoch."""
    train = epoch.get("train") or {}
    val = epoch.get("val") or {}
    lr = epoch.get("lr")
    parts = [
        f"epoch {epoch.get('epoch')}",
        f"train_loss={_fmt(train.get('loss'))}",
        f"val_loss={_fmt(val.get('loss'))}",
        f"lr={lr:.2e}" if isinstance(lr, (int, float)) else "lr=—",
    ]
    if epoch.get("is_best"):
        parts.append("★ best")
    return "   ".join(parts)
