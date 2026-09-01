"""'Extend run…' — give a finished run more epochs (and optionally a new lr).

A finished :class:`~training.models.TrainingRun`'s config YAML is frozen at
creation time (see ``config_gen.write_config``), and its output dir already
holds a ``result.yaml`` that ``friendy_chachkalica.ml.train.train_from_config``
treats as "already complete, skip" the moment ``--resume`` is passed. So just
bumping ``Experiment.epochs`` in admin and clicking "Resume from checkpoint"
does nothing — the frozen config still says the old epoch count, and even if
it didn't, the skip-guard would short-circuit before ever looking at
``last.pt``.

:func:`extend_run` does the two things needed to make resume actually pick up
a higher epoch target (and, if given, a new lr): it raises the Experiment's
``epochs``/``lr``, rewrites the run's config from the updated Experiment, and
moves the stale ``result.yaml`` aside. The other half — the checkpoint's
optimizer/scheduler state silently re-clobbering the new lr and (for a cosine
schedule) the new epoch horizon back to their original values on load — is
fixed on the training side, in ``friendy_chachkalica/ml/train.py``'s resume
block.
"""

from __future__ import annotations

import time
from pathlib import Path

from training.models import TrainingRun
from training.services import config_gen


def extend_run(run: TrainingRun, additional_epochs: int, new_lr: float | None = None) -> TrainingRun:
    """Bump ``run``'s experiment to train ``additional_epochs`` further.

    Raises ``ValueError`` for the same reasons the admin action would want to
    show as a form error rather than a 500: no epochs requested, no output dir
    yet, or no checkpoint to resume from.
    """
    if additional_epochs < 1:
        raise ValueError("Additional epochs must be at least 1.")
    if not run.output_dir:
        raise ValueError(f"Run #{run.pk} has no output_dir yet — nothing to extend.")
    if not any(Path(run.output_dir).rglob("last.pt")):
        raise ValueError(
            f"Run #{run.pk} has no last.pt checkpoint under {run.output_dir} to resume from."
        )

    experiment = run.experiment
    experiment.epochs = experiment.epochs + additional_epochs
    update_fields = ["epochs"]
    if new_lr is not None:
        experiment.lr = new_lr
        update_fields.append("lr")
    experiment.save(update_fields=update_fields)

    # Move any result.yaml(s) aside — their mere presence makes train_from_config
    # skip the run instead of resuming it, regardless of the new epoch target.
    stamp = int(time.time())
    for result_path in Path(run.output_dir).rglob("result.yaml"):
        result_path.rename(result_path.with_name(f"result.yaml.pre-extend-{stamp}"))

    # Re-derive the YAML from the now-updated Experiment. run_paths() is
    # deterministic from experiment.name + run.pk (both unchanged here), so
    # this overwrites the same config_yaml_path/output_dir in place.
    config_gen.write_config(experiment, run)
    return run
