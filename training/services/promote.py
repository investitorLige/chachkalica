"""Promote a finished training result into the model registry.

A :class:`~training.models.RunResult` is one trained model sitting in a run's
output dir. Promoting copies its checkpoint reference, architecture, class space
(read from the train dataset's classes.txt — the model's class order), and a
metrics snapshot into a durable :class:`~training.models.TrainedModel` that can
be evaluated independently later.
"""

from training.models import RunResult, TrainedModel
from training.services import config_gen


def promote_run_result(
    run_result: RunResult, *, name: str | None = None, stage: str = TrainedModel.DEV,
) -> TrainedModel:
    checkpoint = run_result.best_checkpoint or run_result.last_checkpoint
    if not checkpoint:
        raise ValueError(f"{run_result.run_name}: no checkpoint to promote.")

    base = name or run_result.run_name
    name = base
    if TrainedModel.objects.filter(name=name).exists():
        name = f"{base}-{run_result.pk}"
    try:
        classes = config_gen.classes_for_name(run_result.train_dataset_name)
    except Exception:  # noqa: BLE001 - dataset dir may be gone; promote without classes
        classes = []

    return TrainedModel.objects.create(
        name=name,
        stage=stage,
        arch=run_result.model_arch,
        checkpoint_path=checkpoint,
        num_classes=len(classes) or None,
        classes=classes,
        metrics=run_result.primary_metrics,
        source_run_result=run_result,
    )
