"""Auto-evaluate a finished training run's models on the experiment's test set.

Mirrors the manual "promote to registry → evaluate on a dataset" admin flow, but
runs it automatically once a training run finishes: every trained
:class:`~training.models.RunResult` that produced a checkpoint is promoted to a
:class:`~training.models.TrainedModel` and then evaluated against the
experiment's test dataset via a standalone :class:`~training.models.EvalRun`.

No-op when the experiment has no test dataset. Called from
``training.jobs.run_training`` after results are ingested; it enqueues the eval
jobs by dotted path (``training.jobs.run_eval``) so this module need not import
``training.jobs`` back.
"""

import django_rq

from training.models import EvalRun, ExperimentDataset
from training.services import config_gen, promote


def _queue():
    return django_rq.get_queue("default")


def schedule_test_evals(run) -> list[int]:
    """Promote every trained model in ``run`` and enqueue a test-set eval for it.

    Returns the pks of the EvalRuns queued (empty when the experiment has no test
    dataset, or nothing trained a checkpoint). A model whose eval request cannot
    be built is recorded as an errored EvalRun and skipped rather than aborting
    the rest.
    """
    experiment = run.experiment
    if experiment is None:
        return []
    test_ds = experiment.datasets.filter(role=ExperimentDataset.TEST).first()
    if test_ds is None:
        return []

    queue = _queue()
    queued: list[int] = []
    for rr in run.run_results.all():
        if not (rr.best_checkpoint or rr.last_checkpoint):
            continue  # a model that failed to train has no checkpoint to eval
        trained_model = promote.promote_run_result(rr)
        eval_run = EvalRun.objects.create(
            trained_model=trained_model,
            dataset=test_ds.dataset,
            label_source=test_ds.label_source,
            annotator=test_ds.annotator,
            explicit_labels_path=test_ds.explicit_labels_path,
        )
        try:
            config_gen.write_eval_request(eval_run)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            eval_run.status = EvalRun.ERROR
            eval_run.last_error = f"could not build eval request: {exc}"
            eval_run.save(update_fields=["status", "last_error"])
            continue
        queue.enqueue("training.jobs.run_eval", eval_run.pk)
        eval_run.status = EvalRun.QUEUED
        eval_run.save(update_fields=["status"])
        queued.append(eval_run.pk)
    return queued
