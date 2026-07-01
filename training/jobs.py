"""RQ jobs for executing training runs.

Mirrors ``fleet.jobs``: an admin action enqueues one of these, the worker drives
the long-running work and updates the row's status as it goes. A single job owns
a whole run — it launches training on the trainer service, polls until the run
terminates, then ingests the results from the shared output directory.
"""

import time

from django.utils import timezone

from training.models import EvalRun, TrainingRun
from training.services import autoeval, ingest, runner

POLL_INTERVAL = 10       # seconds between status checks
MAX_WAIT = 60 * 60 * 48  # give up after 48h


def _mark(run: TrainingRun, status: str, *, error: str = "", finished: bool = False):
    run.status = status
    run.last_error = error
    if finished:
        run.finished_at = timezone.now()
    run.save(update_fields=["status", "last_error", "finished_at"])


def run_training(run_id: int, resume: bool = False) -> dict:
    run = TrainingRun.objects.get(pk=run_id)
    run.status = TrainingRun.RUNNING
    run.started_at = timezone.now()
    run.last_error = ""
    run.save(update_fields=["status", "started_at", "last_error"])

    try:
        runner.launch(run, resume=resume)
    except Exception as exc:  # noqa: BLE001 - surface launch failures to the row
        _mark(run, TrainingRun.ERROR, error=f"launch failed: {exc}", finished=True)
        raise

    waited = 0
    while waited < MAX_WAIT:
        status = runner.fetch_status(run)
        state = status.get("status")
        if state == "ok":
            break
        if state == "error":
            _mark(run, TrainingRun.ERROR, error=status.get("log_tail", "")[-4000:], finished=True)
            return {"status": "error"}
        if state == "unknown":
            # Service has no record; trust the filesystem if the run finished.
            if ingest.is_complete(run.output_dir):
                break
            _mark(run, TrainingRun.ERROR, error="trainer lost the run and no summary was written",
                  finished=True)
            return {"status": "error"}
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
    else:
        _mark(run, TrainingRun.ERROR, error="timed out waiting for the run to finish", finished=True)
        return {"status": "error"}

    summary = ingest.ingest_run(run)
    _mark(run, TrainingRun.OK, finished=True)

    # Best-effort: training succeeded, so an auto-eval hiccup must not fail the
    # run. Per-eval build errors are already captured on their EvalRun rows.
    try:
        queued = autoeval.schedule_test_evals(run)
    except Exception as exc:  # noqa: BLE001 - never let auto-eval flip an OK run
        run.last_error = f"training ok, but scheduling test evals failed: {exc}"
        run.save(update_fields=["last_error"])
        return {"status": "ok", "auto_eval_error": str(exc), **summary}
    return {"status": "ok", "auto_evals": queued, **summary}


def _mark_eval(eval_run: EvalRun, status: str, *, error: str = "", finished: bool = False):
    eval_run.status = status
    eval_run.last_error = error
    if finished:
        eval_run.finished_at = timezone.now()
    eval_run.save(update_fields=["status", "last_error", "finished_at"])


def run_eval(eval_run_id: int) -> dict:
    eval_run = EvalRun.objects.get(pk=eval_run_id)
    eval_run.status = EvalRun.RUNNING
    eval_run.started_at = timezone.now()
    eval_run.last_error = ""
    eval_run.save(update_fields=["status", "started_at", "last_error"])

    try:
        runner.launch_eval(eval_run)
    except Exception as exc:  # noqa: BLE001
        _mark_eval(eval_run, EvalRun.ERROR, error=f"launch failed: {exc}", finished=True)
        raise

    waited = 0
    while waited < MAX_WAIT:
        status = runner.fetch_eval_status(eval_run)
        state = status.get("status")
        if state == "ok":
            break
        if state == "error":
            _mark_eval(eval_run, EvalRun.ERROR, error=status.get("log_tail", "")[-4000:],
                       finished=True)
            return {"status": "error"}
        if state == "unknown":
            if ingest.eval_is_complete(eval_run.output_dir):
                break
            _mark_eval(eval_run, EvalRun.ERROR, error="trainer lost the eval and wrote no result",
                       finished=True)
            return {"status": "error"}
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
    else:
        _mark_eval(eval_run, EvalRun.ERROR, error="timed out waiting for eval", finished=True)
        return {"status": "error"}

    summary = ingest.ingest_eval(eval_run)
    _mark_eval(eval_run, EvalRun.OK, finished=True)
    return {"status": "ok", **summary}
