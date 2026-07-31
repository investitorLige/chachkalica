"""RQ jobs for executing training runs.

Mirrors ``fleet.jobs``: an admin action enqueues one of these, the worker drives
the long-running work and updates the row's status as it goes. A single job owns
a whole run — it launches training on the trainer service, polls until the run
terminates, then ingests the results from the shared output directory.
"""

import time
from pathlib import Path

from django.utils import timezone

from training.models import EvalRun, ExportRun, TrainingRun
from training.services import autoeval, exports, ingest, runner

POLL_INTERVAL = 10       # seconds between status checks
MAX_WAIT = 60 * 60 * 48  # give up after 48h

# A single run_training/run_eval job polls for the *whole* run (up to MAX_WAIT),
# so its RQ work-horse timeout must comfortably exceed MAX_WAIT. Otherwise RQ
# kills the poller mid-run (the queue DEFAULT_TIMEOUT is only minutes) and the
# row is stranded at "running" while training carries on in the trainer service.
# Enqueue every run_training/run_eval with job_timeout=JOB_TIMEOUT.
JOB_TIMEOUT = MAX_WAIT + 60 * 60  # poll window + an hour for launch/ingest


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
        # A "Pause run" admin action sets this row to paused and asks the
        # trainer to stop out-of-band; check for that before trusting the
        # trainer's status, since a killed process reports as errored there.
        run.refresh_from_db(fields=["status"])
        if run.status == TrainingRun.PAUSED:
            return {"status": "paused"}
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

    return finalize_success(run)


def finalize_success(run: TrainingRun) -> dict:
    """Ingest a finished run, mark it OK, and schedule test evals.

    The tail of a successful run — factored out so the reconcile sweep can
    finalize a run whose poller died before reaching here (see
    ``training.services.reconcile``).
    """
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

    return finalize_eval_success(eval_run)


def finalize_eval_success(eval_run: EvalRun) -> dict:
    """Ingest a finished eval and mark it OK. Shared with the reconcile sweep."""
    summary = ingest.ingest_eval(eval_run)
    _mark_eval(eval_run, EvalRun.OK, finished=True)
    return {"status": "ok", **summary}


def _mark_pipeline(pe, status: str, *, error: str = "", finished: bool = False):
    pe.status = status
    pe.last_error = error
    if finished:
        pe.finished_at = timezone.now()
    pe.save(update_fields=["status", "last_error", "finished_at"])


def run_pipeline_eval(pe_id: int) -> dict:
    """Drive one chachak pipeline eval through the trainer service's /pipeline."""
    from eval_pipelines.models import PipelineEvalRun

    pe = PipelineEvalRun.objects.get(pk=pe_id)
    pe.status = PipelineEvalRun.RUNNING
    pe.started_at = timezone.now()
    pe.last_error = ""
    pe.save(update_fields=["status", "started_at", "last_error"])

    try:
        runner.launch_pipeline(pe)
    except Exception as exc:  # noqa: BLE001
        _mark_pipeline(pe, PipelineEvalRun.ERROR, error=f"launch failed: {exc}", finished=True)
        raise

    waited = 0
    while waited < MAX_WAIT:
        status = runner.fetch_pipeline_status(pe)
        state = status.get("status")
        if state == "ok":
            break
        if state == "error":
            _mark_pipeline(pe, PipelineEvalRun.ERROR, error=status.get("log_tail", "")[-4000:],
                           finished=True)
            return {"status": "error"}
        if state == "unknown":
            if ingest.pipeline_is_complete(pe.output_dir):
                break
            _mark_pipeline(pe, PipelineEvalRun.ERROR,
                           error="trainer lost the pipeline eval and wrote no result",
                           finished=True)
            return {"status": "error"}
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
    else:
        _mark_pipeline(pe, PipelineEvalRun.ERROR, error="timed out waiting for pipeline eval",
                       finished=True)
        return {"status": "error"}

    return finalize_pipeline_success(pe)


def finalize_pipeline_success(pe) -> dict:
    """Ingest a finished pipeline eval and mark it OK. Shared with reconcile."""
    summary = ingest.ingest_pipeline_eval(pe)
    _mark_pipeline(pe, pe.OK, finished=True)
    return {"status": "ok", **summary}


# Each export job runs the primary export (ONNX/TRT) then, best-effort, a bundle —
# so its RQ job_timeout must cover both, not just runner.EXPORT_TIMEOUT/TRT_BUILD_TIMEOUT.
EXPORT_ONNX_JOB_TIMEOUT = runner.EXPORT_TIMEOUT + runner.BUNDLE_TIMEOUT + 60
EXPORT_TRT_JOB_TIMEOUT = runner.TRT_BUILD_TIMEOUT + runner.BUNDLE_TIMEOUT + 60


def _bundle_after_export(run: ExportRun, artifact_path: Path, *, fmt: str, precision: str = "auto"):
    """Best-effort infer bundle next to ``artifact_path``, mirroring what
    ``TrainedModelAdmin._export_bundle`` used to do synchronously in the admin
    request. A bundle failure is recorded on ``run.bundle_error`` but does not
    fail the export itself — the primary artifact already exported fine.

    ``build_bundle_request`` is inside the guard too: it resolves checkpoint paths
    and reads the artifact's sidecar, so it can raise for the same environmental
    reasons the bundle call can, and a raise here used to escape the job entirely.
    """
    try:
        bundle_request = exports.build_bundle_request(run.model, artifact_path)
        if bundle_request is None:
            return
        bundle_dir = artifact_path.parent / f"{artifact_path.stem}-bundle"
        result = runner.export_bundle(bundle_request, bundle_dir, fmt=fmt, precision=precision)
    except Exception as exc:  # noqa: BLE001 - non-fatal, recorded on the row
        run.bundle_error = str(exc)
        return
    run.bundle_dir = result.get("bundle_dir", "")


def _finish_export(run: ExportRun, result: dict, *, fmt: str, precision: str = "auto") -> dict:
    """Run the post-export steps, then mark ``run`` terminal. Never raises.

    Everything here happens *after* the artifact is already on disk, so none of it
    may fail the export — but none of it may strand the row either. Both steps used
    to sit outside any guard: a raise from either left the row at ``RUNNING`` with
    an empty ``last_error`` forever (RQ logs the traceback, but nothing reads it
    back onto the row, and ``reconcile`` has no ExportRun handling), where the old
    synchronous admin action messaged every error to the operator. Failures are
    recorded on ``bundle_error`` — the row's non-fatal error channel — labelled by
    stage, since a missing pipeline sidecar and a missing bundle are different
    problems with the same non-fatal weight.
    """
    artifact_path = Path(run.output_path)
    problems = []
    run.bundle_error = ""  # a retried job reports this attempt, not the last one

    try:
        exports.export_pipeline_sidecar(run.model, artifact_path)
    except Exception as exc:  # noqa: BLE001 - non-fatal, recorded on the row
        problems.append(f"pipeline sidecar: {exc}")

    try:
        _bundle_after_export(run, artifact_path, fmt=fmt, precision=precision)
    except Exception as exc:  # noqa: BLE001 - belt and braces; the helper guards itself
        problems.append(f"bundle: {exc}")
    if run.bundle_error:
        problems.append(f"bundle: {run.bundle_error}")

    run.bundle_error = "; ".join(problems)
    run.status, run.result, run.finished_at = ExportRun.OK, result, timezone.now()
    run.save(update_fields=["status", "result", "bundle_dir", "bundle_error", "finished_at"])
    return result


def run_export_onnx(export_id: int) -> dict:
    """Export one checkpoint to ONNX, then sidecar + bundle it. See ``ExportRun``."""
    run = ExportRun.objects.get(pk=export_id)
    run.status, run.started_at = ExportRun.RUNNING, timezone.now()
    run.save(update_fields=["status", "started_at"])

    try:
        result = runner.export_onnx(run.checkpoint_path, run.output_path)
    except Exception as exc:  # noqa: BLE001 - surface service/network errors on the row
        run.status, run.last_error, run.finished_at = ExportRun.ERROR, str(exc), timezone.now()
        run.save(update_fields=["status", "last_error", "finished_at"])
        raise

    return _finish_export(run, result, fmt="onnx")


def run_export_trt(export_id: int) -> dict:
    """Build one checkpoint's TensorRT engine, then sidecar + bundle it. See ``ExportRun``."""
    run = ExportRun.objects.get(pk=export_id)
    run.status, run.started_at = ExportRun.RUNNING, timezone.now()
    run.save(update_fields=["status", "started_at"])

    input_hw = tuple(run.input_hw) if run.input_hw else None
    try:
        result = runner.export_trt(
            run.checkpoint_path, run.output_path, precision=run.precision, input_hw=input_hw)
    except Exception as exc:  # noqa: BLE001 - surface service/network errors on the row
        run.status, run.last_error, run.finished_at = ExportRun.ERROR, str(exc), timezone.now()
        run.save(update_fields=["status", "last_error", "finished_at"])
        raise

    return _finish_export(run, result, fmt="engine", precision=run.precision)
