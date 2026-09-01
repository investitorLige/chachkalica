"""RQ job for one bundle benchmark.

Mirrors :mod:`training.jobs`: flip the row to ``running``, launch on the trainer, poll
until the subprocess exits, read back the result file it wrote to the shared mount,
then record the outcome. Failures are recorded **and re-raised**, so they also land in
django-rq's failed-job registry (visible under ``/django-rq/`` in the admin).

The result file is the contract: the trainer writes ``benchmark.json`` into the run's
``output_dir``, which every app container sees at the same path (see the SHARED
FILESYSTEM note in docker-compose.yml), so nothing large travels over HTTP.
"""

import json
import time
from pathlib import Path

from django.utils import timezone

from benchmarks.models import BundleBenchmark
from training.services import runner

# The trainer's own filename for the result. Mirrors service.BENCHMARK_RESULT_NAME —
# a hand-maintained mirror, same convention as training.pipelines, because this app's
# environment cannot import the trainer's module.
RESULT_NAME = "benchmark.json"

POLL_INTERVAL = 5
# A sweep is warmup + three timing passes + a stage pass, per cell, and a cell can be
# grown to meet the minimum duration — so a wide batch/concurrency sweep on a slow
# bundle is genuinely long. Two hours is the ceiling before the row is failed rather
# than left stranded at "running".
MAX_WAIT = 2 * 3600
# The RQ timeout has to outlive the poll loop or the job dies while the row still says
# "running" and nothing ever corrects it (same reasoning as training.jobs.JOB_TIMEOUT).
JOB_TIMEOUT = MAX_WAIT + 600


def _mark(benchmark, status, *, error="", finished=False):
    benchmark.status = status
    benchmark.last_error = error[:4000]
    fields = ["status", "last_error"]
    if status == BundleBenchmark.RUNNING and benchmark.started_at is None:
        benchmark.started_at = timezone.now()
        fields.append("started_at")
    if finished:
        benchmark.finished_at = timezone.now()
        fields.append("finished_at")
    benchmark.save(update_fields=fields)


def run_bundle_benchmark(benchmark_id: int) -> dict:
    """Launch, poll, ingest. Returns the headline numbers for the job result."""
    benchmark = BundleBenchmark.objects.get(pk=benchmark_id)

    output_dir = benchmark.resolved_output_dir()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    benchmark.output_dir = output_dir
    benchmark.save(update_fields=["output_dir"])
    _mark(benchmark, BundleBenchmark.RUNNING)

    try:
        runner.launch_benchmark(benchmark)
    except Exception as exc:
        _mark(benchmark, BundleBenchmark.ERROR, error=str(exc), finished=True)
        raise

    deadline = time.time() + MAX_WAIT
    status = {}
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            status = runner.fetch_benchmark_status(benchmark)
        except Exception as exc:  # noqa: BLE001 - a transient HTTP failure is not the answer
            status = {"status": "unknown", "log_tail": str(exc)}
        if status.get("status") in ("ok", "error"):
            break
        if status.get("status") == "unknown":
            # The service restarted. The subprocess is gone with it, but if it had
            # already written the result there is no reason to throw the run away.
            if (Path(output_dir) / RESULT_NAME).is_file():
                break
            _mark(benchmark, BundleBenchmark.ERROR,
                  error="the trainer service lost track of this run (restarted?) and "
                        "no result file was written",
                  finished=True)
            raise RuntimeError("trainer lost the benchmark job")
    else:
        try:
            runner.stop_benchmark(benchmark)
        except Exception:  # noqa: BLE001 - we are already failing the row
            pass
        _mark(benchmark, BundleBenchmark.ERROR,
              error=f"benchmark did not finish within {MAX_WAIT}s", finished=True)
        raise TimeoutError("benchmark timed out")

    result_path = Path(output_dir) / RESULT_NAME
    if not result_path.is_file():
        tail = (status.get("log_tail") or "").strip()[-2000:]
        _mark(benchmark, BundleBenchmark.ERROR,
              error=f"the benchmark wrote no {RESULT_NAME}."
                    + (f"\n\nTrainer log tail:\n{tail}" if tail else ""),
              finished=True)
        raise RuntimeError("benchmark produced no result file")

    try:
        result = json.loads(result_path.read_text())
    except ValueError as exc:
        _mark(benchmark, BundleBenchmark.ERROR,
              error=f"unreadable {RESULT_NAME}: {exc}", finished=True)
        raise

    benchmark.ingest(result)
    benchmark.status = BundleBenchmark.OK
    benchmark.finished_at = timezone.now()
    # A run that produced numbers is still an "ok" run when the harness had caveats;
    # they are surfaced on the report rather than turned into a failure, because a
    # contended GPU or a missing bare-model reference does not invalidate the rest.
    benchmark.last_error = ""
    benchmark.save(update_fields=[
        "results", "p50_ms", "fps", "gpu_mem_mb", "contended", "bundle_name", "fmt",
        "status", "finished_at", "last_error",
    ])
    return {
        "benchmark_id": benchmark.pk,
        "p50_ms": benchmark.p50_ms,
        "fps": benchmark.fps,
        "cells": len(result.get("cells") or []),
        "warnings": len(result.get("warnings") or []),
    }
