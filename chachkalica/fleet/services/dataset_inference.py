"""Run a detection model over every image of a dataset, and time it honestly.

The dataset-shaped sibling of :mod:`videos.services.inference`: the same model
(a catalogued ``.pt``, an exported ``.onnx``/``.engine``, or a whole infer
bundle), reached through the same trainer ``/predict_image`` endpoint with the
same payload builder, over a folder of images instead of a video's frames. The
split of responsibilities matches the video path too — :mod:`fleet.jobs` owns
the status transitions, this owns the loop.

**What the numbers mean**, since that is the whole point of the run:

* ``model_ms`` is what the *trainer* measured around its own inference call —
  image decode, preprocess, forward, postprocess — with a CUDA sync at the end,
  so an async queue can't be mistaken for speed. It excludes HTTP, JSON and this
  worker entirely. That is the number a deployment gets, and ``fps`` is
  ``1000 / mean(model_ms)``.
* ``request_ms`` is what this worker measured around the whole round trip. It is
  always larger, and reporting both is deliberate: the gap *is* the wrapper
  overhead, and hiding it would make the wrapper look free.
* An older trainer service reports no timing of its own. Then ``model_ms`` is
  null, ``timing_source`` says ``request`` and the report says so plainly rather
  than passing a round trip off as the model's speed.

**Warmup is never skipped.** The first call for a model loads it (and
deserializes a TensorRT plan, which can take seconds), so the run makes
``run.warmup`` untimed calls on the first image before the timed loop starts.
Without that, a 200-image run's mean carries a one-off model load and reads
5-10% slow — and a 10-image run reads absurd.

**Images are passed by path, never copied.** A dataset lives under
``data/source/<name>/`` and the trainer mounts the same ``data/`` at the same
place, so the file it opens is the file this loop names. Same contract the bundle
benchmark already relies on. A dataset that lives outside that mount cannot be
run this way, and the admin says so up front rather than failing per image.
"""

import logging
import statistics
import time
from pathlib import Path

from django.utils import timezone

from fleet.models import DatasetInferenceResult, DatasetInferenceRun
from fleet.services import lsapi
from fleet.services.paths import source_root

logger = logging.getLogger(__name__)

#: Consecutive per-image failures that mean the problem is not the images — a
#: wedged trainer, a missing artifact, a model that cannot load. Aborts the run
#: rather than writing thousands of identical failures.
MAX_CONSECUTIVE_ERRORS = 5

#: Images between refreshes of the run's rolling timing summary. Summarizing is a
#: full pass over the run's rows, so doing it per image would be quadratic; doing
#: it only at the end would leave the report blank for the whole run.
SUMMARY_REFRESH_EVERY = 25


def dataset_dir(name: str):
    """Where a dataset by that name lives on disk."""
    return source_root() / name


def shared_data_root():
    """The ``data/`` mount the web, worker and trainer containers all see at one
    path (``/app/data`` under compose)."""
    from django.conf import settings

    return Path(settings.BASE_DIR) / "data"


def visible_to_trainer(directory) -> bool:
    """Whether the trainer can open files under ``directory``.

    Images are handed over as paths, never copied, which only works inside the
    shared mount. A dataset root pointed somewhere else — an absolute
    ``source_dir`` on a disk only the web container has — would fail on every
    single image, so the form refuses it up front instead of queueing a run that
    cannot work.
    """
    try:
        Path(directory).resolve().relative_to(shared_data_root().resolve())
    except ValueError:
        return False
    return True


def select_images(directory, limit: int) -> list:
    """The images this run will go through, in the dataset's canonical order.

    With a ``limit`` below the dataset size, take an evenly spaced subset rather
    than the first N files: the first N files of a sorted dataset are usually one
    scene, one camera or one class, and a latency measurement over one scene is a
    measurement of that scene's resolution and crowd count. Same rule (and the
    same reason) as ``vlm.services.dataset_runner.select_images``.
    """
    images = lsapi.list_dataset_images(directory)
    if not limit or limit >= len(images):
        return images
    step = len(images) / limit
    picked, seen = [], set()
    for i in range(limit):
        index = min(int(i * step), len(images) - 1)
        if index not in seen:
            seen.add(index)
            picked.append(images[index])
    return picked


def _percentiles(values: list[float]) -> dict:
    """``mean``/``p50``/``p90``/``p99``/``min``/``max`` over ``values`` (ms).

    The tail percentiles are nearest-rank off the sorted samples rather than an
    interpolated quantile: with 20-odd images an interpolated p90 invents a
    number between two frames, and every one of these is a real frame's time.
    """
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
        return ordered[index]

    return {
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p90": at(0.90),
        "p99": at(0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def summarize(run: DatasetInferenceRun) -> dict:
    """Aggregate the run's stored rows into ``run.timings`` and save it.

    Failed images are left out of the latency statistics — an image the trainer
    never answered about is a gap in the measurement, not a slow frame, and its
    round-trip time is usually a timeout. ``errors`` is what surfaces them.

    Also writes the two denormalized headline columns (``fps``, ``p50_ms``) so the
    changelist is readable without opening a report.
    """
    rows = list(run.results.filter(error="").values(
        "model_ms", "request_ms", "detections", "stages"))
    model_ms = [r["model_ms"] for r in rows if r["model_ms"] is not None]
    request_ms = [r["request_ms"] for r in rows if r["request_ms"] is not None]

    # The model's own time when the trainer reports it, the round trip when it
    # does not — one of the two, named, rather than a silent mixture.
    timed = model_ms or request_ms
    source = "model" if model_ms else ("request" if request_ms else "")

    stage_totals: dict[str, float] = {}
    for row in rows:
        for stage, value in (row["stages"] or {}).items():
            try:
                stage_totals[stage] = stage_totals.get(stage, 0.0) + float(value)
            except (TypeError, ValueError):
                continue
    stage_sum = sum(stage_totals.values())

    timings = {
        "timing_source": source,
        "images_timed": len(timed),
        "images_failed": run.errors_count,
        "detections_total": sum(r["detections"] for r in rows),
        "model_ms": _percentiles(model_ms),
        "request_ms": _percentiles(request_ms),
        # Per-stage *shares*, not absolutes: the trainer synchronizes at each
        # stage boundary to attribute them, which perturbs the absolute numbers
        # it would otherwise report (the same caveat the bundle benchmark's stage
        # pass carries). A share of the frame budget survives that.
        "stage_share": {
            stage: total / stage_sum for stage, total in sorted(stage_totals.items())
        } if stage_sum else {},
        "stage_ms_mean": {
            stage: total / len(rows) for stage, total in sorted(stage_totals.items())
        } if rows else {},
        "warmup": run.warmup,
    }
    mean = (timings["model_ms"] or timings["request_ms"] or {}).get("mean")
    timings["fps"] = (1000.0 / mean) if mean else None
    timings["model_fps"] = (
        (1000.0 / timings["model_ms"]["mean"]) if timings["model_ms"] else None)
    timings["request_fps"] = (
        (1000.0 / timings["request_ms"]["mean"]) if timings["request_ms"] else None)

    run.timings = timings
    run.fps = timings["fps"]
    run.p50_ms = (timings["model_ms"] or timings["request_ms"] or {}).get("p50")
    run.detections_total = timings["detections_total"]
    run.save(update_fields=["timings", "fps", "p50_ms", "detections_total"])
    return timings


def _call(payload: dict, image_path) -> tuple[dict, float]:
    """One ``/predict_image`` call, with the worker's own stopwatch around it."""
    from training.services import runner

    started = time.perf_counter()
    response = runner.predict_image({**payload, "image_path": str(image_path)})
    return response, (time.perf_counter() - started) * 1000.0


def run_model_on_dataset(run: DatasetInferenceRun) -> dict:
    """Walk the dataset's images, running the model on each and timing it.

    Returns a summary dict; the caller records the terminal status.
    """
    from videos.services.inference import build_predict_payload

    name = run.dataset_name_snapshot
    directory = dataset_dir(name)
    if not directory.is_dir():
        raise RuntimeError(f"{name}: no dataset directory at {directory}")

    images = select_images(directory, run.image_limit)
    if not images:
        raise RuntimeError(f"{name}: no images found under {directory}")

    # Raises RuntimeError for a missing artifact / an inconsistent pipeline, which
    # the job records as the run's error — same pre-flight the video action does
    # at submit time, repeated here because a bundle or artifact can vanish
    # between queueing and running.
    # stage_timings asks the trainer to split its own measurement into
    # load/infer/format. It costs a GPU sync per boundary, which is why the video
    # and camera paths do not ask for it — here, where the whole point is where
    # the frame budget goes, it is worth the perturbation (and the report reads
    # the split as shares for exactly that reason).
    payload = {**build_predict_payload(run), "stage_timings": True}

    run.images_total = len(images)
    run.save(update_fields=["images_total"])

    # Untimed, and on the first image so it is a real frame of the right size.
    # Failures here are not swallowed: if the model cannot run at all, that is
    # the run's error, reported once, rather than N identical per-image ones.
    for _ in range(run.warmup):
        _call(payload, images[0])

    cancelled = False
    consecutive_errors = 0
    seq = 0

    def cancel_requested() -> bool:
        """Whether Stop has been pressed since the last image.

        Cooperative, and checked once per image whether that image produced
        detections or an error — a run failing on every image still has to be
        stoppable. A call already in flight cannot be interrupted, which is why
        the HTTP client bounds every request with a timeout.
        """
        run.refresh_from_db(fields=["status"])
        return run.status == DatasetInferenceRun.CANCEL_REQUESTED

    wall_started = time.perf_counter()
    try:
        for image_path in images:
            seq += 1
            result = DatasetInferenceResult(
                run=run, seq=seq, image_filename=image_path.name)
            try:
                response, request_ms = _call(payload, image_path)
            except Exception as exc:  # noqa: BLE001 - one image must not end the run
                consecutive_errors += 1
                result.error = str(exc)
                result.save()
                run.images_processed = seq
                run.errors_count = run.errors_count + 1
                run.save(update_fields=["images_processed", "errors_count"])
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(
                        f"{consecutive_errors} images in a row failed — giving up. "
                        f"Last error: {exc}"
                    ) from exc
                if cancel_requested():
                    cancelled = True
                    break
                continue

            consecutive_errors = 0
            timings = response.get("timings") or {}
            boxes = response.get("boxes") or []
            result.boxes = boxes
            result.detections = len(boxes)
            result.request_ms = request_ms
            # Absent on a trainer that predates per-call timing; the summary
            # reports which of the two clocks it used rather than guessing.
            total = timings.get("total_ms")
            result.model_ms = float(total) if isinstance(total, (int, float)) else None
            result.stages = {
                key: value for key, value in timings.items()
                if key.endswith("_ms") and key != "total_ms"
                and isinstance(value, (int, float))
            }
            result.save()

            run.images_processed = seq
            run.save(update_fields=["images_processed"])
            if seq % SUMMARY_REFRESH_EVERY == 0:
                summarize(run)

            if cancel_requested():
                cancelled = True
                break
    finally:
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        # Summarize whatever was produced, cancelled or not — the images are
        # already paid for, so their numbers may as well be readable.
        timings = summarize(run)
        timings["wall_ms"] = wall_ms
        timings["wall_fps"] = (
            (1000.0 * run.images_processed / wall_ms) if wall_ms else None)
        run.timings = timings
        run.save(update_fields=["timings"])

    return {
        "cancelled": cancelled,
        "images_total": run.images_total,
        "images_processed": run.images_processed,
        "errors": run.errors_count,
        "fps": run.fps,
        "timings": run.timings,
        "finished_at": timezone.now(),
    }


# ------------------------------------------------------------------ rendering
def render_result(result: DatasetInferenceResult) -> bytes:
    """``result``'s source image as JPEG bytes with its stored boxes drawn on it.

    Drawn on demand rather than written during the run: the boxes are the cheap
    part to keep and re-drawing one image costs milliseconds, so a run over ten
    thousand frames does not leave a second copy of the dataset on disk.

    Reuses ``videos.services.inference.draw_boxes`` — the same overlay the
    inferred videos and the live camera path draw, so a detection looks the same
    wherever it is seen. Raises ``RuntimeError`` when the image cannot be decoded
    or encoded, which the view turns into a 404: a file that is not really an
    image is not a server error.
    """
    import cv2

    from videos.services.inference import draw_boxes

    path = result_image_path(result)
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"could not decode {path.name}")
    draw_boxes(image, result.boxes or [])
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"could not encode {path.name}")
    return buffer.tobytes()


def result_image_path(result: DatasetInferenceResult):
    """The on-disk image a result row names, confined to its own dataset.

    The containment check is what keeps a stored filename from addressing
    anything outside the dataset it was measured on. Raises ``FileNotFoundError``
    when the file is gone (a dataset re-imported since the run) and ``ValueError``
    when the name escapes its directory.
    """
    directory = dataset_dir(result.run.dataset_name_snapshot)
    image_dir = lsapi.image_source_dir(directory)
    candidate = (image_dir / Path(result.image_filename).name).resolve()
    candidate.relative_to(image_dir.resolve())  # raises ValueError if outside
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate
