"""Ask a VLM about every image of a dataset, and score the answers.

The folder-shaped sibling of :mod:`vlm.services.runner`. Same split of
responsibilities — :mod:`vlm.jobs` owns the status transitions, this owns the
loop — and the same shared-mount convention for handing frames over.

Four decisions worth knowing about:

* **Images are passed by path, not copied.** A dataset lives under
  ``data/source/<name>/`` and ``vlm-backend`` mounts the same ``data/`` at the
  same place, so the file the worker sees is the file the backend opens. When
  ``source_dir`` points somewhere outside that mount (it may be absolute), the
  image is staged into the scratch dir instead — otherwise the backend would
  fail on every single image with a path it cannot see.
* **A capped run spans the dataset.** ``image_limit`` takes an even stride
  rather than the first N files, because the first N files of a sorted dataset
  are usually one scene, one camera, or one class.
* **One bad image does not void the run.** A per-image failure is recorded on
  its row and counted; the loop carries on. A *wedged backend* is different, and
  :data:`MAX_CONSECUTIVE_ERRORS` in a row aborts the run rather than writing
  thousands of identical failures.
* **Grading happens inline and again at the end.** Inline, so a half-finished
  run already shows scores; again at the end, so the stored metrics cover every
  row. :func:`regrade` redoes only that cheap half.
"""

import logging
import shutil
import time
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services.paths import source_root
from vlm.models import VlmDatasetResult, VlmDatasetRun
from vlm.services import backend, grading

logger = logging.getLogger(__name__)

#: Consecutive per-image failures that mean the problem is not the images.
MAX_CONSECUTIVE_ERRORS = 5

#: Images between metric refreshes while a graded run is still going. Scoring is
#: a full pass over the rows, so doing it per image would be quadratic; doing it
#: never would leave the report blank for hours on a long run.
METRICS_REFRESH_EVERY = 50


def dataset_dir(name: str) -> Path:
    """Where a dataset by that name lives on disk."""
    return source_root() / name


def shared_data_root() -> Path:
    """The ``data/`` mount both the worker and ``vlm-backend`` see at one path."""
    return Path(settings.BASE_DIR) / "data"


def scratch_dir() -> Path:
    """Where images outside the shared mount are staged for the backend."""
    return shared_data_root() / "vlm_frames"


def read_classes(directory: Path) -> list[str]:
    """The dataset's ``classes.txt``, or an empty list when it has none.

    Only grading needs class names, so a plain "just tell me what the model
    says" run over a folder without a ``classes.txt`` must still work.
    """
    classes_file = directory / "classes.txt"
    if not classes_file.is_file():
        return []
    names, _tools = lsapi.parse_classes_file(classes_file)
    return names


def select_images(directory: Path, limit: int) -> list[Path]:
    """The images this run will ask about, in the dataset's canonical order.

    With a ``limit`` below the dataset size, take an evenly spaced subset: a
    100-image sample of a 10k dataset should look like the dataset, whereas its
    first 100 files are usually one scene, one camera, or one class.
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


def _stage_for_backend(image_path: Path, staging: Path | None) -> str:
    """The path to hand the backend for this image.

    The image itself when it already sits on the shared mount; a copy in the
    scratch dir when it does not. One reused scratch name per run is safe
    because the loop is strictly sequential — the same trick
    :mod:`vlm.services.runner` uses for video frames.
    """
    if staging is None:
        return str(image_path)
    target = staging.with_suffix(image_path.suffix.lower() or ".jpg")
    shutil.copyfile(image_path, target)
    return str(target)


def _staging_path(run: VlmDatasetRun, directory: Path) -> Path | None:
    """A scratch path for this run, or None when staging is unnecessary."""
    try:
        directory.resolve().relative_to(shared_data_root().resolve())
    except ValueError:
        scratch = scratch_dir()
        scratch.mkdir(parents=True, exist_ok=True)
        logger.info("dataset run %s: %s is outside %s — staging each image",
                    run.pk, directory, shared_data_root())
        return scratch / f"dsrun_{run.pk}"
    return None


def _grade_row(result: VlmDatasetResult, matchers, *, negation: bool) -> None:
    """Fill in a row's derived grading fields from its stored text.

    A row that failed has no answer to score, so it gets no verdict either —
    ``matched=None`` rather than a False that would read as "the model got this
    one wrong". Does not save: the caller decides whether this is part of the
    run loop's write or a bulk re-grade.
    """
    if result.error:
        result.predicted_classes = []
        result.matched = None
        return
    result.predicted_classes = grading.predict_classes(
        result.text, matchers, negation=negation,
    )
    result.matched = (
        grading.is_exact_match(result.gt_classes, result.predicted_classes)
        if result.has_label else None
    )


def recompute_metrics(run: VlmDatasetRun) -> dict:
    """Re-aggregate the run's stored rows into ``run.metrics`` and save it.

    Rows that failed are left out: an image the backend never answered about is
    a gap in the measurement, not a wrong answer, and scoring it as "named no
    classes" would charge the model a false negative for an HTTP timeout.
    ``errors_count`` is what surfaces them instead.
    """
    if not run.grade:
        return {}
    run.metrics = grading.metrics(
        run.results.filter(error=""), run.classes_snapshot or [],
    )
    run.save(update_fields=["metrics"])
    return run.metrics


def regrade(run: VlmDatasetRun, *, alias_map: dict, negation: bool) -> dict:
    """Re-score a finished run under a new grading configuration.

    The expensive half of a run is the answers, and they do not change — so
    tuning aliases (or turning the negation heuristic off to see what it was
    doing) costs one pass over the rows and no GPU at all. That is the whole
    reason the raw text is stored per image rather than only its verdict.
    """
    run.alias_map = alias_map or {}
    run.negation_aware = bool(negation)
    run.grade = True
    run.save(update_fields=["alias_map", "negation_aware", "grade"])

    matchers = grading.compile_matchers(run.classes_snapshot or [], run.alias_map)
    rows = list(run.results.all())
    for result in rows:
        _grade_row(result, matchers, negation=run.negation_aware)
    VlmDatasetResult.objects.bulk_update(rows, ["predicted_classes", "matched"])
    return recompute_metrics(run)


def run_vlm_on_dataset(run: VlmDatasetRun) -> dict:
    """Walk the dataset's images, asking the VLM about each one.

    Returns a summary dict; the caller records the terminal status.
    """
    name = run.dataset_name_snapshot
    directory = dataset_dir(name)
    if not directory.is_dir():
        raise RuntimeError(f"{name}: no dataset directory at {directory}")

    images = select_images(directory, run.image_limit)
    if not images:
        raise RuntimeError(f"{name}: no images found under {directory}")

    labels_dir = directory / datasets_svc.LABELS_SUBDIR
    classes = run.classes_snapshot or []
    matchers = grading.compile_matchers(classes, run.alias_map) if run.grade else []
    staging = _staging_path(run, directory)

    run.images_total = len(images)
    run.save(update_fields=["images_total"])

    config = run.model_config_snapshot or {}
    prompt = run.prompt_snapshot
    cancelled = False
    consecutive_errors = 0
    seq = 0

    def cancel_requested() -> bool:
        """Whether the Stop button has been pressed since the last image.

        Cooperative, and checked once per image whether that image produced an
        answer or an error — a run failing on every image still has to be
        stoppable. A call already in flight cannot be interrupted, which is why
        the HTTP client bounds every request with a timeout.
        """
        run.refresh_from_db(fields=["status"])
        return run.status == VlmDatasetRun.CANCEL_REQUESTED

    try:
        for image_path in images:
            seq += 1
            label_file = (
                datasets_svc.find_label_file(labels_dir, image_path.name)
                if labels_dir.is_dir() else None
            )
            gt_classes = (
                grading.ground_truth_classes(
                    label_file.read_text(encoding="utf-8"), classes,
                )
                if label_file is not None else []
            )

            result = VlmDatasetResult(
                run=run,
                seq=seq,
                image_filename=image_path.name,
                has_label=label_file is not None,
                gt_classes=gt_classes,
            )

            call_started = time.monotonic()
            try:
                answer = backend.infer(
                    _stage_for_backend(image_path, staging), config, prompt,
                )
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
            latency_ms = answer.get("latency_ms")
            if latency_ms is None:
                latency_ms = int((time.monotonic() - call_started) * 1000)
            result.text = (answer.get("text") or "").strip()
            result.latency_ms = latency_ms
            if run.grade:
                _grade_row(result, matchers, negation=run.negation_aware)
            result.save()

            run.images_processed = seq
            run.save(update_fields=["images_processed"])
            if run.grade and seq % METRICS_REFRESH_EVERY == 0:
                recompute_metrics(run)

            if cancel_requested():
                cancelled = True
                break
    finally:
        if staging is not None:
            for leftover in staging.parent.glob(f"{staging.name}.*"):
                leftover.unlink(missing_ok=True)
        # Score whatever was produced, cancelled or not — the answers are
        # already paid for, so they may as well be readable.
        recompute_metrics(run)

    return {
        "cancelled": cancelled,
        "images_total": run.images_total,
        "images_processed": run.images_processed,
        "errors": run.errors_count,
        "metrics": run.metrics,
        "finished_at": timezone.now(),
    }
