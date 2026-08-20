"""Delete a run/eval's on-disk artifacts when its database row is deleted.

Wired via ``post_delete`` signals (see :mod:`training.signals`) so it fires for
admin single deletes, bulk "delete selected", and cascades (deleting an
Experiment cascades to its runs) alike.

Guarded: only paths *under* the configured ``runs_root`` / ``configs_root`` (or,
for export artifacts written to an operator-chosen directory, under the project
root) are ever removed, so a blank, relative, or rogue path can never escalate
into deleting the project root or the whole filesystem.

Dataset rows are deliberately excluded — nothing here touches a Dataset's
source/target directory. Every other row with an on-disk artifact (training
run, run result, eval, trained model, export) is covered.
"""

import logging
import shutil
from pathlib import Path

from django.conf import settings

from training.models import TrainingSettings
from training.services.config_gen import _resolve

logger = logging.getLogger(__name__)


def _within(root: Path, path: Path) -> bool:
    """True if ``path`` is strictly inside ``root`` (never ``root`` itself)."""
    root, path = root.resolve(), path.resolve()
    if path == root:
        return False
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _rm_dir(path_str: str, root_str: str) -> None:
    if not path_str:
        return
    path, root = Path(path_str), _resolve(root_str)
    if not _within(root, path):
        logger.warning("cleanup: refusing to delete %s — not under %s", path, root)
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        logger.info("cleanup: removed %s", path)


def _rm_file(path_str: str, root_str: str) -> None:
    if not path_str:
        return
    path, root = Path(path_str), _resolve(root_str)
    if not _within(root, path):
        logger.warning("cleanup: refusing to delete %s — not under %s", path, root)
        return
    if path.is_file():
        path.unlink(missing_ok=True)
        logger.info("cleanup: removed %s", path)


def remove_run_artifacts(run) -> None:
    """Delete a training run's output dir and generated config YAML."""
    ts = TrainingSettings.load()
    _rm_dir(run.output_dir, ts.runs_root)
    _rm_file(run.config_yaml_path, ts.configs_root)


def remove_eval_artifacts(eval_run) -> None:
    """Delete an eval run's output dir and generated request YAML."""
    ts = TrainingSettings.load()
    _rm_dir(eval_run.output_dir, ts.runs_root)
    _rm_file(eval_run.request_yaml_path, ts.configs_root)


def remove_run_result_artifacts(run_result) -> None:
    """Delete one training result's own run subdirectory.

    ``run_dir`` is a subdir of its parent :class:`~training.models.TrainingRun`'s
    ``output_dir`` (see ``training.services.ingest``), exclusive to this result —
    no other row points at it. Deleting the parent run already removes it via
    :func:`remove_run_artifacts`; this covers a lone RunResult delete where the
    run itself survives.
    """
    ts = TrainingSettings.load()
    _rm_dir(run_result.run_dir, ts.runs_root)


def remove_trained_model_artifacts(trained_model) -> None:
    """Delete a promoted model's checkpoint file.

    ``checkpoint_path`` is a live reference into the source TrainingRun's own
    output_dir (``training.services.promote`` never copies it) rather than a
    file this row owns outright — but the model registry entry is still the
    unit of deletion this tab exposes, so its delete removes the checkpoint the
    same way every other non-Dataset tab removes what it points at.
    """
    ts = TrainingSettings.load()
    _rm_file(trained_model.checkpoint_path, ts.runs_root)


def remove_export_artifacts(export_run) -> None:
    """Delete an export's output artifact and assembled bundle dir.

    Both are usually written under ``exports_root``/``bundles_root``, but the
    export forms let an operator type any directory — so, unlike the other
    helpers here, this isn't guarded against one configured root but against
    the project root itself (``settings.BASE_DIR``): still refused outside the
    project, just not pinned to a subtree that isn't actually guaranteed.
    """
    _rm_file(export_run.output_path, settings.BASE_DIR)
    _rm_dir(export_run.bundle_dir, settings.BASE_DIR)
