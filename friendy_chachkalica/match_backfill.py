"""Rebuild an eval's match table from the predictions it already saved.

``metrics.match_table`` is written at eval time (see
``ml/train.py::_write_match_table``), so an eval that finished before that
existed has metrics and a ``*_predictions.pt`` but no table — and without one
there is nothing to slice per annotation tag.

Nothing about the table needs the model. Every eval writes its per-image
predictions, and the ground truth is the same label files on disk, so the whole
artifact can be rebuilt from what is already there: no checkpoint is loaded, no
image is decoded, and the GPU is never touched. That is the difference between
lighting up an old eval in seconds and re-running inference over its dataset.

What it cannot do is invent an evaluation. The thresholds and class spaces must
be the ones that eval actually used, or the rebuilt table describes a different
evaluation than the metrics stored beside it — which is exactly why the caller
passes them in rather than this module guessing. The consumer checks the join:
re-scoring the unfiltered table has to reproduce the eval's own ``map50``.

Reading the ``.pt`` needs torch (the Django app has none), so this runs in the
trainer service behind the synchronous ``POST /match_table`` endpoint, like
:mod:`promote_labels` next to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

try:
    from .data import _read_yolo_label_file
    from .metrics import match_table
    from .ml.train import _atomic_write_json, match_table_path
except ImportError:  # run as a flat script
    from data import _read_yolo_label_file
    from metrics import match_table
    from ml.train import _atomic_write_json, match_table_path


def _as_class_map(classes: Union[Dict, Sequence]) -> Dict[int, str]:
    """Normalize a class list/mapping to ``{index: name}`` (mirrors eval)."""
    if isinstance(classes, dict):
        return {int(k): str(v) for k, v in classes.items()}
    return {index: str(name) for index, name in enumerate(classes)}


def _orig_size(record: Dict[str, Any]) -> tuple[int, int]:
    """``(height, width)`` from a record's stored ``orig_size``."""
    value = record.get("orig_size")
    if value is None:
        return 0, 0
    if torch.is_tensor(value):
        value = value.detach().cpu().flatten().tolist()
    values = list(value)
    return int(values[0]), int(values[1])


def rebuild_targets(records: List[Dict[str, Any]], labels_dir: Optional[str | Path] = None):
    """Re-read each record's ground truth into the target dicts eval scored against.

    Uses ``data._read_yolo_label_file`` — the same parser the eval dataloader
    used — on the ``label_path`` each record stored, so the boxes come back
    identical, in the same order, including the lines that parser skips. Order
    matters more than usual here: a ground-truth box's position in this list is
    the ``row`` the match table publishes, and the annotation tag sidecar keys
    its per-box answers on exactly that row.

    ``labels_dir``, when given, is a *fallback* for records whose recorded
    ``label_path`` no longer exists — an eval whose annotator output has since
    been promoted into the dataset's source labels, say, which moves the files
    out from under the path the eval recorded. The recorded path still wins
    when it resolves, so a relocation never quietly re-points an eval at
    different labels than it scored against.

    Returns ``(targets, missing)``: the target dicts, and how many records had
    no readable label file (an image that was genuinely unlabeled, or a path
    that no longer exists — the caller decides whether that is tolerable).
    """
    targets = []
    missing = 0
    for record in records:
        height, width = _orig_size(record)
        label_path = record.get("label_path")
        path = Path(label_path) if label_path else None
        if path is not None and labels_dir is not None and not path.exists():
            path = Path(labels_dir) / path.name
        if path is None or not path.exists():
            missing += 1
            boxes: list = []
            labels: list = []
        else:
            boxes, labels = _read_yolo_label_file(path, width, height)
        targets.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "image_path": record.get("image_path"),
            "label_path": str(path) if path is not None else None,
        })
    return targets, missing


def _train_classes_from_checkpoint(checkpoint_path: str | Path) -> Dict[int, str]:
    """The class space a checkpoint's predictions are indexed in.

    Read from the checkpoint rather than taken from the caller for the same
    reason ``eval_checkpoint`` reads it: it is the only record of the order the
    model actually learned, and a stale copy of the list would silently score
    every prediction against the wrong class instead of failing.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = (state.get("train_dataset") or {}).get("classes")
    if not raw:
        raise ValueError(
            f"{checkpoint_path} does not record its training class names "
            "(train_dataset.classes), so its saved predictions cannot be mapped "
            "onto the eval classes."
        )
    return _as_class_map(raw)


def backfill_match_table(
    predictions_path: str | Path,
    *,
    classes: Union[Dict, Sequence],
    prediction_classes: Optional[Union[Dict, Sequence]] = None,
    checkpoint_path: Optional[str] = None,
    labels_dir: Optional[str] = None,
    iou_thresholds: Optional[List[float]] = None,
    score_threshold: float = 0.25,
    map_score_threshold: Optional[float] = 0.001,
    operating_nms_threshold: Optional[float] = None,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Write the match table for an existing ``*_predictions.pt``.

    ``classes`` is the eval dataset's class space (the one the labels are in).
    The space the *saved* predictions are indexed in comes from exactly one of
    ``checkpoint_path`` (a single-model eval — its training classes are read
    back out of the checkpoint, as the eval itself did) or
    ``prediction_classes`` (a combined run, whose predictions were remapped
    into the eval space before being merged, so the lookup is an identity).
    Getting that wrong silently scores every prediction against the wrong
    class, which is why neither is defaulted.

    Returns a small summary; the table itself goes to ``output_path``
    (default: ``<split>_matches.json`` beside the predictions, where every
    reader looks for it).
    """
    predictions_path = Path(predictions_path)
    records = torch.load(predictions_path, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{predictions_path} holds no prediction records")

    targets, missing = rebuild_targets(records, labels_dir)
    if not any(int(target["labels"].numel()) for target in targets):
        raise ValueError(
            f"{predictions_path}: no ground-truth boxes found for any of its "
            f"{len(records)} records — a match table needs labels to match against."
        )

    eval_classes = _as_class_map(classes)
    if prediction_classes is not None:
        prediction_map = _as_class_map(prediction_classes)
    elif checkpoint_path:
        prediction_map = _train_classes_from_checkpoint(checkpoint_path)
    else:
        raise ValueError("either checkpoint_path or prediction_classes is required")

    table = match_table(
        [record.get("predictions") for record in records],
        targets,
        iou_thresholds=iou_thresholds,
        score_threshold=score_threshold,
        map_score_threshold=map_score_threshold,
        num_classes=len(eval_classes),
        prediction_classes=prediction_map,
        target_classes=eval_classes,
        eval_classes=eval_classes,
        operating_nms_threshold=operating_nms_threshold,
        image_names=[record.get("image_path") for record in records],
    )
    table["backfilled"] = True

    destination = Path(output_path) if output_path else match_table_path(predictions_path)
    _atomic_write_json(destination, table)
    return {
        "match_table_path": str(destination),
        "images": len(records),
        "images_without_labels": missing,
        "ground_truth_boxes": len(table["gt"]["image"]),
        "predictions": len(table["pred"]["image"]),
    }
