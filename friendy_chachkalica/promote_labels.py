"""Promote a model's saved predictions into a dataset's source ``labels/`` folder.

A finished eval (``eval_checkpoint.py``) or chachak pipeline (``chachak/run.py``)
writes a ``*_predictions.pt`` list of records — ``{image_path, label_path,
orig_size, predictions}`` where ``predictions`` is a Friendy ``(N, 6)`` tensor
``[x_center, y_center, width, height, confidence, class_id]`` normalized to each
image. This module turns those predictions into YOLO label ``.txt`` files under a
dataset's source ``labels/`` folder, so a model's output can become the dataset's
source-of-truth labels (the "auto-label" / "promote to labels" flow).

Two remaps make this correct rather than a blind dump:

* **Class space.** Prediction ``class_id``s are in the *model's* training class
  space (read from the checkpoint's ``train_dataset.classes``); source labels
  must be in the *dataset's* ``classes.txt`` space. Predictions are remapped by
  class *name*; a predicted class with no same-named dataset class is dropped
  (it has no representable index), mirroring how eval scoring remaps by name.
* **Confidence.** Only predictions at or above a caller-chosen confidence make it
  into the labels, matching the operating-point threshold the eval reported.

Reading the ``.pt`` needs torch (the Django app has none), so this runs in the
trainer service's environment behind the synchronous ``POST /promote_labels``
endpoint. The pure conversion (:func:`record_to_lines`) takes plain Python and is
torch-free so it can be unit-tested without loading a checkpoint.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


def _as_class_map(classes: Union[Dict, Sequence]) -> Dict[int, str]:
    """Normalize a class list/mapping to ``{index: name}`` (mirrors eval)."""
    if isinstance(classes, dict):
        return {int(k): str(v) for k, v in classes.items()}
    return {index: str(name) for index, name in enumerate(classes)}


def build_class_remap(
    train_classes: Union[Dict, Sequence],
    dataset_classes: Union[Dict, Sequence],
) -> Dict[int, int]:
    """Map a model's train class id to the dataset ``classes.txt`` index by name.

    Predicted ids whose class name is absent from the dataset space are simply
    omitted from the mapping, so the caller drops those predictions.
    """
    train_id_to_name = _as_class_map(train_classes)
    dataset_name_to_id = {name: idx for idx, name in _as_class_map(dataset_classes).items()}
    return {
        train_id: dataset_name_to_id[name]
        for train_id, name in train_id_to_name.items()
        if name in dataset_name_to_id
    }


def record_to_lines(
    predictions: Sequence[Sequence[float]],
    class_remap: Dict[int, int],
    score_threshold: float,
) -> tuple[list[str], int]:
    """Convert one record's ``(N, 6)`` predictions to YOLO label lines.

    ``predictions`` rows are ``[x_center, y_center, width, height, confidence,
    class_id]`` normalized. Returns ``(lines, dropped)`` where ``lines`` are
    ``"<dataset_class_id> <xc> <yc> <w> <h>"`` for kept boxes and ``dropped`` is
    the count of above-threshold boxes whose class had no dataset-space match.
    """
    lines: list[str] = []
    dropped = 0
    for row in predictions:
        xc, yc, w, h, conf, cls = (float(row[0]), float(row[1]), float(row[2]),
                                   float(row[3]), float(row[4]), int(round(float(row[5]))))
        if conf < score_threshold:
            continue
        mapped = class_remap.get(cls)
        if mapped is None:
            dropped += 1
            continue
        lines.append(f"{mapped} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, dropped


def _label_name(image_path: str, images_dir: Path) -> str:
    """Source-label filename for an image: ``<relative stem>.txt``.

    Uses the stripped convention (drop the image suffix) — the default source
    layout Label Studio project creation and training both read.
    """
    path = Path(image_path)
    try:
        relative = path.relative_to(images_dir)
    except ValueError:
        # image_path may already be relative, or under a differently-rooted mount;
        # fall back to the bare filename so we still land in labels/.
        relative = Path(path.name)
    return str(relative.with_suffix(".txt"))


def _backup_existing_labels(labels_dir: Path, backup_dir: Path) -> int:
    """Move existing ``labels/*.txt`` into ``backup_dir`` before overwriting.

    Returns the number of files backed up. Only ``.txt`` files are moved; any
    other content (e.g. nested backup folders) is left in place.
    """
    if not labels_dir.is_dir():
        return 0
    txts = sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    if not txts:
        return 0
    backup_dir.mkdir(parents=True, exist_ok=True)
    for txt in txts:
        shutil.move(str(txt), str(backup_dir / txt.name))
    return len(txts)


def promote_labels(
    predictions_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]],
    images_dir: Union[str, Path],
    dataset_classes: Union[Dict, Sequence],
    labels_dir: Union[str, Path],
    backup_dir: Union[str, Path],
    *,
    score_threshold: float = 0.25,
    prediction_classes: Optional[Union[Dict, Sequence]] = None,
) -> Dict[str, Any]:
    """Write a run's predictions into ``labels_dir`` as source-of-truth labels.

    Backs up any existing ``labels/*.txt`` into ``backup_dir`` first, then writes
    one ``.txt`` per image whose predictions survived the confidence cut and the
    class-name remap. Returns a summary of what was written.

    Pass either ``checkpoint_path`` (single-model run — its ``train_dataset.classes``
    is read from the checkpoint and remapped by name onto ``dataset_classes``) or
    ``prediction_classes`` (a combined multi-model run, whose saved predictions are
    already indexed in the eval dataset's own class space — see
    ``training.services.config_gen.build_promote_payload`` — so the remap is an
    identity lookup and no checkpoint needs loading). Exactly one should be given.
    """
    import torch  # trainer-env only; kept local so the module imports torch-free

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    if prediction_classes is not None:
        class_remap = build_class_remap(prediction_classes, dataset_classes)
    else:
        checkpoint_path = Path(checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu")
        train_classes_raw = (state.get("train_dataset") or {}).get("classes")
        if not train_classes_raw:
            raise ValueError(
                f"Checkpoint {checkpoint_path} does not record its training class names "
                "(train_dataset.classes), so predictions cannot be remapped by name onto "
                "the dataset classes."
            )
        class_remap = build_class_remap(train_classes_raw, dataset_classes)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    backup_dir = Path(backup_dir)

    records = torch.load(predictions_path, map_location="cpu")

    backed_up = _backup_existing_labels(labels_dir, backup_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = 0
    labels_written = 0
    boxes_written = 0
    dropped_unmapped = 0
    for record in records:
        image_path = record.get("image_path")
        if not image_path:
            continue
        images += 1
        predictions = record.get("predictions")
        rows = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions or [])
        lines, dropped = record_to_lines(rows, class_remap, score_threshold)
        dropped_unmapped += dropped
        out_path = labels_dir / _label_name(image_path, images_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write even when empty: a background-only image is a valid label (no boxes),
        # and an empty file keeps the image paired with the model's verdict.
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        labels_written += 1
        boxes_written += len(lines)

    return {
        "images": images,
        "labels_written": labels_written,
        "boxes_written": boxes_written,
        "dropped_unmapped": dropped_unmapped,
        "backed_up": backed_up,
        "labels_dir": str(labels_dir),
        "backup_dir": str(backup_dir) if backed_up else "",
        "score_threshold": score_threshold,
    }
