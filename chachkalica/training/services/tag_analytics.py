"""Score an eval over slices of its dataset defined by annotation tags.

An eval answers one question — how good is this model on this dataset. The
dataset's annotation tags (``fleet.AnnotationTag``, answered by annotators and
synced out as ``annotation_tags.json`` next to the labels) split that dataset
into slices the headline number hides: rainy frames, night frames, occluded
boxes, small-and-far boxes, rainy *and* night. This module answers the same
question for any of them, with no second inference pass.

It works off two artifacts, joined on nothing more than filenames and row
numbers:

* the eval's ``*_matches.json`` — every prediction's and ground-truth box's
  match outcome, dumped once by the trainer (``metrics.match_table``);
* the dataset's ``annotation_tags.json`` — every image's and box's tag answers.

Why a table rather than re-running the eval per slice: matching is per-image
and threshold-independent up to the stored IoU, so dropping images from the set
cannot change any surviving prediction's verdict. Re-sorting the table's rows
by score and walking them reproduces the trainer's own AP accumulation, for the
whole set or any part of it — which is what makes an arbitrary intersection of
tags cost milliseconds instead of a GPU pass.

**The one thing this cannot do.** A frame tag selects images, so a slice of
frames is a whole little dataset and gets the full metric set. A box tag
selects *ground-truth boxes*, and a prediction carries no tags — an unmatched
false positive belongs to no box, so it cannot be attributed to "occluded".
Box-tag slices therefore report recall, misses, mean IoU and the score
distribution, and no precision or mAP. Reporting one would mean inventing an
attribution nobody measured.

The aggregation here is a second implementation of the trainer's accumulation,
which is a thing worth being nervous about. It checks itself: the unfiltered
slice must reproduce the eval's stored ``map50``/``precision``/``recall``, and
:func:`report` surfaces the comparison rather than hiding it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from fleet.reconcile import tag_values

#: What the trainer names the artifact (``metrics.match_table`` written by
#: ``ml/train.py::_write_match_table``): ``eval_matches.json`` for a base eval,
#: ``predictions_matches.json`` for a chachak pipeline eval.
MATCH_TABLE_SUFFIX = "_matches.json"

#: The bucket an image or box with no answer for a tag falls into. Named, not
#: dropped: "the model is worse on the frames nobody tagged" is a finding about
#: the annotation job, and silently excluding them would hide it.
UNANSWERED = "(unanswered)"

#: Free-text tags can have as many distinct answers as there are images, which
#: is not a grouping. Only the most common ones become groups.
MAX_TEXT_VALUES = 25

#: Slices smaller than this are still computed, but flagged: an AP over a
#: handful of boxes is noise with a decimal point on it.
SMALL_SLICE_BOXES = 30


def match_table_artifact(output_dir: str | Path | None) -> Path | None:
    """The one ``*_matches.json`` in an eval's output directory, or None.

    Located by suffix rather than by name for the same reason the hard-images
    viewer does it: a base eval writes ``eval_matches.json`` and a chachak
    pipeline eval ``predictions_matches.json``, and the caller should not have
    to know which kind of eval it is holding.
    """
    if not output_dir:
        return None
    directory = Path(output_dir)
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob(f"*{MATCH_TABLE_SUFFIX}")):
        return path
    return None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class MatchTable:
    """One eval's match outcomes, as numpy columns.

    Prediction rows arrive here already sorted by descending score (ties in the
    trainer's own concatenation order), because that is the order every metric
    is accumulated in and re-sorting per slice would be the same sort a hundred
    times. Ground-truth rows keep their natural order: they are joined by index
    from ``pred.gt``, and by (image, row) from the tag sidecar.
    """

    path: Path
    iou_thresholds: list[float]
    score_threshold: float
    score_floor: float
    backfilled: bool
    classes: dict[int, str]
    images: list[str]

    gt_image: np.ndarray
    gt_class: np.ndarray
    gt_row: np.ndarray

    # AP set: raw, NMS-free, down to the score floor.
    pred_image: np.ndarray
    pred_class: np.ndarray
    pred_score: np.ndarray
    pred_iou: np.ndarray
    pred_gt: np.ndarray

    # Operating set: the same rows unless the eval used an operating NMS
    # threshold, in which case precision/recall/F1 come from the suppressed set.
    op_class: np.ndarray
    op_image: np.ndarray
    op_score: np.ndarray
    op_iou: np.ndarray
    op_gt: np.ndarray
    operating_is_ap: bool

    #: ``{iou_threshold: (claim_score, claim_iou)}`` per ground-truth box,
    #: filled on first use — only box-tag slices need it.
    claims: dict = field(default_factory=dict)

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_gt(self) -> int:
        return int(self.gt_image.size)


def _columns(block: dict) -> tuple[np.ndarray, ...]:
    """Prediction columns, reordered into descending-score accumulation order."""
    score = np.asarray(block["score"], dtype=np.float64)
    order = _stable_score_order(score)
    return (
        np.asarray(block["image"], dtype=np.int64)[order],
        np.asarray(block["class"], dtype=np.int64)[order],
        score[order],
        np.asarray(block["iou"], dtype=np.float64)[order],
        np.asarray(block["gt"], dtype=np.int64)[order],
    )


def _stable_score_order(scores: np.ndarray) -> np.ndarray:
    """Indices sorting rows by descending score, ties in row order.

    Mergesort on the negated scores: stable, so equal scores keep the order the
    trainer concatenated them in (image order, original order within an image),
    which is the order its own ``argsort(..., stable=True)`` produces.
    """
    return np.argsort(-scores, kind="mergesort")


def _load_table(path: str, _stamp: tuple) -> MatchTable:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("version")
    if version != 1:
        raise ValueError(
            f"{path}: match table version {version!r} is not supported by this build."
        )

    pred_image, pred_class, pred_score, pred_iou, pred_gt = _columns(data["pred"])

    operating = data.get("pred_operating")
    if operating is None:
        op_image, op_class, op_score, op_iou, op_gt = (
            pred_image, pred_class, pred_score, pred_iou, pred_gt)
        operating_is_ap = True
    else:
        op_image, op_class, op_score, op_iou, op_gt = _columns(operating)
        operating_is_ap = False

    return MatchTable(
        path=Path(path),
        iou_thresholds=[float(value) for value in data["iou_thresholds"]],
        score_threshold=float(data["score_threshold"]),
        score_floor=float(data.get("score_floor", data["score_threshold"])),
        backfilled=bool(data.get("backfilled")),
        classes={int(key): (value or str(key)) for key, value in (data.get("classes") or {}).items()},
        images=[str(name) for name in data["images"]],
        gt_image=np.asarray(data["gt"]["image"], dtype=np.int64),
        gt_class=np.asarray(data["gt"]["class"], dtype=np.int64),
        gt_row=np.asarray(data["gt"]["row"], dtype=np.int64),
        pred_image=pred_image, pred_class=pred_class, pred_score=pred_score,
        pred_iou=pred_iou, pred_gt=pred_gt,
        op_image=op_image, op_class=op_class, op_score=op_score,
        op_iou=op_iou, op_gt=op_gt,
        operating_is_ap=operating_is_ap,
    )


@lru_cache(maxsize=4)
def _cached_table(path: str, stamp: tuple) -> MatchTable:
    return _load_table(path, stamp)


def load_table(path: str | Path) -> MatchTable:
    """Parse a match table, memoized on the file's identity.

    Keyed by (path, mtime, size) so a rebuilt table is picked up rather than
    served from the last parse — a backfill writes over the same name.
    """
    path = Path(path)
    stat = path.stat()
    return _cached_table(str(path), (stat.st_mtime_ns, stat.st_size))


def load_tag_document(labels_dir: str | Path | None) -> dict | None:
    """The ``annotation_tags.json`` sitting in a label directory, or None."""
    if not labels_dir:
        return None
    path = tag_values.tags_path(labels_dir)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


# --------------------------------------------------------------------------
# Tag masks
# --------------------------------------------------------------------------


def value_labels(raw) -> list[str]:
    """The grouping buckets one stored tag answer belongs to.

    The sidecar keeps each widget's natural type — a list for the four list
    widgets, an int for a rating, a string for free text. Grouping needs one
    vocabulary, so everything becomes strings here, and a multi-select answer
    lands in *every* value it picked: "weather = rain" means rain is among the
    answers, not that rain was the only one.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(value) for value in raw if str(value)]
    if isinstance(raw, bool):
        return ["yes" if raw else "no"]
    text = str(raw).strip()
    return [text] if text else []


@dataclass
class TagDefinition:
    name: str
    scope: str
    widget: str
    values: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    answered: int = 0
    total: int = 0
    truncated: bool = False

    @property
    def is_frame(self) -> bool:
        return self.scope != tag_values.REGION


@dataclass
class TagIndex:
    """Boolean masks for every (tag, value) pair, over images and over gt rows."""

    tags: list[TagDefinition]
    image_masks: dict[tuple[str, str], np.ndarray]
    gt_masks: dict[tuple[str, str], np.ndarray]
    matched_images: int
    unmatched_images: int
    row_mismatches: int

    def tag(self, name: str) -> TagDefinition | None:
        for definition in self.tags:
            if definition.name == name:
                return definition
        return None


def _order_values(definition: TagDefinition, declared: list[str]) -> None:
    """Settle a tag's value list: declared options first, then observed ones.

    Declared order is the order the annotator saw the options in, which is the
    order a reader expects the rows in. Anything observed but not declared (an
    option since removed from the tag, a rating, free text) follows, most
    common first, so a free-text tag degrades into its top answers instead of a
    thousand one-image groups.
    """
    seen = set()
    ordered: list[str] = []
    for value in declared:
        value = str(value)
        if value in definition.counts and value not in seen:
            ordered.append(value)
            seen.add(value)

    extras = sorted(
        (value for value in definition.counts if value not in seen and value != UNANSWERED),
        key=lambda value: (-definition.counts[value], value),
    )
    limit = MAX_TEXT_VALUES if definition.widget == "text" else len(extras)
    if len(extras) > limit:
        definition.truncated = True
    ordered.extend(extras[:limit])

    if definition.counts.get(UNANSWERED):
        ordered.append(UNANSWERED)
    definition.values = ordered


def build_tag_index(table: MatchTable, document: dict) -> TagIndex:
    """Turn the tag sidecar into masks aligned with the match table's rows.

    The join is by image basename (the sidecar keys images the way Label Studio
    named them, the table by the file the loader read) and, for box tags, by
    row — the box's line in its label file. Rows can only be trusted when the
    two files still describe the same boxes, so every box join is checked
    against the class id the sidecar recorded beside the row; a disagreement
    drops that image's box tags and is counted, rather than quietly attributing
    an answer to the wrong box.
    """
    declared = {str(spec["name"]): spec for spec in (document.get("tags") or [])}
    entries = document.get("images") or {}

    definitions: dict[str, TagDefinition] = {
        name: TagDefinition(
            name=name,
            scope=str(spec.get("scope") or tag_values.FRAME),
            widget=str(spec.get("widget") or "radio"),
        )
        for name, spec in declared.items()
    }

    image_index = {name: index for index, name in enumerate(table.images)}
    # Ground-truth rows indexed by (image, row) so a box tag can find its box
    # in one pass rather than scanning per image.
    gt_position = {
        (int(image), int(row)): position
        for position, (image, row) in enumerate(zip(table.gt_image, table.gt_row))
    }

    image_hits: dict[tuple[str, str], list[int]] = {}
    gt_hits: dict[tuple[str, str], list[int]] = {}
    frame_answered: dict[str, set[int]] = {name: set() for name in definitions}
    box_answered: dict[str, set[int]] = {name: set() for name in definitions}

    matched_images = 0
    row_mismatches = 0
    for filename, entry in entries.items():
        index = image_index.get(filename)
        if index is None:
            continue
        matched_images += 1

        for name, raw in (entry.get("frame") or {}).items():
            if name not in definitions:
                continue
            frame_answered[name].add(index)
            for value in value_labels(raw):
                image_hits.setdefault((name, value), []).append(index)

        boxes = entry.get("boxes") or []
        if not boxes:
            continue
        # One disagreement means the sidecar and the labels have drifted apart
        # (relabelled, re-promoted, hand-edited), and every row after it is
        # suspect too — so the whole image's box tags are dropped.
        positions = []
        drifted = False
        for box in boxes:
            position = gt_position.get((index, int(box.get("row", -1))))
            if position is None or int(table.gt_class[position]) != int(box.get("class_id", -1)):
                drifted = position is not None or drifted
                positions.append(None)
                continue
            positions.append(position)
        if drifted:
            row_mismatches += 1
            continue

        for box, position in zip(boxes, positions):
            if position is None:
                continue
            for name, raw in (box.get("tags") or {}).items():
                if name not in definitions:
                    continue
                box_answered[name].add(position)
                for value in value_labels(raw):
                    gt_hits.setdefault((name, value), []).append(position)

    image_masks: dict[tuple[str, str], np.ndarray] = {}
    gt_masks: dict[tuple[str, str], np.ndarray] = {}

    for name, definition in definitions.items():
        if definition.is_frame:
            definition.total = table.num_images
            definition.answered = len(frame_answered[name])
            for (tag_name, value), indices in image_hits.items():
                if tag_name != name:
                    continue
                mask = np.zeros(table.num_images, dtype=bool)
                mask[np.asarray(indices, dtype=np.int64)] = True
                image_masks[(name, value)] = mask
                definition.counts[value] = int(mask.sum())
            if definition.answered < definition.total:
                mask = np.ones(table.num_images, dtype=bool)
                if frame_answered[name]:
                    mask[np.asarray(sorted(frame_answered[name]), dtype=np.int64)] = False
                image_masks[(name, UNANSWERED)] = mask
                definition.counts[UNANSWERED] = int(mask.sum())
        else:
            definition.total = table.num_gt
            definition.answered = len(box_answered[name])
            for (tag_name, value), indices in gt_hits.items():
                if tag_name != name:
                    continue
                mask = np.zeros(table.num_gt, dtype=bool)
                mask[np.asarray(indices, dtype=np.int64)] = True
                gt_masks[(name, value)] = mask
                definition.counts[value] = int(mask.sum())
            if definition.answered < definition.total:
                mask = np.ones(table.num_gt, dtype=bool)
                if box_answered[name]:
                    mask[np.asarray(sorted(box_answered[name]), dtype=np.int64)] = False
                gt_masks[(name, UNANSWERED)] = mask
                definition.counts[UNANSWERED] = int(mask.sum())

        _order_values(definition, [str(v) for v in (declared[name].get("choices") or [])])

    order = {tag_values.FRAME: 0, tag_values.REGION: 1}
    tags = sorted(
        (d for d in definitions.values() if d.values),
        key=lambda d: (order.get(d.scope, 0), d.name),
    )
    return TagIndex(
        tags=tags,
        image_masks=image_masks,
        gt_masks=gt_masks,
        matched_images=matched_images,
        unmatched_images=len(entries) - matched_images,
        row_mismatches=row_mismatches,
    )


# --------------------------------------------------------------------------
# Scoring a slice
# --------------------------------------------------------------------------


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point interpolated AP — the trainer's ``_average_precision``, in numpy.

    Kept line-for-line equivalent (envelope by suffix maximum, area summed over
    the recall steps) rather than approximated with a 101-point COCO sweep: the
    numbers on this page sit next to the eval's own and have to mean the same
    thing.
    """
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    steps = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1]))


def _true_positives(ious: np.ndarray, gts: np.ndarray, threshold: float) -> np.ndarray:
    """1.0 for each prediction that wins a ground-truth box at ``threshold``.

    A prediction is a true positive when its best same-class box overlaps at or
    above the threshold *and* no higher-scoring prediction already claimed that
    box. Rows are in descending-score order, so the first row mentioning a box
    is its winner — which ``np.unique(..., return_index=True)`` returns,
    because it sorts stably when asked for indices.
    """
    true_positives = np.zeros(ious.size, dtype=np.float64)
    valid = np.nonzero(ious >= threshold)[0]
    if valid.size:
        _, first = np.unique(gts[valid], return_index=True)
        true_positives[valid[first]] = 1.0
    return true_positives


def _curve_stats(scores, ious, gts, gt_count: int, threshold: float, score_cut: float) -> dict:
    """AP over the whole curve, plus precision/recall at the operating cut."""
    if scores.size == 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0}

    true_positives = _true_positives(ious, gts, threshold)
    tp_cumulative = np.cumsum(true_positives)
    fp_cumulative = np.cumsum(1.0 - true_positives)
    precision_curve = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
    recall_curve = tp_cumulative / max(gt_count, 1)

    cut = int(np.count_nonzero(scores >= score_cut))
    if cut == 0:
        precision = recall = 0.0
    else:
        precision = float(precision_curve[cut - 1])
        recall = float(recall_curve[cut - 1]) if gt_count > 0 else 0.0

    return {
        "ap": _average_precision(recall_curve, precision_curve) if gt_count > 0 else 0.0,
        "precision": precision,
        "recall": recall,
    }


def _f1(precision: float, recall: float) -> float:
    total = precision + recall
    return float(2 * precision * recall / total) if total > 0 else 0.0


def _mean(values) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def image_slice_metrics(table: MatchTable, image_mask: np.ndarray) -> dict:
    """The full metric set for a subset of images.

    Mirrors ``metrics.evaluate_detection``: AP integrates the raw NMS-free set
    down to the score floor, while precision/recall/F1 and the counts are read
    at the operating confidence off the deployed (optionally NMS-suppressed)
    set. The mAP means skip classes with no ground truth in the slice, for the
    same reason the trainer's do — averaging a hard zero over a class the slice
    never contains deflates the number without saying anything about the model.
    """
    keep_pred = image_mask[table.pred_image]
    keep_op = image_mask[table.op_image] if not table.operating_is_ap else keep_pred
    keep_gt = image_mask[table.gt_image]
    cut = table.score_threshold

    gt_classes = table.gt_class[keep_gt]
    op_scores_all = table.op_score[keep_op]
    op_classes_all = table.op_class[keep_op]
    above = op_scores_all >= cut

    per_class = []
    ap50: list[float] = []
    ap5095: list[float] = []
    for class_id in sorted(table.classes):
        class_gt = int(np.count_nonzero(gt_classes == class_id))

        select = keep_pred & (table.pred_class == class_id)
        scores = table.pred_score[select]
        ious = table.pred_iou[select]
        gts = table.pred_gt[select]

        by_threshold = [
            _curve_stats(scores, ious, gts, class_gt, threshold, cut)
            for threshold in table.iou_thresholds
        ]
        primary_index = (table.iou_thresholds.index(0.5)
                         if 0.5 in table.iou_thresholds else 0)

        if table.operating_is_ap:
            operating = by_threshold[primary_index]
        else:
            op_select = keep_op & (table.op_class == class_id)
            operating = _curve_stats(
                table.op_score[op_select], table.op_iou[op_select], table.op_gt[op_select],
                class_gt, table.iou_thresholds[primary_index], cut,
            )

        class_ap50 = by_threshold[primary_index]["ap"]
        class_ap5095 = _mean(entry["ap"] for entry in by_threshold)
        if class_gt > 0:
            ap50.append(class_ap50)
            ap5095.append(class_ap5095)

        per_class.append({
            "class_id": class_id,
            "class_name": table.classes.get(class_id, str(class_id)),
            "ap50": class_ap50,
            "ap50_95": class_ap5095,
            "precision": operating["precision"],
            "recall": operating["recall"],
            "f1": _f1(operating["precision"], operating["recall"]),
            "gt": class_gt,
            "predictions": int(np.count_nonzero(above & (op_classes_all == class_id))),
        })

    # Micro precision/recall: one verdict per prediction across all classes at
    # once, which is what the eval reports as its headline precision/recall.
    gt_total = int(np.count_nonzero(keep_gt))
    micro_tp = _true_positives(
        table.op_iou[keep_op], table.op_gt[keep_op],
        table.iou_thresholds[0] if 0.5 not in table.iou_thresholds else 0.5,
    )
    predictions_above = int(np.count_nonzero(above))
    true_above = float(micro_tp[above].sum()) if predictions_above else 0.0
    precision = true_above / predictions_above if predictions_above else 0.0
    recall = true_above / gt_total if gt_total else 0.0

    return {
        "kind": "frame",
        "images": int(np.count_nonzero(image_mask)),
        "gt": gt_total,
        "predictions": predictions_above,
        "map50": _mean(ap50),
        "map50_95": _mean(ap5095),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "classes_with_gt": len(ap50),
        "per_class": per_class,
    }


def gt_claims(table: MatchTable, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Per ground-truth box: its winning prediction's score and IoU at ``threshold``.

    ``-1`` score means no prediction ever claimed it, at any confidence. Because
    matching runs in descending score order, "found at confidence *s*" is
    exactly "claim score >= *s*" — a cut cannot hand the box to a *different*
    prediction, only take its only claimant away. That is what lets one pass
    answer recall at every operating point.
    """
    key = round(float(threshold), 6)
    cached = table.claims.get(key)
    if cached is not None:
        return cached

    score = np.full(table.num_gt, -1.0)
    iou = np.zeros(table.num_gt)
    valid = np.nonzero(table.op_iou >= threshold)[0]
    if valid.size:
        _, first = np.unique(table.op_gt[valid], return_index=True)
        winners = valid[first]
        score[table.op_gt[winners]] = table.op_score[winners]
        iou[table.op_gt[winners]] = table.op_iou[winners]
    table.claims[key] = (score, iou)
    return score, iou


def gt_slice_metrics(table: MatchTable, gt_mask: np.ndarray) -> dict:
    """What a subset of *ground-truth boxes* can honestly be scored on.

    Recall, misses, and how well the boxes that were found were found (IoU,
    confidence). No precision and no AP: a false positive is a prediction, and
    a prediction has no tags, so there is no way to say which slice it belongs
    to. See this module's docstring.
    """
    cut = table.score_threshold
    primary = 0.5 if 0.5 in table.iou_thresholds else table.iou_thresholds[0]
    score, iou = gt_claims(table, primary)

    selected = np.nonzero(gt_mask)[0]
    total = int(selected.size)
    found_mask = score[selected] >= cut
    found = int(np.count_nonzero(found_mask))

    per_class = []
    classes = table.gt_class[selected]
    for class_id in sorted(table.classes):
        in_class = classes == class_id
        class_total = int(np.count_nonzero(in_class))
        if not class_total:
            continue
        class_found = int(np.count_nonzero(found_mask & in_class))
        per_class.append({
            "class_id": class_id,
            "class_name": table.classes.get(class_id, str(class_id)),
            "gt": class_total,
            "found": class_found,
            "missed": class_total - class_found,
            "recall": class_found / class_total,
        })

    recall_by_iou = []
    for threshold in table.iou_thresholds:
        threshold_score, _ = gt_claims(table, threshold)
        hits = int(np.count_nonzero(threshold_score[selected] >= cut))
        recall_by_iou.append({"iou": threshold, "recall": hits / total if total else 0.0})

    return {
        "kind": "box",
        "gt": total,
        "found": found,
        "missed": total - found,
        "recall": found / total if total else 0.0,
        "recall_50_95": _mean(entry["recall"] for entry in recall_by_iou),
        "recall_by_iou": recall_by_iou,
        "mean_iou": float(iou[selected][found_mask].mean()) if found else 0.0,
        "mean_score": float(score[selected][found_mask].mean()) if found else 0.0,
        "per_class": per_class,
        "small": total < SMALL_SLICE_BOXES,
    }


# --------------------------------------------------------------------------
# Filters, cross-tabs, and the report
# --------------------------------------------------------------------------

#: Metrics a slice can be ranked or shaded by, per slice kind. A box-tag slice
#: is deliberately short: see the module docstring on why precision and mAP are
#: not on it.
FRAME_METRICS = [
    ("map50", "mAP50"),
    ("map50_95", "mAP50-95"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("f1", "F1"),
]
BOX_METRICS = [
    ("recall", "recall"),
    ("recall_50_95", "recall (mean over IoU)"),
    ("mean_iou", "mean IoU of found boxes"),
    ("mean_score", "mean confidence of found boxes"),
]


class UnknownClause(ValueError):
    """A filter names a tag/value pair this eval's data has no mask for."""


def resolve_clauses(table: MatchTable, index: TagIndex, clauses) -> tuple[np.ndarray, np.ndarray, str]:
    """Intersect ``[(tag, value), ...]`` into (image mask, gt mask, slice kind).

    Clauses are ANDed — that is what a cross-union of tags means here. Mixing a
    frame clause with a box clause is allowed and useful ("occluded boxes, in
    rainy frames"): the frame clause narrows the images, which narrows the
    boxes. The moment any box clause appears the slice is a set of boxes, and
    the kind drops to ``box`` — the metrics that survive are the ground-truth
    side ones.
    """
    image_mask = np.ones(table.num_images, dtype=bool)
    gt_mask = np.ones(table.num_gt, dtype=bool)
    kind = "frame"

    for tag, value in clauses:
        key = (str(tag), str(value))
        if key in index.image_masks:
            image_mask = image_mask & index.image_masks[key]
        elif key in index.gt_masks:
            gt_mask = gt_mask & index.gt_masks[key]
            kind = "box"
        else:
            raise UnknownClause(f"no boxes or images carry {tag} = {value}")

    # A box only counts when its image survives the frame clauses.
    gt_mask = gt_mask & image_mask[table.gt_image]
    return image_mask, gt_mask, kind


def slice_metrics(table: MatchTable, index: TagIndex, clauses) -> dict:
    """Score one filter. Frame-only filters get everything; box filters get recall."""
    image_mask, gt_mask, kind = resolve_clauses(table, index, clauses)
    if kind == "frame":
        result = image_slice_metrics(table, image_mask)
        result["boxes"] = gt_slice_metrics(table, gt_mask)
        return result
    result = gt_slice_metrics(table, gt_mask)
    result["images"] = int(np.count_nonzero(image_mask))
    return result


def _row(table: MatchTable, index: TagIndex, definition: TagDefinition, value: str) -> dict:
    """One value's row in a tag's breakdown table."""
    metrics = slice_metrics(table, index, [(definition.name, value)])
    return {"value": value, "count": definition.counts.get(value, 0), **metrics}


def tag_breakdown(table: MatchTable, index: TagIndex, definition: TagDefinition) -> dict:
    """Every value of one tag, scored, with the gap against the whole dataset."""
    overall = (image_slice_metrics(table, np.ones(table.num_images, dtype=bool))
               if definition.is_frame
               else gt_slice_metrics(table, np.ones(table.num_gt, dtype=bool)))
    key = "map50" if definition.is_frame else "recall"

    rows = [_row(table, index, definition, value) for value in definition.values]
    for row in rows:
        row["delta"] = row.get(key, 0.0) - overall.get(key, 0.0)

    return {
        "name": definition.name,
        "scope": definition.scope,
        "widget": definition.widget,
        "kind": "frame" if definition.is_frame else "box",
        "answered": definition.answered,
        "total": definition.total,
        "truncated": definition.truncated,
        "headline_metric": key,
        "rows": rows,
        "overall": overall,
    }


def cross_tab(table: MatchTable, index: TagIndex, tag_a: str, tag_b: str, metric: str) -> dict:
    """One metric over every combination of two tags' values.

    The intersection, not the union: a cell is images (or boxes) answering both
    ``tag_a = row`` and ``tag_b = column``. Empty combinations are kept as
    blanks rather than dropped, because "no rainy night frames exist" is itself
    worth seeing in the grid.
    """
    first = index.tag(tag_a)
    second = index.tag(tag_b)
    if first is None or second is None:
        raise UnknownClause(f"unknown tag in cross-tab: {tag_a} x {tag_b}")

    kind = "frame" if (first.is_frame and second.is_frame) else "box"
    allowed = dict(FRAME_METRICS if kind == "frame" else BOX_METRICS)
    if metric not in allowed:
        metric = "map50" if kind == "frame" else "recall"

    rows = []
    values: list[float] = []
    for row_value in first.values:
        cells = []
        for column_value in second.values:
            entry = slice_metrics(
                table, index, [(tag_a, row_value), (tag_b, column_value)]
            )
            population = entry["images"] if kind == "frame" else entry["gt"]
            value = entry.get(metric) if population else None
            if value is not None:
                values.append(float(value))
            cells.append({
                "value": value,
                "population": population,
                "gt": entry["gt"],
                "row": row_value,
                "column": column_value,
            })
        rows.append({"label": row_value, "cells": cells})

    low = min(values) if values else 0.0
    high = max(values) if values else 1.0
    span = (high - low) or 1.0
    for row in rows:
        for cell in row["cells"]:
            cell["intensity"] = (
                None if cell["value"] is None else round((cell["value"] - low) / span, 4)
            )

    return {
        "tag_a": tag_a,
        "tag_b": tag_b,
        "metric": metric,
        "metric_label": allowed[metric],
        "kind": kind,
        "columns": second.values,
        "rows": rows,
        "population_label": "images" if kind == "frame" else "boxes",
    }


def _reproduction_check(computed: dict, stored: dict | None) -> dict | None:
    """Compare the unfiltered slice against the eval's own stored metrics.

    This page re-implements the trainer's accumulation, so it owes the reader
    proof that it agrees with it. With no filter applied the two are computing
    the same thing over the same rows and must land on the same number; a gap
    means this module is wrong, not the eval, and the page says so instead of
    quietly presenting slices built on a broken aggregation.
    """
    if not isinstance(stored, dict):
        return None
    checks = []
    worst = 0.0
    for key, label in (("map50", "mAP50"), ("map50_95", "mAP50-95"),
                       ("precision", "precision"), ("recall", "recall")):
        expected = stored.get(key)
        if not isinstance(expected, (int, float)):
            continue
        actual = float(computed.get(key, 0.0))
        difference = abs(actual - float(expected))
        worst = max(worst, difference)
        checks.append({"label": label, "stored": float(expected),
                       "computed": actual, "difference": difference})
    if not checks:
        return None
    return {"checks": checks, "worst": worst, "agrees": worst <= 1e-4}


def report(table: MatchTable, index: TagIndex, stored_metrics: dict | None = None) -> dict:
    """Everything the tag analytics page renders on first load."""
    overall_frame = image_slice_metrics(table, np.ones(table.num_images, dtype=bool))
    overall_box = gt_slice_metrics(table, np.ones(table.num_gt, dtype=bool))

    breakdowns = [tag_breakdown(table, index, definition) for definition in index.tags]
    frame_tags = [d.name for d in index.tags if d.is_frame]
    box_tags = [d.name for d in index.tags if not d.is_frame]

    return {
        "overall": overall_frame,
        "overall_boxes": overall_box,
        "breakdowns": breakdowns,
        "frame_tags": frame_tags,
        "box_tags": box_tags,
        "tags": [
            {
                "name": d.name, "kind": "frame" if d.is_frame else "box",
                "values": d.values, "counts": d.counts,
                "answered": d.answered, "total": d.total,
            }
            for d in index.tags
        ],
        "coverage": {
            "images": table.num_images,
            "gt": table.num_gt,
            "images_with_tags": index.matched_images,
            "images_not_in_eval": index.unmatched_images,
            "row_mismatches": index.row_mismatches,
        },
        "score_threshold": table.score_threshold,
        "score_floor": table.score_floor,
        "iou_thresholds": table.iou_thresholds,
        "operating_is_ap": table.operating_is_ap,
        "backfilled": table.backfilled,
        "frame_metrics": FRAME_METRICS,
        "box_metrics": BOX_METRICS,
        "check": _reproduction_check(overall_frame, stored_metrics),
    }
