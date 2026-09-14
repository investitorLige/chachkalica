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

#: Match table versions this build can read. Version 2 added ``gt.w``/``gt.h``
#: (each box's extent as a fraction of its frame); version 1 carries no
#: geometry and simply leaves the size tag unavailable, which is a sentence on
#: the page rather than a refusal to render it.
SUPPORTED_MATCH_TABLE_VERSIONS = (1, 2)

#: The bucket an image or box with no answer for a tag falls into. Named, not
#: dropped: "the model is worse on the frames nobody tagged" is a finding about
#: the annotation job, and silently excluding them would hide it.
UNANSWERED = "(unanswered)"

#: The bucket a measured tag's image or box falls into when nothing measured
#: it. Distinct from UNANSWERED: nobody was ever asked, so "the annotators
#: skipped these" would be the wrong story.
UNMEASURED = "(not measured)"

#: The computed tags, named in one place. The ``auto:`` prefix cannot collide
#: with an annotator's tag: ``fleet.AnnotationTag.NAME_PATTERN`` forbids ":", so
#: this is collision-proof by construction rather than by convention -- which
#: matters, because the masks are keyed by name and a collision would silently
#: merge a measured slice into an answered one.
CROWDING_TAG = "auto:crowding"
BOX_SIZE_TAG = "auto:box size"
BRIGHTNESS_TAG = "auto:brightness"
CONTRAST_TAG = "auto:contrast"
POSE_TAG = "auto:pose"
PERSON_SIZE_TAG = "auto:person size"

#: The bucket a box gets when the pass ran and found nobody around it. Kept
#: distinct from UNMEASURED, which means the pass never looked: "the posture
#: model could not see a person here" is an observation, and usually the
#: interesting one.
NO_PERSON = "(no person)"

#: Measured tags that come from the dataset's image-measures sidecar, mapped to
#: the key they are stored under there.
IMAGE_MEASURE_TAGS = [
    (BRIGHTNESS_TAG, "brightness", "brightness", "mean luma of the frame, 0-255"),
    (CONTRAST_TAG, "contrast", "contrast",
     "the frame's 5th-95th percentile luma spread, 0-255"),
]

#: What the computed tags are, for the surfaces that have to describe them
#: before any eval exists -- the evaluate forms. Kept beside the builders below
#: so the description and the thing described cannot drift apart.
AUTO_TAG_CATALOGUE = [
    {
        "name": CROWDING_TAG, "label": "crowding", "scope": tag_values.FRAME,
        "detail": "How many ground-truth boxes share the frame.",
        "needs": "",
    },
    {
        "name": BOX_SIZE_TAG, "label": "box size", "scope": tag_values.REGION,
        "detail": "Each ground-truth box's linear size as a fraction of its frame.",
        "needs": "a match table that records box geometry -- written by every "
                 "eval from now on; an older eval needs 'Build tag analytics data'.",
    },
    {
        "name": BRIGHTNESS_TAG, "label": "brightness", "scope": tag_values.FRAME,
        "detail": "How bright the frame is, as mean luma.",
        "needs": "one 'Measure image statistics' pass over the dataset.",
    },
    {
        "name": PERSON_SIZE_TAG, "label": "person size", "scope": tag_values.REGION,
        "detail": "How big the person this box belongs to is, as a fraction of "
                  "the frame.",
        "needs": "one 'Measure box statistics' pass over this label set.",
    },
    {
        "name": POSE_TAG, "label": "pose", "scope": tag_values.REGION,
        "detail": "What the person this box belongs to is doing -- standing, "
                  "sitting or lying.",
        "needs": "one 'Measure box statistics' pass over this label set.",
    },
    {
        "name": CONTRAST_TAG, "label": "contrast", "scope": tag_values.FRAME,
        "detail": "How much tonal range the frame has -- low means washed out, "
                  "fogged or badly exposed.",
        "needs": "one 'Measure image statistics' pass over the dataset.",
    },
]

#: Buckets a continuous measure is cut into. Three, because low/medium/high is
#: the coarsest split that can show a trend rather than only a contrast.
TERCILE_LABELS = ("low", "medium", "high")

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
    version: int
    iou_thresholds: list[float]
    score_threshold: float
    score_floor: float
    backfilled: bool
    classes: dict[int, str]
    images: list[str]

    gt_image: np.ndarray
    gt_class: np.ndarray
    gt_row: np.ndarray
    #: Box extent as a fraction of its frame, ``-1`` where the table does not
    #: know it (a version 1 table, or a record that never stored a frame size).
    gt_w: np.ndarray
    gt_h: np.ndarray

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
    def has_geometry(self) -> bool:
        """Whether box sizes can be read off this table at all.

        False for a version 1 table, and for a version 2 one whose every box
        came from a record with no frame size to divide by.
        """
        return bool(self.gt_w.size) and bool(np.any(self.gt_w >= 0))

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
    if version not in SUPPORTED_MATCH_TABLE_VERSIONS:
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

    gt = data["gt"]
    gt_image = np.asarray(gt["image"], dtype=np.int64)
    gt_class = np.asarray(gt["class"], dtype=np.int64)
    gt_row = np.asarray(gt["row"], dtype=np.int64)
    if "w" in gt and "h" in gt:
        gt_w = np.asarray(gt["w"], dtype=np.float64)
        gt_h = np.asarray(gt["h"], dtype=np.float64)
    else:
        gt_w = np.full(gt_image.size, -1.0)
        gt_h = np.full(gt_image.size, -1.0)
    # The ground-truth columns are read positionally: a short one would pair a
    # class with another box's row and quietly mis-tag every box after it, so
    # check the lengths rather than let zip() walk off the end of the shortest.
    for name, column in (("class", gt_class), ("row", gt_row),
                         ("w", gt_w), ("h", gt_h)):
        if column.size != gt_image.size:
            raise ValueError(
                f"{path}: gt.{name} has {column.size} entries but gt.image has "
                f"{gt_image.size} -- the table is truncated or hand-edited."
            )

    return MatchTable(
        path=Path(path),
        version=int(version),
        iou_thresholds=[float(value) for value in data["iou_thresholds"]],
        score_threshold=float(data["score_threshold"]),
        score_floor=float(data.get("score_floor", data["score_threshold"])),
        backfilled=bool(data.get("backfilled")),
        classes={int(key): (value or str(key)) for key, value in (data.get("classes") or {}).items()},
        images=[str(name) for name in data["images"]],
        gt_image=gt_image, gt_class=gt_class, gt_row=gt_row,
        gt_w=gt_w, gt_h=gt_h,
        pred_image=pred_image, pred_class=pred_class, pred_score=pred_score,
        pred_iou=pred_iou, pred_gt=pred_gt,
        op_image=op_image, op_class=op_class, op_score=op_score,
        op_iou=op_iou, op_gt=op_gt,
        operating_is_ap=operating_is_ap,
    )


@lru_cache(maxsize=16)
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


@lru_cache(maxsize=4)
def _cached_document(path: str, stamp: tuple) -> dict | None:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


BOX_MEASURES_FILENAME = "auto_measures.json"


def load_box_measures(labels_dir: str | Path | None) -> dict | None:
    """The box-measure sidecar in a label directory, or None.

    Lives beside the labels rather than beside the images, and for the same
    reason ``annotation_tags.json`` does: its rows are line numbers in *those*
    ``.txt`` files, so it is only meaningful next to them.
    """
    if not labels_dir:
        return None
    path = Path(labels_dir) / BOX_MEASURES_FILENAME
    try:
        stat = path.stat()
    except OSError:
        return None
    document = _cached_document(str(path), (stat.st_mtime_ns, stat.st_size))
    return document if isinstance(document, dict) else None


def load_tag_document(labels_dir: str | Path | None) -> dict | None:
    """The ``annotation_tags.json`` sitting in a label directory, or None.

    Memoized on the file's identity the way :func:`load_table` is, and for the
    same reason at a larger scale: the sidecar for a 13k-image dataset is
    several MB of JSON, and it is now read on every tag-analytics render *and*
    on every keystroke-driven refresh of the evaluate forms' tag panel. Keyed by
    (path, mtime, size) so a re-sync is picked up rather than served stale.

    The cached document is shared between callers, so treat it as read-only --
    every consumer here copies before it changes anything.
    """
    if not labels_dir:
        return None
    path = tag_values.tags_path(labels_dir)
    try:
        stat = path.stat()
    except OSError:
        return None
    return _cached_document(str(path), (stat.st_mtime_ns, stat.st_size))


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

    #: Display name. Empty means "use ``name``"; a computed tag sets it to the
    #: half of its name the reader cares about (``auto:crowding`` -> crowding).
    label: str = ""
    #: Where this tag's answers come from. ``annotator`` was answered by a
    #: person, ``auto`` was measured, ``declared`` is defined on the dataset but
    #: has no answers here at all.
    source: str = "annotator"
    #: ``ok`` when the tag can slice. Anything else is a reason the page prints
    #: instead of a table, so a tag never simply vanishes.
    status: str = "ok"
    status_detail: str = ""
    #: Where a measured tag's buckets were cut, as
    #: ``[{"value", "min", "max", "count", "share"}]``. Printed under the tag,
    #: because a bucket named "low" means nothing without the number behind it.
    cuts: list[dict] = field(default_factory=list)
    unit: str = ""
    #: The bucket unanswered/unmeasured entries land in — UNANSWERED for a tag a
    #: person fills in, UNMEASURED for one a machine does.
    missing_label: str = UNANSWERED

    @property
    def is_frame(self) -> bool:
        return self.scope != tag_values.REGION

    @property
    def display(self) -> str:
        return self.label or self.name

    @property
    def usable(self) -> bool:
        """Whether this tag can actually slice anything on this eval."""
        return self.status == "ok" and bool(self.values)


@dataclass
class TagIndex:
    """Boolean masks for every (tag, value) pair, over images and over gt rows."""

    tags: list[TagDefinition]
    image_masks: dict[tuple[str, str], np.ndarray]
    gt_masks: dict[tuple[str, str], np.ndarray]
    matched_images: int
    unmatched_images: int
    row_mismatches: int
    #: Images the measures sidecar could not be joined to because their basename
    #: occurs under more than one subdirectory. Reported rather than guessed.
    ambiguous_filenames: int = 0

    def tag(self, name: str) -> TagDefinition | None:
        for definition in self.tags:
            if definition.name == name:
                return definition
        return None


def tercile_buckets(values, valid=None) -> dict:
    """Split a continuous measure into up to three ordered, non-empty buckets.

    Returns ``{"assignment", "labels", "cuts", "degenerate"}``. ``assignment``
    indexes ``labels``, or is ``-1`` where the measure has no value.

    Cuts fall *between* distinct values, never through a tie -- "1 box" cannot
    be simultaneously low and medium. That matters far more than it sounds,
    because the measures worth slicing by are often discrete: on a dataset where
    80% of frames hold exactly one box the 33rd and 66th percentiles are both
    1.0, so a naive tercile leaves the middle bucket empty and the page renders
    a blank row that reads like a bug. Cutting between distinct values instead
    gives low={1}, medium={2}, high={3 or more} -- three populated buckets that
    describe themselves.

    On a genuinely continuous measure every value is distinct, the candidate
    cuts are a fine grid, and this lands where ``np.quantile(v, [1/3, 2/3])``
    lands. So it is the percentile cut generalised to survive ties, not a
    different rule.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.ones(values.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    assignment = np.full(values.shape, -1, dtype=np.int8)

    sample = values[valid]
    if sample.size == 0:
        return {"assignment": assignment, "labels": [], "cuts": [], "degenerate": "empty"}

    distinct, counts = np.unique(sample, return_counts=True)
    if distinct.size == 1:
        assignment[valid] = 0
        labels = [TERCILE_LABELS[0]]
        return {
            "assignment": assignment,
            "labels": labels,
            "cuts": [{"value": labels[0], "min": None, "max": float(distinct[0]),
                      "count": int(counts[0]), "share": 1.0}],
            "degenerate": "constant",
        }

    # Cumulative share at each distinct value; a cut "after distinct[i]" puts
    # cum[i] of the population below it, so pick the i closest to each target.
    cum = np.cumsum(counts) / counts.sum()
    low_index = int(np.argmin(np.abs(cum[:-1] - 1.0 / 3.0)))
    remaining = np.arange(low_index + 1, distinct.size - 1)
    if remaining.size:
        high_index = int(remaining[np.argmin(np.abs(cum[remaining] - 2.0 / 3.0))])
        edges = [float(distinct[low_index]), float(distinct[high_index])]
        labels = list(TERCILE_LABELS)
        degenerate = ""
    else:
        # Only one place left to cut. Two named buckets beat three with an empty
        # one in the middle, so skip "medium" rather than render nothing in it.
        edges = [float(distinct[low_index])]
        labels = [TERCILE_LABELS[0], TERCILE_LABELS[-1]]
        degenerate = "two"

    # side="left" makes each edge the inclusive top of its bucket.
    assignment[valid] = np.searchsorted(edges, sample, side="left").astype(np.int8)

    cuts = []
    for position, label in enumerate(labels):
        selected = assignment == position
        count = int(np.count_nonzero(selected))
        cuts.append({
            "value": label,
            "min": edges[position - 1] if position else None,
            "max": edges[position] if position < len(edges) else None,
            "count": count,
            "share": count / int(sample.size),
        })
    return {"assignment": assignment, "labels": labels, "cuts": cuts,
            "degenerate": degenerate}


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
        if value in seen:
            continue
        # A declared option nobody ever picked becomes a zero row rather than
        # nothing at all: "no annotator marked a single frame 'heavy'" is a
        # finding about the labelling job, and dropping it hides it.
        definition.counts.setdefault(value, 0)
        ordered.append(value)
        seen.add(value)

    extras = sorted(
        (value for value in definition.counts
         if value not in seen and value != definition.missing_label),
        key=lambda value: (-definition.counts[value], value),
    )
    limit = MAX_TEXT_VALUES if definition.widget == "text" else len(extras)
    if len(extras) > limit:
        definition.truncated = True
    ordered.extend(extras[:limit])

    if definition.counts.get(definition.missing_label):
        ordered.append(definition.missing_label)
    definition.values = ordered


def _measured_tag(*, name, label, unit, scope, values, valid) -> tuple[TagDefinition, dict]:
    """One computed tag: bucket a raw measure and build its masks.

    Measured tags carry no annotator answers, so their "missing" bucket is
    UNMEASURED rather than UNANSWERED -- nobody was asked, so saying the
    annotators skipped these would be the wrong story.
    """
    definition = TagDefinition(
        name=name, scope=scope, widget="radio", label=label,
        source="auto", unit=unit, missing_label=UNMEASURED,
    )
    result = tercile_buckets(values, valid)
    assignment = result["assignment"]
    definition.cuts = result["cuts"]
    definition.total = int(assignment.size)
    definition.answered = int(np.count_nonzero(assignment >= 0))

    masks: dict[tuple[str, str], np.ndarray] = {}
    ordered: list[str] = []
    for position, bucket in enumerate(result["labels"]):
        mask = assignment == position
        masks[(name, bucket)] = mask
        definition.counts[bucket] = int(np.count_nonzero(mask))
        ordered.append(bucket)

    missing = assignment < 0
    if bool(missing.any()):
        masks[(name, UNMEASURED)] = missing
        definition.counts[UNMEASURED] = int(np.count_nonzero(missing))
        ordered.append(UNMEASURED)
    definition.values = ordered

    if result["degenerate"] == "empty":
        definition.status = "no_data"
        definition.status_detail = "Nothing here carries this measure."
    elif result["degenerate"] == "constant":
        # One bucket is not a slice. Say so rather than render a single row
        # holding 100% of the dataset, which looks like a result.
        definition.status = "collapsed"
        definition.status_detail = (
            "Every entry has the same value, so this measure cannot split "
            "this dataset into anything."
        )
    elif result["degenerate"] == "two":
        definition.status_detail = (
            "This measure takes too few distinct values here to cut in three, "
            "so it is split in two."
        )
    return definition, masks


def _crowding_tag(table: MatchTable) -> tuple[TagDefinition, dict]:
    """How many ground-truth boxes share the frame.

    Needs no sidecar and no second pass -- the match table already has one row
    per ground-truth box, so counting them per image is the whole measure. That
    is what lets tag analytics slice an eval of a dataset nobody has tagged.

    A frame with no ground truth counts 0 and stays in the population: it is a
    real frame, it is where false positives live, and folding it into "not
    measured" would hide the most interesting slice on the page.
    """
    counts = np.bincount(table.gt_image, minlength=table.num_images).astype(np.float64)
    return _measured_tag(
        name=CROWDING_TAG, label="crowding",
        unit="ground-truth boxes this eval scored in the frame",
        scope=tag_values.FRAME, values=counts,
        valid=np.ones(counts.shape, dtype=bool),
    )


def _box_size_tag(table: MatchTable) -> tuple[TagDefinition, dict]:
    """How big each ground-truth box is relative to its frame.

    The linear fraction (sqrt of the area fraction), not the area fraction:
    halving a box's size should read as half, and area falls by four.
    """
    valid = (table.gt_w >= 0) & (table.gt_h >= 0)
    values = np.sqrt(np.clip(table.gt_w, 0.0, None) * np.clip(table.gt_h, 0.0, None))
    return _measured_tag(
        name=BOX_SIZE_TAG, label="box size",
        unit="the box's linear size as a fraction of the frame",
        scope=tag_values.REGION, values=values, valid=valid,
    )


#: Why a computed tag is not available, in the operator's terms -- each names
#: the action that fixes it rather than only the artifact that is missing.
NO_BOX_MEASURES_DETAIL = (
    "The boxes in this label set have not been measured. Run 'Measure box "
    "statistics' -- it runs a person detector and the posture engine over the "
    "frames once, and every eval scored against these labels picks it up."
)

NO_MEASURES_DETAIL = (
    "This dataset's images have not been measured. Run 'Measure image "
    "statistics' on the dataset -- one pass over the frames, no GPU and no "
    "re-inference, and every eval of it (including ones already finished) picks "
    "the result up."
)

NO_GEOMETRY_DETAIL = (
    "This eval's match table records no box geometry (it predates it). Select "
    "the eval and run 'Build tag analytics data' -- it rebuilds the table "
    "from the predictions already on disk, with no GPU and no re-inference."
)


DECLARED_NOT_SYNCED_DETAIL = (
    "Defined on this dataset, but absent from the tag answers this eval scored "
    "against -- it was added or renamed after the last sync. Push the tags to "
    "the annotator's projects and sync them."
)

NO_ANSWERS_DETAIL = (
    "Synced, but no image this eval scored carries an answer for it yet."
)


def _measures_by_basename(document: dict) -> tuple[dict, set]:
    """The sidecar's per-image values, keyed the only way a table can join.

    The sidecar keys by path relative to the images directory, because that is
    what uniquely identifies a frame; a match table records only basenames
    (``metrics.match_table`` stores ``Path(name).name``). A basename that occurs
    under two subdirectories therefore cannot be attributed to either, so the
    producer lists those and they are refused here rather than guessed at --
    the same posture the per-box row check takes.
    """
    collisions = set(document.get("basename_collisions") or [])
    by_name = {}
    for relative, values in (document.get("images") or {}).items():
        name = str(relative).rsplit("/", 1)[-1]
        if name not in collisions:
            by_name[name] = values
    return by_name, collisions


def _categorical_tag(*, name, label, scope, values, choices, total) -> tuple:
    """A measured tag whose values are named outcomes rather than a range."""
    definition = TagDefinition(
        name=name, label=label, scope=scope, widget="radio",
        source="auto", missing_label=UNMEASURED)
    definition.total = int(total)
    definition.answered = sum(1 for value in values if value is not None)

    masks = {}
    for choice in choices:
        mask = np.array([value == choice for value in values], dtype=bool)
        masks[(name, choice)] = mask
        definition.counts[choice] = int(np.count_nonzero(mask))
    missing = np.array([value is None for value in values], dtype=bool)
    if bool(missing.any()):
        masks[(name, UNMEASURED)] = missing
        definition.counts[UNMEASURED] = int(np.count_nonzero(missing))

    _order_values(definition, [str(choice) for choice in choices])
    if definition.answered == 0:
        definition.status = "no_data"
        definition.status_detail = "Nothing here carries this measure."
    return definition, masks


def _box_measure_tags(table: MatchTable, document: dict) -> tuple[list, dict, int]:
    """Box-scope tags from the box-measure sidecar.

    Joined on (image, row) and checked against the class id recorded beside the
    row -- the same drift guard the annotation-tag join uses, for the same
    reason: a relabel moves the rows under the sidecar, and attributing a pose
    to the box that now sits on that line is worse than attributing none.
    """
    gt_position = {
        (int(image), int(row)): position
        for position, (image, row) in enumerate(zip(table.gt_image, table.gt_row))
    }
    image_index = {name: index for index, name in enumerate(table.images)}

    poses: list = [None] * table.num_gt
    ratios = np.zeros(table.num_gt, dtype=np.float64)
    ratio_valid = np.zeros(table.num_gt, dtype=bool)
    mismatches = 0

    for filename, entry in (document.get("images") or {}).items():
        index = image_index.get(filename)
        if index is None:
            continue
        boxes = entry.get("boxes") or []
        positions, drifted = [], False
        for box in boxes:
            position = gt_position.get((index, int(box.get("row", -1))))
            if position is None or int(table.gt_class[position]) != int(box.get("class_id", -1)):
                drifted = position is not None or drifted
                positions.append(None)
                continue
            positions.append(position)
        if drifted:
            mismatches += 1
            continue
        for box, position in zip(boxes, positions):
            if position is None:
                continue
            values = box.get("values") or {}
            pose = values.get("pose")
            if pose:
                poses[position] = str(pose)
            ratio = values.get("person size ratio")
            if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
                ratios[position] = float(ratio)
                ratio_valid[position] = True

    declared = []
    for measure in (document.get("measures") or []):
        if measure.get("name") == "pose":
            declared = [str(choice) for choice in (measure.get("choices") or [])]
    if not declared:
        declared = sorted({pose for pose in poses if pose})

    definitions, masks = [], {}
    pose_definition, pose_masks = _categorical_tag(
        name=POSE_TAG, label="pose", scope=tag_values.REGION,
        values=poses, choices=declared, total=table.num_gt)
    definitions.append(pose_definition)
    masks.update(pose_masks)

    size_definition, size_masks = _measured_tag(
        name=PERSON_SIZE_TAG, label="person size",
        unit="the person's area as a fraction of the frame",
        scope=tag_values.REGION, values=ratios, valid=ratio_valid)
    definitions.append(size_definition)
    masks.update(size_masks)
    return definitions, masks, mismatches


def _image_measure_tags(table: MatchTable, document: dict) -> tuple[list, dict, int]:
    """Frame tags for every measure the sidecar carries."""
    by_name, collisions = _measures_by_basename(document)
    ambiguous = sum(1 for name in table.images if name in collisions)

    definitions, masks = [], {}
    for name, key, label, unit in IMAGE_MEASURE_TAGS:
        values = np.zeros(table.num_images, dtype=np.float64)
        valid = np.zeros(table.num_images, dtype=bool)
        for index, image in enumerate(table.images):
            entry = by_name.get(image)
            value = entry.get(key) if isinstance(entry, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[index] = float(value)
                valid[index] = True
        definition, measure_masks = _measured_tag(
            name=name, label=label, unit=unit, scope=tag_values.FRAME,
            values=values, valid=valid)
        definitions.append(definition)
        masks.update(measure_masks)
    return definitions, masks, ambiguous


def build_auto_tags(table: MatchTable, measures: dict | None = None,
                    box_measures: dict | None = None) -> tuple[list[TagDefinition], dict, dict]:
    """Every tag this eval can be sliced by without anyone having annotated it."""
    definitions: list[TagDefinition] = []
    image_masks: dict[tuple[str, str], np.ndarray] = {}
    gt_masks: dict[tuple[str, str], np.ndarray] = {}

    crowding, masks = _crowding_tag(table)
    definitions.append(crowding)
    image_masks.update(masks)

    if table.has_geometry:
        box_size, masks = _box_size_tag(table)
        gt_masks.update(masks)
    else:
        box_size = TagDefinition(
            name=BOX_SIZE_TAG, label="box size", scope=tag_values.REGION,
            widget="radio", source="auto", missing_label=UNMEASURED,
            status="unavailable", status_detail=NO_GEOMETRY_DETAIL,
        )
    definitions.append(box_size)

    if box_measures:
        box_definitions, box_masks, _drift = _box_measure_tags(table, box_measures)
        definitions.extend(box_definitions)
        gt_masks.update(box_masks)
    else:
        for name, label in ((POSE_TAG, "pose"), (PERSON_SIZE_TAG, "person size")):
            definitions.append(TagDefinition(
                name=name, label=label, scope=tag_values.REGION, widget="radio",
                source="auto", missing_label=UNMEASURED,
                status="unavailable", status_detail=NO_BOX_MEASURES_DETAIL))

    ambiguous = 0
    if measures:
        measure_definitions, measure_masks, ambiguous = _image_measure_tags(table, measures)
        definitions.extend(measure_definitions)
        image_masks.update(measure_masks)
    else:
        for name, _key, label, unit in IMAGE_MEASURE_TAGS:
            definitions.append(TagDefinition(
                name=name, label=label, scope=tag_values.FRAME, widget="radio",
                source="auto", unit=unit, missing_label=UNMEASURED,
                status="unavailable", status_detail=NO_MEASURES_DETAIL))
    return definitions, image_masks, gt_masks, ambiguous


def build_tag_index(
    table: MatchTable,
    document: dict | None = None,
    *,
    declared_specs: list[dict] | None = None,
    measures: dict | None = None,
    box_measures: dict | None = None,
    reasons: dict[str, str] | None = None,
    auto: bool = True,
) -> TagIndex:
    """Turn the tag sidecar into masks aligned with the match table's rows.

    The join is by image basename (the sidecar keys images the way Label Studio
    named them, the table by the file the loader read) and, for box tags, by
    row — the box's line in its label file. Rows can only be trusted when the
    two files still describe the same boxes, so every box join is checked
    against the class id the sidecar recorded beside the row; a disagreement
    drops that image's box tags and is counted, rather than quietly attributing
    an answer to the wrong box.
    """
    document = document or {}
    reasons = reasons or {}
    declared = {str(spec["name"]): spec for spec in (document.get("tags") or [])}

    # Tags defined on the dataset but absent from the sidecar were the ones that
    # genuinely could not be seen here: the sidecar is written at sync time, so a
    # tag added or renamed since then carries no answers and never became a
    # definition at all. They are listed now, with the reason and the fix.
    unsynced: set[str] = set()
    for spec in (declared_specs or []):
        name = str(spec.get("name") or "")
        if not name:
            continue
        if name in declared:
            # The admin is live truth for the option list; the sidecar only
            # records what the tag looked like at the last sync.
            declared[name] = {**declared[name], **{k: v for k, v in spec.items() if v}}
        else:
            declared[name] = spec
            unsynced.add(name)

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
            if 0 < definition.answered < definition.total:
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
            if 0 < definition.answered < definition.total:
                mask = np.ones(table.num_gt, dtype=bool)
                if box_answered[name]:
                    mask[np.asarray(sorted(box_answered[name]), dtype=np.int64)] = False
                gt_masks[(name, UNANSWERED)] = mask
                definition.counts[UNANSWERED] = int(mask.sum())

        if definition.answered == 0:
            # Nothing answered it, so there is no slice to show -- but the tag
            # still gets a row on the page saying so. Rendering one
            # "(unanswered)" table covering the whole dataset would be worse
            # than useless: N tags would each claim a full-width result.
            definition.status = "unsynced" if name in unsynced else "no_answers"
            definition.status_detail = reasons.get(name) or (
                DECLARED_NOT_SYNCED_DETAIL if name in unsynced else NO_ANSWERS_DETAIL
            )
            definition.values = []
            continue

        _order_values(definition, [str(v) for v in (declared[name].get("choices") or [])])

        # Every value the page renders has to be sliceable: resolve_clauses
        # looks a value up by mask and refuses one it cannot find, so a declared
        # option nobody picked would render a row that then 400s when clicked.
        # One shared all-false array -- masks are only ever read, never mutated.
        masks = image_masks if definition.is_frame else gt_masks
        empty = np.zeros(
            table.num_images if definition.is_frame else table.num_gt, dtype=bool)
        for value in definition.values:
            masks.setdefault((name, value), empty)

    ambiguous_filenames = 0
    if auto:
        auto_definitions, auto_image_masks, auto_gt_masks, ambiguous_filenames = (
            build_auto_tags(table, measures, box_measures))
        image_masks.update(auto_image_masks)
        gt_masks.update(auto_gt_masks)
        for definition in auto_definitions:
            definitions[definition.name] = definition

    # Everything is kept. A tag that cannot slice carries its reason instead of
    # disappearing -- "which tags could I have used here" is the question this
    # page exists to answer, and a silent drop answers it wrong.
    order = {tag_values.FRAME: 0, tag_values.REGION: 1}
    source_order = {"annotator": 0, "auto": 1}
    tags = sorted(
        definitions.values(),
        key=lambda d: (order.get(d.scope, 0), source_order.get(d.source, 0), d.name),
    )
    return TagIndex(
        tags=tags,
        image_masks=image_masks,
        gt_masks=gt_masks,
        matched_images=matched_images,
        unmatched_images=len(entries) - matched_images,
        row_mismatches=row_mismatches,
        ambiguous_filenames=ambiguous_filenames,
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
        # A value nothing falls into has no score, only zeros. Flag it so the
        # page prints dashes: a row reading "mAP50 0.0000" looks like a model
        # that failed, not like a bucket nobody used.
        row["empty"] = (row["images"] if definition.is_frame else row["gt"]) == 0
        # A frame slice holding no ground truth has no mAP and no recall: both
        # are means over an empty set, which comes out 0.0 and reads as a
        # catastrophic score rather than an undefined one. Empty frames are
        # real, and the predictions they attract are worth seeing, so the row
        # stays -- the metrics that need a denominator do not. (The crowding
        # tag makes this bucket guaranteed rather than incidental.)
        row["no_ground_truth"] = (
            definition.is_frame and not row["empty"] and row["gt"] == 0)
        row["delta"] = (
            None if row["empty"] or row["no_ground_truth"]
            else row.get(key, 0.0) - overall.get(key, 0.0))

    return {
        "name": definition.name,
        "label": definition.display,
        "scope": definition.scope,
        "widget": definition.widget,
        "kind": "frame" if definition.is_frame else "box",
        "source": definition.source,
        "unit": definition.unit,
        "cuts": definition.cuts,
        "status": definition.status,
        "status_detail": definition.status_detail,
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


def comparable_tags(columns) -> list[dict]:
    """Every tag at least one of the compared evals can be sliced by.

    The union, not the intersection: compared evals may be on different datasets
    (the compare page has always allowed that), and a tag only one of them
    carries is still worth looking at — the others simply say so.
    """
    seen: dict[str, dict] = {}
    for column in columns:
        for definition in column["index"].tags:
            if not definition.usable or definition.name in seen:
                continue
            seen[definition.name] = {
                "name": definition.name, "label": definition.display,
                "kind": "frame" if definition.is_frame else "box",
                "source": definition.source,
            }
    return list(seen.values())


def _comparability_warnings(columns) -> list[str]:
    """Ways these evals are not really comparable, per slice.

    The overall table can get away without these; a per-slice number cannot. A
    gap on one bucket is small enough to be explained by a threshold difference,
    and a reader who is not told will explain it with the model instead.
    """
    warnings = []
    datasets = {column.get("dataset") for column in columns if column.get("dataset")}
    if len(datasets) > 1:
        warnings.append(
            "These evals are on different datasets (" + ", ".join(sorted(datasets))
            + "), so a measured tag's buckets were cut over different populations "
              "and 'high' does not mean the same thing in each column.")
    thresholds = {column["table"].score_threshold for column in columns}
    if len(thresholds) > 1:
        warnings.append(
            "They were scored at different operating confidences ("
            + ", ".join(f"{value:g}" for value in sorted(thresholds))
            + "), so a precision or recall gap may be the threshold rather than the model.")
    ious = {tuple(column["table"].iou_thresholds) for column in columns}
    if len(ious) > 1:
        warnings.append("They used different IoU thresholds, so mAP50-95 is not "
                        "comparable between these columns.")
    return warnings


def compare_tag(columns, tag: str, metric: str) -> dict:
    """One metric over one tag's values, with a column per compared eval.

    ``columns`` is ``[{"label", "dataset", "table", "index"}, ...]``. A column
    whose eval does not carry the tag renders as an explicit gap rather than
    being dropped, so a four-way comparison never silently becomes a two-way one.
    """
    present = [c for c in columns
               if c["index"].tag(tag) is not None and c["index"].tag(tag).usable]
    if not present:
        raise UnknownClause(f"none of the compared evals can be sliced by {tag}")

    first = present[0]["index"].tag(tag)
    allowed = dict(FRAME_METRICS if first.is_frame else BOX_METRICS)
    if metric not in allowed:
        metric = "map50" if first.is_frame else "recall"

    # The first column that has the tag sets the row order; anything only a
    # later column carries follows, so the grid reads like its breakdown table.
    values: list[str] = []
    for column in present:
        for value in column["index"].tag(tag).values:
            if value not in values:
                values.append(value)

    rows = []
    for value in values:
        cells = []
        for column in columns:
            definition = column["index"].tag(tag)
            if definition is None or not definition.usable or value not in definition.values:
                cells.append({"value": None, "population": None, "absent": True})
                continue
            entry = slice_metrics(column["table"], column["index"], [(tag, value)])
            population = entry["images"] if entry["kind"] == "frame" else entry["gt"]
            cells.append({
                "value": entry.get(metric) if population else None,
                "population": population, "absent": False,
            })
        numbers = [c["value"] for c in cells if c["value"] is not None]
        best = max(numbers) if len(numbers) > 1 and len(set(numbers)) > 1 else None
        for cell in cells:
            cell["is_best"] = best is not None and cell["value"] == best
        rows.append({"value": value, "cells": cells})

    return {
        "tag": tag, "label": first.display, "metric": metric,
        "metric_label": allowed[metric],
        "kind": "frame" if first.is_frame else "box",
        "population_label": "images" if first.is_frame else "boxes",
        "columns": [{
            "label": column["label"],
            "absent": column["index"].tag(tag) is None
                      or not column["index"].tag(tag).usable,
        } for column in columns],
        "rows": rows,
        "metrics": FRAME_METRICS if first.is_frame else BOX_METRICS,
        "warnings": _comparability_warnings(columns),
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

    # Only tags that can actually slice get a table. The rest are listed with
    # the reason they cannot, which is the half of the picture this page used
    # to leave out entirely.
    usable = [d for d in index.tags if d.usable]
    breakdowns = [tag_breakdown(table, index, definition) for definition in usable]
    frame_tags = [d.name for d in usable if d.is_frame]
    box_tags = [d.name for d in usable if not d.is_frame]
    tags_without_data = [
        {
            "name": d.name, "label": d.display,
            "kind": "frame" if d.is_frame else "box",
            "source": d.source, "status": d.status, "detail": d.status_detail,
        }
        for d in index.tags if not d.usable
    ]

    return {
        "overall": overall_frame,
        "overall_boxes": overall_box,
        "breakdowns": breakdowns,
        "tags_without_data": tags_without_data,
        "frame_tags": frame_tags,
        "box_tags": box_tags,
        "tags": [
            {
                "name": d.name, "label": d.display,
                "kind": "frame" if d.is_frame else "box",
                "source": d.source, "usable": d.usable,
                "values": d.values, "counts": d.counts,
                "answered": d.answered, "total": d.total,
            }
            for d in usable
        ],
        "tag_sources": {
            "annotator": sum(1 for d in index.tags if d.source == "annotator"),
            "computed": sum(1 for d in index.tags if d.source == "auto"),
            "without_data": len(tags_without_data),
        },
        "coverage": {
            "images": table.num_images,
            "gt": table.num_gt,
            "images_with_tags": index.matched_images,
            "images_not_in_eval": index.unmatched_images,
            "row_mismatches": index.row_mismatches,
            "ambiguous_filenames": index.ambiguous_filenames,
            "match_table_version": table.version,
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
