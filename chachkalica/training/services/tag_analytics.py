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

**Three populations, three metric sets.** A frame tag selects images, so a
slice of frames is a whole little dataset and gets everything. A box tag
selects *ground-truth boxes*, and a prediction carries no tags — an unmatched
false positive belongs to no box, so it cannot be attributed to "occluded".
Box-tag slices therefore report recall, misses, mean IoU and the score
distribution, and no precision or mAP. Reporting one would mean inventing an
attribution nobody measured.

A **person** tag is the way out of that, for the questions that are really
about people. Where a box tag can only describe someone who was carrying
something worth labelling, the people the measures pass found are a population
in their own right, and each one is a yes/no per class asked twice — is there a
weapon on them in the labels, and did the model say there was. A false positive
now has an owner (the person it landed on), and a person carrying nothing is a
row rather than an absence, so precision, specificity and accuracy exist here
and only here. Two things make that honest rather than convenient: it needs a
prediction's box, which is why the match table grew a geometry block at
version 3, and it is not an accounting of boxes — a weapon on the ground
belongs to no person, so :func:`person_orphans` counts those separately instead
of letting them disappear between the two views.

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
#: the page rather than a refusal to render it. Version 3 added ``pred.geom``,
#: the operating set's boxes, without which a prediction cannot be attributed
#: to a person and the per-person section says so instead of rendering.
SUPPORTED_MATCH_TABLE_VERSIONS = (1, 2, 3)

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

#: A slice of people rather than of frames or of ground-truth boxes. Not one of
#: ``tag_values``' scopes: those mirror what an annotator can be asked for, and
#: nobody annotates a person the posture pass invented.
PERSON_SCOPE = "person"


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

    #: Operating-set predictions that carry a box, as positions into the ``op_*``
    #: columns above, with their normalized xyxy alongside. Sparse on purpose --
    #: version 3 writes geometry only at and above the operating confidence, so
    #: these are the rows a per-person question can be asked of, and
    #: ``geom_cut`` is the lowest confidence any of them was written at.
    op_geom_row: np.ndarray
    op_geom_box: np.ndarray
    geom_cut: float

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

    @property
    def has_prediction_geometry(self) -> bool:
        """Whether predictions can be placed on the frame at all.

        False for a version 1 or 2 table. Also false for a version 3 one whose
        every record lacked a frame size to normalize against -- the block is
        then present and empty, which is a different sentence to the reader
        than "this eval predates the feature", so the caller checks the version
        separately rather than inferring it from here.
        """
        return bool(self.op_geom_row.size)


def _columns(block: dict) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Prediction columns in descending-score order, and the order itself.

    The order comes back because the sparse geometry block indexes the file's
    rows, not these, and has to be carried across the same permutation.
    """
    score = np.asarray(block["score"], dtype=np.float64)
    order = _stable_score_order(score)
    return (
        np.asarray(block["image"], dtype=np.int64)[order],
        np.asarray(block["class"], dtype=np.int64)[order],
        score[order],
        np.asarray(block["iou"], dtype=np.float64)[order],
        np.asarray(block["gt"], dtype=np.int64)[order],
    ), order


def _geometry(path: str, block: dict, order: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The block's sparse prediction boxes, remapped onto the sorted columns.

    ``geom.index`` points into the file's row order; every consumer here works
    in descending-score order, so the indices are pushed through the same
    permutation once, at load, rather than at every slice.
    """
    geom = block.get("geom")
    empty = (np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float64), 0.0)
    if not isinstance(geom, dict):
        return empty

    index = np.asarray(geom.get("index") or [], dtype=np.int64)
    corners = [np.asarray(geom.get(key) or [], dtype=np.float64)
               for key in ("x0", "y0", "x1", "y1")]
    for key, column in zip(("x0", "y0", "x1", "y1"), corners):
        if column.size != index.size:
            raise ValueError(
                f"{path}: pred.geom.{key} has {column.size} entries but index has "
                f"{index.size} -- the table is truncated or hand-edited.")
    if not index.size:
        return empty
    if int(index.min()) < 0 or int(index.max()) >= order.size:
        raise ValueError(
            f"{path}: pred.geom.index points outside the prediction rows -- "
            f"the table is truncated or hand-edited.")

    # order[i] is the file row that landed at sorted position i, so its inverse
    # takes a file row to where it now lives.
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    rows = rank[index]
    boxes = np.stack(corners, axis=1)

    # Sorted-position order, so a slice can intersect these against a mask with
    # searchsorted rather than a scan.
    shuffle = np.argsort(rows, kind="mergesort")
    return rows[shuffle], boxes[shuffle], float(geom.get("cut") or 0.0)


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

    (pred_image, pred_class, pred_score, pred_iou, pred_gt), pred_order = _columns(data["pred"])

    operating = data.get("pred_operating")
    if operating is None:
        op_image, op_class, op_score, op_iou, op_gt = (
            pred_image, pred_class, pred_score, pred_iou, pred_gt)
        operating_is_ap = True
        geom_row, geom_box, geom_cut = _geometry(path, data["pred"], pred_order)
    else:
        (op_image, op_class, op_score, op_iou, op_gt), op_order = _columns(operating)
        operating_is_ap = False
        geom_row, geom_box, geom_cut = _geometry(path, operating, op_order)

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
        op_geom_row=geom_row, op_geom_box=geom_box, geom_cut=geom_cut,
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
        return self.scope not in (tag_values.REGION, PERSON_SCOPE)

    @property
    def is_person(self) -> bool:
        return self.scope == PERSON_SCOPE

    @property
    def kind(self) -> str:
        """Which population this tag slices, and so which metrics it supports."""
        if self.is_person:
            return "person"
        return "frame" if self.is_frame else "box"

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
    person_masks: dict[tuple[str, str], np.ndarray]
    matched_images: int
    unmatched_images: int
    row_mismatches: int
    #: The people these masks index, or None where this eval has none.
    persons: "PersonTable | None" = None
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


def _person_measure_tags(persons) -> tuple[list, dict]:
    """pose and person size, over people instead of over the boxes on them.

    The same two measures the box-scope builder produces, keyed to a different
    population -- and that change of population is the whole point. A box-scope
    pose tag can only describe people who were carrying something worth
    labelling; this one has a row for everybody the pass saw, which is what a
    false alarm and a correct rejection need in order to exist.
    """
    definitions, masks = [], {}

    pose_definition, pose_masks = _categorical_tag(
        name=POSE_TAG, label="pose", scope=PERSON_SCOPE,
        values=persons.pose, choices=persons.poses_declared, total=persons.count)
    definitions.append(pose_definition)
    masks.update(pose_masks)

    size_definition, size_masks = _measured_tag(
        name=PERSON_SIZE_TAG, label="person size",
        unit="the person's area as a fraction of the frame",
        scope=PERSON_SCOPE, values=persons.size, valid=persons.size > 0)
    definitions.append(size_definition)
    masks.update(size_masks)
    return definitions, masks


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
                    box_measures: dict | None = None,
                    persons=None) -> tuple[list[TagDefinition], dict, dict, dict, int]:
    """Every tag this eval can be sliced by without anyone having annotated it."""
    definitions: list[TagDefinition] = []
    image_masks: dict[tuple[str, str], np.ndarray] = {}
    gt_masks: dict[tuple[str, str], np.ndarray] = {}
    person_masks: dict[tuple[str, str], np.ndarray] = {}

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

    # pose and person size describe a person, so they slice people wherever the
    # data allows it. Where it does not -- an older sidecar that kept no people,
    # or a match table with no prediction boxes to place on them -- they stay
    # box-scope rather than vanishing: the recall-side view they gave before is
    # still true, and an eval does not lose a tag by being old.
    if persons is not None and persons.count:
        person_definitions, masks = _person_measure_tags(persons)
        definitions.extend(person_definitions)
        person_masks.update(masks)
    elif box_measures:
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
    return definitions, image_masks, gt_masks, person_masks, ambiguous


def build_tag_index(
    table: MatchTable,
    document: dict | None = None,
    *,
    declared_specs: list[dict] | None = None,
    measures: dict | None = None,
    box_measures: dict | None = None,
    reasons: dict[str, str] | None = None,
    auto: bool = True,
    persons=None,
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
    person_masks: dict[tuple[str, str], np.ndarray] = {}

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
        (auto_definitions, auto_image_masks, auto_gt_masks, auto_person_masks,
         ambiguous_filenames) = build_auto_tags(table, measures, box_measures, persons)
        image_masks.update(auto_image_masks)
        gt_masks.update(auto_gt_masks)
        person_masks.update(auto_person_masks)
        for definition in auto_definitions:
            definitions[definition.name] = definition

    # Everything is kept. A tag that cannot slice carries its reason instead of
    # disappearing -- "which tags could I have used here" is the question this
    # page exists to answer, and a silent drop answers it wrong.
    order = {tag_values.FRAME: 0, PERSON_SCOPE: 1, tag_values.REGION: 2}
    source_order = {"annotator": 0, "auto": 1}
    tags = sorted(
        definitions.values(),
        key=lambda d: (order.get(d.scope, 0), source_order.get(d.source, 0), d.name),
    )
    return TagIndex(
        tags=tags,
        image_masks=image_masks,
        gt_masks=gt_masks,
        person_masks=person_masks,
        matched_images=matched_images,
        unmatched_images=len(entries) - matched_images,
        row_mismatches=row_mismatches,
        persons=persons,
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


def resolve_score_cut(table: MatchTable, score_cut: float | None) -> float:
    """The confidence to score at: the eval's own, or an operator's override.

    Clamped into ``[score_floor, 1]``. The floor is where the stored table stops
    — asking for less is asking about predictions that were never written, and
    answering it would quietly report a slice as emptier than it is.
    """
    if score_cut is None:
        return table.score_threshold
    return float(min(max(float(score_cut), table.score_floor), 1.0))


def image_slice_metrics(table: MatchTable, image_mask: np.ndarray,
                        score_cut: float | None = None) -> dict:
    """The full metric set for a subset of images.

    Mirrors ``metrics.evaluate_detection``: AP integrates the raw NMS-free set
    down to the score floor, while precision/recall/F1 and the counts are read
    at the operating confidence off the deployed (optionally NMS-suppressed)
    set. The mAP means skip classes with no ground truth in the slice, for the
    same reason the trainer's do — averaging a hard zero over a class the slice
    never contains deflates the number without saying anything about the model.

    ``score_cut`` overrides the eval's own operating confidence, which is what
    lets the page sweep a threshold without re-running anything: the table keeps
    every prediction down to ``score_floor``, so any cut at or above that floor
    is a filter over rows already on disk. Below the floor there is no data to
    read, so it clamps rather than inventing an emptier slice than the truth.
    """
    keep_pred = image_mask[table.pred_image]
    keep_op = image_mask[table.op_image] if not table.operating_is_ap else keep_pred
    keep_gt = image_mask[table.gt_image]
    cut = resolve_score_cut(table, score_cut)

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


def gt_slice_metrics(table: MatchTable, gt_mask: np.ndarray,
                     score_cut: float | None = None) -> dict:
    """What a subset of *ground-truth boxes* can honestly be scored on.

    Recall, misses, and how well the boxes that were found were found (IoU,
    confidence). No precision and no AP: a false positive is a prediction, and
    a prediction has no tags, so there is no way to say which slice it belongs
    to. See this module's docstring.

    ``score_cut`` overrides the eval's operating confidence, as in
    :func:`image_slice_metrics` — here it moves which boxes count as found.
    """
    cut = resolve_score_cut(table, score_cut)
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
# People
# --------------------------------------------------------------------------

#: Slices smaller than this are still computed, but flagged.
SMALL_SLICE_PERSONS = 30

NO_PERSONS_DETAIL = (
    "No people were kept for this label set -- either the box measures have "
    "not been run over it, or they predate per-person rows and recorded only "
    "each box's pose rather than the person behind it. Run 'Measure box "
    "statistics' over this label set."
)
NO_PREDICTION_GEOMETRY_DETAIL = (
    "This eval's match table records no prediction boxes, so a prediction "
    "cannot be attributed to the person it landed on. Run 'Build tag analytics "
    "data' to rewrite it."
)
NO_PEOPLE_FOUND_DETAIL = (
    "The measures pass ran over these frames and found nobody in them."
)

#: Rule defaults, used only when the sidecar does not carry the ones its pass
#: actually ran with. Kept in step with
#: ``friendy_chachkalica/ml/build_measures.py``, which is authoritative.
DEFAULT_CONTAINMENT_MIN = 0.7
DEFAULT_PERSON_IOU_MIN = 0.5
DEFAULT_PERSON_CLASS_NAMES = ("person", "people", "pedestrian")


def _pairwise_intersection(boxes: np.ndarray, others: np.ndarray) -> np.ndarray:
    """``(N, M)`` intersection areas between two sets of xyxy boxes."""
    x0 = np.maximum(boxes[:, None, 0], others[None, :, 0])
    y0 = np.maximum(boxes[:, None, 1], others[None, :, 1])
    x1 = np.minimum(boxes[:, None, 2], others[None, :, 2])
    y1 = np.minimum(boxes[:, None, 3], others[None, :, 3])
    return np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    return (np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
            * np.clip(boxes[:, 3] - boxes[:, 1], 0, None))


def attribute_boxes(boxes: np.ndarray, is_person: np.ndarray, people: np.ndarray,
                    *, containment_min: float, person_iou_min: float) -> np.ndarray:
    """Which person owns each box, as indices into ``people`` (``-1`` for none).

    The rule is ``friendy_chachkalica/ml/build_measures.py::attribute``, applied
    to *predictions* here rather than to ground truth -- vectorized, but the same
    two cases for the same reasons. A box that is itself a person is matched by
    IoU, because the two boxes describe one object and containment would let a
    large foreground figure swallow someone standing behind them; anything else
    is matched by containment, because a helmet's IoU with its own wearer is
    about 0.02 and an argmax over that is noise. Ties go to the smallest
    qualifying person for a contained box and to the best overlap for a person.

    Attributing predictions with the rule the ground truth was attributed by is
    the whole point: score one side by containment and the other by IoU and the
    per-person table would compare two different notions of "on this person".
    """
    if boxes.size == 0 or people.size == 0:
        return np.full(len(boxes), -1, dtype=np.int64)

    intersection = _pairwise_intersection(boxes, people)
    own = _box_areas(boxes)[:, None]
    person_area = _box_areas(people)[None, :]

    contained = intersection / np.maximum(own, 1e-9)
    iou = intersection / np.maximum(own + person_area - intersection, 1e-9)

    person_rows = is_person[:, None]
    score = np.where(person_rows, iou, contained)
    qualifies = score >= np.where(person_rows, person_iou_min, containment_min)
    # A degenerate box has no area to be contained by anything, and dividing by
    # the 1e-9 floor above would hand it a containment of zero or infinity
    # depending on rounding. Excluded outright instead.
    qualifies &= own > 0

    # One argmin over a rank that encodes both tie-breaks: negated overlap for a
    # person box (largest wins), person area for a contained box (smallest wins).
    rank = np.where(person_rows, -score, np.broadcast_to(person_area, score.shape))
    rank = np.where(qualifies, rank, np.inf)
    owner = np.argmin(rank, axis=1).astype(np.int64)
    owner[~qualifies.any(axis=1)] = -1
    return owner


@dataclass
class PersonTable:
    """Everyone the measures pass found, and what belongs to whom.

    The unit here is a person, not a box, which is what makes the questions
    below askable at all: "of the people lying down, how many did the model
    correctly say were unarmed" needs a row for a person carrying nothing, and
    a ground-truth box is not that row -- there is no box.

    ``gt_person`` and ``pred_person`` are the two sides of the same join.
    Ground truth was attributed by the measures pass itself, which numbered the
    boxes with the loader that numbered them for the match table; predictions
    are attributed here, from the match table's own geometry, by the same rule.
    """

    image: np.ndarray            #: (P,) image index per person
    box: np.ndarray              #: (P, 4) normalized xyxy
    pose: list                   #: (P,) posture or None where the engine saw none
    size: np.ndarray             #: (P,) person area as a fraction of the frame
    poses_declared: list         #: postures the pass could emit, in its own order

    gt_person: np.ndarray        #: (num_gt,) owning person, -1 for none
    pred_row: np.ndarray         #: (G,) positions into the table's op_* columns
    pred_person: np.ndarray      #: (G,) owning person per those rows, -1 for none

    #: (num_images,) images the pass actually looked at and whose rows still
    #: line up. Distinguishing these from the rest is what lets "this box sat on
    #: nobody" -- a finding about the frame -- be told apart from "nobody looked
    #: at this frame", which is a finding about the pipeline.
    measured_mask: np.ndarray

    #: Class ids that *are* people, excluded from the per-person questions.
    person_class_ids: frozenset

    #: Images whose sidecar rows no longer line up with the labels, skipped
    #: wholesale rather than attributed to the boxes that now sit on those lines.
    drifted_images: int = 0
    #: Images the sidecar measured, i.e. that could contribute people at all.
    measured_images: int = 0

    #: ``(side, class_id[, cut]) -> per-person presence``, filled on first use.
    #: The confidence is part of the key on the prediction side only — see
    #: :func:`_presence`.
    presence: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.image.size)


def _person_class_ids(table: MatchTable, names) -> set:
    lowered = {str(name).lower() for name in names}
    return {class_id for class_id, name in table.classes.items()
            if str(name).lower() in lowered}


def build_persons(table: MatchTable, document: dict | None) -> PersonTable | None:
    """Join a box-measure sidecar's people onto a match table, or None.

    None means the question cannot be asked here rather than that the answer is
    empty -- a version 1 sidecar kept no people, and a match table before
    version 3 kept no prediction boxes to place on them. Both are sentences the
    page prints; neither is a zero.
    """
    if not isinstance(document, dict):
        return None
    if table.version < 3:
        # Without prediction boxes every prediction would attribute to nobody,
        # and a per-person table built on that reads as a model that fired at
        # no one -- a confident, wrong answer. Refuse to build it, so the page
        # prints why instead.
        return None
    images = document.get("images") or {}
    if not any(isinstance(entry, dict) and "people" in entry for entry in images.values()):
        return None

    containment_min = float(document.get("containment_min") or DEFAULT_CONTAINMENT_MIN)
    person_iou_min = float(document.get("person_iou_min") or DEFAULT_PERSON_IOU_MIN)
    person_ids = _person_class_ids(
        table, document.get("person_class_names") or DEFAULT_PERSON_CLASS_NAMES)

    image_index = {name: index for index, name in enumerate(table.images)}
    gt_position = {
        (int(image), int(row)): position
        for position, (image, row) in enumerate(zip(table.gt_image, table.gt_row))
    }

    person_image: list[int] = []
    person_box: list[list[float]] = []
    person_pose: list = []
    person_size: list[float] = []
    #: image index -> (first person, count), for attributing predictions below
    per_image: dict[int, tuple[int, int]] = {}

    gt_person = np.full(table.num_gt, -1, dtype=np.int64)
    measured_mask = np.zeros(table.num_images, dtype=bool)
    drifted = 0
    measured = 0

    for filename, entry in images.items():
        index = image_index.get(filename)
        if index is None or not isinstance(entry, dict):
            continue
        measured += 1
        people = entry.get("people") or []

        # Same drift guard as the box-scope join, for the same reason: a relabel
        # moves the rows under the sidecar, and hanging a prediction on the
        # person of the box that now sits on that line is worse than hanging it
        # on nobody. One bad row disqualifies the image, people included --
        # their poses are fine but the boxes attributed to them are not.
        boxes = entry.get("boxes") or []
        positions, bad = [], False
        for box in boxes:
            position = gt_position.get((index, int(box.get("row", -1))))
            if position is None or int(table.gt_class[position]) != int(box.get("class_id", -1)):
                bad = position is not None or bad
                positions.append(None)
                continue
            positions.append(position)
        if bad:
            drifted += 1
            continue
        measured_mask[index] = True

        offset = len(person_image)
        # A box's ``person`` is an index into the sidecar's own list, so a
        # malformed entry may not silently shift the ones after it -- dropping
        # person 3 would hand every later box to the person standing next to
        # its owner. Kept as an explicit map rather than as an offset.
        placed: dict[int, int] = {}
        for source_index, person in enumerate(people):
            corners = person.get("box") or []
            if len(corners) != 4:
                continue
            placed[source_index] = len(person_image)
            person_image.append(index)
            person_box.append([float(value) for value in corners])
            pose = person.get("pose")
            person_pose.append(str(pose) if pose else None)
            ratio = person.get("size ratio")
            person_size.append(float(ratio) if isinstance(ratio, (int, float))
                               and not isinstance(ratio, bool) else 0.0)
        per_image[index] = (offset, len(person_image) - offset)

        for box, position in zip(boxes, positions):
            if position is None:
                continue
            owner = box.get("person")
            if isinstance(owner, int) and not isinstance(owner, bool):
                gt_person[position] = placed.get(owner, -1)

    people_boxes = (np.asarray(person_box, dtype=np.float64) if person_box
                    else np.zeros((0, 4), dtype=np.float64))

    pred_row, pred_person = _attribute_predictions(
        table, people_boxes, per_image, person_ids,
        containment_min=containment_min, person_iou_min=person_iou_min)

    declared = []
    for measure in (document.get("person_measures") or []):
        if measure.get("name") == "pose":
            declared = [str(choice) for choice in (measure.get("choices") or [])]
    if not declared:
        declared = sorted({pose for pose in person_pose if pose})

    return PersonTable(
        image=np.asarray(person_image, dtype=np.int64),
        box=people_boxes,
        pose=person_pose,
        size=np.asarray(person_size, dtype=np.float64),
        poses_declared=declared,
        gt_person=gt_person,
        pred_row=pred_row,
        pred_person=pred_person,
        measured_mask=measured_mask,
        person_class_ids=frozenset(person_ids),
        drifted_images=drifted,
        measured_images=measured,
    )


def _attribute_predictions(table: MatchTable, people: np.ndarray, per_image: dict,
                           person_ids: set, *, containment_min: float,
                           person_iou_min: float) -> tuple[np.ndarray, np.ndarray]:
    """Each geometry-carrying prediction's owning person, or ``-1``.

    Walks the geometry rows image by image, because attribution only ever looks
    inside one frame -- the same locality that lets the match table be sliced at
    all.
    """
    rows = table.op_geom_row
    owners = np.full(rows.size, -1, dtype=np.int64)
    if not rows.size or not people.size:
        return rows, owners

    image_of_row = table.op_image[rows]
    order = np.argsort(image_of_row, kind="mergesort")
    grouped = image_of_row[order]
    boundaries = np.flatnonzero(np.diff(grouped)) + 1
    for chunk in np.split(order, boundaries):
        if not chunk.size:
            continue
        image = int(image_of_row[chunk[0]])
        offset, count = per_image.get(image, (0, 0))
        if not count:
            continue
        is_person = np.isin(table.op_class[rows[chunk]], list(person_ids)) \
            if person_ids else np.zeros(chunk.size, dtype=bool)
        local = attribute_boxes(
            table.op_geom_box[chunk], is_person, people[offset:offset + count],
            containment_min=containment_min, person_iou_min=person_iou_min)
        owners[chunk] = np.where(local >= 0, local + offset, -1)
    return rows, owners


def _presence(table: MatchTable, persons: PersonTable, class_id: int,
              side: str, cut: float | None = None) -> np.ndarray:
    """Per person: does this class appear on them, in ground truth or in output.

    Memoized on the table: a breakdown asks for the same two arrays once per
    value of a tag, and they are a scan over every box each time. ``cut`` is
    part of the memo key, not just an argument: the confidence slider asks for
    the same class at a dozen confidences in one page load, and a cache keyed
    on the class alone would answer every one of them with the first cut it
    saw. It is ignored on the ``gt`` side, which no confidence can move.
    """
    cut = table.score_threshold if cut is None else float(cut)
    key = (side, int(class_id)) if side == "gt" else (side, int(class_id), cut)
    cached = persons.presence.get(key)
    if cached is not None:
        return cached

    has = np.zeros(persons.count, dtype=bool)
    if side == "gt":
        selected = (persons.gt_person >= 0) & (table.gt_class == class_id)
        has[persons.gt_person[selected]] = True
    else:
        rows = persons.pred_row
        selected = (
            (persons.pred_person >= 0)
            & (table.op_class[rows] == class_id)
            & (table.op_score[rows] >= cut)
        )
        has[persons.pred_person[selected]] = True
    persons.presence[key] = has
    return has


def _rates(hit: int, miss: int, false_alarm: int, correct_reject: int) -> dict:
    """The five rates a 2x2 confusion table supports, plus what defines them.

    ``has_positives``/``has_predictions`` travel with the numbers because a
    recall of 0.0 over nothing and a recall of 0.0 over forty boxes are the
    same float and opposite findings; the page dashes the first.
    """
    actual = hit + miss
    predicted = hit + false_alarm
    negatives = correct_reject + false_alarm
    decisions = hit + miss + false_alarm + correct_reject
    precision = predicted and hit / predicted
    recall = actual and hit / actual
    return {
        "hit": hit, "miss": miss,
        "false_alarm": false_alarm, "correct_reject": correct_reject,
        "gt_yes": actual, "predicted_yes": predicted,
        "precision": float(precision), "recall": float(recall),
        "f1": _f1(float(precision), float(recall)),
        "specificity": float(negatives and correct_reject / negatives),
        "accuracy": float(decisions and (hit + correct_reject) / decisions),
        "has_positives": bool(actual),
        "has_predictions": bool(predicted),
        "has_negatives": bool(negatives),
    }


def scored_class_ids(table: MatchTable, persons: PersonTable) -> list[int]:
    """The classes a per-person question can be asked about.

    Everything but the person classes themselves: "does this person have a
    person on them" is not a question, and including it would put a column of
    near-perfect hits beside the ones a reader is actually reading.
    """
    return [class_id for class_id in sorted(table.classes)
            if class_id not in persons.person_class_ids]


def person_slice_metrics(table: MatchTable, persons: PersonTable,
                         person_mask: np.ndarray,
                         score_cut: float | None = None) -> dict:
    """Score a subset of *people*: did the model get each one's contents right.

    One binary decision per person per class -- "is there a weapon on this
    person", asked of the labels and of the output at the eval's operating
    confidence -- so this is the one view where a false positive has somewhere
    to belong and precision, specificity and accuracy exist. The headline
    numbers are micro-averaged over the classes: every person-class decision
    counts once, which is what makes them comparable across slices that hold
    different mixes of classes.

    ``score_cut`` overrides that operating confidence, as in
    :func:`image_slice_metrics`. It moves only the prediction side of each
    decision: raising it turns false alarms into correct rejections and hits
    into misses, which is exactly the trade the slider exists to show.
    """
    cut = resolve_score_cut(table, score_cut)
    selected = np.nonzero(person_mask)[0]
    total = int(selected.size)

    per_class = []
    hit = miss = false_alarm = correct_reject = 0
    for class_id in scored_class_ids(table, persons):
        gt_has = _presence(table, persons, class_id, "gt")[selected]
        pred_has = _presence(table, persons, class_id, "pred", cut)[selected]
        counts = (
            int(np.count_nonzero(gt_has & pred_has)),
            int(np.count_nonzero(gt_has & ~pred_has)),
            int(np.count_nonzero(~gt_has & pred_has)),
        )
        entry = _rates(*counts, total - sum(counts))
        per_class.append({
            "class_id": class_id,
            "class_name": table.classes.get(class_id, str(class_id)),
            "persons": total, **entry,
        })
        hit, miss, false_alarm = (hit + counts[0], miss + counts[1],
                                  false_alarm + counts[2])
        correct_reject += total - sum(counts)

    images = np.unique(persons.image[selected]).size if total else 0
    return {
        "kind": "person",
        "persons": total,
        "images": int(images),
        "classes_scored": len(per_class),
        "per_class": per_class,
        "small": total < SMALL_SLICE_PERSONS,
        **_rates(hit, miss, false_alarm, correct_reject),
    }


def person_orphans(table: MatchTable, persons: PersonTable) -> dict:
    """Everything the per-person view could not put on a person.

    The per-person table is a closed accounting of people, which means it is
    silently *not* an accounting of boxes: a weapon lying on the ground, or one
    worn by somebody the detector walked past, belongs to no person and appears
    in none of its rows. Dropping those would let a model look good on people
    while missing half the weapons in the dataset, so they are counted here,
    scored on the one thing they support -- was the box found -- and printed
    beside the per-person numbers rather than behind a link.

    Three reasons a box has no person, kept apart because they call for
    different actions: nobody was there (a finding), the frame was never
    measured (run the pass), or its rows drifted (re-sync, then re-run).
    """
    measured = persons.measured_mask

    primary = 0.5 if 0.5 in table.iou_thresholds else table.iou_thresholds[0]
    claim_score, _ = gt_claims(table, primary)
    found = claim_score >= table.score_threshold

    on_nobody = (persons.gt_person < 0) & measured[table.gt_image]
    unmeasured = (persons.gt_person < 0) & ~measured[table.gt_image]

    per_class = []
    for class_id in scored_class_ids(table, persons):
        in_class = table.gt_class == class_id
        orphans = int(np.count_nonzero(on_nobody & in_class))
        if not orphans and not int(np.count_nonzero(unmeasured & in_class)):
            continue
        hits = int(np.count_nonzero(on_nobody & in_class & found))
        per_class.append({
            "class_id": class_id,
            "class_name": table.classes.get(class_id, str(class_id)),
            "gt": int(np.count_nonzero(in_class)),
            "on_nobody": orphans,
            "found": hits,
            "missed": orphans - hits,
            "recall": float(orphans and hits / orphans),
            "unmeasured": int(np.count_nonzero(unmeasured & in_class)),
        })

    # Predictions the same three ways. An operating-point row with no geometry
    # at all is its own case: the eval knew the box and the table could not
    # normalize it, which no re-run of the measures pass will fix.
    operating = table.op_score >= table.score_threshold
    scored_classes = set(scored_class_ids(table, persons))
    relevant = operating & np.isin(table.op_class, list(scored_classes)) \
        if scored_classes else np.zeros(table.op_image.size, dtype=bool)
    with_geometry = np.zeros(table.op_image.size, dtype=bool)
    with_geometry[persons.pred_row] = True
    placed = np.zeros(table.op_image.size, dtype=bool)
    placed[persons.pred_row[persons.pred_person >= 0]] = True

    predictions = {
        "total": int(np.count_nonzero(relevant)),
        "on_person": int(np.count_nonzero(relevant & placed)),
        "on_nobody": int(np.count_nonzero(
            relevant & with_geometry & ~placed & measured[table.op_image])),
        "unmeasured": int(np.count_nonzero(
            relevant & with_geometry & ~measured[table.op_image])),
        "no_geometry": int(np.count_nonzero(relevant & ~with_geometry)),
    }

    return {
        "gt_total": table.num_gt,
        "gt_on_person": int(np.count_nonzero(persons.gt_person >= 0)),
        "gt_on_nobody": int(np.count_nonzero(on_nobody)),
        "gt_unmeasured": int(np.count_nonzero(unmeasured)),
        "per_class": per_class,
        "predictions": predictions,
        "people": persons.count,
        "images_measured": persons.measured_images,
        "images_total": table.num_images,
        "images_drifted": persons.drifted_images,
        "images_without_people": int(np.count_nonzero(measured)) - int(
            np.unique(persons.image).size if persons.count else 0),
        "score_threshold": table.score_threshold,
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
#: A person slice is the one kind where a false positive has an owner, so it is
#: also the only one that can offer specificity and accuracy -- both need the
#: negatives (people carrying nothing), which neither of the other two
#: populations contains a row for.
PERSON_METRICS = [
    ("f1", "F1"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("specificity", "specificity"),
    ("accuracy", "accuracy"),
]

#: Metric set and headline per slice kind, so a caller never has to re-derive
#: "which metrics does this population support".
METRICS_BY_KIND = {
    "frame": (FRAME_METRICS, "map50"),
    "person": (PERSON_METRICS, "f1"),
    "box": (BOX_METRICS, "recall"),
}

#: The denominator each kind counts in, for the column a table puts first.
POPULATION_BY_KIND = {"frame": "images", "person": "persons", "box": "boxes"}

#: Where that denominator lives in a scored slice. Not the same as the label
#: above: a box slice counts ground-truth rows and calls them "boxes".
POPULATION_KEY = {"frame": "images", "person": "persons", "box": "gt"}


def overall_metrics(table: MatchTable, index: TagIndex, kind: str,
                    score_cut: float | None = None) -> dict:
    """The unfiltered slice of one population — what every row is compared to.

    ``score_cut`` has to travel with it: a row scored at the slider's
    confidence and a baseline scored at the eval's would make every delta on
    the page the sum of two different questions.
    """
    if kind == "person":
        persons = index.persons
        return person_slice_metrics(
            table, persons, np.ones(persons.count, dtype=bool), score_cut)
    if kind == "frame":
        return image_slice_metrics(
            table, np.ones(table.num_images, dtype=bool), score_cut)
    return gt_slice_metrics(table, np.ones(table.num_gt, dtype=bool), score_cut)


class UnknownClause(ValueError):
    """A filter names a tag/value pair this eval's data has no mask for."""


def resolve_clauses(table: MatchTable, index: TagIndex,
                    clauses) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Intersect ``[(tag, value), ...]`` into (image, gt, person masks, kind).

    Clauses are ANDed — that is what a cross-union of tags means here. Mixing a
    frame clause with a box clause is allowed and useful ("occluded boxes, in
    rainy frames"): the frame clause narrows the images, which narrows the
    boxes.

    Which population the result *is* follows from the most specific clause
    present, because that is the one that cannot be expressed in the others:
    person beats box beats frame. A person clause mixed with a box clause
    therefore keeps the people who own at least one box the box clause selected
    — "the people carrying something occluded" — rather than refusing the
    combination, which would leave the cross-tab unable to pair the two.
    """
    persons = index.persons
    person_count = persons.count if persons is not None else 0

    image_mask = np.ones(table.num_images, dtype=bool)
    gt_mask = np.ones(table.num_gt, dtype=bool)
    person_mask = np.ones(person_count, dtype=bool)
    kind = "frame"
    box_clause = False

    for tag, value in clauses:
        key = (str(tag), str(value))
        if key in index.image_masks:
            image_mask = image_mask & index.image_masks[key]
        elif key in index.person_masks:
            person_mask = person_mask & index.person_masks[key]
            kind = "person"
        elif key in index.gt_masks:
            gt_mask = gt_mask & index.gt_masks[key]
            box_clause = True
            if kind != "person":
                kind = "box"
        else:
            raise UnknownClause(f"no people, boxes or images carry {tag} = {value}")

    # A box only counts when its image survives the frame clauses, and a person
    # likewise — both are sub-populations of the images.
    gt_mask = gt_mask & image_mask[table.gt_image]
    if person_count:
        person_mask = person_mask & image_mask[persons.image]
        if box_clause:
            owned = np.zeros(person_count, dtype=bool)
            owners = persons.gt_person[gt_mask & (persons.gt_person >= 0)]
            owned[owners] = True
            person_mask = person_mask & owned
    return image_mask, gt_mask, person_mask, kind


def slice_metrics(table: MatchTable, index: TagIndex, clauses,
                  score_cut: float | None = None) -> dict:
    """Score one filter, on whatever population the clauses resolved to."""
    image_mask, gt_mask, person_mask, kind = resolve_clauses(table, index, clauses)
    if kind == "person":
        result = person_slice_metrics(table, index.persons, person_mask, score_cut)
        result["gt"] = int(np.count_nonzero(gt_mask))
        return result
    if kind == "frame":
        result = image_slice_metrics(table, image_mask, score_cut)
        result["boxes"] = gt_slice_metrics(table, gt_mask, score_cut)
        return result
    result = gt_slice_metrics(table, gt_mask, score_cut)
    result["images"] = int(np.count_nonzero(image_mask))
    return result


def _row(table: MatchTable, index: TagIndex, definition: TagDefinition, value: str) -> dict:
    """One value's row in a tag's breakdown table."""
    metrics = slice_metrics(table, index, [(definition.name, value)])
    return {"value": value, "count": definition.counts.get(value, 0), **metrics}


def tag_breakdown(table: MatchTable, index: TagIndex, definition: TagDefinition) -> dict:
    """Every value of one tag, scored, with the gap against the whole dataset."""
    kind = definition.kind
    overall = overall_metrics(table, index, kind)
    key = METRICS_BY_KIND[kind][1]
    population = POPULATION_BY_KIND[kind]

    rows = [_row(table, index, definition, value) for value in definition.values]
    for row in rows:
        # A value nothing falls into has no score, only zeros. Flag it so the
        # page prints dashes: a row reading "mAP50 0.0000" looks like a model
        # that failed, not like a bucket nobody used.
        row["empty"] = row[POPULATION_KEY[kind]] == 0
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
        "kind": kind,
        "population": population,
        "source": definition.source,
        "unit": definition.unit,
        "cuts": definition.cuts,
        "status": definition.status,
        "status_detail": definition.status_detail,
        "answered": definition.answered,
        "total": definition.total,
        "truncated": definition.truncated,
        "headline_metric": key,
        "metrics": METRICS_BY_KIND[kind][0],
        "rows": rows,
        "overall": overall,
    }


#: What the scorecard reports per row, by slice kind. Frame rows carry the full
#: set; box rows carry only what a set of ground-truth boxes can answer, so the
#: same column reads as a real number in one row and a dash in the next.
#:
#: ``swept`` says whether the confidence slider moves the column. Average
#: precision integrates the whole precision-recall curve down to the score
#: floor, so a confidence cut cannot change it — the mAP columns sit still
#: while the slider repaints the rest. Saying so in the data keeps the page
#: from looking broken at exactly the moment someone first drags it.
SCORECARD_COLUMNS = [
    ("map50", "mAP50", False),
    ("map50_95", "mAP50-95", False),
    ("precision", "precision", True),
    ("recall", "recall", True),
    ("f1", "F1", True),
]


def scorecard(table: MatchTable, index: TagIndex,
              score_cut: float | None = None) -> dict:
    """Every value of every usable tag, scored, in one table.

    The per-tag breakdowns answer "how does this one tag split the model"; this
    answers "which slice anywhere in the eval is worst", which is the question
    that actually starts an investigation and which no single breakdown can be
    read for.

    Every cell is shaded, and the bounds are per (tag, column): one scale for
    the whole table would rank a box row's recall against a frame row's mAP,
    which do not mean the same thing, and a single scale per tag would flatten
    precision against recall when the slider pulls them in opposite directions.
    Shading a column against itself, within one tag, is the only comparison on
    this page that is honest in both directions.

    Recomputed per ``score_cut`` rather than cached: that is the point of the
    slider, and it costs one pass over masks already built.
    """
    cut = resolve_score_cut(table, score_cut)
    groups = []
    for definition in index.tags:
        if not definition.usable:
            continue
        # Three populations share these columns, and each dashes out the ones
        # it cannot answer (a box row has no precision, a person row no mAP).
        # The alternative — a table per population — would put the worst slice
        # in the eval in whichever of three tables you had not scrolled to,
        # which is the one thing this view exists to prevent.
        kind = definition.kind
        overall = overall_metrics(table, index, kind, cut)
        key = METRICS_BY_KIND[kind][1]

        rows = []
        for value in definition.values:
            metrics = slice_metrics(table, index, [(definition.name, value)], cut)
            population = metrics[POPULATION_KEY[kind]]
            empty = population == 0
            # A frame slice with predictions but no ground truth has no mAP and
            # no recall — both are means over an empty set. It is a real slice
            # worth seeing (it is where false positives live), so the row stays
            # and only the undefined numbers drop out. The crowding tag makes
            # this bucket guaranteed rather than incidental.
            no_gt = kind == "frame" and not empty and metrics["gt"] == 0
            rows.append({
                "value": value,
                "population": int(population),
                "gt": int(metrics["gt"]),
                "empty": bool(empty),
                "no_ground_truth": bool(no_gt),
                "headline": None if (empty or no_gt) else metrics.get(key),
                "delta": (None if (empty or no_gt)
                          else metrics.get(key, 0.0) - overall.get(key, 0.0)),
                "metrics": {
                    name: (None if (empty or no_gt) else metrics.get(name))
                    for name, _label, _swept in SCORECARD_COLUMNS
                },
            })

        # Shading bounds per column, over this tag's rows only. A column whose
        # values are all equal (or has one usable row) spans nothing; it is sent
        # as null so the page leaves those cells unpainted rather than painting
        # an arbitrary extreme across a tag that does not actually vary.
        intensities = {}
        for name, _label, _swept in SCORECARD_COLUMNS:
            seen = [row["metrics"][name] for row in rows
                    if row["metrics"].get(name) is not None]
            low, high = (min(seen), max(seen)) if seen else (None, None)
            span = (high - low) if seen else 0.0
            for row in rows:
                value = row["metrics"].get(name)
                row.setdefault("intensity", {})[name] = (
                    None if value is None or not span else round((value - low) / span, 4)
                )
            intensities[name] = {"low": low, "high": high} if seen else None

        scored = [row["headline"] for row in rows if row["headline"] is not None]
        groups.append({
            "name": definition.name,
            "label": definition.display,
            "kind": kind,
            "source": definition.source,
            "unit": definition.unit,
            "headline_metric": key,
            "population_label": POPULATION_BY_KIND[kind],
            "bounds": intensities,
            "worst": min(scored) if scored else None,
            "best": max(scored) if scored else None,
            "overall": overall.get(key),
            "rows": rows,
        })

    return {
        "score_cut": cut,
        "score_threshold": table.score_threshold,
        "score_floor": table.score_floor,
        "at_eval_threshold": abs(cut - table.score_threshold) < 1e-9,
        "columns": [{"key": key, "label": label, "swept": swept}
                    for key, label, swept in SCORECARD_COLUMNS],
        "groups": groups,
    }


def cross_tab(table: MatchTable, index: TagIndex, tag_a: str, tag_b: str, metric: str,
              score_cut: float | None = None) -> dict:
    """One metric over every combination of two tags' values.

    The intersection, not the union: a cell is images (or boxes) answering both
    ``tag_a = row`` and ``tag_b = column``. Empty combinations are kept as
    blanks rather than dropped, because "no rainy night frames exist" is itself
    worth seeing in the grid.

    ``score_cut`` scores the whole grid at a confidence other than the eval's,
    as everywhere else on this page.
    """
    first = index.tag(tag_a)
    second = index.tag(tag_b)
    if first is None or second is None:
        raise UnknownClause(f"unknown tag in cross-tab: {tag_a} x {tag_b}")

    # The same precedence resolve_clauses applies, worked out once up front so
    # the grid can label its cells before it has scored any of them.
    if first.is_person or second.is_person:
        kind = "person"
    elif first.is_frame and second.is_frame:
        kind = "frame"
    else:
        kind = "box"
    allowed = dict(METRICS_BY_KIND[kind][0])
    if metric not in allowed:
        metric = METRICS_BY_KIND[kind][1]

    rows = []
    values: list[float] = []
    for row_value in first.values:
        cells = []
        for column_value in second.values:
            entry = slice_metrics(
                table, index, [(tag_a, row_value), (tag_b, column_value)], score_cut
            )
            population = entry[POPULATION_KEY[kind]]
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
        "metrics": METRICS_BY_KIND[kind][0],
        "population_label": POPULATION_BY_KIND[kind],
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
                "kind": definition.kind,
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
    # The first column that carries the tag decides which population it is, and
    # so which metrics the grid may offer. Compared evals can disagree -- one
    # old enough to have no people still has a box-scope pose tag -- so a column
    # whose kind differs is scored on its own terms and flagged, rather than
    # being read off a metric it does not have.
    allowed = dict(METRICS_BY_KIND[first.kind][0])
    if metric not in allowed:
        metric = METRICS_BY_KIND[first.kind][1]

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
            population = entry[POPULATION_KEY[entry["kind"]]]
            mismatched = entry["kind"] != first.kind
            cells.append({
                "value": None if mismatched or not population else entry.get(metric),
                "population": population, "absent": False,
                "mismatched_kind": mismatched, "kind": entry["kind"],
            })
        numbers = [c["value"] for c in cells if c["value"] is not None]
        best = max(numbers) if len(numbers) > 1 and len(set(numbers)) > 1 else None
        for cell in cells:
            cell["is_best"] = best is not None and cell["value"] == best
        rows.append({"value": value, "cells": cells})

    return {
        "tag": tag, "label": first.display, "metric": metric,
        "metric_label": allowed[metric],
        "kind": first.kind,
        "population_label": POPULATION_BY_KIND[first.kind],
        "columns": [{
            "label": column["label"],
            "absent": column["index"].tag(tag) is None
                      or not column["index"].tag(tag).usable,
        } for column in columns],
        "rows": rows,
        "metrics": METRICS_BY_KIND[first.kind][0],
        "warnings": _comparability_warnings(columns) + (
            ["One of these evals scores this tag over a different population "
             "(people vs ground-truth boxes), so its column is left blank "
             "rather than compared against a number that means something else."]
            if any(cell.get("mismatched_kind")
                   for row in rows for cell in row["cells"]) else []),
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


# --------------------------------------------------------------------------
# Test sets: one model, several datasets
# --------------------------------------------------------------------------
#
# An eval queued against several test datasets at once is several ordinary
# evals sharing a ``test_group`` (see ``training.services.test_sets``), so each
# set keeps its own page, hard images and comparison. What none of them can
# show on its own is the spread between sets, and the number for all of them
# together -- which is the same one-property argument as every slice on this
# page, run the other way: matching is per image, so appending one set's rows
# to another's changes no verdict, and the concatenation scores exactly as one
# eval over the union would have.


def pool_tables(tables: list[MatchTable]) -> tuple[MatchTable | None, str]:
    """Concatenate several evals' match tables into one, or say why not.

    Only for the frame metrics: the pooled table carries no prediction
    geometry, so it has no people, and no tag index, so it has no slices. Its
    image names are prefixed with the set's position, because two test sets can
    both hold a ``000001.jpg``.

    Refuses rather than approximates when the tables do not describe the same
    scoring -- a different class space would add one set's class 3 to another
    set's class 3, and a different operating confidence would make the pooled
    precision a blend of two questions.
    """
    if not tables:
        return None, "no test set has a match table yet"
    first = tables[0]
    for other in tables[1:]:
        if other.classes != first.classes:
            return None, ("the sets were scored in different class spaces, so their "
                          "classes cannot be added together")
        if list(other.iou_thresholds) != list(first.iou_thresholds):
            return None, "the sets were scored at different IoU thresholds"
        if other.score_threshold != first.score_threshold:
            return None, ("the sets were scored at different operating confidences "
                          f"({first.score_threshold:g} vs {other.score_threshold:g})")

    images, gt_parts, pred_parts, op_parts = [], [], [], []
    image_offset = gt_offset = 0
    for position, table in enumerate(tables):
        images.extend(f"{position}/{name}" for name in table.images)
        gt_parts.append((table.gt_image + image_offset, table.gt_class, table.gt_row,
                         table.gt_w, table.gt_h))
        for parts, columns in (
            (pred_parts, (table.pred_image, table.pred_class, table.pred_score,
                          table.pred_iou, table.pred_gt)),
            (op_parts, (table.op_image, table.op_class, table.op_score,
                        table.op_iou, table.op_gt)),
        ):
            image, klass, score, iou, gt = columns
            # -1 is "matched nothing" and has to stay that, not become a real row.
            parts.append((image + image_offset, klass, score, iou,
                          np.where(gt >= 0, gt + gt_offset, gt)))
        image_offset += table.num_images
        gt_offset += table.num_gt

    def _joined(parts):
        columns = [np.concatenate(column) for column in zip(*parts)]
        # Each part is already in descending-score order; a stable sort over
        # the concatenation keeps ties in set order, which is the order a
        # single eval over the sets laid end to end would have met them in.
        order = _stable_score_order(columns[2])
        return [column[order] for column in columns]

    pred_image, pred_class, pred_score, pred_iou, pred_gt = _joined(pred_parts)
    operating_is_ap = all(table.operating_is_ap for table in tables)
    if operating_is_ap:
        op_image, op_class, op_score, op_iou, op_gt = (
            pred_image, pred_class, pred_score, pred_iou, pred_gt)
    else:
        op_image, op_class, op_score, op_iou, op_gt = _joined(op_parts)
    gt_image, gt_class, gt_row, gt_w, gt_h = (np.concatenate(column)
                                              for column in zip(*gt_parts))

    return MatchTable(
        path=first.path,
        version=min(table.version for table in tables),
        iou_thresholds=list(first.iou_thresholds),
        score_threshold=first.score_threshold,
        # The pooled table only holds every set's rows above the highest floor.
        score_floor=max(table.score_floor for table in tables),
        backfilled=any(table.backfilled for table in tables),
        classes=dict(first.classes),
        images=images,
        gt_image=gt_image, gt_class=gt_class, gt_row=gt_row, gt_w=gt_w, gt_h=gt_h,
        pred_image=pred_image, pred_class=pred_class, pred_score=pred_score,
        pred_iou=pred_iou, pred_gt=pred_gt,
        op_image=op_image, op_class=op_class, op_score=op_score,
        op_iou=op_iou, op_gt=op_gt,
        operating_is_ap=operating_is_ap,
        op_geom_row=np.zeros(0, dtype=np.int64),
        op_geom_box=np.zeros((0, 4), dtype=np.float64),
        geom_cut=0.0,
    ), ""


#: The per-set table's columns, in the order the page prints them.
TEST_SET_METRICS = [
    ("map50", "mAP50"),
    ("map50_95", "mAP50-95"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("f1", "F1"),
]


def _whole(table: MatchTable) -> dict:
    return image_slice_metrics(table, np.ones(table.num_images, dtype=bool))


def test_set_breakdown(entries: list[dict]) -> dict:
    """Every test set's headline metrics side by side, plus all of them pooled.

    ``entries`` is one dict per set: ``label`` (the dataset), ``table`` (its
    :class:`MatchTable`, or None while it is still running or never wrote one)
    and ``problem`` (why there is no table). Everything else on an entry --
    links, status, which one is the current page -- rides through untouched.

    Every set's row is scored from its table by the same function the pooled row
    is, not read off the eval's stored metrics, so the rows and the pool add up
    by construction; each eval's own page already checks that function against
    the trainer's numbers.

    Two summaries, because they answer different questions: **pooled** is the
    model over every image of every set (a set twice the size counts twice),
    **mean of sets** weighs each set once (a small, hard set is not drowned out).
    A gap between them is itself worth reading.
    """
    rows, scored = [], []
    for entry in entries:
        table = entry.get("table")
        row = {key: value for key, value in entry.items() if key != "table"}
        if table is None:
            row["metrics"] = None
        else:
            row["metrics"] = _whole(table)
            scored.append((row, table))
        rows.append(row)

    # Best/worst per metric across the sets, marked rather than colour-ramped:
    # with two to five rows a ramp says nothing a marker doesn't.
    for key, _label in TEST_SET_METRICS:
        values = [row["metrics"][key] for row, _table in scored
                  if not (key.startswith("map") and not row["metrics"]["gt"])]
        if len(values) < 2 or len(set(values)) < 2:
            continue
        for row, _table in scored:
            value = row["metrics"][key]
            row.setdefault("best", {})[key] = value == max(values)
            row.setdefault("worst", {})[key] = value == min(values)

    pooled, pooled_problem = None, ""
    if len(scored) >= 2:
        table, pooled_problem = pool_tables([table for _row, table in scored])
        if table is not None:
            pooled = _whole(table)
    elif len(entries) >= 2:
        pooled_problem = "fewer than two sets have results yet"
    if pooled is not None and len(scored) < len(entries):
        pooled_problem = (f"only {len(scored)} of {len(entries)} sets have results, so "
                          "this pools those alone")

    mean = None
    if len(scored) >= 2:
        # Sets with no ground truth have an undefined mAP rather than a zero one
        # (see image_slice_metrics), so they sit the mAP means out.
        mean = {}
        for key, _label in TEST_SET_METRICS:
            values = [row["metrics"][key] for row, _table in scored
                      if not (key.startswith("map") and not row["metrics"]["gt"])]
            mean[key] = _mean(values) if values else None
        for key in ("images", "gt", "predictions"):
            mean[key] = sum(row["metrics"][key] for row, _table in scored)

    # Per class: AP50 and recall per set, keyed by class *name* so a set whose
    # table numbers classes differently still lines up (pool_tables refuses those;
    # this grid does not have to).
    names: list[str] = []
    for row, _table in scored:
        for entry in row["metrics"]["per_class"]:
            if entry["class_name"] not in names:
                names.append(entry["class_name"])
    per_class = []
    for name in names:
        cells = []
        for row, _table in scored:
            match = next((c for c in row["metrics"]["per_class"]
                          if c["class_name"] == name), None)
            cells.append(None if match is None else {
                "ap50": match["ap50"] if match["gt"] else None,
                "recall": match["recall"] if match["gt"] else None,
                "gt": match["gt"],
            })
        pooled_cell = None
        if pooled is not None:
            match = next((c for c in pooled["per_class"] if c["class_name"] == name), None)
            if match is not None:
                pooled_cell = {"ap50": match["ap50"] if match["gt"] else None,
                               "recall": match["recall"] if match["gt"] else None,
                               "gt": match["gt"]}
        per_class.append({"class_name": name, "cells": cells, "pooled": pooled_cell})

    # Flattened to one list of cells per row so the template walks columns
    # instead of looking keys up in a dict, which Django templates cannot do.
    def _cells(metrics, marked=None):
        cells = []
        for key, _label in TEST_SET_METRICS:
            undefined = key.startswith("map") and metrics.get("gt") == 0
            cells.append({
                "value": None if undefined else metrics.get(key),
                "undefined": undefined,
                "best": bool(marked and marked.get("best", {}).get(key)),
                "worst": bool(marked and marked.get("worst", {}).get(key)),
            })
        return cells

    for row in rows:
        if row["metrics"] is not None:
            row["cells"] = _cells(row["metrics"], row)

    return {
        "rows": rows,
        "scored": len(scored),
        "pooled_cells": _cells(pooled) if pooled is not None else None,
        "mean_cells": _cells(mean) if mean is not None else None,
        "pooled": pooled,
        "pooled_problem": pooled_problem,
        "mean": mean,
        "per_class": per_class,
        "per_class_columns": [row["label"] for row, _table in scored],
        "metrics": TEST_SET_METRICS,
    }


#: How each population is named where a heading has to name it.
KIND_LABELS = {
    "frame": "Frame tags",
    "person": "Person tags",
    "box": "Box tags",
}


def all_tags_table(breakdowns: list[dict]) -> list[dict]:
    """Every value of every tag on one grid, split by the population it scores.

    One table per population rather than one table overall, because the three
    do not share a metric set and never can: a frame slice has mAP, a person
    slice has specificity, a box slice has neither. Forcing them into one grid
    would mean a column that is blank for two thirds of its rows, which reads
    as missing data rather than as an undefined quantity.

    Shading is normalized **within each tag**, per column: the question a reader
    brings here is "which of this tag's values does the model struggle on",
    and scaling across tags would answer a different one -- the darkest cell
    would just be whichever tag happens to contain the dataset's hardest slice,
    every time.
    """
    groups = []
    for kind, label in KIND_LABELS.items():
        metrics = METRICS_BY_KIND[kind][0]
        entries = []
        for breakdown in breakdowns:
            if breakdown["kind"] != kind:
                continue
            rows = []
            for row in breakdown["rows"]:
                undefined = row["empty"] or row.get("no_ground_truth")
                rows.append({
                    "value": row["value"],
                    "population": row[POPULATION_KEY[kind]],
                    "empty": row["empty"],
                    "delta": row["delta"],
                    "cells": [{"key": key, "share": None,
                               "value": None if undefined else row.get(key)}
                              for key, _label in metrics],
                })
            for position in range(len(metrics)):
                numbers = [entry["cells"][position]["value"] for entry in rows
                           if entry["cells"][position]["value"] is not None]
                # One number is not a spread, and a row of identically shaded
                # cells would imply a comparison that was never made.
                if len(numbers) < 2 or min(numbers) == max(numbers):
                    continue
                low, span = min(numbers), max(numbers) - min(numbers)
                for entry in rows:
                    cell = entry["cells"][position]
                    if cell["value"] is not None:
                        cell["share"] = round((cell["value"] - low) / span, 4)
            entries.append({
                "name": breakdown["name"], "label": breakdown["label"],
                "source": breakdown["source"], "unit": breakdown["unit"],
                "rows": rows,
                "overall": [{"key": key, "value": breakdown["overall"].get(key)}
                            for key, _label in metrics],
            })
        if entries:
            groups.append({
                "kind": kind, "label": label, "metrics": metrics,
                "population_label": POPULATION_BY_KIND[kind], "tags": entries,
            })
    return groups


def person_section(table: MatchTable, index: TagIndex) -> dict:
    """The per-person view, or the reason this eval cannot have one.

    Everything keyed to people lives here rather than beside the box tables:
    the numbers answer a different question over a different population, and
    the one thing guaranteed to mislead is a precision from this section read
    next to a recall from that one as though they shared a denominator.
    """
    persons = index.persons
    if persons is None:
        return {"available": False, "detail": (
            NO_PREDICTION_GEOMETRY_DETAIL if table.version < 3 else NO_PERSONS_DETAIL)}
    if not persons.count:
        return {"available": False, "detail": NO_PEOPLE_FOUND_DETAIL}

    return {
        "available": True,
        "detail": "",
        "persons": persons.count,
        "overall": overall_metrics(table, index, "person"),
        "orphans": person_orphans(table, persons),
        "classes": [table.classes.get(class_id, str(class_id))
                    for class_id in scored_class_ids(table, persons)],
        "metrics": PERSON_METRICS,
    }


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
    box_tags = [d.name for d in usable if d.kind == "box"]
    person_tags = [d.name for d in usable if d.is_person]
    tags_without_data = [
        {
            "name": d.name, "label": d.display, "kind": d.kind,
            "source": d.source, "status": d.status, "detail": d.status_detail,
        }
        for d in index.tags if not d.usable
    ]

    return {
        "overall": overall_frame,
        "overall_boxes": overall_box,
        "breakdowns": [b for b in breakdowns if b["kind"] != "person"],
        "person_breakdowns": [b for b in breakdowns if b["kind"] == "person"],
        "all_tags": all_tags_table(breakdowns),
        "persons": person_section(table, index),
        "tags_without_data": tags_without_data,
        "frame_tags": frame_tags,
        "box_tags": box_tags,
        "person_tags": person_tags,
        "tags": [
            {
                "name": d.name, "label": d.display, "kind": d.kind,
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
        "person_metrics": PERSON_METRICS,
        "check": _reproduction_check(overall_frame, stored_metrics),
    }
