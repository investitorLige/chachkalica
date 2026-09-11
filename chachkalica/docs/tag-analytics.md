# Tag analytics

Where does the model actually fail — at night, in the rain, on the boxes an
annotator marked occluded? An eval answers one number for a whole dataset;
**Tag analytics** re-scores that same eval over the slices the dataset's
[annotation tags](annotation-tags.md) define, including arbitrary intersections
of them.

Fleet → any eval list → select one eval → **Tag analytics…**. It is also a
button on each column of the "Analyze / compare" page.

## Nothing is re-run

The page loads in milliseconds and costs no GPU, because the model is never
asked anything. Every eval writes a **match table** next to its predictions —
one row per prediction (its score, its best same-class ground-truth box, and
how much they overlap) and one per ground-truth box — and the page re-scores
that table over whatever subset you name.

That works because matching is *per image* and independent of the confidence
cut:

- a prediction's verdict only ever involves boxes in its own image, so dropping
  images cannot change the verdict of the ones that stay;
- matching runs in descending score order, so raising the cut can only remove a
  claimant, never hand its box to someone else.

So re-sorting the rows by score and walking them reproduces the trainer's own
accumulation — for the whole dataset or for any part of it.

**The page proves this to you on every load.** With no filter applied it must
reproduce the eval's stored mAP50, mAP50-95, precision and recall exactly, and
it prints the comparison. On a 13,374-image eval the largest disagreement is
6×10⁻⁸ — float32 against float64, nothing else. If it ever disagrees by more,
the banner turns red and says the slices are not to be trusted, rather than
quietly showing numbers built on a broken aggregation.

## Frame tags get everything; box tags get recall

This is the one asymmetry worth understanding before reading the tables.

A **frame tag** selects images. A slice of images is a small dataset like any
other, so it gets the whole metric set: mAP50, mAP50-95, precision, recall, F1,
per class.

A **box tag** selects ground-truth boxes — and a *prediction* carries no tags.
A false positive is a box the model invented; there is no annotator answer
attached to it, so it cannot be charged to "occluded" rather than to "clear".
Precision and mAP are therefore not defined for a box-tag slice and the page
does not print one. What is measurable is whether each tagged box was found,
and how well:

| | |
|---|---|
| **recall** | share found, at the eval's operating confidence and IoU 0.5 |
| **recall (mean IoU)** | the same, averaged over every IoU threshold — sensitive to loose boxes |
| **mean IoU** | how tightly the found boxes were found |
| **mean conf** | how confident the model was about them |

Mixing the two is allowed and is usually the interesting question: *occluded
boxes, in rainy frames*. A frame clause narrows the images, which narrows the
boxes, and the slice is scored as a set of boxes.

## The page

- **Coverage** — how many of the eval's images carry answers at all, and
  anything that did not line up (see *When the join breaks* below).
- **Whole dataset** — the baseline every row is compared against.
- **One table per tag** — every value scored, with a signed gap against the
  whole dataset so an outlier is visible without arithmetic.
- **Cross-tab** — pick two tags and a metric, get the grid. Cells are
  intersections, shaded from the grid's own worst cell to its best; an empty
  combination stays blank rather than disappearing, because "there are no rainy
  night frames" is itself worth seeing.
- **Custom slice** — add as many `tag = value` conditions as you like. This is
  the one that answers three-way questions the fixed tables cannot.

Every value of a multi-select tag counts independently: an image answering
`weather = rain, fog` is in both the rain group and the fog group.

Images or boxes with **no answer** for a tag become their own `(unanswered)`
group rather than vanishing. "The model is worse on the frames nobody tagged"
is a finding about the annotation job, and dropping them silently would hide
it.

A free-text tag keeps only its most common answers as groups — a tag with one
distinct value per image is not a grouping.

## Getting the data

Two files have to meet, and the page names whichever is missing.

**The match table** (`eval_matches.json`, or `predictions_matches.json` for a
chachak pipeline eval) is written by every eval from now on. Older evals have
none: select them and run **Build tag analytics data** — it rebuilds the table
in the trainer from the `*_predictions.pt` the eval already saved plus the
labels on disk. No checkpoint is loaded, no image is decoded, the GPU is never
touched: rebuilding the 13,374-image eval above takes **5 seconds** and
produces a 25 MB file.

**The tag answers** (`annotation_tags.json`) are written into the dataset's
label directory by `fleet sync` — see
[Annotation Tags](annotation-tags.md#where-the-answers-end-up). The page looks
for them next to whichever labels *that eval scored against*, so an eval on
annotator output reads that annotator's answers and an eval on source labels
reads the promoted ones.

An eval whose thresholds the rebuild cannot recover — one old enough that its
stored metrics predate `operating_nms_threshold`, or that swept for a best-F1
confidence rather than using a fixed one — will rebuild fine and then fail the
agreement check. That is the check working: re-run the eval instead.

## When the join breaks

Frame answers join by image filename. Box answers join by **row** — the box's
line in its `.txt` — because a geometry-only format leaves nothing else to join
on. The sidecar stores each row's class id beside it, and the page checks it
against the ground truth it is about to attach the answer to.

If they disagree, the labels and the answers have drifted apart: the dataset was
relabelled, a different annotator's output was promoted over it, or a `.txt`
was hand-edited after the sync. That image's *box* tags are then dropped rather
than attributed to whatever box now sits on that line, its *frame* tags are
kept, and the count appears under Coverage. Re-sync the annotator's project to
line them up again.

## Reading small slices

A slice of fifteen boxes has an mAP with three decimal places and no
information. Anything under 30 ground-truth boxes is flagged on the custom-slice
result. Cross-tab cells carry their population for the same reason — a dark
green cell over four images is not a finding.
