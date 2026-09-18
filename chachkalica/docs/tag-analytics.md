# Tag analytics

Where does the model actually fail — at night, in the rain, on the boxes an
annotator marked occluded, on the small ones, in the crowded frames? An eval
answers one number for a whole dataset; **Tag analytics** re-scores that same
eval over slices of it, including arbitrary intersections of them.

Slices come from two places:

- the dataset's [annotation tags](annotation-tags.md), answered by annotators;
- **computed tags**, measured from the eval itself and needing no annotation at
  all — so this page is useful on a dataset nobody has tagged.

Fleet → any eval list → select one eval → **Tag analytics…**. It is also a
button on each column of the "Analyze / compare" page, which additionally has a
**Per-tag comparison** section: one tag, one metric, one column per compared
eval.

## Computed tags

| tag | scope | what it is | what it costs |
|---|---|---|---|
| **crowding** | frame-wide | ground-truth boxes sharing the frame | nothing — it is in the match table |
| **box size** | box-wide | the box's linear size as a fraction of its frame | nothing, on a table that records geometry |
| **brightness** | frame-wide | mean luma of the frame | one image pass, no GPU |
| **contrast** | frame-wide | the frame's 5th–95th percentile luma spread | the same pass |
| **person size** | box-wide | how big the person this box belongs to is | one GPU pass over the frames |
| **pose** | box-wide | standing / sitting / lying | the same pass |

The first two cost nothing at all: they are read straight off the match table,
so they work on every eval that exists — crowding on any table, box size on one
written by a build that records geometry (older evals get it from **Build tag
analytics data**, which rebuilds from the predictions already on disk).

The other four need a pass over the images, run once per dataset and reused by
every eval of it afterwards — including evals that finished months ago.

### The two measuring passes

**Image statistics** (brightness, contrast) is a decode and a histogram per
frame. No GPU, no trainer, no model: *Datasets → Measure image statistics*, or
the **Measure now** button on this page's Data sources row. It depends only on
the images, so one pass serves every annotator's labels and every eval.

**Box statistics** (person size, pose) needs the trainer's GPU: a person
detector and the RTMO posture engine over each frame. It is keyed to *one label
set*, because its rows are line numbers in those `.txt` files, so start it from
the eval whose labels you want measured. It takes the trainer's single job slot
for its duration, so queued evals wait. It runs over **every** frame, including
the ones with no labels at all: a person standing in an empty frame is the only
evidence there is that the model was right to say nothing about them.

Where the person boxes come from, in order: the pipeline's own person detector
if it has one (so "person size" means the crop the model actually saw), else the
dataset's own `person` labels, else the posture engine's own detections — it is
a person detector too. The pass records which it used, and the page shows it.

A box the pass found no person around gets the named bucket `(no person)`,
which is *not* the same as "not measured": the pass looked and found nobody,
and on a person dataset that bucket is mostly the small and occluded people the
detector missed — often the interesting slice.

The pass also keeps the **people themselves** — each one's box, posture and
size, and which person every ground-truth box was attributed to. That is what
makes the per-person section below possible: "these two helmets are on the same
standing man" cannot be recovered from two rows that each say only "standing".

Continuous measures (everything but pose) are cut into **terciles** over that eval's own population,
and the cut points are printed under the tag, because a bucket called "low"
means nothing without the number behind it. The cuts fall *between* distinct
values, never through a tie: on a dataset where most frames hold exactly one
box the 33rd and 66th percentiles are both 1, and a naive tercile would render
an empty middle bucket that reads like a bug. Where a measure takes too few
distinct values to split in three it is split in two and says so; where it takes
one, it says it cannot slice at all rather than showing a single full-width row.

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

## Three populations, three metric sets

This is the one thing worth understanding before reading the tables. A tag does
not just select a slice — it selects *what kind of thing* the slice is made of,
and that decides which metrics can honestly be computed.

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

A **person tag** — `pose` and `person size` — selects *people*, and this is the
population where precision comes back. See the next section.

Mixing them is allowed and is usually the interesting question: *occluded
boxes, in rainy frames*. The most specific scope wins: a frame clause narrows
the images, which narrows the boxes and the people; a person clause makes the
whole slice a set of people; a person clause paired with a box clause keeps the
people who own at least one box the box clause selected.

## People: the population where a false positive has an owner

A box tag can only ever describe somebody who was carrying something worth
labelling. There is no row for the man lying down with no weapon on him, which
is exactly the row a precision needs.

So the people themselves are the population. Every person the box-measures pass
found is one row, and every class is one yes/no question asked twice of them —
*is there a helmet on this person in the labels*, and *did the model say there
was*, at the eval's operating confidence. That cross-tabulates:

| | model says yes | model says no |
|---|---|---|
| **labels say yes** | hit | miss |
| **labels say no** | false alarm | correct reject |

which gives **precision**, **recall**, **F1**, **specificity** and **accuracy**
per `pose = lying / standing / sitting` and per person-size tercile. The
headline numbers are micro-averaged over the classes, so every person-class
decision counts once and slices holding different class mixes stay comparable.

A prediction is attributed to a person by the same rule the ground truth was:
containment (≥ 0.7 of the box inside the person), or IoU (≥ 0.5) when the box
is itself a person, ties to the smallest qualifying person. Scoring one side by
containment and the other by IoU would compare two different notions of "on
this person".

### Boxes on nobody

The per-person tables are a closed accounting of *people*, which makes them
silently **not** an accounting of boxes. A weapon on the ground, or one worn by
somebody the person detector walked past, belongs to no row in them. Those are
counted in their own panel and scored on the one thing they support — was the
box found — so a model cannot look good on people while missing half the
objects in the dataset. Three reasons a box has no person are kept apart,
because they call for different actions:

| | |
|---|---|
| **on nobody** | the pass looked and there was no person there — a finding |
| **not measured** | the pass never saw that frame — run it |
| **drifted** | labels and sidecar disagree about which box is on which line — re-run the measures pass |

Predictions above the operating confidence are accounted for the same way, plus
a fourth case: a prediction whose record never stored a frame size carries no
box in the match table at all, which no re-run of the measures pass will fix.

### What it needs

Both halves, or the section prints the reason instead of a table:

- **person boxes** — box measures at sidecar version 2 or later. Version 1 kept
  only each box's pose, not the person behind it, so re-run *Measure box
  statistics*.
- **prediction boxes** — match table version 3 or later. Earlier tables record
  a prediction's outcome but not where it was, so it cannot be placed on
  anybody; run *Build tag analytics data* to rewrite the table from the saved
  predictions (no GPU, no re-inference).

Where either is missing, `pose` and `person size` stay box-scope and give the
recall-side view they always did — an eval does not lose a tag by being old.

## The page

- **Data sources** — what the page had to read and what each gap costs. The
  match outcomes are the only hard requirement; a missing tag sidecar removes
  some tags, not the page.
- **Coverage** — how many of the eval's images carry answers at all, and
  anything that did not line up (see *When the join breaks* below).
- **Whole dataset** — the baseline every row is compared against.
- **One table per tag** — every value scored, with a signed gap against the
  whole dataset so an outlier is visible without arithmetic.
- **People** — the per-person confusion above, its per-pose and per-size
  breakdowns, and the *Boxes on nobody* panel. Boxed off from the rest of the
  page on purpose: a precision read out of this section and a recall read out
  of a box table do not share a denominator.
- **All tags** — every value of every tag on one grid, shaded worst → best down
  each column *within each tag*. Scaling the shading across tags would just make
  the darkest cell whichever tag happens to hold the dataset's hardest slice.
  Each tag is banded under its own population and scored on that population's
  metrics, dashing the columns it cannot answer — a box band has no precision, a
  person band no mAP. A **confidence slider** re-scores the whole grid at any
  operating point down to the match table's score floor; nothing is re-run for
  it, because every prediction above that floor is already on disk. The mAP
  columns are marked as the ones the slider cannot move: average precision
  integrates the whole curve, so a confidence cut does not change it.
- **Cross-tab** — pick two tags and a metric, get the grid. Cells are
  intersections, shaded from the grid's own worst cell to its best; an empty
  combination stays blank rather than disappearing, because "there are no rainy
  night frames" is itself worth seeing.
- **Custom slice** — add as many `tag = value` conditions as you like. This is
  the one that answers three-way questions the fixed tables cannot.
- **Tags with no data here** — every tag that exists but cannot slice this eval,
  with the reason and the fix. A tag defined on the dataset but never synced, an
  option nobody has picked, a measure this eval's match table is too old to
  carry: all of them are listed rather than silently omitted, because "what
  could I have cut this by?" is the question the page exists to answer.

A declared option nobody ever picked renders as a **zero row**, not as nothing:
"no annotator marked a single frame `heavy`" is a finding about the labelling
job.

A frame slice holding no ground truth at all — the empty frames, which crowding
gives you a bucket of — reports its predictions and precision but says *mAP
undefined* rather than printing a mAP of 0.0000. A mean over no boxes is not a
score.

Every value of a multi-select tag counts independently: an image answering
`weather = rain, fog` is in both the rain group and the fog group.

Images or boxes with **no answer** for a tag become their own `(unanswered)`
group rather than vanishing. "The model is worse on the frames nobody tagged"
is a finding about the annotation job, and dropping them silently would hide
it.

A free-text tag keeps only its most common answers as groups — a tag with one
distinct value per image is not a grouping.

## Getting the data

One file is required; the page names anything else that is missing and says
what it costs.

**The match table** (`eval_matches.json`, or `predictions_matches.json` for a
chachak pipeline eval) is written by every eval from now on. Older evals have
none: select them and run **Build tag analytics data** — it rebuilds the table
in the trainer from the `*_predictions.pt` the eval already saved plus the
labels on disk. No checkpoint is loaded, no image is decoded, the GPU is never
touched: rebuilding the 13,374-image eval above takes **5 seconds** and
produces a 25 MB file.

That is the only hard requirement: with no match table there are no images and
no ground-truth rows to score. Everything below is optional, and its absence
costs tags rather than the page.

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
