# VLM dataset runs

Asking a vision-language model a question about every image of a dataset, and —
when the dataset is labeled — scoring what it said. The companion to
`vlm-live-inference.md`, which does the same over a video and does not score
anything.

## The action

**Models → select one → "Run on a dataset…"**. Three fields: which dataset, how
many images, and whether to grade. The prompt is not asked for; it lives on the
model row and travels with it, the same way the video action works, and the run
snapshots the prompt, the model configuration and the dataset's `classes.txt` so
that editing any of them later cannot rewrite what a finished run did.

The result is a row in the third tab, **Dataset runs**, which opens a report:
the answers image by image, and the analytics.

It is one action with a grading checkbox rather than two separate actions
because whether a dataset *can* be graded is a fact about the folder on disk,
not about the operator's intent — the form checks for a `labels/` folder itself
and says so. The grading *configuration* is deliberately not on that form
either; see "Re-grading is free" below.

`image_limit` takes an **evenly spaced** subset rather than the first N files: a
sorted dataset's first 100 images are usually one scene, one camera, or one
class, so "the first 100" measures something other than the dataset. At roughly
a second an answer, a 10 000-image dataset is hours of GPU — the field defaults
to a cap for that reason.

## How a free-text answer is scored

The two sides speak different languages. Ground truth is a YOLO
`labels/*.txt` — class ids and boxes. The model answers in prose. There are no
boxes in an answer, so nothing here pretends to score detection: **a label file
is reduced to the set of class names it annotates, and the answer is searched
for those names.**

```
labels/a.txt: "0 0.5 0.5 0.2 0.2"   →  {helmet}          ← ground truth
answer:       "A worker in a hard hat, no vest."          ← what the model said
                          └── alias of helmet   └── negated, so not a claim
                                              →  {helmet}  ← predicted
                                   {helmet} == {helmet}    →  success
```

An image is a **success** (an *exact match*) when the answer names every class
the label file annotates and no class it does not. A right answer with an extra
claim is not a success — which is why the report also breaks the result down
per class, so a mismatch says *which* half went wrong.

Three details in that reduction are load-bearing, and all three live in
`vlm/services/grading.py`:

**Aliases.** `classes.txt` says `helmet`; the model says "hard hat", or just
"yes". Each class carries extra terms, and separators and simple plurals are
already flexible, so `hard_hat` matches "hard hat", "hard-hats" and "hardhat"
without configuration.

**Negation.** "No helmet is being worn" *contains the word helmet*. A plain
substring test scores that as helmet-present — the exact inverse of the truth —
so a mention preceded by a negation cue in the same clause is read as an
absence claim instead. Without this, any question-shaped prompt scores
backwards, which makes the whole page a lie rather than merely imprecise. It is
a heuristic (five words back, not across a clause boundary) and it can misfire,
so it is a checkbox, and the report can re-grade with it off to see exactly what
it was doing.

**A missing label file is not an empty one.** No file means *unlabeled*: the
image is answered but left out of the scoring entirely, and counted separately
so a half-labeled dataset cannot quietly inflate an accuracy. An empty file
means *nothing is present* and is a real negative the model can get right or
wrong. `fleet.services.analytics` draws the same distinction over the same
folders.

Images whose call **failed** are also left out of the scoring, and counted as
failures instead: an image the backend never answered about is a gap in the
measurement, not a wrong answer, and scoring it as "named no classes" would
charge the model a false negative for an HTTP timeout.

## What the report shows

* **Exact-match accuracy**, successes, and how many images were graded.
* **Micro and macro F1** — micro over all class decisions, macro over the
  classes that took part. A class in `classes.txt` that is never labeled and
  never claimed is left out of the macro average rather than scored zero.
* **Per class**: precision, recall, F1, TP/FP/FN, and how many images label it.
* **Confusion**, over the images whose label file annotates *exactly one* class
  — the only case where a wrong answer has a single meaning. Columns add
  `(none)` and `(multiple)` for answers that named nothing or several things,
  because on a two-class dataset those two columns are usually where the story
  is. Same green-diagonal shading as the eval-analytics matrices.
* **Every answer**, with the image, the labeled classes, the classes the answer
  claimed, and a ✓/✗ — plus a "mismatches only" filter, which is what makes a
  several-thousand-image run navigable.
* **Clicking an image** opens it full size with its **ground-truth annotation
  burned on** — the label file's own boxes and polygons, captioned with the class
  name, in the same palette the dataset label preview uses so a class is the same
  colour in both places. That is the question a mismatch always raises ("what was
  actually labeled here?") and it is one click from the row. The thumbnails
  themselves stay unannotated: 300 rows must not mean 300
  decode-draw-encode round trips per page load, so only the click pays.

  This deliberately does not reuse `videos.services.inference.draw_boxes` — that
  one draws *model* output and bakes a confidence into every caption, and a
  ground-truth box captioned "helmet 0.00" would be worse than no caption. It
  also cannot draw polygons, which label files can carry.
  `vlm/services/label_render.py` is the ~40 lines that can.

## Re-grading is free

The expensive half of a run is the answers, and they do not change. So the raw
text is stored per image and grading is a pure function over it: the report's
alias table has a **Re-grade** button that re-scores a finished run under a new
configuration, no GPU involved.

This is also why the aliases are not on the run form. The terms worth setting
are the ones you pick *after* seeing what this model actually says about these
images — guessing them beforehand is how you end up re-running an hour of
inference to fix a synonym.

## Differences from a video run

Same worker, same backend, same one-model-warm-at-a-time constraint
(`ml_backends/vlm/service.py` — alternating two VLM rows will reload the model
on every image and be unusably slow). Three deliberate differences:

* **Nothing pauses it.** A video run pauses when its live page is closed,
  because a frame costs GPU time and a closed tab is watching nothing. A dataset
  run is a measurement you start and come back to, so it keeps going; its poll
  is *not* a heartbeat, and the Stop button is the only way to end it early.
* **Images are not copied.** A dataset lives under `data/source/<name>/` and
  `vlm-backend` mounts the same `data/` at the same path, so the file the worker
  sees is the file the backend opens. Only when `source_dir` points outside that
  mount is each image staged into `data/vlm_frames` first.
* **One bad image does not void the run.** A per-image failure is recorded on
  its row and counted, and the loop carries on; five failures *in a row* mean
  the problem is not the images, and abort rather than writing thousands of
  identical rows.

## Known limits

* No batching — images go one at a time, as on the video path.
* Scores are recomputed every 50 images while a run is going, so a mid-run
  refresh shows numbers without making the scoring pass quadratic.
* The report renders the first 300 answer rows; the mismatch filter is the way
  through a bigger run.
* Grading is presence-of-class only. Nothing here can tell you the model put the
  helmet in the wrong place, only that it did or did not mention one.
* Weights are still the gate: a model whose snapshot is not in `data/hf_cache`
  cannot load on this network, and the action refuses to start (see
  `vlm-live-inference.md` under "Weights are the gate").
