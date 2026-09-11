# Dataset inference runs

Run one model over a dataset's images and find out **how fast it actually is**,
in whatever format it will be deployed in — a catalogued `.pt`, an exported
`.onnx`, a TensorRT `.engine`, or a whole infer bundle.

Datasets → select one → **Run model inference…**. Results land in
**Fleet → Dataset inference runs**, one row per run, each opening a report.

## Why it exists

Three surfaces already run a model, and none of them answered this question:

| Surface | Runs | Answers |
|---|---|---|
| Models → *Preview model on selected dataset…* | a catalogued `.pt` only | what does it see, on this image? |
| Models → *Evaluate…* | a catalogued `.pt` only | how **accurate** is it, in mAP? |
| Videos → *Run model inference…* | all three formats | what does it see, over a clip? |
| Bundle benchmarks | one bundle | how fast, in a synthetic loop? |
| **Datasets → *Run model inference…*** | **all three formats** | **how fast, per real frame, through the pipeline it will be served with?** |

The preview viewer is the closest relative and the reason this exists: it only
ever loads a `.pt`, while what ships is an engine or a bundle, and an engine's
frame cost is the number anyone actually asks about.

It is deliberately **not** an eval. There is no mAP here — that is what
*Evaluate…* is for, and a second, worse accuracy number for the same question
would only invite arguments about which one is right.

## The form

Two steps, the same ones the video action uses (both are built and parsed by
`training/services/inference_form.py`, so they cannot drift apart):

1. **Model type** — trained catalogue, exported artifact, or infer bundle.
2. **The model, and the pipeline the images go through.** The page arrives
   already filled in from the model's own recorded pipeline (see
   [Pipeline Metadata](pipeline-metadata.md)); submitting as-is serves the model
   the way it was trained. For a bundle the geometry is the bundle's — press
   *Sync bundle* and the fields fill and lock, and they are re-read from the
   manifest when the run is submitted.

Two knobs belong to the *measurement* rather than to the model:

- **Images to run** (default 200, `0` = all). A smaller number takes an
  **evenly spaced** subset, not the first N files — the first N files of a
  sorted dataset are usually one scene, and that scene's resolution and crowd
  count would be what the timing measured.
- **Warmup calls** (default 3). Untimed calls on the first image before the
  timed loop. Leave it at 3 or more: the first call for a model loads it, and a
  TensorRT engine deserializes its plan then too. Measured on a yolox-m `.pt`
  here: **290 ms for the first call, ~12 ms for every one after it.**

## What the numbers mean

The report shows three clocks, and the distinction between them is the whole
point:

| Clock | Measured by | Includes |
|---|---|---|
| **model** | the trainer, around its own inference call, GPU synchronized | image read, preprocess, forward, postprocess |
| **round trip** | the RQ worker, around the HTTP call | the model, plus HTTP + JSON + the client |
| **wall clock** | the whole run ÷ its images | the round trip, plus the per-image database writes |

`fps` on the changelist is `1000 / mean(model ms)`. The gap between the model
and the round trip *is* the wrapper overhead (~2 ms per call on this box), which
is why both are shown rather than only the flattering one.

**A trainer that reports no per-call timing** (an image built before this
feature) leaves the report with only the round trip. It says so in a banner, the
changelist marks the fps `(round trip)`, and nothing pretends a round trip is a
frame time. Rebuild the trainer image to get the model's own clock:

```bash
docker compose up -d --build trainer
```

### Where the frame goes

With the trainer's own clock the report also breaks the frame into
`load` / `infer` / `format` — the trainer is asked for the split
(`stage_timings: true` on `/predict_image`, opt-in because attributing stages
needs a GPU sync at each boundary, which inflates the total it splits). Read the
**shares**, not the absolutes; the same caveat the
[bundle benchmark](bundle-benchmarks.md)'s stage pass carries.

A `load` share far above `infer`'s is the usual finding, and it is real: on a
yolox-m `.pt` at 4K, image decode + upload was **6.0 ms against 5.6 ms of
inference**. The fix is on the input side (smaller frames, a GPU-resident
preprocess — see the bundle benchmark's 67%-preprocess finding), not a faster
card.

## The report

- **Speed** — headline tiles, then the per-image distribution (mean, p50, p90,
  p99, min, max) for both clocks. Tail percentiles are nearest-rank: every one
  of them is a real frame's time, not an interpolation between two.
- **Where the frame goes** — the stage table above.
- **What ran** — every pipeline knob the run actually used. Only the knobs its
  pipeline uses are listed, and a blank one reads "chachak default".
- **Images** — one row per image with its detections and both latencies. Click a
  thumbnail for the full-size image with the model's boxes burned on.

Boxes are stored per image (as `/predict_image` returned them) and drawn on
demand, so a run over ten thousand frames costs ten thousand small rows rather
than a second copy of the dataset on disk.

The page polls while a run is live and reloads once when it finishes — the
statistics are server-rendered, and rebuilding them in JS would be two
implementations of the same numbers. The poll is **not** a heartbeat: the run
keeps going whether or not the page is open, and **Stop** is the only thing that
ends it early. A stopped run keeps what it measured.

## Limits worth knowing

- **Images are handed to the trainer as paths, never copied**, so the dataset
  must live under the shared `data/` mount (`/app/data` under compose). The form
  refuses a dataset outside it up front rather than failing on every image.
- **Sequential, batch 1.** The trainer serializes `/predict_image` behind one
  lock and keeps one model warm. This measures the latency of a frame, not the
  throughput of a saturated GPU or a batched stream — for those, and for GPU
  memory and concurrency, use [Bundle Benchmarks](bundle-benchmarks.md), which
  drives a bundle's own runtime in a proper timing loop.
- **The GPU is shared.** A training run on the same card makes these numbers
  smaller-looking but not comparable. Unlike the bundle benchmark, nothing here
  refuses to run while the trainer is busy — a dataset run is cheap enough to be
  worth having on demand — so check what else is running before comparing two
  runs.
- One failed image is recorded on its row and counted, and the run carries on;
  five failures in a row abort it, because that is a wedged trainer rather than
  five bad images.
