# Evaluating a model from the Datasets tab

Score a model against a dataset's labels — mAP, precision, recall, the confusion
matrix — in whatever format it will be deployed in: a catalogued `.pt`, an
exported `.onnx`, a TensorRT `.engine`, or a whole infer bundle.

Datasets → select one → **Evaluate a model on this dataset…**. The result is an
ordinary eval and lands in **Eval Pipelines** next to every model-side eval.

## Why it exists

Models → *Evaluate…* starts from a catalogued checkpoint and asks which dataset.
It can only ever score a `.pt`, because an `EvalRun` was born pointing at a
`TrainedModel`. But a `.pt`'s mAP is not the deployed model's mAP:

- **export quantizes** — fp16, and on some archs a cast that underflows;
- **export replaces the NMS** — an EfficientNMS plugin with its own top-K and
  IoU, not the framework's;
- **a bundle pins a geometry** — the tiling / person-crop pipeline its weights
  were tuned with, which is not necessarily the one you would pick by hand.

Each of those is a place accuracy can move, and the only way to know it didn't
is to score the artifact that ships. This action starts from the dataset instead
and asks which model, which is what lets it offer all three formats.

| Surface | Scores | Answers |
|---|---|---|
| Models → *Evaluate…* | a catalogued `.pt` (one, or several combined) | how accurate is this checkpoint? |
| **Datasets → *Evaluate a model on this dataset…*** | **`.pt`, artifact, or bundle** | **how accurate is the thing we actually deploy?** |
| Datasets → *Run model inference…* | `.pt`, artifact, or bundle | how *fast* is it, per real frame? |

## The form

Two steps, and both halves are the shared ones
(`training/services/inference_form.py`) that the video, camera and
dataset-inference forms use — so the model list and the pipeline knobs cannot
drift from the serving surfaces:

1. **Model type** — trained catalogue, exported artifact, or infer bundle.
2. **The model, the pipeline, and the scoring.** The page arrives prefilled from
   the model's own recorded pipeline (see [Pipeline Metadata](pipeline-metadata.md)),
   so submitting as-is scores the model through the geometry it was trained and
   is served with.

The scoring's own knobs:

- **Score threshold** (the shared half's field) — the operating point precision,
  recall, F1, the prediction counts and the confusion matrix are reported at.
- **mAP confidence floor** (default `0.001`) — the floor for AP/mAP. Keep it low
  so the precision-recall curve is swept rather than truncated.
- **Label source** — the dataset's own `labels/`, one annotator's output, or an
  explicit path.

## What it creates

The **pipeline** decides the row, exactly as the Models tab decides it:

| Pipeline | Row | Trainer endpoint |
|---|---|---|
| raw | `training.EvalRun` ("Base Eval") | `/eval` → `ml/eval_checkpoint.py` |
| anything else | `eval_pipelines.PipelineEvalRun` | `/pipeline` → `chachak/run.py` |

Both rows now carry `model_source` + `artifact_path` / `bundle_path` (the same
three columns every serving row carries), and a `model_label_snapshot` so a
finished eval still says what it scored after the artifact behind it moves.

**A bundle's pipeline is the bundle's.** It is read back out of the manifest when
the form is submitted, not taken from the page — so an operator who never pressed
*Sync bundle* still gets the geometry the bundle ships, rather than a raw eval of
a model that was tuned for tiling.

Because these are ordinary evals, everything downstream already works on them:
the **compare** page (an artifact eval next to its parent checkpoint's is the
whole point), [tag analytics](tag-analytics.md), hard images, and
promote-to-labels.

## How an artifact gets evaluated

`ml/eval_checkpoint.py::load_eval_adapter` branches on the suffix. A `.pt` is
rebuilt from its own `model_name` / `num_classes` / `params` as always; an
`.onnx` or `.engine` is handed to `chachak.infer.load_checkpoint_adapter` — the
same resolution the trainer's predict endpoint and every chachak pipeline use, so
the eval runs the artifact through the runtime that serves it.

Three things follow from that:

- **Class names come from the artifact's `.meta.json`** (`class_map`), the way
  they come from `train_dataset.classes` for a checkpoint. An artifact that
  records none is *refused*, not assumed into the eval dataset's class space —
  a wrong class order relabels every box silently.
- **The eval batch is narrowed to a TensorRT engine's built profile** (1 unless
  the engine was deliberately built wider), because the raw path hands the
  loader's whole batch to the adapter in one call.
- **The suffix branch is deliberate.** chachak's loader prefers a sibling
  `<name>.onnx` when one exists beside a `.pt` — right for serving, wrong here,
  where a `.pt` eval has to score the checkpoint that was named.

The two services that *re-read* a finished eval's saved predictions — promote to
labels, and the match-table rebuild — need to know the class space those ids are
in. For a catalogued model the trainer reads it back out of the checkpoint; for
an artifact or bundle, Django sends the `.meta.json` class list instead
(`config_gen.prediction_space`).
