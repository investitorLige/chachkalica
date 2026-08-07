# Pipeline Metadata

A model is only meaningful together with the pipeline it was trained through.
Serve a person-crop model whole frames, or tile it at a size it never saw, and
the detections are quietly wrong — no error, just worse numbers.

So the convention in this project is:

> **The pipeline and its parameters are stored as metadata on the model, and
> travel with every artifact the model exports. Every action that runs a model
> prefills itself from that record — the operator confirms, they don't retype.**

This holds regardless of format: `.pt`, `.onnx`, `.engine`, or a chachak bundle.

## Where the record lives

| Format | Where the metadata is | Written by |
| --- | --- | --- |
| `.pt` (catalogued model) | `TrainedModel.pipeline_metadata` (JSON field) | `training.services.promote.promote_run_result`, at promotion |
| `.onnx` / `.engine` | `<artifact>.pipeline.json` beside the file | `training.services.exports.export_pipeline_sidecar`, at export |
| chachak bundle | `pipeline.json` in the bundle (Contract C) | `chachak.bundle_export.manifest`, from `exports.build_bundle_request` |

All three carry the same schema, defined in exactly one place:
`training/services/pipeline_meta.py`. `FIELDS` there is the whole vocabulary —
adding a field flows into new metadata, new sidecars and every consumer's
prefill at once.

```json
{
  "schema_version": 1,
  "pipeline": "people_detect_first",
  "detector_checkpoint": "/models/person.engine",
  "detector_expand_ratio": 0.15,
  "detector_min_box_size": 96.0,
  "tile_size_px": 640,
  "tile_width_pct": null,
  "tile_height_pct": null,
  "overlap": 0.2,
  "merge_nms_iou": 0.55,
  "chain": [],
  "score_threshold": 0.3
}
```

`null` means "chachak's default", and a partial record (a sidecar written before
a field existed) reads back the same way — `pipeline_meta.normalize` fills the
rest in rather than raising. `"pipeline": "raw"` is an explicit "full frames",
not a hole in the record.

## Why frozen, not derived

The record is copied off the source `Experiment` **once**, at promotion. It is
not re-derived by walking `TrainedModel → RunResult → TrainingRun → Experiment`
on every read, because a promoted model has to keep describing the run it came
from after the experiment is retuned or deleted. Resolution order in
`pipeline_meta.for_trained_model`:

1. the frozen `pipeline_metadata` field;
2. the source experiment, if the field is empty (a fixture-created row, or one
   the backfill migration missed);
3. `raw()`.

`export_pipeline_sidecar` also copies the person-detector checkpoint next to the
artifact and points the metadata at the copy, so the detector travels with the
export rather than depending on a path that may not exist wherever the artifact
ends up running.

## Consumers

Everything below reads the record above. None of them keep their own field list;
that duplication is what this module exists to prevent.

- **Video inference** (`videos.admin.run_inference`) — the newest model is
  preselected and the form is rendered *already filled* server-side, so running
  a model the way it was trained takes zero selections. Switching models
  reapplies that model's own record client-side.
- **Camera live inference** (`cameras.admin.CameraInferenceForm`) — the same map
  is attached to both model pickers as `data-pipeline-defaults` and applied by
  `camera_inference_form.js` when a model is picked.
- **Preview on dataset** and **Evaluate** (`training.admin.TrainedModelAdmin`) —
  both default to the trained pipeline via `_experiment_pipeline_defaults`.
- **Bundle export** — `exports.build_bundle_request` builds chachak's Contract C
  request from the record, so a bundle can still be produced after the source
  experiment is gone.
- **Bundle *import*** — the same road travelled backwards:
  `bundles._manifest_defaults` translates a bundle's `pipeline.json` into this
  schema, so a bundle copied in from another machine prefills the same forms with
  no DB row and no sidecar behind it. It is the one source whose record *wins*
  over the form rather than seeding it — see [Infer Bundles](infer-bundles.md).

Artifacts exported before sidecars existed still prefill:
`exports.read_pipeline_defaults` falls back to the catalogued model matched by
the `<model name>-best` / `-last` filename stem the export actions produce.

## One vocabulary, four request builders

Every path that runs a model now accepts the full `FIELDS` set, so a parameter
recorded at training time is actually applied at serving time rather than
silently falling back to a chachak default:

| Builder | Feeds | Carries |
| --- | --- | --- |
| `config_gen.pipeline_block` | training / experiment YAML | all |
| `config_gen.build_pipeline_request` | pipeline **eval** (`chachak/run.py`) | all |
| `config_gen.build_predict_request` | `/predict_image` — video, camera, preview | all |
| `exports.build_bundle_request` | chachak bundle (Contract C) | all |

Two things that had to be closed for this to hold, worth knowing if you add
another parameter:

- The trainer service's `PredictImageRequest`
  (`friendy_chachkalica/service.py`) is the schema gate for anything reaching
  `/predict_image` — a field absent there is dropped in transit, silently.
  Adding one also means adding it to `_predict_key`, or the warm-runtime cache
  will keep serving the previous value after an edit.
- `PipelineEvalRun` needed `tile_size_px`, `detector_min_box_size` and
  `merge_nms_iou` before an eval could reproduce training geometry.

**Null is not the same as absent.** chachak parses `merge_nms_iou`,
`tile_size_px` and friends with an unconditional `float()` / `int()`, so a key
present with value `None` raises rather than falling through to the default.
Every builder omits unset keys instead of emitting nulls — keep that pattern.

`detector.min_box_size` is emitted for **both** person-crop pipelines. It was
once scoped to `people_detect_first` on the theory that `batch_people` crops
fixed-size tiles and so can't shrink to the degenerate sizes the floor exists to
catch — but `BatchPeoplePipeline` only *finds* people in tiles and then crops the
original frame (`chachak.pipeline.BatchPeoplePipeline.process_batch`), so its
crops are exactly as small. chachak applies the floor for both regardless
(`crop_regions` has no pipeline gate), so scoping it in the builders meant a
`batch_people` model trained with no floor and was then served with one.

**A blank `detector_checkpoint` is resolved when the record is frozen**, not by
each consumer. `config_gen.pipeline_block` substitutes
`DEFAULT_PERSON_DETECTOR_CHECKPOINT` when it writes the training YAML, so
`pipeline_meta.from_experiment` must record that same path — otherwise the record
describes a pipeline the model was never trained through, and every consumer has
to re-guess the fallback. Three did; `config_gen.build_predict_request` raised
instead, so video and camera inference refused to run any model whose experiment
had simply left the field on its default.

## Adding a parameter

1. Add it to `pipeline_meta.FIELDS` with its "chachak default" value.
2. Map it in `pipeline_meta.from_experiment` (and add the `Experiment` field if
   it's new there).
3. Add it to the request builders in the table above, omitting it when unset.
   For `/predict_image` that also means `PredictImageRequest` **and**
   `_predict_key` in `friendy_chachkalica/service.py`.
4. Add the storage field to whichever run models record it (`InferenceJob`,
   `CameraInference`, `PipelineEvalRun`) and the input to their forms. For
   `CameraInference`, add it to `worker_fingerprint` too, or editing it won't
   restart the camera's worker.

Steps 1–2 alone make every new export and every prefill carry it. Existing
sidecars keep working — a missing key reads as the default.
