# .pt Bundles

A **`.pt` bundle** is a single `.tar.gz` holding one catalogued model's promoted
checkpoint plus every fact needed to recreate its `TrainedModel` row on another
machine:

```
name-ptbundle.tar.gz
└── name-ptbundle/
    ├── manifest.json              # everything below, machine-readable
    ├── checkpoint.pt               # verbatim copy of the promoted "best" checkpoint
    ├── checkpoint.pipeline.json    # pipeline_meta.for_trained_model(), standalone sidecar form
    ├── checkpoint.detector.*       # the person detector, only if the pipeline uses one
    ├── training_config.yaml        # the exact friendy_chachkalica YAML it was trained from, if still on disk
    └── README.md                   # human/agent walkthrough
```

Built by `training.services.exports.build_pt_bundle`, queued from the admin via
`TrainedModelAdmin.export_pt_bundle` → `training.jobs.run_export_pt_bundle`
(`ExportRun.kind = "pt"`) — same shape as the ONNX/TensorRT export actions
(confirm form → queued job → row you can watch under "Export runs"), but the job
itself never touches the trainer service: it's a plain-file operation (copy,
hash, tar) against the checkpoint already on the shared filesystem, so it runs
in seconds regardless of what else is training.

## Why this exists, and how it differs from an [Infer Bundle](infer-bundles.md)

An infer bundle (`chachak/bundle_export/`) converts a model to a torch-free
ONNX/TensorRT runtime for **serving** — it deliberately drops everything about
how the weights were trained, because a consumer only needs to run inference.

A `.pt` bundle is the opposite trade: it keeps the raw checkpoint (so it needs
torch + `friendy_chachkalica` to load, unlike an infer bundle) but carries
**training provenance** an infer bundle has no schema for at all — hyperparameters,
optimizer/scheduler settings, dataset roster, augmentation config, per-run
metrics. The goal is "hand this to another machine (or another engineer, or an
agent) and they can catalog it as a Trained model and understand exactly how it
got there" — not "run this with no dependencies."

There is deliberately **no import action**. A `.pt` bundle is read by a human or
an agent, who recreates the `TrainedModel` row by hand on the destination. The
job here is only to make sure nothing they'd need is left out of the bundle.

## manifest.json

```jsonc
{
  "schema_version": 1,
  "bundle_kind": "pt_bundle",
  "exported_at": "<iso8601>",
  "trained_model": {
    "name": "...", "description": "...", "stage": "...",
    "arch": "...", "num_classes": 7, "classes": ["hardhat", ...],
    "metrics": {"map50": 0.91, ...}
  },
  "checkpoint": {
    "filename": "checkpoint.pt",
    "original_path": "<absolute path on the exporting machine>",
    "sha256": "...", "size_bytes": 123456789,
    "inspected": {"arch": "rfdetr", "trained_size": [640, 640]}  // best-effort, may be absent
  },
  "pipeline_metadata": { /* same blob as checkpoint.pipeline.json — see Pipeline Metadata */ },
  "provenance": {
    "run_result": { "run_name", "run_index", "model_arch", "train_dataset_name",
                     "best_epoch", "best_loss", "run_dir", "val_metrics", "test_metrics" },  // or null
    "training_run": { "status", "config_yaml_path", "output_dir", "started_at", "finished_at" },  // or null
    "experiment": { /* every Experiment hyperparameter — epochs, batch_size, lr,
                       optimizer_name/_params, scheduler_name/_params, pipeline,
                       detector_*, tile_*, overlap, merge_nms_iou, chain, eval_*,
                       iou_thresholds, seed, amp, gradient_clip_norm,
                       early_stopping_patience, best_metric, val_interval, device */ },  // or null
    "experiment_models": [ { "arch", "num_classes", "pretrained", "params" }, ... ],
    "experiment_datasets": [ { "dataset_name", "storage_type", "storage_root", "role",
                                "label_source", "annotator", "explicit_labels_path",
                                "aug_hflip", "aug_hflip_fraction",
                                "aug_scale_crop", "aug_scale_crop_fraction" }, ... ],
    "training_config_included": true
  }
}
```

`trained_model` and `pipeline_metadata` mirror the `TrainedModel` row and
`pipeline_meta.for_trained_model()` verbatim — see [Pipeline
Metadata](pipeline-metadata.md) for that schema's own rules (resolution order,
what each field means). `provenance.*` is reached by walking
`trained_model.source_run_result → run (TrainingRun) → experiment`; a model
promoted from an untracked checkpoint has no run to walk, so those fields are
`null` rather than the export failing.

`checkpoint.sha256` is there to verify the copy landed intact after a transfer
(`sha256sum checkpoint.pt` on the destination should match).

## Only the best checkpoint

Unlike ONNX/TensorRT export (which bundles both `best` and `last` when they
differ), a `.pt` bundle only ever carries `best` — the checkpoint the model was
actually promoted with. One bundle is meant to answer to one `TrainedModel` row;
if you need `last.pt` to resume training elsewhere, export it separately.

## Code

| Piece | Where |
| --- | --- |
| Admin action | `training/admin.py:TrainedModelAdmin.export_pt_bundle` |
| Queued job | `training/jobs.py:run_export_pt_bundle` |
| Bundle builder | `training/services/exports.py:build_pt_bundle` |
| `ExportRun` kind | `training/models.py:ExportRun.PT` |
