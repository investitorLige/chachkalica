# Infer Bundles as a Model Source

A **bundle** is one directory holding a complete inference pipeline:

```
ppe-bundle/
├── pipeline.json        # the manifest — Contract C (chachak.bundle_export.manifest)
├── models/
│   ├── model.engine     # + model.meta.json (Contract B: preprocessing, class map)
│   └── detector.engine  # + detector.meta.json — the person detector, when the pipeline uses one
├── runtime/             # vendored chachak + onnx_infer/trt_infer
├── infer.py             # runs the whole thing standalone: python infer.py IMAGES
├── requirements.txt
└── README.md
```

The export jobs write one beside every artifact (`training.jobs._bundle_after_export`
→ `<name>-bundle/` under `exports_root`), and because it is self-contained it
**travels**: scp a bundle onto another machine and it is still a runnable pipeline.

This page is about the consequence of that: a bundle is a first-class **model
source** in this app, alongside a catalogued `.pt` and a loose `.onnx`/`.engine`.

## Drop it in, pick it, sync it

1. **Drop it in.** Copy the bundle directory anywhere under the bundle root —
   `data/bundles` by default (`TrainingSettings.bundles_root`). Subdirectories are
   scanned too, so bundles can be grouped per site or per model. Anything with a
   `pipeline.json` in it is a bundle; anything else is ignored.
2. **Pick it.** Both inference surfaces offer a *Model source* of "infer bundle",
   which switches the model picker to a dropdown of what is on disk:
   - **Video inference** — Videos → "Run model inference…"
   - **Camera live inference** — the *Live inference* inline on a camera
3. **Sync it.** Press **Sync bundle**. It reads the manifest, checks the bundle,
   fills the pipeline fields from it, and locks them.

Nothing else is configured. The manifest supplies the model, the detector *and*
the geometry, so a bundle from another machine needs no knowledge of how it was
trained.

## What "Sync bundle" checks

`training.services.bundles.validate` — one JSON endpoint
(`admin/bundles/sync/`) shared by both forms. Each line comes back as ok / fail /
info, where info is a note or a skipped check rather than a problem:

| Check | Fails when |
| --- | --- |
| Manifest | no `pipeline.json`, unparseable, or a `schema_version` newer than this app supports |
| Pipeline | the manifest names a pipeline (or chain member) this app can't serve |
| Model artifact | `artifacts.model` is absent, points outside the bundle, is not `.onnx`/`.engine`, or isn't there |
| Model / detector metadata | a `.meta.json` exists but is unreadable |
| Detector artifact | the pipeline crops around detected people and the bundle carries no detector |
| Classes | the manifest carries no class names, so detections would be unlabelled |
| Format | `bundle.default_format` contradicts the model file's suffix |
| Standalone runtime | never — a bundle with no `runtime/`/`infer.py` still serves here, it just won't run on its own |
| **Load test** | the pipeline doesn't build, or doesn't run |

Everything except the load test is filesystem work: instant, and safe to run
while a camera is inferring.

One exception to the table: a bundle built on a **build node** is reported as
*info*, not *fail*, for the load test. Its engine was compiled for another GPU on
purpose, so it cannot load here and never will — see
[Build Nodes](build-nodes.md).

### The load test, and why it's the one that matters

`.engine` files are TensorRT plans **tied to the GPU model and TensorRT version
they were built on**. No amount of reading files can tell you whether the engine
in a bundle you just copied will load here — so the load test builds the
pipeline through the trainer's `/predict_image` and runs it on a synthetic
640×480 grey frame. Zero detections is a pass: what is being proven is that the
artifacts load and the pipeline assembles.

It is opt-in (a checkbox, default on) because it takes the trainer's GPU lock and
evicts its single-entry warm model — see `docs/camera-live-inference.md` on why
that matters for a camera that is already running.

**Saving never runs it.** Both surfaces re-validate structurally on submit and
never touch the GPU; the load test happens only when the operator asks.

## The bundle owns its geometry

Bundle-sourced pipeline fields are read-only in the form, and re-derived from the
manifest on save (`bundles.apply_defaults`, via `InferenceJob.sync_bundle` /
`CameraInference.sync_bundle`). Whatever a form posted for them is discarded.

That is deliberate, and it is the one place a bundle behaves differently from the
other two sources. A bundle's tile geometry, crop expansion and merge NMS were
tuned together with its weights — its own README says as much, offering only
`--conf` as a command-line option. So:

- **Locked:** `pipeline`, `detector_checkpoint`, `detector_expand_ratio`,
  `detector_min_box_size`, `tile_size_px`, `tile_width_pct`, `tile_height_pct`,
  `overlap`, `merge_nms_iou`, `chain` (`bundles.GEOMETRY_FIELDS`).
- **Yours:** `score_threshold` (prefilled from the manifest's `defaults.conf`),
  plus per-run settings like `frame_stride` and `target_fps`.

The lock in the browser is a *statement* of that; the server-side re-derivation is
what enforces it. A stale page, a hand-edited POST, or a bundle that changed on
disk since the sync all end up running the bundle as it is now — or failing.

The detector path always points at the bundle's **own** copy
(`models/detector.engine`), never at the host path recorded under `provenance`.
That's the point of a bundle: the detector travels with it.

## Manifest → pipeline metadata

The manifest is chachak's flattened `PipelineConfig`; the app's own vocabulary is
`training/services/pipeline_meta.py` (see [Pipeline Metadata](pipeline-metadata.md)).
`bundles._manifest_defaults` is the only place the two meet:

| `pipeline.json` | pipeline metadata |
| --- | --- |
| `pipeline` | `pipeline` |
| `artifacts.detector` (bundle-relative) | `detector_checkpoint` (absolute) |
| `detector.expand_ratio` / `.min_box_size` | `detector_expand_ratio` / `detector_min_box_size` |
| `tiling.tile_size_px` / `.tile_width_pct` / `.tile_height_pct` / `.overlap` | same names, flat |
| `merge_nms_iou` | `merge_nms_iou` |
| `chain` | `chain` |
| `defaults.conf` (or legacy `score_threshold`) | `score_threshold` |

## Where the paths have to be

`/predict_image` receives a **path string** and opens the file itself, so the
bundle must live where the trainer can see it. `data/` is bind-mounted at
`/app/data` in web, worker *and* trainer (see the SHARED FILESYSTEM note in
`docker-compose.yml`), which is why the bundle root defaults inside it. A bundle
root outside that tree will validate fine and fail at inference.

## A bundle that wasn't exported here — `rtmo-posture`

Most bundles in `data/bundles` came out of `chachak.bundle_export`, which means
their arch, their preprocessing and their runtime were all decided in this repo.
`rtmo-posture` is the exception worth knowing about, because it shows what a
bundle from outside has to bring with it.

RTMO is a bottom-up, whole-frame, multi-person **pose** estimator — an
mmpose/mmdeploy `end2end.onnx`, not anything trained here. Its graph emits
`(dets[B,N,5], keypoints[B,N,17,3])` rather than Contract A's
`(boxes, scores, labels)`, so three things had to exist before it could be a
model source at all:

* `onnx_infer/arch/rtmo.py` — the arch handler. Contract A has no keypoint field,
  so it *spends* each person's 17 joints on a geometry heuristic that emits one
  of three classes — `standing`, `sitting`, `lying` — and drops the joints. In
  the app this is a posture classifier that happens to be a pose model inside.
* `trt_infer/session.py` — a third output layout (`RTMO_OUTPUTS`), detected by
  tensor name, handing the ArchHandler that pair per image instead of unpacking a
  triple. **If a re-export ever renames those tensors, nothing raises**: the pair
  is silently read as `(boxes, scores, labels)`.
* `pipeline: "raw"` in the manifest — whole frame straight to the model, no
  tiling and no person-crop, because the model finds every person itself. Note
  `raw` is a *serving* pipeline (`InferenceJob.RAW`), not one of chachak's, which
  is why this bundle carries no `runtime/` + `infer.py`: chachak has no `raw`
  pipeline to generate a standalone entrypoint for. **Sync bundle** reports that
  as an `info`, not a failure — this app serves it fine, it just doesn't travel
  as a self-running directory.

Its engine is rebuilt the ordinary way, from the ONNX the bundle carries beside
it — rtmo needs no EfficientNMS surgery, so there is no `.pt` involved:

```bash
docker compose run --rm --no-deps trainer python -m ml.trt_export.cli \
    /app/data/bundles/rtmo-posture-bundle/models/rtmo.trt.onnx \
    -o /app/data/bundles/rtmo-posture-bundle/models/rtmo.engine
```

Two things that bite here and are written up in the bundle's own `README.md`:
the build needs a few GB of free VRAM and reports a shortage as
`Could not find any implementation for node`, which reads like a graph problem;
and `meta.layout: "bgr"` makes preprocess *reverse* the channel axis, so the
graph only sees the order it wants because every caller in this repo hands it
RGB. Feed it BGR and it loses about ten points of recall without erroring.

## Bundles vs loose artifacts

Both are offered; they answer different needs.

| | loose `.onnx` / `.engine` | bundle |
| --- | --- | --- |
| Addressed by | the model file, relative to `exports_root` | the directory, relative to `bundles_root` |
| Pipeline record | `<artifact>.pipeline.json` sidecar, or the catalogued model matched by filename | its own `pipeline.json` — always present |
| Detector | a sidecar copy, if the export made one | always inside the bundle |
| Prefill | silently, on change | explicitly, via **Sync bundle**, with a report |
| Geometry | editable | the bundle's |
| Survives a copy to another machine | only with its sidecars | yes, whole |

Bundles built remotely land under `bundles_root/<node name>/` — already inside the
bundle root, so they appear in the dropdowns with no copying. The node-name
directory is not decoration: an engine only runs on the GPU that compiled it, so
which machine a bundle came from is part of what it *is*. See
[Build Nodes](build-nodes.md).

The bundles the export jobs write under `exports_root` are found through *that*
setting, as loose artifacts (`<bundle>/models/model.engine` shows up in the
exported-artifact dropdown, without the manifest being read). Copy or move one
into `bundles_root` to have it treated as a bundle.

## Code

| Piece | Where |
| --- | --- |
| Discovery, translation, validation | `training/services/bundles.py` |
| Sync endpoint | `fleetsite/admin_views.py:bundle_sync_view` (`admin/bundles/sync/`) |
| Sync button behaviour | `training/static/training/bundle_sync.js` (shared by both forms) |
| Video surface | `videos/admin.py:run_inference`, `templates/admin/videos/run_inference.html` |
| Camera surface | `cameras/admin.py:CameraInferenceForm`, `cameras/static/cameras/camera_inference_form.js` |
| Storage | `InferenceJob.bundle_path`, `CameraInference.bundle_path` |
| Bundle writer (the other end) | `chachak/bundle_export/`, via `exports.build_bundle_request` |
