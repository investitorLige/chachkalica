# Bundles — the model format this project runs

A **bundle** is a directory containing everything needed to run one trained
detection pipeline: the model weights, an optional person detector, and the
exact pipeline (tiling, person-crop expansion, thresholds) it was trained and
exported with. You register a bundle once in the admin; each camera then picks
which registered bundle it runs, falling back to a system-wide default.

This document is the format reference. For the day-to-day admin workflow, see
the [README](README.md#quick-start) and [Configuration
layering](README.md#configuration-layering).

## Why a bundle, not a loose file

A detection model is not just weights — *how a frame reaches it* is part of
what was trained. A model trained on cropped-out people produces nonsense on a
whole frame; a model trained on 640px tiles produces nonsense on a downscaled
4K frame. That pipeline shape has to travel with the weights, not be
reconstructed by whoever wires the camera up later. A bundle is that pairing,
made portable: one directory, one `pipeline.json`, done.

## Directory layout

```
ppe-v0.1/                              ← bundle_path, relative to BUNDLES_DIR
├── pipeline.json                      ← required: the whole pipeline, see below
├── models/
│   ├── model.engine                   ← required: the detection model
│   ├── model.meta.json                ← required: Contract B (pre/post-processing)
│   ├── model.engine.json              ← optional: build provenance
│   ├── detector.engine                ← optional: a person detector, if the
│   ├── detector.meta.json             │   pipeline crops people first
│   └── detector.engine.json           ← optional
└── per-camera-overrides/              ← optional
    ├── gate-1.json                    ← matched to Camera.name == "gate-1"
    └── loading-dock.json
```

Everything under `BUNDLES_DIR` is bind-mounted **read-only** into every
container — nothing in the running system ever writes into a bundle. Artifacts
may be `.engine` (TensorRT, tied to the exact GPU + TensorRT version they were
built on) or `.onnx` (portable, slower). A bundle can carry both; `Bundle`
registration only requires whichever one `pipeline.json`'s `artifacts` block
actually names.

## `pipeline.json`

The whole pipeline, in one file, at the bundle's root.

```json
{
  "schema_version": 1,
  "name": "ppe-v0.1",
  "artifacts": {
    "model": "models/model.engine",
    "detector": "models/detector.engine"
  },
  "pipeline": "people_detect_first",
  "chain": [],
  "score_threshold": 0.4,
  "infer_batch_size": 4,
  "merge_nms_iou": 0.3,
  "tiling": {
    "tile_size_px": null,
    "tile_width_pct": 50.0,
    "tile_height_pct": 50.0,
    "overlap": 0.2,
    "nms_iou": 0.5
  },
  "detector": {
    "score_threshold": 0.5,
    "person_class_name": "person",
    "person_class_id": null,
    "expand_ratio": 0.1,
    "nms_iou": 0.5,
    "min_box_size": 224.0,
    "batch_size": 4
  }
}
```

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes | Must be `1`. Guarded so an older backend refuses a newer bundle loudly instead of mis-reading it. |
| `name` | no | Cosmetic; the registered `Bundle.name` in the admin is what's actually used. |
| `artifacts.model` | yes | Path to the detection model, relative to the bundle root. |
| `artifacts.detector` | only if the pipeline needs one | Path to the person-detector model. |
| `pipeline` | **yes** | One of `raw`, `batch_detect`, `people_detect_first`, `batch_people`, `chain`. See [Pipelines](#pipelines) below. |
| `chain` | only for `pipeline: chain` | Ordered list of the other pipeline names to run and merge. |
| `score_threshold` | no (default `0.5`) | Confidence floor for the *model's* detections. |
| `infer_batch_size` | no (default `4`) | Tiles/crops per forward pass. Clamped down to the model artifact's exported max batch — see `models/model.engine.json`'s `batch.max`. |
| `merge_nms_iou` | no (default `0.5`) | De-duplicates overlapping detections after tiling/cropping/chaining. |
| `tiling.*` | only for `batch_detect` / `batch_people` | Tile geometry. `tile_size_px: null` means "use the percentages below", not "unset" — an *explicit* null is a real value here. |
| `detector.*` | only if the pipeline needs a detector | The person-detector's own threshold, class, and crop-expansion settings. `person_class_id: null` means "resolve by `person_class_name`", also a real value, not absence. |

Unlike a per-camera override file, `pipeline.json` should be a **complete**
document — every field the pipeline actually needs. Anything you leave out
still gets a sane default (chachak's own), but a bundle exists precisely so a
consumer never has to guess what a model needs; write down everything you know.

## `models/*.meta.json` — Contract B

Each artifact (`model.*`, `detector.*`) carries its own `<stem>.meta.json`
next to it — how to pre-process an input and interpret that graph's raw
output. This is unrelated to the pipeline and doesn't change based on how the
model is wired up; it's produced by whatever exported the `.engine`/`.onnx` in
the first place. `Bundle.clean()` validates it (`vendor/onnx_infer/meta.py`)
but never writes it.

`models/*.engine.json` (optional) is build provenance — precision, the
TensorRT version it was built with, the batch/input-size profile it was
optimized for. Purely informational, shown read-only in the admin; used to
diagnose a TensorRT version mismatch (`scripts/smoke_engine.py` checks it too).

## `per-camera-overrides/<camera name>.json`

Optional, one file per camera that needs to run this bundle differently.
Matched by `Camera.name` exactly (`gate-1` → `per-camera-overrides/gate-1.json`).
A camera with no matching file just runs the bundle's `pipeline.json` as-is.

The file is read from **whichever bundle that camera is set to run**. Pointing a
camera at a different bundle therefore leaves its old override behind — that is
the intent (an override describes how a frame reaches *this* model, and says
nothing about another one), but it does mean a camera moved between two bundles
needs an override in each if it wants one in both.

An override is the **same document shape** as `pipeline.json`'s pipeline
fields, but **sparse** — list only what this camera changes:

```json
{
  "schema_version": 1,
  "tiling": { "overlap": 0.35 }
}
```

This camera inherits everything else — pipeline choice, thresholds, the rest
of `tiling`, all of `detector` — from the bundle. A field is merged
independently within `tiling`/`detector`, so overriding `tiling.overlap`
doesn't reset `tiling.tile_width_pct`.

A more complete override, switching this one camera to a different pipeline
entirely:

```json
{
  "schema_version": 1,
  "pipeline": "people_detect_first",
  "detector": { "expand_ratio": 0.2, "min_box_size": 96 }
}
```

**Absent vs. explicit `null` still matters.** Leaving `tiling` out of an
override means "inherit the bundle's tiling entirely." Writing
`"tiling": {"tile_size_px": null}` is a real instruction ("use percentage
tiling for this camera, even if the bundle uses a fixed pixel size") — the
same distinction `pipeline.json` itself relies on.

**Validation.** Every override file present when a `Bundle` is registered (or
re-saved) is validated then — merged onto the bundle's own config and checked
the same way `pipeline.json` is, so a bad override is caught in the admin, not
at 3am in the GPU worker's log. An override dropped onto disk *after*
registration, without re-saving the bundle, is only caught live: the GPU
worker's reconcile loop reports it against that one camera
(`inference_status: error`) rather than crashing — it degrades one camera, not
the whole system.

### Writing an override without hand-editing JSON

```bash
python manage.py write_camera_override ppe-v0.1 gate-1 \
    --pipeline people_detect_first \
    --detector-expand-ratio 0.1 --detector-min-box-size 64 \
    --notes "wide entrance, needs bigger crops"
```

Run `python manage.py write_camera_override --help` for every flag — they
mirror `pipeline.json`'s fields one-to-one (`--tile-width-pct`, `--overlap`,
`--merge-nms-iou`, `--chain`, …). Only the flags you pass are written; add
`--dry-run` to preview without writing.

## Pipelines

| Pipeline | What it does | When |
|---|---|---|
| `raw` | Whole frame straight to the model | Subjects are already large in frame |
| `batch_detect` | Overlapping tiles, detect each, NMS-merge | Wide views; distant subjects |
| `people_detect_first` | Detect people, crop each, detect in the crop | Per-person attributes (PPE) |
| `batch_people` | Tile, find people per tile, then crop | Wide views *and* per-person work |
| `chain` | Several of the above, merged | Recall matters more than GPU cost |

`people_detect_first` and `batch_people` are the two that need
`artifacts.detector` — `Bundle.clean()` rejects a bundle (or an override) that
picks one of these without a detector artifact.

Those two also change how danger zones read: because every detection they emit was
found *on* a specific person, a zone is evaluated at **that person's feet** rather
than at the detection's own bottom edge, and the person themselves is never emitted
as a detection. See [Danger zones](README.md#danger-zones).

## Registering a bundle

1. Prove it loads first: `python scripts/smoke_engine.py /data/bundles/ppe-v0.1`
   inside the `gpu-inference` container (see the [README](README.md)).
2. **Models & engines → Bundles → Add** in the admin. Enter a `name` and the
   `bundle_path` (relative to `BUNDLES_DIR`). Save.
3. Saving parses `pipeline.json`, every artifact's `meta.json`, and every
   `per-camera-overrides/*.json` file already present — reject on anything
   that wouldn't actually run, otherwise cache the resolved pipeline and class
   maps into read-only fields you can see on the same page.
4. On that same page, set the **class severities** and confidence thresholds for
   this bundle. They are per bundle, not global: a class name only means what the
   model emitting it means by it, so two bundles are free to grade the same word
   differently. A brand-new bundle needs one save before its classes appear here,
   since the list comes from the manifest that save just read.
5. Point cameras at it — **Cameras → the camera → Inference → Bundle**, or the
   *"Assign a detector to the selected cameras…"* action on the camera list. To
   make it the fallback for every camera that hasn't chosen, set it as the
   **default bundle** on **Models & engines → Active model**.
6. Re-saving a `Bundle` row re-reads everything from disk. That's how you pick
   up a re-exported bundle, or a `pipeline.json` you hand-edited. The class
   severities you set in step 4 survive it; the thresholds do not, since those
   are a database-only override of what `pipeline.json` says.

**What each registered bundle costs while in use.** The GPU worker holds one set
of engines per *distinct* bundle some enabled camera resolves to. Registering a
bundle costs nothing until a camera runs it; five cameras sharing one bundle cost
one model's worth of VRAM; five cameras on five bundles cost five. A bundle the
last camera stops using is released within about 15 seconds.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "No such directory" | `bundle_path` doesn't resolve under `BUNDLES_DIR`, or the bind mount doesn't have it yet. |
| "pipeline.json not found" | Wrong `bundle_path`, or the file isn't at the bundle's root (not inside `models/`). |
| "artifacts.model is missing" | `pipeline.json`'s `artifacts.model` path doesn't exist relative to the bundle root. |
| "is missing its Contract-B sidecar" | The artifact has no `<stem>.meta.json` beside it. |
| "needs a person detector, but artifacts.detector is not set" | `pipeline` (or an override's `pipeline`) is `people_detect_first`/`batch_people`/a `chain` containing one, but no detector artifact is named. |
| A camera's resolved pipeline isn't what you expect | Check *Resolved configuration* on the camera's admin page — it names the bundle it resolved to and shows the override file status and the fully merged config, so you can see which layer supplied which value. |
| An override file is ignored | It has to live in the bundle *that camera is set to run*, not the one it used to. The camera page's *Per-camera override* row names the path it actually looked at. |
| One camera errors, the rest keep running | Expected: a bundle that fails to load takes down only the cameras pointing at it. The error names the bundle. |
| Engine fails to deserialize in the GPU worker but not in the admin | TensorRT version mismatch — the admin never loads the artifact itself (`Bundle.clean()` only reads sidecars), so this only surfaces at runtime. Compare `tensorrt_version` on the bundle's page against `TENSORRT_IMAGE`. |
