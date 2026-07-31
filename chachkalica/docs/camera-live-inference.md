# Camera live inference (models over RTSP)

## Context

`cameras/` already registers an RTSP camera and shows a live MJPEG preview of
it, fed by a `stream_cameras` worker that holds the one RTSP connection and
keeps the latest JPEG in Redis (see `camera-live-preview.md`). Separately,
`videos/services/inference.py` runs a model frame-by-frame over a *recorded*
mp4 via the trainer's `POST /predict_image`.

Nothing connected the two — `cameras` had no relation to `videos`/`training` at
all. This adds that: pick a model in the camera's admin page, turn it on, and
watch the live stream with detection boxes burnt in.

Goal for this first pass is deliberately narrow: **an annotated live preview**,
one camera, one model. No persisted detections, no events/alerts, no recording.

## Design

```
      camera (RTSP)
           │  one connection, held by stream_cameras (unchanged)
           ▼
   SETEX camera:<id>:frame  ─────────┬──────────────────┐   ~6-7 fps
   SETEX camera:<id>:preview          │                  │
           │                         │ GET (full res)   │ GET (full res)
           │ (raw preview, 1280w)    │                  │
           ▼                         ▼                  ▼
   admin MJPEG view      ┌─ inference thread ──┐  ┌─ compositor thread ─┐
                         │  ~2/s (target_fps)  │  │  every raw frame    │
                         │                     │  │                     │
                         │ write raw bytes →   │  │  decode             │
                         │  camera_<id>.jpg    │  │  downscale to 1280w │
                         │        │            │  │  draw_boxes(newest) │
                         │        │            │  │  encode JPEG        │
                         │        ▼            │  │        │            │
                         │ POST /predict_image │  │        │            │
                         │  (warm model, GPU)  │  │        │            │
                         │        │ boxes      │  │        │            │
                         └────────┼────────────┘  └────────┼────────────┘
                                  │  _BoxState             │
                                  └───────────────────────►│
                                  │                        │
                       SETEX :boxes              SETEX :annotated
                                                           │
                                                           ▼
                                        admin MJPEG view  ?annotated=1
```

### Why two loops instead of one

The obvious single loop — infer, draw, publish — makes the annotated stream
advance only as fast as the model, i.e. judder along at `target_fps` (2/s, a
500 ms step). Splitting them means the annotated stream tracks the **camera**
while the boxes track the **model**:

| Stage | Rate | Set by |
|---|---|---|
| camera → RTSP → decode | native (25/30 fps) | the camera |
| → `camera:<id>:frame` | ~6-7/s | `FRAME_PUSH_INTERVAL_SECONDS` (pre-existing) |
| → `camera:<id>:annotated` | **~6-7/s** | the compositor, i.e. the push rate |
| → boxes refresh | **~2/s** | `target_fps` |

Cost is one JPEG decode + downscale + draw + encode per frame, no extra GPU.
Measured on a real 3840x2160 showroom frame: **31 ms** before preview
downscaling and **32 ms** after — the 4K *decode* dominates, so shrinking the
output (14 ms of 4K encode → 2.2 ms resize + 1.5 ms encode) buys bandwidth, not
CPU. So ~31 fps sustainable on one core —
the compositor is nowhere near the bottleneck, and `FRAME_PUSH_INTERVAL_SECONDS`
is what actually caps smoothness. Verified against the live camera: 37 annotated
frames in 6.0 s (**6.2 fps**, median gap 149 ms) while boxes were refreshing at
2/s — exactly the 150 ms push interval.

The trade-off: between inferences the boxes are up to one inference interval
stale, so on fast motion a box can lag the thing it's drawn around. This is the
same trade the recorded-video path already makes with `frame_stride`.

Raising `target_fps` above ~6-7 does nothing on its own — the model would just be
handed frames it already ran, which `frame_digest` skips. Genuinely faster
inference needs `FRAME_PUSH_INTERVAL_SECONDS` lowered too.

### Why inference reads Redis instead of opening its own RTSP stream

Two things fall out of the indirection for free, and both are the whole reason
for it:

- **No second RTSP session.** Most IP cameras cap concurrent sessions at 2-4.
  Spending one of those on a model, when a frame is already sitting in Redis,
  is the wrong trade — and it would put the model in competition with the
  preview for the camera's own connection budget.
- **Frames can never queue up.** There is only ever *one* latest frame under
  the Redis key, so a model slower than the camera drops the intermediate
  frames instead of falling progressively further behind reality. A queue (or a
  direct `VideoCapture` read loop, which buffers internally) does the opposite:
  latency grows without bound and the operator watches the past.

A stalled stream is handled too — the worker digests each frame
(`live_inference.frame_digest`) and skips inference when it's byte-identical to
the one it just ran, so a frozen camera doesn't burn GPU re-detecting the same
image.

### Why it reuses `/predict_image` rather than loading the model itself

`/predict_image` already handles `.pt`, `.onnx` and `.engine` (chachak's
`load_checkpoint_adapter` dispatches on suffix) and every chachak pipeline
(raw / tiled / person-crop / chain), and keeps the model warm in GPU memory
between calls. Reusing it means the live path needed no model-loading code at
all, and a model configured for a video or an eval works unchanged on a camera.

Two costs come with it, both accepted at this scale and both worth knowing:

- **The trainer's model cache holds exactly one entry** (`_predict_cache` in
  `friendy_chachkalica/service.py`, size 1 to bound GPU memory). So a camera
  running model A and an admin previewing model B **evict each other, and each
  eviction is a full model rebuild**. One camera on one model is fine. Two
  cameras on *different* models would thrash badly — that configuration needs a
  multi-entry cache or a dedicated in-process inference service, and is the
  natural next step if it's ever wanted.
- **`_predict_lock` serializes the GPU call** with the interactive preview, and
  the GPU itself is shared with `sam-backend` and any live training run. So
  latency is bursty, and `CameraInference.target_fps` is an upper bound, never a
  guarantee.

### Why frames go through the filesystem

`/predict_image` takes an image *path*, not bytes — the trainer runs in its own
container and opens the file itself. So each frame is written to
`data/videos/camera_frames/camera_<id>.jpg` on the tree that web/worker/trainer
all bind-mount at the same path (see the SHARED FILESYSTEM note in
`docker-compose.yml`). The **raw JPEG bytes are written through unchanged**
rather than decoded and re-encoded: the model only needs the pixels, so
re-encoding would cost CPU and a generation of quality for nothing. The frame is
decoded exactly once, for drawing.

One file per camera, reused every frame — the loop is sequential per process, so
that's safe, and it's unlinked on worker exit.

## Files

- **`cameras/models.py`** — `CameraInference`, a `OneToOneField` on `Camera`.
  A persistent *config*, not a run: unlike `videos.InferenceJob` (one row per
  finished video) it describes a loop the worker keeps alive while `enabled` is
  set. Carries the same model/pipeline field set as `InferenceJob` and
  `eval_pipelines.PipelineEvalRun` (so a model configured for one works for the
  others), plus `enabled`/`target_fps` and worker-written status fields
  (`status`, `last_error`, `last_inference_at`, `last_latency_ms`,
  `last_box_count`).
  `worker_fingerprint()` is what the reconcile loop diffs to decide a worker
  needs restarting — deliberately *excludes* the status fields, or the worker's
  own writeback would trigger an endless restart cycle.

- **`cameras/services/frame_cache.py`** — gained `set/get_annotated_frame`,
  `set/get_boxes`, `clear_annotated`, and later `set/get_preview_frame`. The
  annotated TTL is much longer than the raw one (30s vs 8s): annotated frames
  arrive at `target_fps` (default 2/s) and a single slow call stretches the gap
  further, so the raw TTL would blank a perfectly healthy stream. The preview
  key shares the raw 8s TTL — the two are pushed together, so a preview
  outliving its source would only mislead a viewer.

- **`cameras/services/live_inference.py`** — the per-frame work, split by rate:
  `build_predict_payload` (mirrors the video path's namesake — resolve model +
  pipeline knobs once), `predict_frame` (write → predict → cache boxes),
  `composite_frame` (decode → downscale → draw → encode → cache annotated),
  `frame_digest`.

- **`cameras/management/commands/infer_cameras.py`** — the worker, same shape as
  `stream_cameras`: reconcile every 15s against `CameraInference` rows,
  one **process** per camera (a slow/hung GPU-bound HTTP call can be killed;
  a thread can't), escalating stop (event → terminate → kill). A model that
  fails to *build* reports the error and idles for 60s rather than crash-looping
  the trainer — the reconcile loop restarts it as soon as the config is edited.
  Inside each process, **two threads** sharing a `_BoxState` (boxes + a version
  counter under a lock). They have to be concurrent: the inference thread is
  blocked on its HTTP call most of the time, and the compositor must keep
  publishing throughout. Both watch the same stop event, so the parent's
  shutdown ends both. The compositor skips work when neither the frame digest
  nor the box version has moved — the one cost worth avoiding is re-encoding a
  4K frame to byte-identical output.

- **`cameras/admin.py`** — `CameraInferenceInline` (StackedInline) on
  `CameraAdmin`, with the pipeline knobs in a collapsed fieldset and the worker
  status read-only. `CameraInferenceForm` turns `artifact_path` into a dropdown
  of what's actually in the export root (same source as the video-inference
  page, keeping a since-deleted saved value selectable so an unrelated edit
  can't silently blank the model), and on **enable** builds the exact
  `/predict_image` payload the worker would build — so a missing checkpoint or a
  pipeline missing its detector is a form error on save, not a `status=error`
  row discovered later. A disabled config skips that probe, so a half-finished
  config still saves.
  `mjpeg_view`/`live_view` grew an `?annotated=1` switch rather than new URLs;
  the change page's inline preview and the action's link both default to the
  annotated stream once inference is on, since that's what the operator just
  turned on.

- **`videos/services/inference.py`** — `_draw_boxes` → `draw_boxes` (public),
  now shared by the video and camera paths. Only call site updated.
  It also had to grow **resolution scaling**: it was written for ~1080p video,
  and the showroom camera is **3840x2160**, where a fixed 2px box and 0.5 font
  scale render as hairlines with unreadable labels. Strokes and text now scale
  with frame height off a 1080p reference (never *down*, so smaller frames are
  unchanged), and box/label coordinates are clamped into the frame — a detection
  whose box extended past the top edge previously had its label drawn at a
  negative y and silently vanished, which is exactly what the first real 4K
  frame did. `videos/tests.py::DrawBoxesTests` covers both.

- **`templates/admin/cameras/camera_live.html`** — shows the model, inference
  status, ms/frame and a detections snapshot, plus a raw/annotated toggle. Warns
  explicitly when `?annotated=1` is requested but inference is off, since the
  stream would otherwise just sit blank with no explanation.

- **Both `docker-compose.yml` files** — new `camera-inference` service. Not
  `<<: *app` and not a copy of `camera-workers` either: it needs the `data/`
  mount (frames reach the trainer as paths) and `TRAINING_SERVICE_URL`, which
  `camera-workers` deliberately does without, but it needs no RTSP access, no
  docker socket, and **no GPU of its own** — the trainer owns the GPU. The
  standalone `chachkalica/` stack has no trainer service, so there it targets
  one on the host via `host.docker.internal`.

## Operating it

Once deployed (see the deployment note at the bottom), in the admin:
**Cameras → (a camera) → Live inference** → pick a model,
tick `enabled`, save. Up to ~15s (one reconcile cycle) before the worker picks
it up, then the change page's preview and the "View live stream…" action show
boxes. `camera-workers` must be running too — with nothing in
`camera:<id>:frame` there is nothing to infer, and the config reports
`idle: Waiting for a live frame from the camera.`

## Known limits (flagged, not fixed)

- **One model at a time, globally** — the size-1 trainer cache described above.
  Two cameras on different models, or a camera plus an interactive preview on a
  different model, will thrash.
- **No persistence.** Detections live in Redis with a 30s TTL and are gone
  after that. No events, no alerts, no saved clips — that was explicitly out of
  scope for this pass.
- **Latency is not measured end-to-end.** `last_latency_ms` covers the
  `/predict_image` round trip (including any wait for the GPU lock), not the
  camera→browser glass-to-glass delay, which also includes the RTSP buffer,
  `stream_cameras`' push interval, and the MJPEG poll.
- **Stale boxes between inferences.** The annotated stream is smooth (~6-7 fps)
  but the boxes it draws refresh at `target_fps` (2/s), so on fast motion a box
  lags the thing it surrounds by up to 500 ms. Raising `target_fps` narrows the
  gap until the model's own latency becomes the limit.
- ~~**Full-resolution annotated frames**~~ — **fixed.** The compositor now
  downscales before drawing and publishes `:annotated` preview-sized (1280 wide;
  see `cameras/services/preview.py` and the follow-up section of
  `camera-live-preview.md`), because a browser is the only consumer of that key.
  Drawing *after* the resize also keeps labels crisp, since `draw_boxes` scales
  its lines to the frame it's handed. Full resolution is still what the model
  gets, via `:frame`.

## Verification done

Without deploying (see the deployment note on why), against the **live** stack:

1. Pulled the real latest frame for camera 11 out of Redis (1,360,420 bytes,
   JPEG magic confirmed) and wrote it to
   `/app/data/videos/camera_frames/camera_11.jpg` — the exact path the worker
   uses — then confirmed from inside the **trainer** container that it sees that
   file, at that path, with those bytes.
2. `POST /predict_image` with that path and a real checkpoint
   (`PPE_v0.4_ppl_first-37/01-...-rtdetr/best.pt`, 502 MB) returned
   `200` with the real class space
   (`helmet`/`vest`/`no_helmet`/`no_vest`) — 0 boxes at threshold 0.5, one
   `helmet` at 0.074 with threshold dropped to 0.05 (an empty showroom, so that
   is the expected answer). This confirms the raw-JPEG passthrough is readable
   by the model and the shared path contract holds.
3. Ran the real predict + composite path on those real frame bytes with only the
   trainer call and the Redis writes stubbed: confirmed the file handed to the
   trainer is byte-identical to the camera frame, and inspected the resulting
   annotated 3840x2160 JPEG visually. That inspection is what surfaced the
   4K drawing problem fixed above.
4. Ran the **real `_composite_loop`** against the **live** camera for 6 s with
   nothing stubbed — real Redis reads of camera 11's changing frames, real
   digest skipping, real 4K compositing, real `camera:11:annotated` writes:
   37 frames, 6.2 fps, median gap 149 ms, ~0.90 MB per annotated frame, while
   boxes were being refreshed at 2/s. Also benchmarked `composite_frame` at
   31 ms median on a real 4K frame (~32 fps/core) and `frame_digest` at 1.5 ms.
   Verification artifacts (`bench.jpg`, the annotated/boxes Redis keys) removed
   afterwards.
5. `videos` + `cameras` suites: 96/96 passing (`--keepdb` against real
   Postgres). Three `training.tests.ConfigGenTests` failures are pre-existing,
   from the already-modified `config_gen.py` in the working tree, and unrelated.

## Deployment note

Not deployed. Two things need a decision first:

- **The running `web` image is built from a different tree state.** It contains
  `training/migrations/0025_alter_experiment_detector_checkpoint.py`, which does
  **not** exist in the working tree — the tree has a different `0025`
  (`0025_trainingsettings_exports_root_and_more`). Neither is applied to the DB
  yet, so nothing divergent is baked in, but rebuilding `web` ships the whole
  uncommitted working tree, not just this feature.
- **`cameras/0002_camerainference` depends on that tree-side `training/0025`**,
  so bringing this up applies both (plus `videos/0005`) to the live database.

Once that's resolved, deploying is:

```bash
docker compose up -d --build web              # applies migrations, admin changes
docker compose up -d --build camera-inference # the new worker
docker compose logs -f camera-inference
```
