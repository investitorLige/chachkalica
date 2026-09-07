# RTMO pose estimation — the complete manual

Everything in this repo that belongs to the **pose estimation detector**, copied
into one directory, plus the manual for building its TensorRT engine and for
understanding what it actually does.

The detector is **RTMO** — a bottom-up, whole-frame, multi-person pose estimator
(mmpose/mmdeploy export). It finds every person in a frame and their 17 COCO
keypoints. This project doesn't serve keypoints: a geometry heuristic turns each
person's joints into one of three labels — `standing`, `sitting`, `lying` — and
that label is what reaches the database, because the app's detection contract has
a field for a class name and none for a keypoint array. So in the app this is a
**posture classifier that happens to be a pose model inside**.

Nothing here is a fork. Every file under `vendor/`, `bundle/` and `app_glue/` is a
**verbatim copy** of a file in the repo, at the path named in the map below. Fix a
bug in one and you must carry it across by hand.

---

## Contents

- [Quick start](#quick-start)
- [What is in this directory](#what-is-in-this-directory)
- [The detector, end to end](#the-detector-end-to-end)
  - [1. The frame arrives](#1-the-frame-arrives-bgr-uint8)
  - [2. Preprocessing — Contract B](#2-preprocessing--contract-b)
  - [3. The graph](#3-the-graph)
  - [4. Postprocessing — the arch handler](#4-postprocessing--the-arch-handler)
  - [5. The posture heuristic](#5-the-posture-heuristic)
  - [6. What the app actually stores](#6-what-the-app-actually-stores)
- [Building the TensorRT engine](#building-the-tensorrt-engine)
  - [Why the TensorRT version is not negotiable](#why-the-tensorrt-version-is-not-negotiable)
  - [The build, step by step](#the-build-step-by-step)
  - [Verifying the build](#verifying-the-build)
  - [Why this engine is fp32](#why-this-engine-is-fp32)
  - [Why no graph surgery](#why-no-graph-surgery-and-no-efficientnms_trt)
  - [Build troubleshooting](#build-troubleshooting)
- [Wiring it into the app](#wiring-it-into-the-app)
- [Tuning the posture rules](#tuning-the-posture-rules)
- [Where the ONNX came from](#where-the-onnx-came-from)
- [Measured numbers](#measured-numbers)
- [Known limits and gotchas](#known-limits-and-gotchas)

---

## Quick start

```bash
cd pose_estimation_manual
pip install -r requirements.txt

# CPU, no TensorRT needed — runs the ONNX through onnxruntime:
python run_pose.py path/to/images --model bundle/models/rtmo.trt.onnx --device cpu --draw

# GPU, the real engine. Must run under TensorRT 11.1.0.106 — use the container:
docker run --rm --gpus all -v "$PWD:/manual" -v "$PWD/imgs:/imgs" -w /manual \
    eyefrommercury-gpu-inference:latest \
    python run_pose.py /imgs --draw
```

`run_pose.py` writes `results/poses.json` — per image, every person with a pixel
box, a normalized box, a score, the posture label, and all 17 keypoints — and
with `--draw`, annotated copies under `results/annotated/`.

The keypoints are in that JSON and **nowhere in the app**; see
[What the app actually stores](#6-what-the-app-actually-stores). That is the whole
reason this script exists: when a posture call looks wrong, the joints are the
only way to see why.

---

## What is in this directory

| path | copied from | what it is |
| --- | --- | --- |
| `bundle/` | `bundles_dir/rtmo-posture-bundle/` | the registered bundle, verbatim |
| `bundle/models/rtmo.engine` | " | TensorRT **11.1.0.106**, fp32, static `[1,3,640,640]`, 222 MB |
| `bundle/models/rtmo.trt.onnx` | " | the graph it was compiled from, 176 MB — unmodified mmdeploy export |
| `bundle/models/rtmo.meta.json` | " | **Contract B**: preprocessing + the 3-class posture map |
| `bundle/models/rtmo.engine.json` | " | build provenance (precision, TensorRT version, GPU, profile) |
| `bundle/pipeline.json` | " | **Contract C**: `pipeline: "raw"` — whole frame, no tiling, no crops |
| `bundle/README.md` | " | the bundle's own README (**partly stale** — see [the heuristic](#5-the-posture-heuristic)) |
| `vendor/onnx_infer/arch/rtmo.py` | `vendor/onnx_infer/arch/rtmo.py` | **the arch handler** — output layout + the posture heuristic |
| `vendor/onnx_infer/` | `vendor/onnx_infer/` | Contract-B loader, preprocess, postprocess, the ORT session, all arch handlers |
| `vendor/trt_infer/` | `vendor/trt_infer/` | `TrtModel` (engine + context + output allocator) and `TrtAdapter` |
| `vendor/chachak/` | `vendor/chachak/` | the pipelines; RTMO uses none of them (`raw`), but `config.py` defines what `raw` means |
| `app_glue/engine_runtime.py` | `detections/services/engine_runtime.py` | the GPU worker's model pool and the `raw` short-circuit |
| `app_glue/pipeline_config.py` | `engines/pipeline_config.py` | parses/validates `pipeline.json` |
| `app_glue/annotate.py` | `detections/services/annotate.py` | Friendy `(N,6)` → the box dicts stored and streamed |
| `app_glue/smoke_engine.py` | `scripts/smoke_engine.py` | "does this artifact deserialize here?" — run it first |
| `app_glue/BUNDLES.md` | `BUNDLES.md` | the bundle format reference |
| `build/build_engine.py` | *new here* | ONNX → `.engine` + provenance sidecar |
| `build/verify_engine.py` | *new here* | engine vs. onnxruntime, on real images |
| `pose_runtime.py` | *new here* | the shared path: image → letterbox → session → people (keypoints kept) |
| `run_pose.py` | *new here* | the CLI over `pose_runtime` |

The four `*new here*` files are the only things written for this directory. The
repo has no RTMO build script — the engine was built by one that lived outside it
(`bundle/models/rtmo.engine.json` records `"builder": "rtmo build_engine.py"`), so
`build/build_engine.py` reconstructs that build. It has been run: it reproduces a
byte-comparable-sized engine that verifies against the ONNX (see
[Measured numbers](#measured-numbers)).

`app_glue/` is **reference material**, not a runnable copy: those files import
`vendor.chachak…` and Django-app siblings that only resolve from the repo root.
Read them here; run them there.

---

## The detector, end to end

```
camera frame (BGR uint8 HWC)
  │
  ├─ _bgr_to_chw ................ engine_runtime.py — /255, HWC→CHW, still BGR
  │
  ├─ preprocess ................. onnx_infer/preprocess.py, driven by meta.json
  │     channel flip (layout=bgr) → ×255 → bilinear letterbox to 640 → pad 114
  │     records Transform(scale_x, scale_y, pad_x=0, pad_y=0, orig_w, orig_h)
  │
  ├─ TrtModel.run ............... trt_infer/session.py
  │     [1,3,640,640] → dets [1,N,5], keypoints [1,N,17,3]   (already NMS'd)
  │     _split_rtmo strips the batch axis → (dets[N,5], keypoints[N,17,3])
  │
  ├─ RTMOHandler.adapt_outputs .. onnx_infer/arch/rtmo.py
  │     boxes = dets[:, :4]; scores = dets[:, 4]
  │     labels = _classify_posture(keypoints[i], boxes[i])  ← keypoints spent here
  │
  ├─ to_friendy ................. onnx_infer/postprocess.py
  │     score floor → inverse letterbox → xyxy→cxcywh normalized → (N,6)
  │
  └─ annotate.friendy_to_dicts .. detections/services/annotate.py
        {class_name: "standing"|"sitting"|"lying", confidence, cx, cy, w, h}
```

### 1. The frame arrives (BGR uint8)

The GPU worker decodes a cached JPEG with OpenCV, so frames are **HWC uint8 BGR**.
`_bgr_to_chw` transposes to CHW, divides by 255, and — deliberately — **does not
convert to RGB**:

> Channel order is left as BGR: the model's `meta.json` declares its own `layout`,
> and `vendor/onnx_infer/preprocess` performs the swap when the model wants RGB.
> Converting here as well would swap twice and silently feed every model
> wrong-ordered channels.

**Read this carefully if you write your own harness.** In the app's path the
tensor entering `preprocess` is BGR, and `meta.layout == "bgr"` makes `preprocess`
reverse the channel axis — so the graph is fed channels in **RGB** order relative
to what OpenCV read. (`preprocess`'s own module docstring describes the opposite
starting convention — an RGB input flipped to BGR — because it was written for
callers that hand it RGB. The two readings disagree on the name, not on the
bytes: what matters is that a frame must reach `preprocess` **as BGR**, exactly as
`cv2.imread` produced it.) `pose_runtime.load_image_bgr` reproduces that: it reads
with PIL, converts to RGB, then reverses to BGR so the rest of the chain sees what
the app sees. Feed it an RGB array and you get quietly worse detections, never an
error.

### 2. Preprocessing — Contract B

`bundle/models/rtmo.meta.json`, read by `vendor/onnx_infer/meta.py`:

```json
{
  "schema_version": 1,
  "arch": "rtmo",
  "num_classes": 3,
  "class_map": {"0": "standing", "1": "sitting", "2": "lying"},
  "score_threshold": 0.4,
  "input": {
    "resize_mode": "letterbox", "size": 640, "multiple": 0,
    "pad_value": 114.0, "input_scale": "byte"
  },
  "normalize": null,
  "layout": "bgr",
  "box_coords": "input_pixels",
  "clip_boxes": true
}
```

| field | value | consequence |
| --- | --- | --- |
| `resize_mode` | `letterbox` | scale (up **or** down) so the longest side is exactly 640, aspect preserved, then pad **bottom-right** to 640×640. `pad_x`/`pad_y` stay 0, which is why the coordinate inverse is a plain divide. |
| `size` | 640 | the graph's H and W are fixed at 640 in the ONNX itself; this is not a knob. |
| `pad_value` | 114.0 | the YOLO-lineage grey. Applied **after** the ×255, so it is 114/255 grey, not 114× that. |
| `input_scale` | `byte` | the CHW `[0,1]` float gets multiplied back to `[0,255]`. RTMO was trained on raw pixel values with no normalization. |
| `normalize` | `null` | no mean/std subtraction at all. |
| `layout` | `bgr` | reverse the channel axis — see the warning above. |
| `box_coords` | `input_pixels` | boxes come out in the 640×640 letterboxed frame and must be mapped back. |
| `clip_boxes` | `true` | after the inverse, clamp to the original image bounds — a person against the frame edge can otherwise be predicted into the pad margin. |
| `score_threshold` | 0.4 | the *service* floor. The graph applies its own 0.15 first. |

The resize is a hand-written `align_corners=False` half-pixel bilinear
(`preprocess._resize_chw`) that matches `torch.nn.functional.interpolate`'s
default — parity with the training-side adapter, not an OpenCV resize.

### 3. The graph

Facts read out of `bundle/models/rtmo.trt.onnx` itself:

| property | value |
| --- | --- |
| producer | `pytorch 1.12.1`, ONNX **opset 11**, IR 6 |
| nodes | 851 |
| input | `input`, `[batch, 3, 640, 640]`, float32 — batch dynamic, H/W static |
| output 1 | `dets`, `[batch, N, 5]` — `x1, y1, x2, y2, score`, N data-dependent |
| output 2 | `keypoints`, `[batch, N, 17, 3]` — `x, y, score` per COCO joint |
| pre-NMS top-k | `min(2000, candidates)` |
| NMS | ONNX `NonMaxSuppression`: max 200 boxes/class, **IoU 0.65**, **score 0.15** |

Everything is baked in: anchor decode, the top-2000 selection, NMS, and the gather
that pulls `dets` and `keypoints` by the **same** post-NMS indices. Row *i* of one
is the same person as row *i* of the other — no re-association anywhere in this
codebase, by construction.

Two thresholds live in the graph and cannot be changed without re-exporting:
**score 0.15** and **IoU 0.65**. The bundle's 0.4 is a second, higher floor applied
in `to_friendy`; lowering it below 0.15 does nothing.

### 4. Postprocessing — the arch handler

`vendor/trt_infer/session.py` picks its output layout **by tensor name**:

```python
RTMO_OUTPUTS = ("dets", "keypoints")
...
self._rtmo = set(RTMO_OUTPUTS) == set(self._output_names)
```

Four layouts are recognised — `EfficientNMS_TRT` (4 fixed outputs), passthrough
`(boxes, scores, labels)`, `rtmo` `(dets, keypoints)`, and `rfdetr_raw`
`(dets, labels)`. RTMO's pair is *not* a detection triple, so it is handed to the
handler unchanged, one pair per image (`_split_rtmo`). **If your rebuilt graph's
outputs are named anything but `dets` and `keypoints`, this detection fails
silently** and the pair is mis-read as `(boxes, scores, labels)`;
`build/build_engine.py` warns about exactly this at the end of a build.

`RTMOHandler.adapt_outputs` then produces Contract A:

```python
boxes  = dets[:, :4]                      # xyxy, 640×640 letterboxed frame
scores = dets[:, 4]
labels = [_classify_posture(keypoints[i], boxes[i]) for i in range(len(boxes))]
```

and the keypoints are dropped. The `[N,17,3]` array has nowhere to go: Contract A
is `(boxes, scores, labels)`, `Detection.boxes` stores one flat dict per box
(`cx, cy, w, h, class_id, class_name, confidence`), and the danger-zone evaluator
reads a box's bottom edge. So the
handler *spends* the keypoints on the one number that fits — a class id.

Because the graph already NMS'd, this handler runs no suppression of its own
(unlike `arch/scrfd.py`, whose graph stops before NMS).

### 5. The posture heuristic

A hand-tuned geometric rule in `vendor/onnx_infer/arch/rtmo.py` — not a trained
classifier. Joints below `KPT_SCORE_MIN = 0.3` are treated as **not visible** and
excluded from every measurement (the conventional COCO visibility floor; a real
unoccluded RTMO joint scores 0.8–1.0). A midpoint is built from both sides when
both are visible, from whichever one is visible otherwise, and is `None` when
neither is — never one side silently standing in for both.

```
shoulder = mid(L_SHOULDER, R_SHOULDER)     hip = mid(L_HIP, R_HIP)
knee     = mid(L_KNEE, R_KNEE)

if shoulder and hip:
    torso = angle_from_vertical(hip - shoulder)          # 0 = upright, 90 = flat

    if torso >= 65°:                       # LYING_TORSO_DEG
        if knee is None:            -> LYING
        hip_flexion = angle(shoulder-hip, knee-hip)      # 180 = extended, 90 = bent
        -> SITTING if hip_flexion <= 140°  # HIP_FLEXION_SITTING_MAX
           else LYING

    if knee:
        thigh = angle_from_vertical(knee - hip)
        -> SITTING if thigh >= 50°         # SITTING_THIGH_DEG
           else STANDING
    # torso upright but no visible knee: fall through to the box-aspect fallback

# fallback — only when the joints above weren't confidently visible
aspect = box_height / box_width
-> LYING    if aspect < 0.6                # LYING_ASPECT_MAX
   STANDING if aspect >= 1.5               # STANDING_ASPECT_MIN
   SITTING  otherwise
```

Why each threshold is what it is (the constants carry their own rationale in the
source; this is the short version):

- **65° torso** — a flat torso alone is not enough to call someone down. Someone
  slumped over a desk or reclined in a chair tips just as far. 55° (the original
  value) caught ordinary seated lean; 65° leaves only genuinely extreme postures
  for the hip check to arbitrate.
- **140° hip flexion** — the one measurement that does not care how the body is
  oriented in frame. Lying is an *extended* body: shoulder, hip and knee near a
  straight line, ~180°. Sitting is a *bent* body: thigh out from under the torso
  at close to a right angle, rarely past ~140° even in a recliner. Deliberately
  generous toward sitting, because sitting-misread-as-lying is the failure this
  bundle actually produced. The cost: someone lying with knees drawn up reads as
  `sitting`.
- **50° thigh** — a standing thigh hangs near vertical; a seated one runs level
  with the seat. Only consulted once the torso already reads upright.
- **0.6 / 1.5 aspect** — a last resort for legs under a desk or an overhead
  camera. A seated box is the squarest of the three, so 0.8 (the original) pushed
  merely-square seated boxes into the prone branch; a genuinely prone box is wider
  still.

> ⚠️ **`bundle/README.md` is stale on this section.** It documents the older
> two-step rule (torso ≥ 55° → lying, else thigh ≥ 50° → sitting) and the older
> 0.8 aspect fallback, with no hip-flexion check. **The code above is what runs.**
> Nothing else in that README is known to be out of date.

### 6. What the app actually stores

`to_friendy` maps each surviving detection back to the original frame and emits
Friendy `(N, 6)` = `[cx, cy, w, h, confidence, class_id]`, normalized to the
frame. `annotate.friendy_to_dicts` turns that into the dict shape used everywhere
downstream, resolving `class_id` through the bundle's class map:

```json
{"cx": 0.41, "cy": 0.78, "w": 0.30, "h": 0.12,
 "class_id": 2, "class_name": "lying", "confidence": 0.66}
```

That is the whole record. **No keypoints, no per-joint confidence, no skeleton** —
they end at `RTMOHandler`. Danger zones evaluate this box's bottom edge like any
other detection (the `raw` pipeline emits no person association, so the
person-anchored zone rule that `people_detect_first` bundles get does not apply
here). Severity per class is set in the admin, not in the bundle — a `lying`
person as a fall signal is a policy call, and `bundle/README.md` leaves it
deliberately unset.

---

## Building the TensorRT engine

### Why the TensorRT version is not negotiable

A TensorRT engine deserializes **only** under the TensorRT version that built it.
There is no compatibility shim, no retry that helps, and the failure surfaces
inside the GPU worker's reconcile loop where it reads like a configuration
problem. So:

1. Read `docker/Dockerfile.gpu`'s `TENSORRT_IMAGE` — and `.env`'s override, which
   wins. Today both say `nvcr.io/nvidia/tensorrt:26.07-py3` = **TensorRT 11.1.0.106**.
2. Build **inside that container**, not against whatever `pip install tensorrt`
   put on the build box. They have drifted apart before and it fails silently at
   deserialize time, not at build time.
3. The version you built with goes into `<engine>.json` as `tensorrt_version`,
   which the admin shows and `app_glue/smoke_engine.py` checks.

Changing `TENSORRT_IMAGE` means re-building every engine in every bundle. The
comment block at the top of `docker/Dockerfile.gpu` is the history of that
happening twice; read it before touching the tag.

### The build, step by step

```bash
cd pose_estimation_manual

docker run --rm --gpus all \
    -v "$PWD:/manual" -w /manual \
    nvcr.io/nvidia/tensorrt:26.07-py3 \
    python build/build_engine.py bundle/models/rtmo.trt.onnx \
        -o bundle/models/rtmo.engine
```

(`eyefrommercury-gpu-inference:latest` works just as well and already has torch,
which the script uses only to record the GPU name in the sidecar.)

What it does, in order:

1. `trt.init_libnvinfer_plugins` — registers the standard plugins. RTMO needs
   none, but the other archs' engines embed `EfficientNMS_TRT` and this script is
   usable for them too.
2. `builder.create_network()` with **no flags**. In TensorRT 11 every network is
   strongly typed; `trt.BuilderFlag` has no `FP16` member at all (checked: the
   enum is `DEBUG, DIRECT_IO, DISABLE_COMPILATION_CACHE, …, TF32, VERSION_COMPATIBLE,
   WEIGHT_STREAMING`). Precision is a property of the graph's tensor dtypes.
3. `trt.OnnxParser.parse_from_file` on the **unmodified** export. Every parser
   error is printed; a Float/Half mismatch here means you are on the `--fp16` path.
4. A single static optimization profile, `min == opt == max == (1, 3, 640, 640)`.
   A wider profile makes TensorRT size its device memory for the widest case —
   that is the failure `TrtModel`'s execution-context error message warns about.
5. `build_serialized_network`, written to the `-o` path.
6. `<engine>.json` provenance written beside it — precision, requested precision,
   TensorRT version, GPU, input name, profile, batch, source, UTC timestamp.
7. A last check that the outputs are still named `dets` and `keypoints`, because
   `vendor/trt_infer/session.py` dispatches on exactly those names.

Expected console output (measured on an RTX 4070, 2026-09-02):

```
[build] tensorrt 11.1.0.106
[build] parsing /manual/bundle/models/rtmo.trt.onnx
[build] input  input (-1, 3, 640, 640)
[build] output dets (-1, -1, 5)
[build] output keypoints (-1, -1, 17, 3)
[build] profile input min=opt=max=(1, 3, 640, 640)
[build] wrote /manual/bundle/models/rtmo.engine (222 MB) in 65s
[build] wrote /manual/bundle/models/rtmo.engine.json
```

Flags worth knowing: `--batch` (default 1), `--hw` (default 640 — only meaningful
if you re-exported the ONNX at another size, since the graph fixes H/W),
`--workspace-gib` (default 4), `--verbose` (TensorRT VERBOSE; worth it the first
time a build fails), `--fp16` (see below).

### Verifying the build

**Never skip this.** A build is a recompilation and can pick a tactic that changes
numbers. The reference is onnxruntime on CPU, which shares nothing with TensorRT
but the graph.

```bash
docker run --rm --gpus all -v "$PWD:/manual" -v "$PWD/imgs:/imgs" -w /manual \
    eyefrommercury-gpu-inference:latest bash -c \
    "pip install -q onnxruntime; \
     python build/verify_engine.py bundle/models/rtmo.engine --images /imgs"
```

```
image                               trt  ort     Δbox    Δscore     Δkpt  posture
---------------------------------------------------------------------------------
people.jpg                            1    1    0.128  1.19e-06    0.176  1x standing
---------------------------------------------------------------------------------
compared     6/6 images
worst Δbox   0.128 px (tolerance 1.0)
worst Δscore 1.19e-06 (tolerance 0.01)
worst Δkpt   0.176 px

OK: the engine matches its ONNX within tolerance.
```

Read it in this order:

1. **Detection count** — a mismatch is *the* finding. It means NMS or the score
   floor landed differently, and nothing below it is meaningful. Exit code 1.
2. **Δbox / Δscore** — sub-pixel and ~1e-6…1e-4 is a normal fp32 recompilation.
   Whole pixels, or 1e-2 on scores, is not.
3. **Δkeypoint** — same, over all 17 joints of every person.
4. **Posture agreement** — the only number that matters downstream, since the
   label is all the app keeps. A disagreement is a *warning*, not a failure: a
   person sitting exactly on a threshold can flip on a 1e-4 keypoint difference.
   More than a stray image means the thresholds are too tight for that footage,
   not that the build is bad.

Use real footage from the cameras this will run on. A synthetic frame proves the
engine loads and binds; it proves nothing about detections.

Then prove it deserializes the way the app will load it:

```bash
docker compose run --rm gpu-inference python scripts/smoke_engine.py /data/bundles/rtmo-posture-bundle
```

### Why this engine is fp32

fp16 was requested and the build fell back — recorded honestly in the sidecar as
`"requested_precision": "fp16"`, `"precision": "fp32"`, so it is never mistaken
for the fp16 build that was actually asked for.

Because TensorRT 11 networks are strongly typed, fp16 means **casting the ONNX
graph's tensors first**, not setting a builder flag. Two casters were tried:

| caster | result |
| --- | --- |
| `onnxruntime.transformers.float16` | produced a graph TensorRT's parser rejects — duplicate cast-node output names |
| `onnxconverter_common.float16` | got two nodes further, then a genuine `Float`/`Half` type mismatch on an ElementWise `Mul` inside the decode/NMS machinery |

The second is a real type error in the cast graph, not a tactic-search failure, so
it would need a hand-patched graph to fix. That was not worth chasing: fp32
already runs at ~8.8 ms/frame single-stream on an RTX 4070, far inside any
camera's frame rate. `build/build_engine.py --fp16` reproduces the attempt (with
`onnxconverter_common`) and falls back to fp32 with the sidecar recording both,
so the failure stays reproducible rather than becoming folklore.

### Why no graph surgery, and no `EfficientNMS_TRT`

The other archs in this repo needed export-time surgery to become
TensorRT-buildable — a fixed top-K rewrite, or the NMS+gather rebuilt as an
`EfficientNMS_ONNX_TRT` node. **RTMO needs none.** TensorRT's own ONNX parser
converts the exporter's `NonMaxSuppression` node straight into an internal NMS
layer, with dynamic per-image output and no fixed `keep_top_k` padding: TensorRT
reports each image's real row count through the output allocator, the same
mechanism `TrtModel` already uses for the `EfficientNMS_TRT` engines. So
`bundle/models/rtmo.trt.onnx` is the unmodified `end2end.onnx` and builds as-is.

The one change the repo needed was on the runtime side:
`vendor/trt_infer/session.py` learned the 2-tensor `(dets, keypoints)` layout,
since everything there previously assumed a 3-tensor detection triple.

### Build troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `deserialize_cuda_engine` returns None, or a serialization-version error at load | The engine was built with a different TensorRT than the loader has. Compare `<engine>.json`'s `tensorrt_version` with `trt.__version__` in the container. Rebuild, or pin `TENSORRT_IMAGE` to match. |
| Parser errors mentioning `Float`/`Half` | You are on `--fp16`. Drop it; see [Why this engine is fp32](#why-this-engine-is-fp32). |
| `build_serialized_network` returns None | TensorRT logged the reason above it. Re-run with `--verbose`. Usually workspace or an unsupported node. |
| `failed to create a TensorRT execution context` at load | Device-memory allocation sized far larger than the model needs — almost always an optimization profile that is too wide. Rebuild with the static profile this script defaults to. |
| Engine builds, then every detection is nonsense | Check the output names. `vendor/trt_infer/session.py` dispatches on `{dets, keypoints}`; anything else falls through to the passthrough path and the pair is read as `(boxes, scores, labels)`. |
| Detections are plausible but consistently poor | Channel order. The chain must be BGR into `preprocess`; see [1. The frame arrives](#1-the-frame-arrives-bgr-uint8). |
| Build succeeds on the dev box, fails in the container | You built against a pip `tensorrt`. Build inside the container. |

---

## Wiring it into the app

The bundle format is documented in full in `app_glue/BUNDLES.md`; this is the
RTMO-specific path.

**`pipeline.json` — Contract C.** `pipeline: "raw"`: the whole frame goes straight
to the model. No tiling (RTMO is a whole-frame model), no person-crop (it *is* the
person finder). `raw` is not a chachak pipeline at all — `InferenceConfig.is_raw()`
makes `ModelRuntime.run` short-circuit to `adapter.predict([chw])` and skip
`build_pipeline` entirely (`app_glue/engine_runtime.py`, and `RAW` in
`vendor/chachak/config.py`).

```json
{
  "schema_version": 1,
  "name": "rtmo-posture-bundle",
  "artifacts": {"model": "models/rtmo.engine"},
  "pipeline": "raw",
  "chain": [],
  "score_threshold": 0.4,
  "infer_batch_size": 1,
  "merge_nms_iou": 0.5
}
```

`infer_batch_size: 1` is structural: the engine's profile is a static
`[1,3,640,640]`. `merge_nms_iou` is inert for `raw` (nothing to merge).

**Registering it**, once the engine verifies:

```bash
# 1. does it load here at all?
docker compose run --rm gpu-inference python scripts/smoke_engine.py /data/bundles/rtmo-posture-bundle

# 2. does it register cleanly?
docker compose run --rm web python manage.py shell
>>> from engines.models import Bundle, available_bundle_dirs
>>> available_bundle_dirs()
>>> b = Bundle(name="rtmo-posture", bundle_path="rtmo-posture-bundle"); b.clean(); b.refresh_from_bundle()
```

3. **Models & engines → Bundles → Add** in the admin; save; then set class
   severities for `standing` / `sitting` / `lying` on that same page (they only
   appear after the first save).
4. Point a camera at it — **Cameras → the camera → Inference → Bundle**.
5. If you touched anything under `vendor/`:
   `docker compose build gpu-inference && docker compose up -d gpu-inference`,
   then tail the logs and confirm the bundle loads with no traceback.
6. Not done until a real camera on this bundle has produced a `Detection` row with
   a plausible `class_name` and confidence.

**Cost while resident.** The GPU worker holds one set of engines per *distinct*
bundle that some enabled camera resolves to — five cameras on this bundle cost one
copy; a bundle nothing points at is released within ~15 seconds.

---

## Tuning the posture rules

The thresholds are the first place to look when a camera's real footage disagrees
with the labels. They live at the top of `vendor/onnx_infer/arch/rtmo.py` (and, to
take effect, in the repo's own copy at the same path — this directory is a copy).

| constant | now | raise it to… | lower it to… |
| --- | --- | --- | --- |
| `KPT_SCORE_MIN` | 0.3 | trust fewer joints; more people fall through to the box-aspect fallback | trust noisier joints; angles get computed from bad points |
| `LYING_TORSO_DEG` | 65 | fewer people reach the flat-torso branch at all | ordinary seated lean starts being arbitrated by the hip check |
| `HIP_FLEXION_SITTING_MAX` | 140 | more flat-torso people read as `sitting` | more read as `lying` |
| `SITTING_THIGH_DEG` | 50 | more upright people read as `standing` | more read as `sitting` |
| `LYING_ASPECT_MAX` | 0.6 | more fallback cases read as `lying` | fewer |
| `STANDING_ASPECT_MIN` | 1.5 | fewer fallback cases read as `standing` | more |

Workflow: collect frames from the camera that is getting it wrong, run
`python run_pose.py <frames> --draw`, and look at the joints in
`results/poses.json` before moving a number. The drawn skeleton usually shows the
answer immediately — a "lying" false positive is nearly always a forward-bent
crouch whose hip is bent but whose knees weren't visible enough to be measured.

Because the labels are just class ids, a change here needs no re-export and no
rebuild — only a `gpu-inference` restart (`vendor/` is baked into the image, so
rebuild it too).

---

## Where the ONNX came from

`bundle/models/rtmo.trt.onnx` is an **mmpose/mmdeploy `end2end.onnx`**, exported
outside this repo — the export kit is not in the tree, and the graph's own
metadata (`producer: pytorch 1.12.1`, opset 11, decode + top-k + NMS + gather all
baked in, outputs named `dets`/`keypoints`) is the mmdeploy pose-detection export
signature. Everything in this manual downstream of that file is reproducible here;
the export itself is not, and the model checkpoint it came from is not in this
repo either.

If you need to re-export from an mmpose checkpoint, that is an mmdeploy job
(`tools/deploy.py` with a `pose-detection_rtmo_onnxruntime_dynamic` style config)
and its output must land with:

- input `input`, shape `[batch, 3, 640, 640]`;
- outputs named exactly **`dets`** `[B,N,5]` and **`keypoints`** `[B,N,17,3]`,
  gathered by the same post-NMS indices;
- COCO-17 joint order.

Anything else and `vendor/trt_infer/session.py`'s name-based dispatch and
`RTMOHandler`'s joint indices both break. **This re-export path has not been run
here** — treat it as a starting point, and re-verify with
`build/verify_engine.py` against the old engine's numbers before trusting it.

---

## Measured numbers

**From the bundle's original verification** (17 real images — this project's own
camera frames, COCO/mmpose test images, and a public fall-detection sequence):

- detection count matched onnxruntime CPU **exactly on every image** (2–9 people
  per frame);
- worst box delta **0.65 px** (on a 9-person frame), most under 0.2 px;
- worst score delta **9.8e-4**, most under 1e-4;
- spot checks: two chair-sitters → `sitting`/`sitting`; a batter mid-swing →
  `standing` with the crouching catcher → `sitting`; a standing skier →
  `standing`; an office frame with one person standing at a desk and one reclining
  with legs on a table → `standing` / `lying` (0.66 conf).

**Re-measured on 2026-09-02** on this machine (RTX 4070, TensorRT 11.1.0.106,
container `eyefrommercury-gpu-inference:latest`), using the scripts in this
directory:

| what | result |
| --- | --- |
| `build/build_engine.py` on the shipped ONNX | 222 MB engine in **65 s**, fp32, static `[1,3,640,640]` |
| rebuilt engine vs. onnxruntime CPU (6 images) | counts equal; worst Δbox **0.126 px**, Δscore **1.19e-06**, Δkpt **0.198 px** |
| **shipped** engine vs. onnxruntime CPU (same images) | counts equal; worst Δbox **0.128 px**, Δscore **1.19e-06**, Δkpt **0.176 px** |
| shipped engine, per-image latency | **14 ms** steady state (best of 5), 76 ms first call including warm-up — this is preprocess + engine + decode, not engine alone; `bundle/README.md` records ~8.8 ms for the engine by itself |
| ONNX on CPU (onnxruntime, this box) | ~330 ms/image |
| posture agreement TRT vs. ORT | identical on every image |

The images were a single-person COCO test frame (a standing skier, correctly
labelled `standing` by both runtimes) repeated six times — enough to exercise the
build/verify loop and time steady state, **not** a re-run of the 17-image
verification above. Use your own footage before trusting a rebuild.

---

## Known limits and gotchas

- **The engine is not portable.** One TensorRT version, and effectively the GPU
  family it was built on. Rebuild from `bundle/models/rtmo.trt.onnx` anywhere else.
- **Keypoints stop at the arch handler.** Nothing downstream — database, SSE
  payload, admin, danger zones — can carry them. `run_pose.py` is the only way to
  see them.
- **Batch is 1**, structurally: the engine's profile is static `[1,3,640,640]`.
  The *graph's* batch axis is dynamic, so a wider profile is buildable
  (`--batch N`), but `pipeline.json` and every whole-frame bundle here run one
  image per forward pass.
- **Input is fixed at 640×640** in the graph. `--hw` only matters for a graph
  re-exported at another size.
- **fp32 only** — see [Why this engine is fp32](#why-this-engine-is-fp32).
- **The posture rule is hand-tuned geometry**, not a trained classifier. It has
  known failure shapes: a person lying with knees drawn up reads as `sitting`; a
  deep forward crouch reads as `lying`; legs under a desk drop you into the
  box-aspect fallback, which is a guess.
- **The ONNX escape hatch does not load through the app's own loader for this
  bundle.** `load_onnx_adapter` resolves the sidecar as
  `path.with_suffix(".meta.json")` — correct for `rtmo.engine` → `rtmo.meta.json`,
  but for `rtmo.trt.onnx` it looks for `rtmo.trt.meta.json`, which the bundle does
  not carry. Copy `rtmo.meta.json` to `rtmo.trt.meta.json` if you need that path
  in the app. `pose_runtime.resolve_meta_path` works around it here, so
  `run_pose.py --model bundle/models/rtmo.trt.onnx` is fine.
- **`bundle/README.md`'s posture section is stale** — see
  [the heuristic](#5-the-posture-heuristic).
- **These are copies.** A fix made in this directory does not reach the running
  system. The live files are `vendor/onnx_infer/arch/rtmo.py`,
  `vendor/trt_infer/session.py`, and `bundles_dir/rtmo-posture-bundle/` in the
  repo root.
