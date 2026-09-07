# rtmo-posture-bundle — RTMO pose -> standing / sitting / lying

A whole-frame, multi-person pose estimator (RTMO, mmpose/mmdeploy export)
repurposed as a **posture classifier**. It finds every person in the frame and
their 17 COCO keypoints, and a geometry heuristic
(`vendor/onnx_infer/arch/rtmo.py::RTMOHandler`) turns those keypoints into one
of three classes per person — `standing`, `sitting`, `lying` — since that's
what actually has somewhere to go: Contract A (`boxes, scores, labels`) has no
field for a raw keypoint array, only a class label, so this is what gets
computed once per person and kept.

## What's in here

| path | what it is |
| --- | --- |
| `models/rtmo.engine` | TensorRT 11.1.0.106, **fp32**, static `[1, 3, 640, 640]` |
| `models/rtmo.trt.onnx` | the unmodified export it was compiled from (`end2end.onnx` — no graph surgery was needed, see below) |
| `models/rtmo.meta.json` | preprocessing contract (Contract B) + the 3-class posture map |
| `models/rtmo.engine.json` | build provenance |
| `pipeline.json` | `pipeline: "raw"` — whole frame straight to the model, no tiling, no person-crop |

No `runtime/infer.py` — unlike the fully-portable bundles here, this one
needed no export-time graph surgery to make it TensorRT-buildable, so there's
no standalone inference code worth vendoring; the app runs it entirely through
`vendor/onnx_infer` / `vendor/trt_infer`.

## Why fp32, not fp16

Requested fp16; the build fell back. TensorRT 11 is strongly typed, so fp16
here means casting the ONNX graph's tensors first, not a builder flag.
`onnxruntime.transformers.float16`'s cast produced a graph TensorRT's parser
rejected (duplicate cast-node output names); `onnxconverter_common`'s cast
avoided that but produced a different one two nodes later — a genuine
`Float`/`Half` type mismatch on an `ElementWise Mul` inside the NMS/decode
machinery, not a tactic-search failure. Chasing a hand-patched fp16 graph
further wasn't worth it: fp32 already runs at ~8.8ms/frame (single stream) on
an RTX 4070, far inside any camera's real frame rate. `models/rtmo.engine.json`
records `"requested_precision": "fp16"` / `"precision": "fp32"` so this is
never silently mistaken for the fp16 build that was actually asked for.

## No graph surgery, and no `EfficientNMS_TRT` plugin

The exporter bakes anchor decode, a 2000-wide pre-NMS top-k, and NMS itself
(`NonMaxSuppression`, ONNX opset 11) into the graph, then gathers **both**
`dets` (`x1,y1,x2,y2,score`) and `keypoints` (17x3) by the same post-NMS
indices — so the two outputs are already row-for-row matched, no
re-association needed.

TensorRT's own ONNX parser converts that `NonMaxSuppression` node straight into
an internal NMS layer, with dynamic per-image output (no fixed
`keep_top_k` padding — TensorRT reports each real count through its output
allocator, the same mechanism this app already uses for the EfficientNMS_TRT
engines). No plugin substitution, no hand rebuilding the NMS+gather as an
`EfficientNMS_ONNX_TRT` node, no fixed top-K rewrite — the unmodified export
builds and runs correctly as-is. `vendor/trt_infer/session.py` needed one
addition to recognize this arch's 2-tensor `(dets, keypoints)` output shape
(everything else there assumed a 3-tensor `(boxes, scores, labels)` detection
triple).

## The posture heuristic

Per detected person, using confidence-gated (`score >= 0.3`) COCO keypoints:

1. **Torso angle** (shoulder-midpoint to hip-midpoint, degrees off vertical)
   `>= 55°` -> `lying`. An upright person's torso is never that close to
   horizontal; a person down, fallen, or slumped flat is.
2. Otherwise, **thigh angle** (hip-midpoint to knee-midpoint) `>= 50°` ->
   `sitting`; less -> `standing`.
3. If the needed keypoints aren't confidently visible (legs under a desk, a
   steep camera angle), falls back to box aspect ratio (`height/width`):
   `< 0.8` -> `lying`, `>= 1.5` -> `standing`, else `sitting`.

This is a **hand-tuned geometric rule**, not a trained classifier — thresholds
live in `vendor/onnx_infer/arch/rtmo.py` with their own rationale, and are the
first place to look if a camera's real footage disagrees with them.

## Verified

Against 17 real images (this project's own camera frames, plus COCO/mmpose
test images and a public fall-detection sequence — a clean flat-on-the-floor
photo wasn't available locally; the "lying" cases exercised are two men deep
in a forward-bent crouch/fall, which the heuristic correctly reads as
torso-horizontal):

* **Detection count matched onnxruntime CPU exactly on every image** (2-9
  people per frame).
* **Box delta**: worst 0.65px (a 9-person frame), most under 0.2px.
* **Score delta**: worst 9.8e-4, most under 1e-4.
* Visual spot-check (this project's own office camera, and COCO/mmpose test
  images): two chair-sitters -> `sitting`/`sitting`; a batter mid-swing ->
  `standing`, the crouching catcher -> `sitting`; a standing skier ->
  `standing`; an office camera frame with one person standing at a desk and
  one reclining with legs up on a table -> `standing` / `lying` (0.66 conf).

## Classes

| id | name | severity |
| --- | --- | --- |
| 0 | standing | *(set in the admin — Bundles → this bundle → class severities)* |
| 1 | sitting | *(set in the admin)* |
| 2 | lying | *(set in the admin)* |

Not set here on purpose — severity is a policy call (BUNDLES.md: "a class name
only means what the model emitting it means by it"), and this bundle's obvious
candidate use (a `lying` person as a fall/collapse signal) is exactly the kind
of judgment that belongs in the admin, not hardcoded at export time.
