# ONNX inference service (`onnx_infer`) — design & plan

## Context

Trained detectors are saved as `best.pt`/`last.pt` training checkpoints
(`friendy_chachkalica/ml/train.py:311`). To run one for inference, `chachak`
reconstructs the model architecture in Python — `load_checkpoint_adapter`
(`chachak/infer.py:29`) calls `build_model(model_name, num_classes, **params)`
then `load_state_dict`. That means every consumer needs the *architecture code*
importable: `friendy_chachkalica` plus each arch's heavy deps (`transformers`
for RT-DETR, the `rfdetr` package, torchvision, the vendored YOLOX, and — once
added — `ultralytics`).

**Goal:** be able to load and run a trained model **without the architecture
class definitions or their packages**. We do this by exporting each trained
model to **ONNX** (graph + weights, self-contained) plus a small `meta.json`
sidecar, and adding a **new `onnx_infer` service** in `chachkalica_unified`
whose only deps are `onnxruntime`, `numpy`, `pillow`. Inference (image → boxes)
then needs none of the training stack.

**Non-goal / important:** `best.pt`/`last.pt` stay exactly as they are. They
carry optimizer/scheduler/scaler/epoch/history for training resume
(`train.py:213`, `:357`) which ONNX cannot hold. The ONNX artifact is a
**separate, additive deployment artifact**, not a replacement for the checkpoint.

### Key facts that shape the design (verified)

- Every adapter's `predict()` returns the **same** `[N,6]` "friendy" tensor:
  `(x_center, y_center, width, height, confidence, class_id)`, boxes normalized
  cxcywh (`friendy_chachkalica/formats.py:6`, `xyxy_prediction_to_friendy`).
- Every consumer touches the model **only** through `.predict(images)`:
  `chachak/detector.py:37/67`, `pipeline.py:71/125/239/266`, `preview.py:92`.
  `predict_adapter` (`infer.py:78`) already tolerates adapters with or without a
  `score_threshold` kwarg via try/except.
- Registry archs today: `retinanet`, `rfdetr`, `rtdetr`, `yolox`
  (`friendy_chachkalica/registry.py:13`). `adapters/yolo.py` is **empty** and
  there are zero `ultralytics`/`yolov*` references — so "regular YOLO" is a new
  5th arch, added here for completeness.
- Post-processing (the arch-specific hard part) lives **outside** the
  `nn.Module`, in each adapter's `predict()`:
  - RetinaNet (`adapters/retinanet.py:48`): torchvision emits boxes directly.
  - YOLOX (`adapters/yolox.py:81`): vendored `postprocess()` = decode + conf + **NMS**.
  - RT-DETR (`adapters/rtdetr.py:73`): HF `post_process_object_detection` = sigmoid + top-k (no NMS).
  - RF-DETR (`adapters/rfdetr.py:90`): DETR-style sigmoid + top-k, then drop background class (`label == num_classes`).

## Two frozen contracts (the whole design hinges on these)

The exporter and the service are decoupled by two contracts. Get these right and
the rest is mechanical.

### Contract A — the ONNX graph (uniform across all 5 archs)

- **input**: `pixel_values`, `float32`, `[1, 3, H, W]`, already pre-processed
  (resized/normalized/padded as that arch wants). Dynamic axes on `H`/`W`
  (fixed square for RF-DETR). Batch is always 1.
- **outputs** (three named tensors):
  - `boxes` `[N,4]` — xyxy in the graph's **own input-tensor pixel space**
    (`box_coords: "input_pixels"`), i.e. decode is fully baked in.
  - `scores` `[N]` float32
  - `labels` `[N]` int64
- Decode, sigmoid/top-k, NMS, and the RF-DETR background-class drop are all
  **baked into the graph** at export time. No arch emits grid coords, normalized
  cxcywh, or letterboxed-with-hidden-pad — always input-pixel xyxy.

### Contract B — `meta.json` (sidecar next to the `.onnx`)

```json
{
  "schema_version": 1,
  "arch": "rtdetr",
  "num_classes": 3,
  "class_map": {"0": "helmet", "1": "head", "2": "person"},
  "score_threshold": 0.5,
  "input": {
    "resize_mode": "square | longest_side | none",
    "size": 560,          // square
    "max_size": 640,      // longest_side
    "multiple": 32,        // pad each side up to this (0 = none)
    "pad_value": 0.0,
    "input_scale": "unit | byte"
  },
  "normalize": { "mean": [0.485,0.456,0.406], "std": [0.229,0.224,0.225] },
  "layout": "rgb",
  "box_coords": "input_pixels"
}
```

`normalize: null` for archs that don't normalize at the tensor boundary (YOLOX
feeds raw pixels; RetinaNet's normalize is folded into the export wrapper).
`schema_version` is guarded on load so an old service fails loudly on a newer
artifact instead of silently mis-decoding.

`clip_boxes` (bool, default false) clamps mapped-back boxes to the original
image bounds. Set per-arch to match whether that arch's torch path clips: YOLOX
`true` (its `clip_xyxy`), RetinaNet `false` (already clipped internally — a
no-op either way), RT-DETR / RF-DETR `false` (they don't clip, so the service
must not either, or it diverges). It's a meta flag, not arch branching in the
core.

### Coordinate mapping (decided: bake decode in graph, back-map in service)

- The graph resolves the **decode frame** → boxes in input-tensor pixel space.
- Input-pixels → original-image-normalized happens in the **service**, because
  only the service knows the original size and the resize it applied. It is a
  **single uniform inverse** for all archs: `preprocess` records the exact
  `(scale_x, scale_y, pad_x, pad_y)` it applied per image; `postprocess` inverts
  precisely that. No per-arch coordinate logic anywhere.

## Package layout (`chachkalica_unified/onnx_infer/`)

One `.py` file per architecture under `arch/`, dispatched through a registry —
mirroring the existing `friendy_chachkalica/ml/adapters/<arch>.py` +
`registry.py::MODEL_REGISTRY` convention. The generic core stays generic; each
arch file supplies only what is genuinely arch-specific.

```
onnx_infer/
  __init__.py
  meta.py         # ModelMeta dataclass + load/validate (schema_version guard)
  session.py      # OnnxModel: load .onnx + .meta.json, own the InferenceSession, provider selection
  preprocess.py   # generic meta-driven resize/normalize/pad -> (ndarray[1,3,H,W], Transform)  [pure numpy]
  postprocess.py  # generic (boxes,scores,labels)+Transform+orig size+threshold -> [N,6]        [pure numpy]
  adapter.py      # OnnxAdapter: .predict()/.to()/.eval() — drop-in for torch adapters (lazy torch)
  errors.py
  arch/
    __init__.py   # ARCH_REGISTRY: {"retinanet": RetinaNetHandler, ...}; get_handler(name)
    base.py       # ArchHandler protocol
    retinanet.py
    yolox.py
    rtdetr.py
    rfdetr.py
    yolo.py       # new (ultralytics)
```

- `ArchHandler` (in `base.py`) is a thin protocol each arch file implements:
  - `default_input_spec()` / `default_normalize()` — the arch's preprocess
    defaults (used when building/validating `meta.json`; `meta` still wins).
  - `adapt_outputs(raw_ort_outputs) -> (boxes, scores, labels)` — maps the raw
    ORT session output list to the canonical 3-tensor of **Contract A**. Every
    arch (retinanet/yolox/rtdetr/rfdetr) exports through our own wrapper that
    emits the three tensors in order, so all four use `PassthroughHandler`
    (identity). RF-DETR does **not** use the `rfdetr` package's native exporter
    (see the RF-DETR build-order note): its head math is baked into our wrapper
    like the others, so no arch-specific `adapt_outputs` is needed after all.
- `preprocess.py`/`postprocess.py` keep the **uniform** math (resize/pad +
  the single coordinate inverse); the handler only feeds params and the output
  shim. `adapter.py` looks up the handler via `get_handler(meta["arch"])`.
- The core (meta/session/preprocess/postprocess/arch handlers) is **pure numpy**
  — no torch, no training packages. Usable from a genuinely torch-free service.
- `OnnxAdapter.predict()` **lazily** imports torch and returns `torch.Tensor`
  so `chachak` consumers are untouched (decision: lazy torch in the adapter only).

### `OnnxAdapter` surface (mirrors the torch adapters exactly)

```python
class OnnxAdapter:
    name: str                    # meta["arch"]
    num_classes: int
    score_threshold: float
    def to(self, device): ...    # picks ORT providers (CUDA/CPU); returns self
    def eval(self): return self  # no-op
    def predict(self, images, score_threshold=None) -> list[Tensor]:
        # images: list of CHW float tensors (same input the torch adapters take)
        # returns: list of [N,6] friendy tensors — identical shape/semantics
```

`preprocess` returns `(tensor, transform)`; `postprocess` for **every** arch:
`boxes_input_px → subtract pad → divide by scale → clip → xyxy_to_xywhn(orig_w, orig_h)`
→ append score/label → `[N,6]`. Runtime `score_threshold` applied here so
callers keep per-call override behavior.

## Wire-in (auto-detect, zero consumer changes)

In `chachak/infer.py::load_checkpoint_adapter`:

```python
onnx_path = Path(checkpoint_path).with_suffix(".onnx")
if onnx_path.exists():
    return load_onnx_adapter(onnx_path, device)   # new
# ... existing torch build_model path ...
```

`load_onnx_adapter` returns `(OnnxAdapter, info)` with `info` shaped like the
existing dict (`model_name`, `num_classes`, `params`, `train_classes` from
`meta["class_map"]`). `detector.py:67` and everything below is untouched.

## Export side (separate, out of the service — `friendy_chachkalica/ml/onnx_export/`)

Same one-file-per-arch structure, own registry:

```
friendy_chachkalica/ml/onnx_export/
  __init__.py
  cli.py          # export_onnx CLI: checkpoint path -> best.onnx + best.meta.json
  registry.py     # EXPORT_REGISTRY: {"retinanet": export_retinanet, ...}
  arch/
    retinanet.py  # ExportWrapper(nn.Module) + build_meta() for this arch
    yolox.py
    rtdetr.py
    rfdetr.py     # our wrapper: LWDETR.export() for traceability + head baked in-graph
    yolo.py
```

`cli.py` rebuilds the adapter the same way `chachak/infer.py:29` does, then
dispatches to the arch's exporter (like `build_model`): wrap `adapter.model` in
the per-arch `ExportWrapper` (Contract A output), `torch.onnx.export` with
dynamic axes, write `best.onnx` + `best.meta.json` next to the checkpoint. Runs
in the training env (which already has every arch dep). Standalone CLI,
re-runnable on existing checkpoints. The service-side `arch/<arch>.py` and the
export-side `arch/<arch>.py` are the two halves that must agree on Contract A/B
for that arch — kept adjacent by name on purpose.

## Per-arch specifics (5 archs)

| Arch | resize_mode | normalize | multiple | Graph internals (baked at export) | Risk |
|------|-------------|-----------|----------|-----------------------------------|------|
| retinanet | none | null (folded into wrapper) | 32 | wrapper prepends normalize; torchvision postproc yields boxes | low |
| yolox | none | null (raw px) | 32 | decode grid + `torchvision.ops.batched_nms` in graph | med |
| rtdetr | none | mean/std | 32 | sigmoid + top-k (no NMS) in graph | med (opset≥16, eval) |
| rfdetr | square | mean/std | 0 | our wrapper: `LWDETR.export()` for traceability, then sigmoid+top-k+cxcywh→xyxy+clamp+bg-drop in-graph | med |
| yolo (new) | longest_side (letterbox) | null | 32 | ultralytics native export; letterbox-undo folded so output is input-pixels | low |

Notes:
- **RetinaNet normalize** stays inside the wrapper; graph input is raw
  (`input_scale` records unit vs byte); `normalize: null`.
- **RF-DETR** background filter (`label == num_classes`) baked into the wrapper.
- **YOLO/YOLOX letterbox**: the letterbox-internal offset is folded into the
  graph so output boxes are in input-pixel space; the service's uniform inverse
  still undoes the service-side resize/pad it recorded.

## Testing

- `tests/test_onnx_parity.py` — **the gate**. Per arch with a fixture
  checkpoint: run identical images through the torch adapter and `OnnxAdapter`,
  assert same box count (after identical thresholds), boxes within `atol` (few
  normalized px), scores within tol, labels exact. An arch is not "done" until
  parity passes.
- `tests/test_preprocess.py` — resize/pad/normalize + the transform-inverse math
  (round-trip a box through preprocess→postprocess).
- `tests/fixtures/` — a tiny exported `.onnx` + `.meta.json` per arch + a few
  sample images.

## Dependencies

- Add to `chachkalica_unified`: `onnxruntime` (or `onnxruntime-gpu`), `numpy`,
  `pillow`. None conflict with the training stack.
- Export-side deps (`torch.onnx`, per-arch packages) live only where
  `export_onnx.py` runs; the service repo installs none of them.

## Build order (each slice = `onnx_export/arch/<arch>.py` + `onnx_infer/arch/<arch>.py` + parity test)

1. **RetinaNet** — ✅ done (parity passing). Proved the seam + `meta.json`.
2. **YOLOX** — ✅ done (parity passing at 640² + 480×640 + 512²). NMS-in-graph;
   dynamic H/W export verified working in torch 2.12 (decode grids regenerate at
   runtime). Added the `clip_boxes` meta flag for it.
3. **RT-DETR** — ✅ done (parity passing at 512² + 480×640 + 704×512 resized).
   HF export with a float32 sine-embedding twin (onnxruntime has no double
   Sin/Cos kernel); sigmoid + top-k query selection baked into the graph.
   **Fixture gotcha:** a random-init RT-DETR gives every encoder proposal the
   same constant score (ln(1/C)), so the encoder's `topk(num_queries)` selects a
   different query set under torch vs onnxruntime (tie-break divergence) — not an
   export bug (graph is bit-identical up to the topk). Fixed by spreading
   `enc_score_head` + final `class_embed` in the parity fixture (small std so the
   256-dim dot product doesn't saturate sigmoid), mirroring retinanet/yolox.
4. **RF-DETR** — ✅ done (parity passing at 480×640 + 512² + 720×480; verified
   end-to-end on real trained nano/base/large checkpoints, worst L1 ~1e-6–2e-4).
   **Design change from the original plan:** we do *not* use the `rfdetr`
   package's native `.export()` (which emits raw `dets`/`labels` and would need a
   heavy `adapt_outputs`). Instead we wrap `adapter.model` like the other archs:
   call `LWDETR.export()` (on a `deepcopy`) purely to switch the module to
   `forward_export` and put deformable-attention into an ONNX-traceable mode, then
   bake the whole head — sigmoid + top-k(`num_select`) over `Q*(C+1)` +
   `cxcywh→xyxy` + clamp `[0,1]` + **background-class drop** (`label ==
   num_classes`) — into our wrapper. Output is normalized xyxy
   (`box_coords: "input_normalized"`), service handler is a plain
   `PassthroughHandler`. RF-DETR's square (aspect-changing) resize makes the
   normalized box identical in the input and the original image, so the torch
   path's scale-to-original then re-normalize cancels — no coordinate bookkeeping.
   Random-init parity fixture needs no head surgery (unlike RT-DETR).
5. **YOLO (ultralytics)** — deferred (new arch, not currently trained).

## Verification (end-to-end)

1. Export a real trained checkpoint: `python friendy_chachkalica/export_onnx.py <run>/best.pt`
   → confirm `best.onnx` + `best.meta.json` written.
2. Run the parity test for that arch: `pytest onnx_infer/tests/test_onnx_parity.py -k <arch>`
   → torch vs ONNX outputs match within tolerance.
3. Drive a real consumer with the ONNX artifact present (e.g. `chachak`
   detector/preview on a sample image) and confirm boxes render identically to
   the torch path — auto-detect (`.onnx` sibling) selects `OnnxAdapter` with no
   caller change.
4. Confirm no training packages imported by the service:
   run the consumer in an env with only `onnxruntime`/`numpy`/`pillow`/`torch`
   (no `transformers`/`rfdetr`/`ultralytics`/`friendy_chachkalica`) and verify it
   still predicts.

## Open follow-ups (out of scope for v1)

- Wiring `export_onnx.py` into the Django runner / `promote` service so the ONNX
  artifact is produced automatically on training completion or promotion.
