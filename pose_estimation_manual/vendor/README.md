# vendor/ — inference runtime copied from `chackalica_unified`

These three packages were copied from the sibling repo rather than written fresh,
because they already encode a lot of hard-won detail: EfficientNMS output
unpacking, the letterbox transform inverse, per-architecture output contracts,
and the tile/crop remap geometry.

**Source:** `/home/mercury/luka_RD/chackalica_unified`
**Commit:** `df75e5200309180f7d5f5d48b87cd37466b625fa` ("final benchmark", 2026-07-27)

Nothing outside `vendor/` should be imported from in here — that self-containment
is the point, and the test suite asserts it.

## What was copied and what changed

### `onnx_infer/` — verbatim

The architecture-free ONNX runtime. Two frozen contracts live here:

- **Contract A** — the ONNX graph's output layout, per architecture (`arch/`).
- **Contract B** — the `meta.json` sidecar schema (`meta.py`): resize mode, input
  scale, normalization, box coordinate frame, class map.

Kept in full, including `session.py` / `adapter.py` (the onnxruntime path). That
path is the escape hatch for when the host's TensorRT can't deserialize an engine
built elsewhere — it runs the same graph on CPU with no TensorRT at all.

Only the module docstring changed, to drop a reference to a `PLAN.md` that wasn't
copied.

### `trt_infer/` — imports rewritten, plus two fixes made here

The TensorRT runtime: `TrtModel` (engine deserialize, plugin registration,
data-dependent output allocation via `IOutputAllocator`, EfficientNMS unpacking,
real `[B,3,H,W]` batching) and `TrtAdapter` (the torch-facing drop-in).

Its `from onnx_infer.x import y` lines became `from ..onnx_infer.x import y`.

Beyond that, `session.py` has diverged from the source commit twice — both from
production failures seen here, so **do not overwrite this file with a fresh copy**:

| Divergence | Status upstream |
|---|---|
| `_split_passthrough` detects the batch axis by rank instead of assuming one. The passthrough exporters index the batch away, so `boxes[0]` returned detection *zero* and silently dropped every other detection — an RF-DETR crop emitting 300 delivered 1. | **Still present upstream** as of `chackalica_unified` @ `6e44b53`. Not yet reported/fixed there. |
| The output allocator is built once in `__init__` and registered once, and `_reallocate` only allocates when it needs more room. Rebuilding it per call leaked ~9.45GiB of live GPU memory over ~18h and wedged this service permanently on 2026-08-07. | **Fixed upstream** in `chackalica_unified` after `6e44b53`, with tests. |

The upstream fix also had to `.clone()` each output, because that repo's newer
`run_torch` hands back GPU views. This copy predates `run_torch` and copies
everything to host numpy inside `run()`, so it needs no clone — but a torch-out path
added here later would.

### `chachak/` — trimmed

The pipelines that decide *how a frame is presented to the model*: whole-frame,
tiled, cropped around detected people, or a chain of those merged.

Trimmed to the inference path only. The original's dataset-eval machinery was the
sole reason it depended on the `friendy_chachkalica` **training** package — and
dragging that in would mean `transformers`, `rfdetr`, `ultralytics`, and
torchvision detection heads inside a serving container that never trains
anything. Removing four things cut the dependency entirely:

| Removed | Why |
|---|---|
| `Pipeline.run()` | Walked a Friendy eval dataloader and scored with `evaluate_detection`. Live inference calls `process_batch` per frame instead. |
| `run.py`, `preview.py`, `configs/` | CLI + YAML entry points for eval runs. Django is the config source now. |
| `_friendy.py` | The `sys.path` bridge into the training package. Replaced by `formats.py`. |
| the `.pt` branch of `load_checkpoint_adapter` | Rebuilding a torch architecture needs `build_model`. Exported artifacts only now — `.engine` or `.onnx`, each with its `meta.json`. |

Diverged here, deliberately — **do not overwrite these two with a fresh copy**:

| Divergence | Why it exists |
|---|---|
| `boxes.py`: `remap_local_preds_to_frame` and `merge_predictions` treat columns past the sixth as opaque payload and carry them through, instead of hard-coding width 6. | Nothing in the geometry needs to know what the extra columns are, and this is the only place a per-detection value can ride through the coordinate lift and the NMS merge — after the merge the rows are no longer in crop order. |
| `pipeline.py`: `process_batch` takes an optional `context` out-param. Given one, it reports each frame's person boxes and widens its output to a seventh column naming which person each detection was found on. `crop_regions` keeps its original 3-tuple shape (training parity); `_crop_regions_indexed` is the indexed version behind it. | A person-crop pipeline computes the item→person association and then throws it away. Zones need it: an item is in a danger zone when the *person it was found on* is standing in one, and the person is never emitted as a detection. Off by default, so the six-column contract is unchanged for anything that doesn't ask. |

Added / rewritten:

| File | Change |
|---|---|
| `formats.py` | Verbatim copy of `friendy_chachkalica/formats.py` (97 lines of pure torch box geometry). `boxes.py` needs exactly `clip_xyxy` and `xyxy_to_xywhn` from it. |
| `config.py` | `PipelineConfig` (a whole dataset-eval run, loaded from YAML) replaced by `InferenceConfig` — only the fields the pipelines actually read. `TilingConfig` and `DetectorConfig` are verbatim; the pipelines reach into their attributes by name and their defaults encode real tuning knowledge. |
| `__init__.py` | Rewritten for the trimmed surface; dropped the flat-script `except ImportError` fallbacks throughout, since nothing here runs as a loose script anymore. |

## Keeping it in sync

There is no automation, deliberately — these have diverged and are expected to.
When pulling a fix across, check the table above first: a change to
`Pipeline.run()` or the `.pt` loader has no counterpart here.

`models/people/best_ckpt.engine` in the sibling repo is a working yolox person
detector with all its sidecars, and is the fixture to smoke-test any change to
`trt_infer` against.
