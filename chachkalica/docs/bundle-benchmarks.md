# Bundle Benchmarks

Measures an **exported bundle** — the thing that actually ships — on **real frames**,
through the bundle's own vendored runtime. Admin section: **Bundle Benchmarks**, or the
"Bundle benchmarks" tab in the header.

## This is not the Arch console

There are two benchmarks in this admin and they answer different questions.

| | Arch console (`/admin/benchmarks/`) | Bundle Benchmarks (this one) |
|---|---|---|
| what it measures | architecture variants | one exported bundle |
| weights | random-init | the trained weights that will deploy |
| input | synthetic random tensors | real images you choose |
| pipeline | the model alone | decode → detector → crop → model → NMS |
| stored as | static JSON in `data/benchmarks/` | a DB row per run, with history |
| answers | "which arch should we pick" | "how will this deploy, and where does the time go" |

Neither replaces the other. A bundle number is always slower than its arch number, and
that gap is the point — it is the pipeline the operator will actually run.

## Running one

Add a row (the add form *is* the launcher — saving it queues the job):

- **Bundle** — a scan of the bundle root from Training settings. Copying a bundle
  directory in there is all it takes for it to appear.
- **Dataset** or **Images path** — the frames. Sorted then truncated to
  **Max images**, so two runs on the same source see the same frames in the same
  order; without that, runs are not comparable.
- **Batch sizes** / **Concurrency** — comma-separated. Each batch is capped by the
  artifacts' own profile (a skipped batch says so). Each concurrency level loads a
  *separate copy* of the bundle, one per worker thread — a `box_cache` is a plain dict
  and a TensorRT execution context is not safe to share — so more streams cost
  proportionally more GPU memory, and the report shows both.
- **Warmup** / **Timed calls** / **Min duration** — warmup is never zero: an engine's
  first call carries deserialization, and on a real bundle here that was 512 ms against
  a 47 ms steady state. The timed loop is grown if needed to span the minimum
  duration, because NVML's utilization counter refreshes about once a second.

Runs execute on the trainer (web and worker have no GPU) and one at a time: the trainer
refuses a second GPU job, because a benchmark sharing the card does not fail, it
produces plausible numbers that are wrong. For contention it cannot prevent — another
stack on the same GPU — it records a co-tenant census and stamps the run **contended**.

## Reading the report

**Latency and throughput come from two different loops** and are not convertible into
each other. Latency synchronizes every call, which is what makes a sample mean
anything; throughput does not, because the synchronize destroys the pipelining that
sets it.

**Four instruments feed the consensus verdict, and the disagreements are the
payload:**

| | what it is | when it is unavailable |
|---|---|---|
| T1 | host wall clock around the call | never |
| T2 | CUDA events — the device's own elapsed time | CPU runs |
| T3 | CUPTI kernel time (separate pass); sees onnxruntime *and* TensorRT kernels | CPU runs |
| T4 | the model artifact alone, outside this harness — `polygraphy` for an engine, a bare onnxruntime session for ONNX | reported with the reason |

- T1 ≈ T2 → GPU-bound, the host adds nothing.
- T1 ≫ T2 → host-bound; the stage table names the stage.
- T3 ≪ T2 → the device is idle inside the call: launch overhead, sync bubbles, or a
  swarm of tiny kernels.
- T4 is compared **per model call**, because a tiling or person-crop pipeline runs the
  model several times per frame and a whole-frame comparison would invent a wrapper
  overhead that is really just several inferences.

**T5, opt-in: an Nsight Systems kernel profile.** Tick *Nsight kernel profile* and a
final pass re-runs one cell in a **child process** under `nsys`, after every timed pass
has finished — so it perturbs none of the numbers above. It produces:

- a per-kernel table on the report (share, total, count, avg, median), which is the one
  thing T3's aggregate cannot give: *which* kernels the device time went into;
- the CUDA API and host↔device transfer tables beside it;
- a downloadable `.nsys-rep` for the Nsight Systems GUI, plus the three CSVs, in
  `<output_dir>/nsys/`.

Nsight is **bind-mounted, not baked in** — it is ~1.1 GB. The trainer service gets
`${NSYS_HOST_DIR:-/opt/nvidia/nsight-systems/2025.5.2}:/opt/nsight:ro`; point
`NSYS_HOST_DIR` at whatever this host has. If it is absent the run still completes and
records "Nsight profile unavailable" as a warning.

On a real bundle here it corroborated the stage table from a completely independent
tool: `cudaMemcpyAsync` was 52.6% of host API time and Host→Device copies 88.8% of
memory-op time — the same host-transfer story the 67% `model.preprocess` stage told.

Two limits worth knowing. It runs **once per benchmark**, on the first cell, not once
per cell: the question it answers is not very sensitive to batch size and a capture is
expensive. And **GPU performance counters are never requested** — SM occupancy and
tensor-core activity need `CAP_SYS_ADMIN` and fail with `ERR_NVGPUCTRPERM` in an
unprivileged container. Asking for them makes the whole capture fail rather than degrade,
so the pass sticks to tracing, which works unprivileged. Getting occupancy means
`cap_add: [SYS_ADMIN]` on the trainer service and is a security decision, not a code one.

**Still not used:**

- **`trtexec`** — the pip `tensorrt` wheel ships no binary. `polygraphy` (already present
  as a nvidia-modelopt dependency) fills the same role through the Python builder API.
- **DCGM** — absent here: no `dcgmi`, no `libdcgm`, no Python bindings, and
  `nv-hostengine` is inactive. It is a host *daemon*, not a library you add to a
  container, and its interesting field groups need the same profiling permission the
  counters above do. Nsight already covers the per-kernel question without it.

So: the stage table says *which stage*, the kernel profile says *which kernel*, and
neither says *why a kernel is slow inside itself* — that is Nsight Compute (`ncu`)
territory and deliberately out of scope.

## Where the frame budget goes

The stage table is a tree, and siblings plus residual equal the parent at every level:

```
run_batch
  ├─ pipeline
  │    ├─ person          the detector, inclusive (its own pre/forward/post below it)
  │    ├─ assemble        expand / crop / pad / tile   (residual)
  │    └─ model           the trained model, inclusive (pre/forward/post below it)
  ├─ merge_nms
  ├─ remap
  └─ unattributed         (residual)
```

Each stage reports how it was obtained. **`unavailable` is not zero** — it means this
bundle's vendored runtime has no such call, so the cost is unknown, not free.

This pass **synchronizes at every boundary**, without which a stage's wall time is the
time to *queue* its work rather than to do it. That inflates the total, so the report
prints the stage pass's own total next to the headline latency and labels the
difference *instrumentation overhead*. Read stage **shares**, not stage absolutes.

A worked example from a real `people_detect_first` engine bundle in `data/bundles`:
`model.preprocess` was **67% of the frame** while the engine forward was 21%. That
bundle was exported before the GPU-resident preprocessing path existed, so its
`TrtAdapter` still runs numpy on the host — which is invisible in any measurement of
the model alone, and obvious here.

## Comparing runs

"Compare selected…" on two or more finished runs puts them side by side with deltas
against the oldest, coloured by direction (lower is better for latency and memory,
higher for throughput). It compares the batch-1 single-stream cell, the only one every
sweep is guaranteed to have. Stage **shares** are compared only for stage ids present
in every run, because two bundle versions may legitimately declare different stages.

A run is never edited — a row whose parameters changed after the fact would describe a
benchmark that never happened. "Re-run selected…" clones instead, so the history that
shows a bundle got slower survives.

## Bundles that declare their own stages

The stage set is not fixed here. A bundle may ship
`runtime/benchmark_stages.py` with

```python
BENCHMARK_STAGES = [
    {"id": "stage1", "label": "Preprocess", "target": "onnx_infer.adapter:preprocess",
     "parent": "pipeline", "role_aware": True},
    {"id": "stage3", "label": "In-between", "parent": "pipeline", "kind": "residual"},
]
```

(or the same list under a `benchmark_stages` key in `pipeline.json`), and that
declaration wins over the built-in probe table. The harness wraps what it declares and
the report renders whatever arrives, in the order it arrives — so a new bundle version
can add or rename stages with no change to the harness and no change to the template. A
malformed declaration falls back to the probe table rather than failing the run.

`target` is the **binding site**, not the definition site: `onnx_infer/adapter.py` does
`from .preprocess import preprocess` at module scope, so patching
`onnx_infer.preprocess.preprocess` resolves fine and then never fires. A probe that
installs and reports zero is the one failure mode worth being careful about.

## Where things live

| | |
|---|---|
| harness | `inferlica/benchmark/bundle_bench.py`, `bundle_stages.py` (baked into the trainer image) |
| trainer endpoints | `POST /benchmark_bundle`, `GET /benchmarks/{id}`, `POST /benchmarks/{id}/stop` |
| result file | `<runs_root>/benchmarks/<run id>/benchmark.json`, plus the trainer's `service.log` beside it |
| nsight capture | `<runs_root>/benchmarks/<run id>/nsys/profile.nsys-rep` + three CSVs; downloadable from the report |
| trainer log | `logs/benchmark/benchmark-<id>.log` |
| app | `chachkalica/benchmarks/` (model, admin, job) |
| templates | `templates/admin/benchmarks/bundle_report.html`, `compare.html` |

Run the harness by hand inside the trainer container:

```
docker exec chackalica_unified-trainer-1 python3 -m inferlica.benchmark.bundle_bench \
  --bundle /app/data/bundles/<name>-bundle \
  --images /app/data/source/<dataset>/images \
  --batch-sizes 1 --concurrency 1,2 --nsys --out /tmp/benchmark.json
```

`--nsys-worker` is the internal mode `nsys` actually profiles (it loads the bundle and
runs the timed loop and nothing else); there is no reason to invoke it by hand.

## Known gotchas

- **An ONNX bundle runs one image per session call** regardless of batch size, so on
  ONNX a larger batch only ever moves the person detector.
- **A `.engine` built elsewhere will not deserialize here.** A bundle from a build node
  is refused with the reason rather than a stack trace; `data/bundles/01-…-bundle`'s
  engine is TensorRT 11.2.1.2 against the image's pinned 11.1.0.106 and needs a rebuild.
- **onnxruntime's CUDA provider is currently broken in the trainer image** — it wants
  CUDA 12's `libcublasLt` while the image carries CUDA 13 — so an ONNX bundle silently
  runs on the CPU. The report reads the session's actual providers and says so rather
  than publishing CPU numbers under a `cuda` label. TensorRT bundles are unaffected.
