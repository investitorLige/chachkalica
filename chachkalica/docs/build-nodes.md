# Build Nodes — compiling engines on someone else's GPU

A TensorRT engine only deserializes on the **GPU model and TensorRT version that
built it**. Until now this system had exactly one machine that could build them —
the box running `trainer` — so every engine and every infer bundle it produced was
locked to that one card.

A **build node** is a small service you deploy on another GPU machine. Django
sends it an exported model graph; it compiles the engine on *its* GPU, assembles a
complete infer bundle around it, and hands the bundle back. The bundle is then
valid on that machine.

```
 Django (web/worker)                         build node (other machine)
 ───────────────────                         ──────────────────────────
  .pt  ──► trainer /export_trt_onnx
           (CPU, no GPU)
             │
             ▼
        model.onnx ──── upload ──────────►  compile engine (its GPU)
        + meta.json                         compile person detector (baked in)
                                            assemble bundle
        bundle/  ◄───── download ─────────  bundle.tar.gz
        under bundles_root/<node>/
```

## What a node is, and is not

It is **slim on purpose**: no torch, no training stack, no architectures. It
cannot train, cannot evaluate, and **cannot accept a `.pt`**. Exporting a
checkpoint to ONNX is CPU work the trainer already does safely while the GPU is
busy, so that stays on the trainer and only the GPU-bound half travels.

The image is ~5 GB against the trainer's ~11 GB, and needs nothing on the host but
Docker and the NVIDIA Container Toolkit.

## Standing one up

**1. Generate the person-detector graphs** — once, on the box that has the trainer
image and the person checkpoint. The script and the checkpoint are not *in* the
trainer image (it only carries the training packages), so they are mounted in:

```bash
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$PWD/friendy_chachkalica:/app/friendy_chachkalica:ro" \
  -v "$PWD/buildnode:/app/buildnode:ro" \
  -v "$PWD/person_model_test:/app/person_model_test:ro" \
  -v "$PWD/models:/app/models" \
  -w /app chackalica_unified-trainer \
  python buildnode/scripts/make_person_graphs.py
```

This writes four files into `models/people/`. They are baked into the node image
so every node can compile a person detector **for itself** — see [Why the person
detector is special](#why-the-person-detector-is-special). `models/` is
gitignored, so these are out-of-band artifacts like `best_ckpt.engine` already is.

**2. Get the image onto the target machine.** Two ways; the first is usually
easier, because the other machine then needs neither the repo nor the person
graphs (which are gitignored and ~200 MB).

*Ship the built image.* Build it here, then move it by whichever of these suits —
a registry scales best past a couple of nodes, the ssh pipe leaves no file behind,
the tarball is for an offline target:

```bash
docker build -f buildnode/Dockerfile -t chachkalica-buildnode .

# a) registry
docker tag chachkalica-buildnode <registry>/chachkalica-buildnode:v1
docker push <registry>/chachkalica-buildnode:v1        # target: docker pull <same>

# b) over ssh, nothing left on disk
docker save chachkalica-buildnode | gzip | ssh <machine> 'gunzip | docker load'

# c) a file
docker save chachkalica-buildnode | gzip > buildnode.tar.gz     # ~5.4 GB before gzip
scp buildnode.tar.gz <machine>: && ssh <machine> 'gunzip -c buildnode.tar.gz | docker load'
```

*Or build there from the repo* — copy the repo across **including
`models/people/detector*.onnx`** (gitignored, so `git clone` will not bring them,
and the image build fails without them by design):

```bash
docker build -f buildnode/Dockerfile -t chachkalica-buildnode .
```

**3. Start it** on the target machine:

```bash
BUILDNODE_TOKEN=$(openssl rand -hex 20) docker compose -f buildnode/docker-compose.yml up -d
# no repo there? the image is enough:
#   docker run -d --name buildnode --gpus all --restart unless-stopped \
#     -e BUILDNODE_TOKEN=<token> -p 8300:8300 \
#     -v buildnode-work:/var/lib/buildnode chachkalica-buildnode
```

Keep that token. Check it came up:

```bash
curl -H "Authorization: Bearer $BUILDNODE_TOKEN" http://localhost:8300/health
```

`status` must be `ok`. `degraded` means no GPU or no TensorRT is visible — usually
a missing NVIDIA Container Toolkit — and the node will refuse builds with a 503
rather than fail several minutes in.

**4. Register it** in the admin under **Training → Build nodes**: name, base URL
(`http://<host>:8300`), and the same token. Then run **Ping selected build nodes**;
the GPU, TensorRT version and person-detector columns fill in.

The URL has to be reachable **from the `worker` container**, not from your
browser — the RQ worker is what talks to the node. On a LAN that is just the
machine's address; check with
`docker compose exec worker curl -sf -H "Authorization: Bearer <token>" http://<host>:8300/health`.
If the ping fails, that curl tells you whether it is the network or the token.

> The **node** is authoritative for the token — it is that container's env var.
> The field in Django is a copy. "Generate a token for selected nodes" mints one
> to paste into the node's environment; it refuses to touch a node that is
> currently healthy, where a new token would silently break a working link.

## Using one

**Trained models → "Export to TensorRT…" → Build on: `<node>`.**

Everything else on that form works as before. Two things differ for a remote build:

- The **output directory is ignored.** A remote build produces a whole bundle, not
  a loose artifact, and it lands under `bundles_root/<node name>/`. It shows up in
  the video and camera bundle dropdowns immediately.
- The engine inside it **will not load on this box**, by design. That is the point.

Watch progress under **Export runs**, which gains a *built on* column. The row
records the node, its GPU, and the node's own build id — enough to go read that
build's log on the node if it failed.

## Where things actually land

**The bundle comes back here.** It is downloaded to the Django box and unpacked
under `bundles_root/<node name>/<name>-bundle/` — by default
`chachkalica/data/bundles/<node>/…`. That is the deliverable, and it is what the
video and camera dropdowns read.

**Almost nothing stays on the node.** During a build it uses a scratch directory
inside its `buildnode-work` volume:

```
/var/lib/buildnode/
├── builds/build-<hex>/       one build: uploads, the compiled engine, bundle.tar.gz
│                             deleted as soon as Django downloads it (DELETE /builds/{id}),
│                             or reaped after BUILDNODE_BUILD_TTL (24 h) if it isn't
└── person-cache/<digest>-<precision>-<profile>/detector.engine
                              the ONE thing that persists, on purpose
```

The person-detector engine is kept because it takes a couple of minutes to compile
and never changes between builds — so only the *first* person-crop build on a
fresh node pays for it. Everything else is transient. A node is a compiler, not a
model store; if you wipe the volume you lose nothing but that cache.

## Why the person detector is special

Every person-crop pipeline (`people_detect_first`, `batch_people`, and chains
containing them) needs a person detector inside the bundle. The shipped one is
`models/people/best_ckpt.engine` — **an engine**, compiled for this box. A bundle
built elsewhere that carried it would fail the moment it ran.

So a node carries the detector's *graphs* and compiles its own, once, cached by
graph digest and precision. Django never uploads a detector for the standard
pipeline; the spec just says `builtin`.

Two graphs are baked in, because the two bundle formats need different things:

| File | Used for | Why |
| --- | --- | --- |
| `detector.onnx` | ONNX bundles | Runs under onnxruntime anywhere |
| `detector.trt.onnx` | engine bundles | Raw outputs + `EfficientNMS_TRT`; the only form TensorRT can compile for a YOLOX |

The node reproduces the shipped engine's profile and batch range
(64–1024 px, batch 1/8/16 — `buildnode/person.py`). Left to derive them it would
build a static-640 batch-1 engine, which is a real throughput and resolution
regression against the detector every existing bundle carries.

The detector's meta sidecar is **hand-patched** (`layout: bgr`,
`input_scale: byte`, `pad_value: 114`) because it is a Megvii checkpoint, not a
friendy-trained one. `make_person_graphs.py` copies it verbatim; regenerating it
from the exporter would write `rgb`/`unit` and silently mis-preprocess every frame.

### When the person model changes

The baked graphs are derived artifacts, and **nothing detects that they are
stale** — no build fails, no test goes red. Nodes keep compiling detector engines
from the old graph and every bundle they produce quietly carries the previous
person model. The full sequence:

1. Regenerate the graphs (step 1 of [Standing one up](#standing-one-up)).
2. Rebuild and redeploy the node image **on every node** — the graphs are baked in
   at image build time, so a running node uses the old ones until it is replaced.
3. Nothing to clear: each node's compiled-detector cache is keyed by the graph's
   SHA-256, so a new graph lands in a new cache directory and recompiles on the
   next build. The old entry is dead weight, not a stale hit.
4. Rebuild `models/people/best_ckpt.engine` separately — that is what the **local**
   trainer path bundles, and it is not produced by this script.
5. Re-export any bundle you care about. Existing ones keep the detector they were
   built with, which is correct behaviour but probably not what you want.

The same list appears in the root [`README.md`](../../README.md) under *Data And
Generated Files*, because that is where someone changing the person model is
likely to be looking.

## Which architectures a node can build

| Arch | From | Why |
| --- | --- | --- |
| rtdetr, rfdetr, ecdet | the standard ONNX | Fixed-size top-k, NMS-free; TensorRT compiles it directly |
| yolox, retinanet, fasterrcnn | a **prepared** graph | Their standard ONNX bakes a data-dependent NMS TensorRT rejects; it must be re-exported as raw outputs + `EfficientNMS_TRT`, which needs the `.pt` and torch |

You do not have to think about this. `jobs.run_export_remote` calls the trainer's
**`/export_trt_onnx`** first, which decides per arch and does the re-export when
needed — CPU-only, GPU-free, safe while training. A node handed an unprepared
graph for one of the bottom three refuses it with a 400 explaining exactly that,
before taking the GPU.

## Load-testing a remote bundle

Don't expect to. "Sync bundle" runs its load test through the **local** trainer, so
a bundle built elsewhere cannot pass it — its engine is for another GPU. `validate()`
detects this (via `built_by: buildnode` in the engine's `.engine.json`) and reports
the load test as **info**, naming the GPU it was built for, rather than a failure.

To actually prove a bundle works, run it on the machine it was built for — it is
self-contained:

```bash
python infer.py <images> --conf 0.4
```

## Keeping TensorRT versions in lockstep

`buildnode/requirements.txt` pins `tensorrt==11.2.1.2` and it **must match**
`friendy_chachkalica/requirements-trt.txt`. An engine built against a different
runtime will not load in this stack — `trt_infer/session.py` checks the
`.engine.json` sidecar and fails loudly rather than mysteriously. The ping surfaces
each node's version, so a drifted node is visible in the changelist.

Bump the two files together, and rebuild every cached engine — including each
node's cached person detector, which the cache key invalidates automatically only
for profile changes, not for a TensorRT bump. Clearing the node's
`buildnode-work` volume is the blunt way to do it.

## Testing without a second machine

The repo's compose has a profile-gated node so the whole path can be exercised on
one box. It is not started by a plain `docker compose up`, because on this box it
would contend with `trainer` for the same GPU:

```bash
BUILDNODE_TOKEN=dev-token docker compose --profile buildnode up -d --build buildnode
```

Register it at `http://buildnode:8300` — the compose service name, which is how
the worker reaches it. Its host port is bound to **127.0.0.1 only**, unlike every
other service in that file: the token defaults to a guessable `dev-token`, and a
build node compiles arbitrary uploaded ONNX on the GPU, so it must not be
LAN-reachable. A real node inverts both (all interfaces, mandatory token).

The node's own tests need no GPU and can run in the image, which is the quickest
way to tell a broken deployment from a broken build request:

```bash
docker run --rm chachkalica-buildnode python -m unittest discover -s buildnode/tests -t .
```

## Code

| Piece | Where |
| --- | --- |
| The node service | `buildnode/service.py` |
| Build execution, path rewriting | `buildnode/builds.py` |
| Torch-free TensorRT compile | `buildnode/compile.py` |
| Baked person detector + engine cache | `buildnode/person.py` |
| Person graph regeneration | `buildnode/scripts/make_person_graphs.py` |
| Django HTTP client | `training/services/buildnode.py` |
| The job | `training.jobs.run_export_remote` |
| Model + admin | `training.models.BuildNode`, `training.admin.BuildNodeAdmin` |
| Prepared-graph endpoint | `friendy_chachkalica/service.py` → `/export_trt_onnx` |

See also [Infer Bundles](infer-bundles.md) for what a bundle is and how it is
consumed once it lands.
