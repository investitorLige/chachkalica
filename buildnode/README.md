# buildnode

A slim service that compiles TensorRT engines and assembles chachak infer bundles
**on the machine it runs on**, so a bundle can be produced for the GPU that will
actually run it. A TensorRT engine plan only deserializes on the GPU model and
TensorRT version that built it, which is why this exists at all.

Deployed on other GPU machines and driven from the Django admin
(**Training → Build nodes**). The operator-facing documentation, including how to
stand one up, is **[`chachkalica/docs/build-nodes.md`](../chachkalica/docs/build-nodes.md)** —
read that first. This file is about the code.

## Deploy to another machine

Target machine needs **Docker + NVIDIA Container Toolkit**. Nothing else — no repo,
no Python, no weights.

### On the repo box, once ever

```bash
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$PWD/friendy_chachkalica:/app/friendy_chachkalica:ro" \
  -v "$PWD/buildnode:/app/buildnode:ro" \
  -v "$PWD/person_model_test:/app/person_model_test:ro" \
  -v "$PWD/models:/app/models" \
  -w /app chackalica_unified-trainer \
  python buildnode/scripts/make_person_graphs.py
```

Skip if `models/people/detector.trt.onnx` already exists. Redo it whenever the
person model changes — see [When the person model
changes](../chachkalica/docs/build-nodes.md#when-the-person-model-changes).

### On the repo box, per image version

```bash
docker build -f buildnode/Dockerfile -t chachkalica-buildnode .
```

Then get it to the target — **any one** of these:

```bash
# a) registry, if you have one. Best for more than one or two nodes.
docker tag chachkalica-buildnode <registry>/chachkalica-buildnode:v1
docker push <registry>/chachkalica-buildnode:v1
#    ...and on the target: docker pull <registry>/chachkalica-buildnode:v1

# b) straight over ssh, no file left anywhere
docker save chachkalica-buildnode | gzip | ssh <machine> 'gunzip | docker load'

# c) a file you carry (offline/airgapped target)
docker save chachkalica-buildnode | gzip > buildnode.tar.gz   # ~5.4 GB before gzip
scp buildnode.tar.gz <machine>:
#    ...and on the target: gunzip -c buildnode.tar.gz | docker load
```

### On each target machine

```bash
TOKEN=$(openssl rand -hex 20); echo "$TOKEN"     # keep it

docker run -d --name buildnode --gpus all --restart unless-stopped \
  -e BUILDNODE_TOKEN="$TOKEN" -p 8300:8300 \
  -v buildnode-work:/var/lib/buildnode \
  chachkalica-buildnode

curl -H "Authorization: Bearer $TOKEN" http://localhost:8300/health   # want "status":"ok"
```

### In Django

**Training → Build nodes → Add**: name, `http://<machine-ip>:8300`, that token →
select it → **Ping selected build nodes**. GPU and TensorRT columns fill in.

Then use it: **Trained models → Export to TensorRT… → Build on: `<node>`**.

### If the ping fails

The URL must be reachable from the **worker** container, not your browser:

```bash
docker compose exec worker curl -sf -H "Authorization: Bearer $TOKEN" http://<ip>:8300/health
```

Connection error → network/firewall. `401` → token mismatch. `"degraded"` → the
container cannot see the GPU (NVIDIA Container Toolkit).

### Upgrading a node

Load/pull the new image, then replace the container:

```bash
docker rm -f buildnode && <the docker run above>
```

Keep the `buildnode-work` volume; it holds the compiled person-detector cache,
which is keyed by graph hash and recompiles itself when the graph changes.

## No torch, and why that constrains things

The image carries TensorRT, onnxruntime and the graph-surgery tools, but **not
torch** — that is the difference between ~5 GB and ~11 GB, and it is enforced by
an import check in the Dockerfile that fails the build if torch ever creeps back
onto the compile or bundle path.

The cost is a real limit. Compiling an engine for **yolox, retinanet or
fasterrcnn** means first re-exporting the graph with an `EfficientNMS_TRT` node in
place of its baked NMS, and that re-export runs off the torch model. A node cannot
do it. Django therefore asks the trainer's `/export_trt_onnx` (CPU-only) to
prepare those graphs and uploads the result; rtdetr and rfdetr compile straight
from their standard ONNX. A node handed an unprepared graph for one of the three
refuses it with a 400 that says so, before taking the GPU.

Keeping that subset reachable required making three package `__init__`s lazy
(`chachak`, `friendy_chachkalica`, `trt_export.arch`) — each eagerly imported torch
on behalf of modules that never needed it.

## Layout

| Module | Does |
| --- | --- |
| `service.py` | The HTTP surface. Auth on every endpoint including `/health`. |
| `builds.py` | One build: scratch dir, path rewriting, compile, bundle, tarball. |
| `compile.py` | The torch-free tail of `trt_export.cli.build_engine`. |
| `person.py` | The baked person detector and its per-node engine cache. |
| `gpu.py` | Identity + capability for `/health`. Never raises. |
| `auth.py` | Bearer token from `BUILDNODE_TOKEN`. |
| `settings.py` | Env-driven paths, TTL, upload cap. |
| `scripts/make_person_graphs.py` | Regenerates the baked graphs. Runs on the trainer. |

## API

All endpoints require `Authorization: Bearer $BUILDNODE_TOKEN`.

```
GET    /health               node identity, capability, current load
POST   /builds               multipart: spec + model + model_meta
                             [+ detector + detector_meta]  ->  {build_id, status}
GET    /builds/{id}          {status, log_tail, error, result}
GET    /builds/{id}/artifact the bundle, as a tar.gz stream
DELETE /builds/{id}          drop the scratch dir
GET    /builds               everything this node still has
```

Builds are asynchronous and serialized behind one GPU lock. `POST /builds` returns
as soon as the uploads are on disk.

The `spec` part is JSON:

```json
{
  "name": "ppe-medium-best",
  "fmt": "engine",
  "precision": "fp16",
  "conf": 0.4,
  "input_hw": [640, 640],
  "model_prepared": true,
  "request": { "…a chachak pipeline request, paths as bare filenames…" }
}
```

Every path inside `request` is rewritten to point at the uploaded files before
chachak parses it, and any path component in a filename is rejected — the request
came from another machine, so a path in it is either a mistake or an attempt to
read this node's disk. Set `detector.checkpoint` to anything (`"builtin"` by
convention) and omit the detector upload to use the baked one.

## Environment

| Variable | Default | |
| --- | --- | --- |
| `BUILDNODE_TOKEN` | — | **Required.** No token, no service (503). |
| `BUILDNODE_PORT` | `8300` | |
| `BUILDNODE_WORK_DIR` | `/var/lib/buildnode/builds` | Build scratch. |
| `BUILDNODE_CACHE_DIR` | `/var/lib/buildnode/person-cache` | Compiled person detector. |
| `BUILDNODE_PERSON_DIR` | `/opt/buildnode/person` | Baked graphs. |
| `BUILDNODE_BUILD_TTL` | `86400` | Seconds a finished build is kept. |
| `BUILDNODE_MAX_UPLOAD_MB` | `2048` | Per-file upload cap. |
| `BUILDNODE_LOG_LINES` | `400` | Log tail length. |

## Tests

No GPU needed; they run in the image:

```bash
docker run --rm chachkalica-buildnode python -m unittest discover -s buildnode/tests -t .
```
