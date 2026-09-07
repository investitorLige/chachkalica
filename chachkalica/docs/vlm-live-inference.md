# VLM live inference

Running a vision-language model over a video and watching its commentary arrive
as the video plays. The companion to `camera-live-inference.md`, which does the
same for RTSP cameras and detection models, and to `vlm-dataset-runs.md`, which
runs the same models over a labeled dataset and scores what they say.

## The tabs

**Models** (`vlm.VlmModel`) — a pretrained VLM, configured. Nothing is trained
here, so a row is just: which family, which checkpoint, and the prompt to ask of
every frame. The prompt lives on the model rather than being asked at run time,
which is what makes "Run VLM Inference…" a two-field form instead of a wizard.

**Videos** (`vlm.VlmVideo`) — an mp4 library of its own, separate from the
detection Videos tab and rooted at `FleetSettings.vlm_videos_dir`. Three ways to
add one: upload from the browser, import a file already in the folder, or paste a
link for a worker to download. The upload path is the only place in this project
that accepts a browser file upload.

**Dataset runs** (`vlm.VlmDatasetRun`) — the history of the Models tab's "Run on
a dataset…" action, each row opening a report of the answers and their scores.
See `vlm-dataset-runs.md`; nothing else in this file applies to it.

`VlmRun` and `VlmAlert` are real tables but not tabs — a run is reached from the
action's redirect and from the past-runs panel on its video's change page, the
same way `videos.Video`'s player hangs off `VideoAdmin` rather than being its own
section.

## The shape of a run

```
  admin (web)                worker (rq, default queue)     vlm-backend (GPU)
  ───────────                ──────────────────────────     ─────────────────
  Run VLM Inference…         cv2 decode, every stride'th    transformers
   → VlmRun (snapshotted)      frame → JPEG under data/     one model warm
   → enqueue(6h timeout)       → POST /infer  ────────────►  + a lock
   → redirect to live view   ◄── {text, latency_ms}         HF_HUB_OFFLINE=1
  live view                  → VlmAlert row
   → poll alerts JSON  ◄───── VlmAlert rows (postgres)
   → each poll = heartbeat ──► worker pauses when unwatched
```

Nothing about this is novel in this codebase: heavy models already live in their
own container behind HTTP (`trainer`, `sam-backend`), frames are already handed
over as paths on the shared `data/` mount, and the "stop burning GPU when nobody
is looking" heartbeat is lifted from the camera path.

## Why polling, not websockets

This project is WSGI-only — no Channels, no ASGI, no SSE anywhere. The camera
live view gets away with an `<img>` pointed at an MJPEG endpoint because frames
are *images*; discrete alert events have no such trick available.

So the live page polls a small JSON endpoint once a second, paging on a per-run
`seq` cursor seeded from what the server rendered. That fits the existing
gunicorn-with-threads + rq shape and adds no infrastructure. It also means the
poll can double as the viewer heartbeat, so there is no second endpoint that
could drift out of sync with the first.

## Alerts trail the video, on purpose

The video plays at its own speed. A VLM answer takes on the order of a second, so
at anything above ~1 fps the analysis falls progressively behind the playhead.
Rather than pretend otherwise, every alert carries the timestamp of the frame it
came from, the page shows `analysed N/M frames`, and when the video ends while the
run is still going a banner says so. Alerts are not gated to `video.currentTime`;
the rail is a plain feed.

Every answer is an alert, unfiltered. A prompt that always answers produces one
card per sampled frame, so the feed and the table grow as `duration × fps`. That
is why the fps default is low.

## One model warm at a time

`ml_backends/vlm/service.py` caches exactly one adapter and evicts on any change
of `(backend, family, weights, quantization)`, with a lock serializing
generation. Same trade-off the trainer's `_predict_cache` makes, for the same
reason: these are multi-gigabyte models on a GPU now shared three ways
(`trainer`, `sam-backend`, `vlm-backend`). Alternating two VLM rows against one
video will reload on every frame and be unusably slow — run them one at a time.

## Weights are the gate

**huggingface.co does not verify from this network.** It resolves and answers —
the obstacle is TLS, not routing. A intercepting proxy re-signs every
certificate with `CN=atlas-MASTER`, a sub-CA of the internal
`atlas-ATLASMASTER-CA`, and that root is published only over LDAP inside
`atlas.local`, so it reaches no container's trust store and every
`from_pretrained` dies on `CERTIFICATE_VERIFY_FAILED`. (Pinning the intermediate
via `SSL_CERT_FILE` *and* setting `ssl.VERIFY_X509_PARTIAL_CHAIN` does make the
Hub reachable — Python, unlike curl, will not accept a non-self-signed anchor on
its own — but trusting an interception CA is an operator's call, not a default.)

So every checkpoint must already sit in `data/hf_cache/hub` as a
`models--<org>--<repo>` directory, and the backend runs with `HF_HOME` +
`HF_HUB_OFFLINE=1` so `from_pretrained` fails fast instead of paying for a
connection that cannot succeed.

`python manage.py fetch_vlm_weights` lists what is present, what is missing, and
the exact commands to run elsewhere and copy in. Two details it encodes, both
of which cost real time to rediscover: the SmolVLM repos ship an `onnx/` folder
of transformers.js exports larger than the weights themselves (25 GB for the
2B), so they are downloaded with `--exclude "onnx/*"`; and the downloader is `hf`
(`pip install -U huggingface_hub`), since the `huggingface-cli` alias is
deprecated and on huggingface_hub 1.x exits without downloading. The copy must use
`rsync -a` or `tar`, never `scp -r`, because the HF cache stores each file once
under `blobs/` and symlinks it from `snapshots/` — scp dereferences those and
doubles the transfer. The admin gates on the same
check twice: uncached options are disabled in the weights dropdown, and the run
action refuses to start with uncached weights.

## Cancellation and pausing

Two different things:

* **Pausing** is automatic. The worker stops between frames while the heartbeat
  is stale and resumes when the page is reopened, after an initial grace period
  so the redirect and first paint aren't mistaken for an abandoned tab.
* **Cancelling** is explicit — the Stop button sets `cancel_requested`, which the
  worker honours after its current frame. A call already in flight cannot be
  interrupted, which is why the HTTP client bounds every request with a timeout.

## Adding the Ollama backend

`VlmAdapter.infer(image_path, prompt, max_new_tokens) -> str` is the entire
contract. `ml_backends/vlm/adapters/ollama.py` is a stub that raises; filling it
in and nothing else is the whole change. The Django app, the worker, the HTTP
client, and the admin do not know which backend a model uses.

## Known limits

* No batching — frames go one at a time.
* A long run holds one of the three `default` rq workers for its duration, the
  same trade-off `videos.InferenceJob` already accepts.
* `cv2.CAP_PROP_POS_MSEC` returns 0 for some codecs; the runner falls back to
  `frame_index / source_fps`, which is exact for constant-frame-rate video and
  approximate for VFR.
* Uploads are bounded by gunicorn's `--timeout 120`, not by any configured size
  limit — there is no proxy in front of it.
