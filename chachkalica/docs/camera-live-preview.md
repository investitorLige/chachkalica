# Camera live preview (MJPEG via Redis)

## Context

`cameras/` (added 2026-07-22) lets an admin register an RTSP camera by URL and
does a one-shot connectivity check on save (`cameras/services/rtsp.py`) — it
opens the stream, reads a single frame, and closes it. There's no live view
and no relation yet to `videos`/`training`/`eval_pipelines`.

Goal: a live preview embedded in the camera's admin page. Chosen approach —
MJPEG served from a Redis-cached "latest frame," written continuously by a
background worker — over a WebRTC/go2rtc sidecar, because this repo is a
small single-host Docker Compose app (Django + Postgres + Redis/RQ already,
no Kubernetes) and:
- it reuses infra that's already here (Redis for RQ, cv2 for the existing
  connectivity check) instead of adding a new external service/binary
- one persistent RTSP connection per camera (held by the worker) serves any
  number of admin viewers from Redis, avoiding most IP cameras' 2-4
  concurrent-session RTSP limit
- no WebRTC/ICE/STUN/TURN networking to get stuck debugging
- the same "grab latest frame" mechanism is a natural stepping stone toward
  pulling frames into `data/source/<dataset>/` for the actual training
  pipeline later, if that's ever wanted

Trade-off accepted: ~5-15fps JPEG refresh, not smooth 25-30fps video. Fine for
"confirm the camera's alive and see roughly what it's pointed at."

## Design

```
cv2.VideoCapture (RTSP)  --every frame-->  stream_cameras worker (1 thread/camera)
                                              |
                        SETEX camera:<id>:frame    (full res, for the model)
                        SETEX camera:<id>:preview  (downscaled, for viewers)
                                              |
                                              v
                                            Redis
                                              ^
                                              |
                                 GET camera:<id>:preview (poll)
                                              |
                       admin:cameras_camera_mjpeg  (custom admin URL)
                                              |
                        <img> inline on the change page, or full-page
                        via the "View live stream…" admin action
```

The worker is the only thing that ever opens an RTSP connection. Everything
downstream just reads Redis, so viewer count is decoupled from the camera's
own connection limit.

## Files (as built)

- **`cameras/services/frame_cache.py`** — thin Redis wrapper: `set_frame`
  (`SETEX camera:<id>:frame <ttl>`, ttl a bit longer than the worker's push
  interval so a dead worker's frame expires naturally) and `get_frame`.
  Reuses `django_rq.get_connection("default")` — the same Redis connection
  RQ already holds — rather than opening a second client, so no new
  dependency was needed (`redis-py` already ships transitively via
  `django-rq`/`rq`; `requirements.txt` is unchanged).

- **`cameras/management/commands/stream_cameras.py`** — long-running worker,
  run the same way `rqworker` already is (a management command, one
  docker-compose service). Every ~15s, reconciles a `{camera_id: thread}` map
  against `Camera.objects.all()` — starts a thread for new/changed cameras,
  stops+joins threads for removed ones. Each per-camera thread: open
  `cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)` (same open/read timeout
  properties as `cameras/services/rtsp.py`), read continuously, push a JPEG
  to Redis at a capped rate (~150ms, ~6-7fps — bounds CPU/bandwidth
  regardless of the camera's native framerate). On failure: backoff and
  retry; write `status`/`last_error`/`last_checked_at` back to the `Camera`
  row, throttled (on transition or at most every 30s) to avoid hammering
  Postgres. Handles SIGTERM/SIGINT for clean shutdown.

- **`cameras/admin.py`** — followed `videos/admin.py`'s `VideoAdmin` pattern
  (custom admin URLs + an action that redirects to a `TemplateResponse`
  page) instead of a separate app-level `views.py`/`urls.py`, to match this
  repo's existing convention for "select one row, view something full-page":
  - `get_urls()` adds two URLs wrapped in `self.admin_site.admin_view(...)`
    (Django admin's own auth, no separate `@staff_member_required` needed):
    `admin:cameras_camera_mjpeg` (`mjpeg_view` — the `StreamingHttpResponse`,
    `multipart/x-mixed-replace; boundary=frame`, polling `frame_cache` every
    ~100ms) and `admin:cameras_camera_live` (`live_view` — a full-page
    `TemplateResponse` using the new `templates/admin/cameras/camera_live.html`,
    modeled on `templates/admin/videos/video_player.html`).
  - `live_preview` — readonly field on the change form, an inline `<img>`
    pointing at the mjpeg URL plus an "Open full live view" link, or a
    "save first to preview" placeholder on the add form (`obj is None`).
  - `actions = ["view_live_stream"]` — `@admin.action(description="View
    live stream…")`, mirrors `VideoAdmin.play_video`: requires exactly one
    selected row (warns otherwise) and redirects to the live view page.

- **`templates/admin/cameras/camera_live.html`** — extends
  `admin/base_site.html` (this repo's convention for every custom admin
  page), shows the `<img>` plus `camera.status`/`last_error`/
  `last_checked_at` for diagnostics, and a "Back to cameras" link.

- **Two `docker-compose.yml` files updated** — this repo has both a
  standalone one in `chachkalica/` (for running the Django app alone) and a
  parent-level unified one at `chackalica_unified/docker-compose.yml` (the
  one actually running day-to-day, which also builds `trainer`/
  `sam-backend`). Both got the same new `camera-workers` service running
  `python manage.py stream_cameras`, **not** reusing the `x-app`/anchor
  wholesale — that anchor mounts the host docker socket (needed by `worker`
  for Label Studio provisioning) and other services' data mounts, none of
  which this worker touches. Its env is just `POSTGRES_*` (to read `Camera`
  rows via the Django ORM) and `REDIS_URL`.

- **`cameras/tests.py`** — added `LiveStreamTests`: mjpeg requires login and
  404s for an unknown camera, correct `Content-Type`, live view renders the
  camera and 404s for unknown ids, change page shows the preview/link, and
  the action both rejects multi-selection and redirects correctly for one.

## Sequencing (as it happened)

1. `frame_cache.py` — Small.
2. `stream_cameras` management command — Medium.
3. `cameras/admin.py` (urls, views, action, inline preview) — Medium.
4. `camera_live.html` template — Small.
5. Both `docker-compose.yml` files — Small.
6. Tests — Small.

## Verification (done)

Ran against the actual live stack (`chackalica_unified`), not a fresh one —
`camera-workers` was added and built there, `web` was rebuilt to pick up the
admin changes:

1. `docker compose up -d --build camera-workers`, then `docker compose up -d
   --build web` — confirmed via `docker compose logs camera-workers` that it
   found the existing registered camera (id 2) and started its worker.
2. Confirmed via `psql` that the camera's `status` flipped to `online`, and
   via `redis-cli GET`/`TTL camera:2:frame` that real JPEG bytes were present
   with the TTL counting down.
3. Ran the full `cameras` test suite in the `web` container against real
   Postgres — 16/16 passing (8 pre-existing + 8 new).
4. Used a throwaway superuser + Django's test `Client` (created and deleted
   within the same shell session) to hit the real admin over HTTP: the
   change page renders the `live_preview` `<img>` and "Open full live view"
   link; `GET .../mjpeg/?camera=2` returns `200`, the right `Content-Type`,
   and the first few streamed chunks are genuine `--frame` boundaries wrapping
   real JPEG data (verified via the `JFIF` marker); `POST` the
   `view_live_stream` action redirects to `.../live/?camera=2`; that page
   renders the camera's name and the correct mjpeg URL.

## Follow-up: preview downscaling + duplicate suppression (done)

The original version of this page flagged full-resolution streaming as a
deferred concern. It stopped being theoretical once a 4K camera was attached:
measured against the live showroom stream, a single viewer pulled **194 MB in
12 s (~16 MB/s)** — full-resolution frames, re-sent every poll.

Two changes, both in the producer rather than the viewer:

- **`cameras/services/preview.py`** — `resize` (to `PREVIEW_MAX_WIDTH`, 1280,
  `INTER_AREA`) + `encode` (quality 85). `stream_cameras` publishes
  `camera:<id>:preview` beside the full-resolution `:frame`, and the compositor
  publishes `:annotated` preview-sized outright, since a browser is its only
  consumer. `:frame` deliberately stays untouched — `predict_frame` hands those
  exact bytes to the trainer, and downscaling before inference would throw away
  the detail small/distant detections depend on.
- **`_mjpeg_frames` skips repeats.** The view polls faster than the worker
  pushes (0.1 s vs 0.15 s) precisely so no frame is missed, which means the
  same frame is regularly found twice; a stalled camera holds one frame for its
  whole TTL. Frames are now digested (`blake2b`) and sent once, with a
  `_MJPEG_KEEPALIVE_SECONDS` (5 s) re-send so a static scene can't leave the
  connection silent long enough for an intermediary to time it out.

The resize is paid once per frame in the worker that was already
decoding/encoding, not once per frame per viewer in a web worker shared with
the rest of the admin. Measured on the same 4K stream afterwards:

| | before | after |
|---|---|---|
| per viewer | 16.2 MB/s | **0.58 MB/s** (28× less) |
| per frame | ~1.6 MB (3840w) | ~124 KB (1280w) |
| frames/s delivered | ~10 (about half duplicates) | 4.8 (every distinct frame, no repeats) |
| producer cost | — | +3.7 ms/frame (2.2 resize + 1.5 encode) |

The delivered frame rate matches the worker's measured 4.9 fps push rate — the
drop from ~10 is duplicates no longer being sent, not frames being lost. That
push rate is bounded by `FRAME_PUSH_INTERVAL_SECONDS` (150 ms) plus one 4K
`capture.read()`, giving a ~219 ms median gap; the +3.7 ms of preview work is
inside the noise of that.

Cost of the fallback: `get_preview_frame` falls back to `:frame` when no
preview exists (a cold cache right after a `camera-workers` restart), which
streams 4K — a stopgap for a few seconds, not a mode to sit in.
