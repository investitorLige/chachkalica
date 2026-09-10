"""Run a model frame-by-frame over a video and burn boxes onto it.

Mirrors ``videos.services.frame_extraction``'s cv2 read loop and its
``training.services.runner.predict_image`` call (see ``_has_person`` there),
but writes every frame back out into a real annotated mp4 instead of sampling
a few frames into a dataset.

The model is whatever the job points at — a trained ``.pt`` checkpoint or an
exported ``.onnx`` / ``.engine`` (see ``InferenceJob.model_checkpoint``) — and
each frame reaches it either whole (``pipeline="raw"``) or through a chachak
pipeline that tiles it / crops around detected people. Both are just fields on
the ``/predict_image`` payload, assembled once per job by
``config_gen.build_predict_request``, so the frame path is the only thing that
changes between frames.

``predict_image`` takes a file path (the trainer runs in its own container,
reached over HTTP — see ``training/services/runner.py``), so each inferred
frame is written to a reusable temp jpg before the call; the loop is
sequential so a single path is safe to reuse.

How the boxes *look* is a second, separable question: a job with an empty
``render_style`` gets :func:`draw_boxes` below (one line width, hashed colors —
the overlay for looking at what a model did), and a job carrying a style dict
gets :class:`videos.services.render_style.MarketingRenderer` instead (see the
"Run model inference for marketing…" action). Only the drawing call and the
encode settings differ; the frame loop is the same one.

Encoding goes through an ``ffmpeg`` subprocess (already installed in this image
for the video downloader's merge step) rather than ``cv2.VideoWriter``:
``opencv-python-headless`` wheels don't ship a licensed H.264 encoder, and its
portable fallback codec is not reliably playable in a browser ``<video>`` tag.
"""

import hashlib
import re
import subprocess
from pathlib import Path

from fleet.services.paths import videos_root
from videos.services.frame_extraction import _resolve_checkpoint

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_BOX_COLOR = (0, 255, 0)  # BGR — lime green, fallback for an unnamed class

# Distinct, readable BGR colors, one per class label. Kept away from near-black
# and near-white so every color is visible against a video frame.
_CLASS_COLOR_PALETTE = [
    (0, 255, 0),      # green
    (0, 165, 255),    # orange
    (255, 0, 0),      # blue
    (0, 0, 255),      # red
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (128, 0, 255),    # rose
    (255, 128, 0),    # azure
    (0, 128, 128),    # olive
]

# Box/label sizes below are tuned for a 1080p frame and scaled up from there, so
# a 4K source (the showroom camera is 3840x2160) doesn't get hairline boxes and
# unreadable labels. Never scaled *down* — sub-1080p frames keep the 1080p sizes,
# since 1px boxes and sub-pixel text would be worse.
_REFERENCE_HEIGHT = 1080
_BOX_THICKNESS = 2
_FONT_SCALE = 0.5


def _color_for_class(class_name: str):
    """A BGR color for ``class_name``, stable across processes and runs.

    Keyed by name rather than ``class_id`` so a label keeps its color even if a
    model's class ordering differs from another's. Hashed with md5 instead of
    the builtin ``hash()``, which is salted per-process and would reshuffle
    colors on every restart.
    """
    if not class_name:
        return _BOX_COLOR
    digest = hashlib.md5(class_name.encode()).digest()
    return _CLASS_COLOR_PALETTE[digest[0] % len(_CLASS_COLOR_PALETTE)]


def output_dir(root: Path | None = None) -> Path:
    """Where a job's rendered artifacts go: ``<root>/inferred``.

    ``root`` defaults to the videos root, which is every caller in this app. It
    is a parameter because the Marketing Studio section keeps its own video
    library and reuses this whole module over its own directory — and the two
    must not share one output folder: :func:`unique_output_filename` only sees
    the directory it is given, and :func:`run_inference_on_video`'s scratch frame
    is named after the job's pk, which collides across two tables.
    """
    d = (root or videos_root()) / "inferred"
    d.mkdir(parents=True, exist_ok=True)
    return d


def unique_output_filename(video_name: str, root: Path | None = None) -> str:
    """An ``<stem>_inferred.mp4`` filename under :func:`output_dir` that collides
    with nothing already there — used to prefill/compute the output path up
    front, before the job runs."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", (video_name or "").strip()) or "video"
    if not _NAME_RE.match(stem):
        stem = "video"
    d = output_dir(root)
    candidate = f"{stem}_inferred.mp4"
    n = 2
    while (d / candidate).exists():
        candidate = f"{stem}_inferred_{n}.mp4"
        n += 1
    return candidate


def draw_boxes(frame, boxes: list[dict]) -> None:
    """Burn normalized center-xywh boxes onto ``frame`` (BGR ndarray) in place.

    Shared with the live camera path (``cameras.services.live_inference``), which
    draws the same ``/predict_image`` box dicts onto RTSP frames.

    Line widths and label text scale with the frame height (see
    :data:`_REFERENCE_HEIGHT`), and both the box and its label are clamped into
    the frame — a detection whose box extends past the top edge would otherwise
    have its label drawn at a negative y and simply vanish.

    Each box is colored by its ``class_name`` (see :func:`_color_for_class`), so
    different labels are visually distinguishable at a glance.
    """
    import cv2

    height, width = frame.shape[:2]
    scale = max(1.0, height / _REFERENCE_HEIGHT)
    thickness = max(1, round(_BOX_THICKNESS * scale))
    font_scale = _FONT_SCALE * scale
    text_thickness = max(1, round(scale))
    gap = round(6 * scale)

    for box in boxes:
        cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
        class_name = box.get("class_name", "?")
        color = _color_for_class(class_name)
        x1 = min(max(int((cx - w / 2) * width), 0), width - 1)
        y1 = min(max(int((cy - h / 2) * height), 0), height - 1)
        x2 = min(max(int((cx + w / 2) * width), 0), width - 1)
        y2 = min(max(int((cy + h / 2) * height), 0), height - 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name} {box.get('confidence', 0):.2f}"
        (_, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        # Above the box when it fits, otherwise just inside the top edge.
        text_y = y1 - gap if y1 - gap - text_h > 0 else y1 + text_h + gap
        text_y = min(text_y, height - 1)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, text_thickness, cv2.LINE_AA)


def build_predict_payload(job) -> dict:
    """The ``/predict_image`` payload for ``job``, minus the per-frame image path.

    Resolves the job's model (trained checkpoint, exported artifact, or the model
    inside a bundle) and its pipeline/detector/tiling knobs into one payload,
    reused for every frame with
    only ``image_path`` swapped in. Raises ``RuntimeError`` for an unusable model
    or an inconsistent pipeline configuration.
    """
    from training.services import config_gen

    try:
        checkpoint = job.model_checkpoint()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if not Path(checkpoint).exists():
        raise RuntimeError(f"Model artifact not found: {checkpoint}")

    detector_checkpoint = (job.detector_checkpoint or "").strip()
    if detector_checkpoint:
        detector_checkpoint = str(_resolve_checkpoint(detector_checkpoint))

    try:
        return config_gen.build_predict_request(
            checkpoint,
            job.pipeline,
            image_path="",  # filled in per frame
            detector_checkpoint=detector_checkpoint,
            detector_expand_ratio=job.detector_expand_ratio,
            detector_min_box_size=job.detector_min_box_size,
            tile_size_px=job.tile_size_px,
            tile_width_pct=job.tile_width_pct,
            tile_height_pct=job.tile_height_pct,
            overlap=job.overlap,
            merge_nms_iou=job.merge_nms_iou,
            chain=job.chain,
            score_threshold=job.score_threshold,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def run_inference_on_video(job, *, root: Path | None = None) -> dict:
    """Run ``job``'s model over ``job.video`` frame-by-frame, writing the annotated
    result to ``output_dir() / job.output_filename``.

    ``job.output_filename`` must already be set (the admin action computes it
    up front via :func:`unique_output_filename`, before enqueuing).
    """
    import cv2

    from training.services import runner

    if not job.output_filename:
        raise RuntimeError("InferenceJob.output_filename must be set before running.")

    payload = build_predict_payload(job)

    video_path = job.video.path()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    out = output_dir(root)
    output_path = out / job.output_filename
    tmp_frame_path = out / f".{job.pk}_frame.jpg"

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        capture.release()
        raise RuntimeError(f"Could not read frame size from video: {video_path}")

    # A styled job draws through the marketing renderer and may also be cut down
    # to a delivery resolution; a plain one keeps every existing default.
    renderer = None
    crf = None
    out_width, out_height = width, height
    if job.render_style:
        from videos.services import render_style

        renderer = render_style.MarketingRenderer(job.render_style)
        encode = render_style.encode_options(job.render_style)
        crf = encode["crf"]
        if encode["output_height"] and encode["output_height"] < height:
            # Resized before drawing, not by ffmpeg afterwards: the style's line
            # widths and label sizes are in output pixels, so a 4K frame drawn at
            # 4K and then squeezed to 1080p would hand back a third of the stroke
            # the operator chose. Inference still sees the full-resolution frame.
            out_height = encode["output_height"] - (encode["output_height"] % 2)
            out_width = round(width * out_height / height / 2) * 2

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_width}x{out_height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            *(["-crf", str(crf), "-preset", "slow"] if crf is not None else []),
            str(output_path),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    frame_stride = max(1, job.frame_stride)
    frames_total = 0
    frames_processed = 0
    current_boxes: list[dict] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frames_total % frame_stride == 0:
                cv2.imwrite(str(tmp_frame_path), frame)
                response = runner.predict_image(
                    {**payload, "image_path": str(tmp_frame_path)}
                )
                current_boxes = response.get("boxes", [])
                frames_processed += 1

            if (out_width, out_height) != (width, height):
                frame = cv2.resize(frame, (out_width, out_height),
                                   interpolation=cv2.INTER_AREA)
            if renderer is not None:
                # Every frame, not only inferred ones: the renderer's easing is
                # per rendered frame, which is what turns a held box (stride > 1)
                # into one that glides to its next position.
                renderer.draw(frame, current_boxes)
            else:
                draw_boxes(frame, current_boxes)
            ffmpeg.stdin.write(frame.tobytes())
            frames_total += 1
    finally:
        capture.release()
        tmp_frame_path.unlink(missing_ok=True)
        ffmpeg.stdin.close()
        stderr = ffmpeg.stderr.read()
        ffmpeg.wait()

    if ffmpeg.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg exited with {ffmpeg.returncode}: {stderr.decode(errors='replace')[-2000:]}"
        )
    if frames_total == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("No frames read from the video.")

    return {
        "output_filename": job.output_filename,
        "frames_total": frames_total,
        "frames_processed": frames_processed,
    }


# --------------------------------------------------------------------- preview

#: Where :func:`preview_frame` parks the model's answer for one frame, so that
#: re-rendering the same frame with a different *look* costs no GPU. Under the
#: output directory (already ours, already excluded from the video scanner by its
#: leading dot) rather than a Django cache: the marketing form is served by the
#: web container and there may be several of them, while this survives a restart
#: and is shared between them.
_PREVIEW_CACHE_ENTRIES = 200

#: Previews are drawn at most this tall. The style's sizes scale with the frame
#: in both directions (see render_style), so a 1080p preview of a 4K render is a
#: faithful proportional preview of it — and a 4K jpeg base64'd into a page is
#: not something to hand a browser on every slider nudge.
_PREVIEW_MAX_HEIGHT = 1080


def _preview_cache_dir(root: Path | None = None) -> Path:
    d = output_dir(root) / ".preview_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _preview_cache_key(payload: dict, video_path: Path, frame_index: int) -> str:
    """Identity of "these boxes": the exact model request, the video's bytes (by
    size + mtime, cheap and good enough for a preview) and which frame."""
    import json

    try:
        stat = video_path.stat()
        fingerprint = f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        fingerprint = "?"
    material = json.dumps(
        {**payload, "image_path": ""}, sort_keys=True, default=str
    ) + f"|{video_path}|{fingerprint}|{frame_index}"
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def _prune_preview_cache(root: Path | None = None) -> None:
    entries = sorted(_preview_cache_dir(root).glob("*.json"), key=lambda p: p.stat().st_mtime)
    for path in entries[:-_PREVIEW_CACHE_ENTRIES]:
        path.unlink(missing_ok=True)


def preview_frame(job, position: float = 0.5, *, use_cache: bool = True,
                  root: Path | None = None) -> dict:
    """Render one frame of ``job``'s video the way the finished video will look.

    The feedback loop the marketing action is built around: thirty style knobs
    are unusable if seeing their effect means re-encoding a five-minute clip, so
    the form previews a single frame instead. ``position`` is a fraction of the
    video's duration.

    The model runs only on a cache miss — the inference result for a given
    (payload, video, frame) is stable, so dragging a slider re-draws from the
    cached boxes and never touches the trainer's GPU. ``use_cache=False`` forces
    a re-run, which is what the form's explicit "re-run the model" does.

    ``job`` need not be saved; nothing here reads its pk. Returns
    ``{"jpeg", "width", "height", "boxes", "frame_index", "frames_total",
    "cached"}``. Raises ``RuntimeError`` for an unreadable video or model.
    """
    import json

    import cv2

    from training.services import runner
    from videos.services import render_style

    payload = build_predict_payload(job)
    video_path = job.video.path()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        frames_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        position = min(max(float(position), 0.0), 1.0)
        # Never the very last frame: some containers report a frame count one
        # past what actually decodes, and a preview that fails at the far end of
        # the slider looks like a broken feature.
        frame_index = int(position * max(frames_total - 2, 0)) if frames_total > 2 else 0
        if frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            # A seek can land on a non-decodable position in a damaged file; the
            # first frame always decodes if anything does.
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = 0
            ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read a frame from {video_path.name}.")
    finally:
        capture.release()

    cache_path = _preview_cache_dir(root) / f"{_preview_cache_key(payload, video_path, frame_index)}.json"
    boxes, cached = None, False
    if use_cache and cache_path.is_file():
        try:
            boxes = json.loads(cache_path.read_text())
            cached = True
        except (OSError, json.JSONDecodeError):
            boxes = None
    if boxes is None:
        tmp_frame_path = _preview_cache_dir(root) / f".preview_{cache_path.stem}.jpg"
        try:
            cv2.imwrite(str(tmp_frame_path), frame)
            response = runner.predict_image({**payload, "image_path": str(tmp_frame_path)})
        finally:
            tmp_frame_path.unlink(missing_ok=True)
        boxes = response.get("boxes", [])
        cache_path.write_text(json.dumps(boxes))
        _prune_preview_cache(root)

    height, width = frame.shape[:2]
    encode = render_style.encode_options(job.render_style)
    target_height = min(encode["output_height"] or height, height, _PREVIEW_MAX_HEIGHT)
    if target_height != height:
        target_width = round(width * target_height / height)
        frame = cv2.resize(frame, (target_width, target_height),
                           interpolation=cv2.INTER_AREA)

    drawn = render_style.MarketingRenderer(job.render_style, still=True).draw(frame, boxes)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("Could not encode the preview frame.")

    return {
        "jpeg": buffer.tobytes(),
        "width": frame.shape[1],
        "height": frame.shape[0],
        "boxes": drawn,
        "detections": len(boxes),
        "frame_index": frame_index,
        "frames_total": frames_total,
        "cached": cached,
    }
