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


def output_dir() -> Path:
    d = videos_root() / "inferred"
    d.mkdir(parents=True, exist_ok=True)
    return d


def unique_output_filename(video_name: str) -> str:
    """An ``<stem>_inferred.mp4`` filename under :func:`output_dir` that collides
    with nothing already there — used to prefill/compute the output path up
    front, before the job runs."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", (video_name or "").strip()) or "video"
    if not _NAME_RE.match(stem):
        stem = "video"
    d = output_dir()
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


def run_inference_on_video(job) -> dict:
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

    output_path = output_dir() / job.output_filename
    tmp_frame_path = output_dir() / f".{job.pk}_frame.jpg"

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        capture.release()
        raise RuntimeError(f"Could not read frame size from video: {video_path}")

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
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
