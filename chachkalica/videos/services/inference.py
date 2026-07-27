"""Run a trained model frame-by-frame over a video and burn boxes onto it.

Mirrors ``videos.services.frame_extraction``'s cv2 read loop and its
``training.services.runner.predict_image`` call (see ``_has_person`` there),
but writes every frame back out into a real annotated mp4 instead of sampling
a few frames into a dataset.

``predict_image`` takes a file path (the trainer runs in its own container,
reached over HTTP — see ``training/services/runner.py``), so each inferred
frame is written to a reusable temp jpg before the call; the loop is
sequential so a single path is safe to reuse.

Encoding goes through an ``ffmpeg`` subprocess (already installed in this image
for the video downloader's merge step) rather than ``cv2.VideoWriter``:
``opencv-python-headless`` wheels don't ship a licensed H.264 encoder, and its
portable fallback codec is not reliably playable in a browser ``<video>`` tag.
"""

import re
import subprocess
from pathlib import Path

from fleet.services.paths import videos_root
from videos.services.frame_extraction import _resolve_checkpoint

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_BOX_COLOR = (0, 255, 0)  # BGR — lime green


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


def _draw_boxes(frame, boxes: list[dict]) -> None:
    """Burn normalized center-xywh boxes onto ``frame`` (BGR ndarray) in place."""
    import cv2

    height, width = frame.shape[:2]
    for box in boxes:
        cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"{box.get('class_name', '?')} {box.get('confidence', 0):.2f}"
        text_y = y1 - 6 if y1 - 6 > 10 else y1 + 16
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _BOX_COLOR, 1, cv2.LINE_AA)


def run_inference_on_video(job) -> dict:
    """Run ``job.trained_model`` over ``job.video`` frame-by-frame, writing the
    annotated result to ``output_dir() / job.output_filename``.

    ``job.output_filename`` must already be set (the admin action computes it
    up front via :func:`unique_output_filename`, before enqueuing).
    """
    import cv2

    from training.services import runner

    if not job.output_filename:
        raise RuntimeError("InferenceJob.output_filename must be set before running.")

    checkpoint = _resolve_checkpoint(job.trained_model.checkpoint_path)
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")

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
                response = runner.predict_image({
                    "model_checkpoint": str(checkpoint),
                    "image_path": str(tmp_frame_path),
                    "pipeline": "raw",
                    "score_threshold": job.score_threshold,
                })
                current_boxes = response.get("boxes", [])
                frames_processed += 1

            _draw_boxes(frame, current_boxes)
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
