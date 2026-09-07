"""The whole RTMO path in one place: image -> letterbox -> engine -> people.

This is the module ``run_pose.py`` and ``build/verify_engine.py`` share. It is
deliberately thin — every step it performs is the *same code the app performs*,
imported from ``vendor/``:

    image (BGR uint8 HWC)
      -> _bgr_to_chw            (app_glue/engine_runtime.py, copied here)
      -> onnx_infer.preprocess  (letterbox to 640x640, pad 114, x255, BGR flip)
      -> TrtModel / OnnxModel   (dets[N,5], keypoints[N,17,3], already NMS'd)
      -> _classify_posture      (onnx_infer/arch/rtmo.py — standing/sitting/lying)
      -> inverse letterbox      (input pixels -> original image pixels)

The one thing it does that the app does *not* is keep the keypoints. The app's
adapter (``TrtAdapter.predict``) hands the graph's pair to ``RTMOHandler``, which
spends the keypoints on a posture label and drops them, because Contract A
(``boxes, scores, labels``) has nowhere to put 17 points per person. Here they
are the output, so the coordinate inverse is applied to them too.

Import from the manual's root directory (``vendor/`` has to be importable):

    cd pose_estimation_manual && python -c "import pose_runtime"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vendor.onnx_infer.arch.rtmo import (
    KPT_SCORE_MIN,
    POSTURE_CLASSES,
    _classify_posture,
)
from vendor.onnx_infer.meta import ModelMeta
from vendor.onnx_infer.preprocess import Transform, preprocess

# COCO-17, in the order RTMO emits them. Same indices as the constants at the top
# of vendor/onnx_infer/arch/rtmo.py.
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# Skeleton edges for drawing only — the posture heuristic uses joints, not edges.
SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Person:
    """One detected person, in original-image pixel coordinates."""

    box: tuple[float, float, float, float]   # x1, y1, x2, y2
    score: float
    posture: str
    posture_id: int
    keypoints: np.ndarray                    # [17, 3] — x, y, score

    def to_dict(self, width: int, height: int) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "score": round(float(self.score), 4),
            "posture": self.posture,
            "posture_id": int(self.posture_id),
            "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            "box_normalized": [
                round(float(x1 / width), 5), round(float(y1 / height), 5),
                round(float(x2 / width), 5), round(float(y2 / height), 5),
            ],
            "keypoints": {
                name: [round(float(x), 1), round(float(y), 1), round(float(s), 4)]
                for name, (x, y, s) in zip(KEYPOINT_NAMES, self.keypoints)
            },
        }


# ────────────────────────────────────────────────────────────── input ──

def load_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image as HWC uint8 **BGR** — the layout the app's frames arrive in.

    OpenCV would give this directly; PIL is used so this module needs no cv2. The
    channel order matters: ``_bgr_to_chw`` leaves it alone and
    ``preprocess`` flips it because ``meta.layout == "bgr"``. Handing this
    function an RGB array instead feeds the graph its channels backwards, which
    shows up as quietly worse detections rather than an error.
    """
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def bgr_to_chw(frame_bgr: np.ndarray) -> np.ndarray:
    """HWC BGR uint8 -> CHW float32 in [0, 1]. Verbatim from the app's
    ``detections/services/engine_runtime.py::_bgr_to_chw`` (minus the torch
    wrapper, which ``preprocess`` does not need)."""
    return np.ascontiguousarray(frame_bgr.transpose(2, 0, 1), dtype=np.float32) / 255.0


def iter_images(target: str | Path):
    """One image path, or every image under a directory, recursively, sorted."""
    target = Path(target)
    if target.is_file():
        return [target]
    return sorted(
        p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )


# ──────────────────────────────────────────────────────────── session ──

def resolve_meta_path(model_path: str | Path) -> Path:
    """The Contract-B sidecar for an artifact, tolerating the ``.trt.onnx`` stem.

    The app resolves this as ``path.with_suffix(".meta.json")``, which is right
    for ``rtmo.engine`` -> ``rtmo.meta.json`` but misses for
    ``rtmo.trt.onnx`` -> ``rtmo.trt.meta.json``, a file the bundle does not
    carry. So the ONNX escape hatch cannot be loaded through
    ``vendor/onnx_infer/load_onnx_adapter`` against this bundle as shipped —
    see README.md, "The ONNX escape hatch". Here the extra ``.trt`` segment is
    stripped as a fallback instead.
    """
    model_path = Path(model_path)
    direct = model_path.with_suffix(".meta.json")
    if direct.exists():
        return direct
    stem = model_path.stem
    if stem.endswith(".trt"):
        stripped = model_path.with_name(stem[: -len(".trt")] + ".meta.json")
        if stripped.exists():
            return stripped
    return direct  # let ModelMeta.load report the miss, with the expected name


def load_session(model_path: str | Path, device: str = "cuda"):
    """``(session, meta)`` for a ``.engine`` (TensorRT) or ``.onnx`` (onnxruntime).

    Both expose the same ``run(batched)``. Their return shapes differ slightly —
    ``normalize_outputs`` below squares that away.
    """
    model_path = Path(model_path)
    meta = ModelMeta.load(resolve_meta_path(model_path))
    if meta.arch != "rtmo":
        raise ValueError(
            f"{model_path.name}'s meta.json says arch={meta.arch!r}; this module "
            "only knows the rtmo (dets, keypoints) output layout."
        )
    if model_path.suffix.lower() == ".engine":
        from vendor.trt_infer.session import TrtModel

        return TrtModel(model_path, meta, device=device), meta
    from vendor.onnx_infer.session import OnnxModel

    return OnnxModel(model_path, meta, device=device), meta


def normalize_outputs(raw) -> tuple[np.ndarray, np.ndarray]:
    """Whatever the session returned -> ``(dets[N,5], keypoints[N,17,3])``.

    ``TrtModel.run`` already splits the batch axis off for B == 1 (see
    ``_split_rtmo``), while ``OnnxModel.run`` hands back the graph's own
    ``[1,N,5] / [1,N,17,3]``. ``RTMOHandler`` flattens both the same way, so this
    does too.
    """
    dets_raw, kpts_raw = raw[0], raw[1]
    dets = np.asarray(dets_raw, dtype=np.float32).reshape(-1, 5)
    keypoints = np.asarray(kpts_raw, dtype=np.float32).reshape(-1, 17, 3)
    if dets.shape[0] != keypoints.shape[0]:
        raise ValueError(
            f"dets ({dets.shape[0]}) and keypoints ({keypoints.shape[0]}) row "
            "counts disagree — the graph gathers both by the same post-NMS "
            "indices, so this should be impossible."
        )
    return dets, keypoints


# ───────────────────────────────────────────────────────────── decode ──

def decode(
    dets: np.ndarray,
    keypoints: np.ndarray,
    transform: Transform,
    score_threshold: float,
    clip: bool = True,
) -> list[Person]:
    """Graph output (input-pixel frame) -> ``Person`` list (original pixels).

    Posture is classified **before** the coordinate inverse, exactly as
    ``RTMOHandler`` does it. That is safe rather than lucky: letterboxing scales
    both axes by the same factor, so every angle the heuristic measures and the
    one aspect ratio it falls back on are unchanged by the inverse.
    """
    people: list[Person] = []
    for index in range(dets.shape[0]):
        score = float(dets[index, 4])
        if score < score_threshold:
            continue
        box_input = dets[index, :4].astype(np.float32)
        posture_id = _classify_posture(keypoints[index], box_input)

        box = box_input.copy()
        box[[0, 2]] = (box[[0, 2]] - transform.pad_x) / transform.scale_x
        box[[1, 3]] = (box[[1, 3]] - transform.pad_y) / transform.scale_y
        kpts = keypoints[index].copy()
        kpts[:, 0] = (kpts[:, 0] - transform.pad_x) / transform.scale_x
        kpts[:, 1] = (kpts[:, 1] - transform.pad_y) / transform.scale_y
        if clip:
            box[[0, 2]] = np.clip(box[[0, 2]], 0.0, transform.orig_w)
            box[[1, 3]] = np.clip(box[[1, 3]], 0.0, transform.orig_h)

        people.append(
            Person(
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                score=score,
                posture=POSTURE_CLASSES[posture_id],
                posture_id=posture_id,
                keypoints=kpts,
            )
        )
    return people


def run_image(
    session,
    meta: ModelMeta,
    frame_bgr: np.ndarray,
    score_threshold: float | None = None,
) -> tuple[list[Person], tuple[np.ndarray, np.ndarray], Transform]:
    """One BGR frame -> ``(people, (dets, keypoints), transform)``.

    The raw pair comes back alongside the decoded people so a verification run
    can compare two runtimes numerically without decoding twice.
    """
    threshold = meta.score_threshold if score_threshold is None else float(score_threshold)
    batched, transform = preprocess(bgr_to_chw(frame_bgr), meta)
    dets, keypoints = normalize_outputs(session.run(batched))
    people = decode(dets, keypoints, transform, threshold, clip=meta.clip_boxes)
    return people, (dets, keypoints), transform


__all__ = [
    "KEYPOINT_NAMES", "KPT_SCORE_MIN", "POSTURE_CLASSES", "SKELETON",
    "Person", "bgr_to_chw", "decode", "iter_images", "load_image_bgr",
    "load_session", "normalize_outputs", "resolve_meta_path", "run_image",
]
