"""RTMO (bottom-up pose estimator, mmpose/mmdeploy export) service handler.

Unlike every other arch here, the graph's own two raw outputs are not
``(boxes, scores, labels)`` at all — they're a pose model's native pair:

  * ``dets``      ``[K, 5]``  — ``x1, y1, x2, y2, score`` per person, already
    NMS'd (see below), in the graph's 640x640 letterboxed input-pixel frame.
  * ``keypoints`` ``[K, 17, 3]`` — COCO-17 ``x, y, score`` per person, gathered
    by the *same* NMS indices as ``dets`` row-for-row, in the same pixel frame.

Contract A has no field for a per-detection keypoint array, and nothing
downstream of it knows what to do with one — a Friendy ``(N,6)`` row is
``cx, cy, w, h, confidence, class_id`` and that is all. But the keypoints are
the reason this arch exists: this handler spends them on a **posture
classification** (standing / sitting / lying), computed once per person from
joint geometry, and emits *that* as Contract A's ``labels`` — three classes
rather than the raw keypoints Contract A could never carry anyway. The
keypoints themselves are discarded right after.

Graph-side NMS: the exporter's ``NonMaxSuppression`` node is baked in and
TensorRT's own ONNX parser converts it to an internal NMS layer with no graph
surgery needed — no ``EfficientNMS_TRT`` prep in
``ml/trt_export/arch/`` for this one, unlike retinanet/yolox — so this handler
runs no suppression of its own. The two outputs it receives are already the
NMS'd, index-matched pair.
"""

from __future__ import annotations

import numpy as np

from .base import ArchHandler

# COCO-17 keypoint order, as mmpose/RTMO emit it.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

STANDING, SITTING, LYING = 0, 1, 2
POSTURE_CLASSES = {STANDING: "standing", SITTING: "sitting", LYING: "lying"}

# A joint below this confidence is "not visible" — occluded, out of frame, or a
# false low-confidence blip — and is left out of every geometry computation
# rather than trusted. 0.3 matches the conventional COCO keypoint visibility
# floor (mmpose's own visualizers default to the same value); RTMO's real
# keypoints score in the 0.8-1.0 range on an unoccluded joint (see this
# handler's own verification notes), so this is a generous floor, not a tight
# one.
KPT_SCORE_MIN = 0.3

# Torso angle from vertical (degrees; 0 = upright, 90 = flat). At or above
# this, the shoulder-to-hip line reads flat rather than upright. On its own
# that is *not* enough to call someone lying down: a person slumped forward
# over a desk or reclined back in a chair tips their torso just as far while
# still plainly sitting, which is why a flat torso now has to survive the hip
# check below before it becomes LYING. 65 (was 55) already keeps ordinary
# seated lean — leaning in to a screen, sitting back — out of the flat branch
# entirely, so the hip check only has to arbitrate the genuinely extreme
# postures.
LYING_TORSO_DEG = 65.0

# Hip flexion: the angle at the hip between the torso (hip->shoulder) and the
# thigh (hip->knee), in degrees. This is the one measurement that separates
# the two things a flat torso can mean, because it does not care which way the
# body is oriented in the frame:
#
#   * lying down is an *extended* body — shoulder, hip and knee fall on
#     roughly one straight line, so the angle sits near 180.
#   * sitting is a *bent* body — the thigh runs out from under an upright-ish
#     torso at close to a right angle, so the angle sits near 90, and even a
#     recliner rarely opens past ~140.
#
# At or below this, a flat torso is read as a slumped or reclined sitter
# instead of a fallen one. Deliberately generous (a bent hip wins) because
# sitting misread as lying is the failure this bundle actually produces; the
# cost is that someone lying with their knees drawn up can read as SITTING.
HIP_FLEXION_SITTING_MAX = 140.0

# Thigh angle from vertical (hip-to-knee line). A standing leg's thigh hangs
# near-vertical; a seated thigh runs roughly level with the seat, well past
# this. Only consulted once the torso itself already reads upright.
SITTING_THIGH_DEG = 50.0

# Box aspect ratio (height / width) fallback, used only when the keypoints
# needed for the angle checks above aren't confidently visible (legs under a
# desk, a person shot from directly overhead, a low-confidence pass). A
# standing/sitting person's box is usually taller than wide; a prone one is
# usually wider than tall. These are deliberately loose — a last resort, not
# the primary signal. 0.6 (was 0.8) because a seated person's box is the
# squarest of the three postures — knees forward, legs cropped by a desk — and
# at 0.8 the merely-square seated boxes were landing in the prone branch;
# a genuinely prone box is wider still.
LYING_ASPECT_MAX = 0.6
STANDING_ASPECT_MIN = 1.5


def _angle_from_vertical(vec: np.ndarray) -> float:
    """``vec`` (dx, dy) image-space -> degrees off vertical, in [0, 90]."""
    dx, dy = float(vec[0]), float(vec[1])
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle at the shared origin between two image-space vectors, in
    [0, 180]. Degenerate (either vector ~zero-length, e.g. a hip and knee
    predicted on top of each other) returns 180 — "extended" — so a
    measurement this can't actually make never counts as a bent hip."""
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 180.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _joint(kpts: np.ndarray, index: int, min_score: float) -> np.ndarray | None:
    point = kpts[index]
    return point[:2] if point[2] >= min_score else None


def _midpoint(
    kpts: np.ndarray, left: int, right: int, min_score: float
) -> np.ndarray | None:
    """Mean of the two sides if both are visible; whichever one is if only
    one is; ``None`` if neither is — never a midpoint built from one side
    silently standing in for both."""
    a, b = _joint(kpts, left, min_score), _joint(kpts, right, min_score)
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def _classify_posture(kpts: np.ndarray, box: np.ndarray) -> int:
    shoulder = _midpoint(kpts, L_SHOULDER, R_SHOULDER, KPT_SCORE_MIN)
    hip = _midpoint(kpts, L_HIP, R_HIP, KPT_SCORE_MIN)

    if shoulder is not None and hip is not None:
        knee = _midpoint(kpts, L_KNEE, R_KNEE, KPT_SCORE_MIN)
        torso_angle = _angle_from_vertical(hip - shoulder)

        if torso_angle >= LYING_TORSO_DEG:
            # Flat torso — lying down, or a sitter slumped forward / reclined
            # back far enough to look like one. The hip decides: a bent hip is
            # a sitter, an extended one is prone. With no visible knee there
            # is no hip angle to read, and a flat torso alone stays LYING.
            if knee is None:
                return LYING
            hip_flexion = _angle_between(shoulder - hip, knee - hip)
            return SITTING if hip_flexion <= HIP_FLEXION_SITTING_MAX else LYING

        if knee is not None:
            thigh_angle = _angle_from_vertical(knee - hip)
            return SITTING if thigh_angle >= SITTING_THIGH_DEG else STANDING
        # Torso is upright but legs aren't visible (crop, occlusion, desk) —
        # fall through to the box-aspect fallback below rather than guess.

    x1, y1, x2, y2 = box
    width, height = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    aspect = height / width
    if aspect < LYING_ASPECT_MAX:
        return LYING
    if aspect >= STANDING_ASPECT_MIN:
        return STANDING
    return SITTING


class RTMOHandler(ArchHandler):
    """``(dets[K,5], keypoints[K,17,3])`` -> Contract A, labels = posture.

    ``dets``/``keypoints`` row ``i`` describe the same person (the graph
    gathers both by the same post-NMS indices), so no re-association is
    needed here — just read them across in lockstep.
    """

    name = "rtmo"

    def adapt_outputs(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(outputs) != 2:
            raise ValueError(
                f"{self.name}: expected 2 graph outputs (dets, keypoints), "
                f"got {len(outputs)}"
            )
        dets = np.asarray(outputs[0], dtype=np.float32).reshape(-1, 5)
        keypoints = np.asarray(outputs[1], dtype=np.float32).reshape(-1, 17, 3)
        if dets.shape[0] != keypoints.shape[0]:
            raise ValueError(
                f"{self.name}: dets ({dets.shape[0]}) and keypoints "
                f"({keypoints.shape[0]}) row counts disagree"
            )

        boxes = dets[:, :4]
        scores = dets[:, 4]
        labels = np.array(
            [_classify_posture(keypoints[i], boxes[i]) for i in range(boxes.shape[0])],
            dtype=np.int64,
        )
        return boxes, scores, labels
