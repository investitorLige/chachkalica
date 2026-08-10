"""Generic post-processing (pure NumPy): the canonical 3-tensor graph output
plus the applied :class:`Transform` -> a Friendy ``(N, 6)`` prediction array.

Uniform for every architecture — no per-arch logic here. The graph already
decoded, thresholded (to its arch default), NMS'd, and emitted xyxy boxes in its
own input-pixel frame (Contract A). This module only:

  1. applies the caller's runtime ``score_threshold`` (on top of the graph's),
  2. inverts the preprocessing transform back to original-image pixels,
  3. converts to normalized ``(x_center, y_center, width, height)`` and appends
     ``(confidence, class_id)`` — matching ``friendy_chachkalica.formats``.

Clipping to image bounds is opt-in per arch via the meta's ``clip_boxes``:
``xyxy_prediction_to_friendy`` does not clip, so this must clip exactly for the
archs whose adapter clips before calling it, and no others.
"""

from __future__ import annotations

import numpy as np

from .preprocess import Transform

# Column order of the Friendy prediction tensor (mirrors formats.FRIENDY_PREDICTION_COLUMNS).
FRIENDY_COLUMNS = ("x_center", "y_center", "width", "height", "confidence", "class_id")


def to_friendy(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    transform: Transform,
    score_threshold: float,
    clip_boxes: bool = False,
    box_coords: str = "input_pixels",
) -> np.ndarray:
    """boxes ``[N,4]`` xyxy, scores ``[N]``, labels ``[N]`` -> Friendy ``[M,6]``.

    ``box_coords``:
      * ``"input_pixels"`` — boxes are xyxy in the graph's input-tensor pixel
        frame; invert the recorded resize+pad to original pixels, then normalize.
      * ``"input_normalized"`` — boxes are already xyxy in ``[0,1]`` over the model
        input (DETR-family). The transform is irrelevant: the value maps straight
        to Friendy xywhn (matches the torch path, where scaling to the original
        size and re-normalizing cancels out).

    ``clip_boxes`` clamps boxes to the original image's bounds, to match archs
    whose torch path clips (YOLOX; RF-DETR under its letterbox canvas, since
    padded-margin predictions can land outside the original image; RT-DETR and
    ECDet, whose sigmoid-decoded centres can put a box past the canvas edge with
    no padding involved). A no-op for archs already in-bounds (RetinaNet).

    It applies under **both** ``box_coords``, in whichever frame the boxes are
    then normalized against: ``[0, orig_w] x [0, orig_h]`` after the inverse for
    ``input_pixels``, and ``[0, 1]`` for ``input_normalized`` — which is the same
    clamp, since for the normalized archs the model input's frame *is* the
    original image's (they stretch, so there is no padding to offset). Skipping
    it for ``input_normalized`` used to make the flag a silent no-op for
    RT-DETR/ECDet, whose torch paths do clip: the exported graph then kept boxes
    the torch path had clamped, so the same checkpoint scored differently per
    format — an unclipped box has a larger area, so IoU against an edge-touching
    ground truth drops and near-threshold matches flip to misses.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if boxes.shape[0] == 0:
        return np.zeros((0, len(FRIENDY_COLUMNS)), dtype=np.float32)

    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.shape[0] == 0:
        return np.zeros((0, len(FRIENDY_COLUMNS)), dtype=np.float32)

    boxes = boxes.copy()
    if box_coords == "input_pixels":
        # Invert preprocessing: input-pixel -> original-image pixel.
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform.pad_x) / transform.scale_x
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform.pad_y) / transform.scale_y
        if clip_boxes:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, transform.orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, transform.orig_h)
        norm_w, norm_h = transform.orig_w, transform.orig_h
    else:  # input_normalized — already [0,1] over the model input.
        if clip_boxes:
            boxes = np.clip(boxes, 0.0, 1.0)
        norm_w = norm_h = 1.0

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2

    xywhn = np.stack(
        [x_center / norm_w, y_center / norm_h, width / norm_w, height / norm_h],
        axis=1,
    )
    return np.concatenate(
        [xywhn, scores.reshape(-1, 1), labels.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
