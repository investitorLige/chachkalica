"""SCRFD (face detector) service handler.

Every other handler here is a :class:`PassthroughHandler`: the export wrapper
bakes decode *and* NMS into the graph, so the three outputs are already
``(boxes, scores, labels)``, suppressed, ready for :func:`onnx_infer.postprocess
.to_friendy` to threshold and map back. SCRFD's exporter (see
``bundles_dir/face_det_bundle``'s ``scrfd_export`` kit) bakes the anchor decode
and a fixed top-K selection into the graph, but deliberately leaves NMS out of
it: the graph also emits a 5-point landmark per row, ``EfficientNMS_TRT``
returns no indices to gather those landmarks by, and TensorRT rejects a
data-dependent NMS shape as a graph op. So the three raw outputs here are
``(boxes[K,4], scores[K], kps[K,5,2])`` — unsuppressed, in the engine's
descending-score ``TopK`` order — and this handler is the one place in the
service that runs NMS itself instead of trusting the graph already did.

Landmarks are read off ``outputs[2]`` and then dropped: Contract A is exactly
``(boxes, scores, labels)``, and neither ``Detection`` nor anything reading it
downstream has anywhere to put a per-face 5-point landmark yet. They survive
NMS under the same ``keep`` mask as the boxes, so wiring them through later is
a matter of widening Contract A, not redoing the suppression.
"""

from __future__ import annotations

import numpy as np

from .base import ArchHandler

# SCRFD's graph has one class and no class output at all — "face" is this
# handler's own constant, not read off anything.
_FACE_CLASS_ID = 0

# The NMS IoU gate. Contract B (`ModelMeta`) has no field for it — only
# `score_threshold` travels per bundle, applied downstream by `to_friendy` —
# and `adapt_outputs` is only ever handed the graph's raw tensors, not the
# meta.json dict. 0.4 matches `face_det_bundle/models/face_det.meta.json`'s
# `nms_threshold` and the `scrfd_export` kit's own `--nms-iou` default; if a
# future SCRFD bundle is exported with a different value, update it here too.
_NMS_IOU = 0.4


class SCRFDHandler(ArchHandler):
    """``(boxes[K,4], scores[K], kps[K,5,2])``, unsuppressed -> Contract A.

    Running NMS on the *full* K rows rather than only the ones that will pass
    the caller's score threshold gives the same kept set as thresholding
    first: NMS only ever lets a higher-scoring row suppress a lower one, never
    the reverse, so a sub-threshold row can't change which above-threshold
    rows survive — `to_friendy`'s own threshold drops it afterward regardless.
    That means this handler needs no threshold of its own.
    """

    name = "scrfd"

    def adapt_outputs(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(outputs) != 3:
            raise ValueError(
                f"{self.name}: expected 3 graph outputs (boxes, scores, kps), "
                f"got {len(outputs)}"
            )
        boxes = np.asarray(outputs[0], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
        # outputs[2] is kps[K,5,2] — landmarks; see module docstring.

        keep = _nms(boxes, scores, _NMS_IOU)
        boxes, scores = boxes[keep], scores[keep]
        labels = np.full(scores.shape, _FACE_CLASS_ID, dtype=np.int64)
        return boxes, scores, labels


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy IoU NMS — this arch's graph does not do its own (see above).

    Reproduces the exporter's own reference NMS exactly rather than a generic
    IoU convention: areas and intersections are measured ``(x1 - x0 + 1)``,
    the InsightFace inclusive-pixel convention SCRFD was trained and verified
    against (``scrfd_export``'s own ``scrfd_infer/nms.py``). Dropping the
    ``+1`` shifts IoU by ~0.5% on a 30px face — enough to change what survives
    at this threshold — so it stays. Suppression is strict ``iou > threshold``,
    matching the same reference.
    """
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x1 - x0 + 1) * (y1 - y0 + 1)
    order = np.argsort(-scores)

    keep = np.zeros(boxes.shape[0], dtype=bool)
    suppressed = np.zeros(boxes.shape[0], dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep[i] = True
        xx0 = np.maximum(x0[i], x0[order])
        yy0 = np.maximum(y0[i], y0[order])
        xx1 = np.minimum(x1[i], x1[order])
        yy1 = np.minimum(y1[i], y1[order])
        inter = np.clip(xx1 - xx0 + 1, 0, None) * np.clip(yy1 - yy0 + 1, 0, None)
        union = areas[i] + areas[order] - inter
        iou = inter / np.clip(union, np.finfo(np.float32).eps, None)
        # Includes i's own self-comparison (iou == 1.0, always > threshold), but
        # that only ever re-flags suppressed[i] — harmless, since `keep[i]` is
        # already recorded above and i is never revisited as a candidate seed.
        suppressed[order] |= iou > iou_threshold
    return keep
