"""RF-DETR (raw torch export) service handler.

Unlike the ``rfdetr`` handler, this arch's graph was never run through the
``onnx_export/arch/rfdetr.py`` wrapper — it's a plain ``torch.onnx.export`` of
the RF-DETR head, so none of the postprocessing math is baked in. The graph
emits exactly ``(dets[B,300,4], labels[B,300,5])``:

* ``dets`` — normalized ``[0,1]`` ``cxcywh`` boxes over the model input, one
  per query. No score column (unlike ``rtmo``'s ``dets``).
* ``labels`` — raw per-class logits, 4 real classes plus one background slot
  in position 4 (**not** a class id — RF-DETR has no "argmax" output at all).

So this handler does the decode a wrapped export would have baked into the
graph: per-class sigmoid (RF-DETR classifies each class independently, not a
softmax over a mutually-exclusive set — a query can plausibly fire two
classes at once), drop the background slot, flatten to (query, class) pairs
and take the top ``_NUM_SELECT`` by score. No NMS: RF-DETR is a set-prediction
model trained not to duplicate a query onto one object, so nothing here
suppresses overlapping boxes (matches ``from_my_bitches/weapon/
onnx_inference.py``, the checkpoint's own reference/verify script, which
does the same three things in the same order and explicitly skips NMS).

Boxes come out already normalized over the model input, so meta.json for this
arch uses ``box_coords: "input_normalized"`` — same convention as ``rtdetr``,
no resize/pad inverse needed.
"""

from __future__ import annotations

import numpy as np

from .base import ArchHandler

# RF-DETR's own default `num_select`: how many (query, class) pairs survive
# the flatten-and-sort, before `to_friendy`'s score threshold is even applied.
# Matches the reference script's default exactly — see module docstring.
_NUM_SELECT = 300


class RFDetrRawHandler(ArchHandler):
    """``(dets[N,4], labels[N,5])`` raw -> Contract A ``(boxes, scores, labels)``.

    ``vendor/trt_infer/session.py`` recognizes this arch's two raw output names
    (``dets``, ``labels``) and hands them here per-image, batch axis already
    stripped — see its ``RFDETR_RAW_OUTPUTS``/``_split_rfdetr_raw``.
    """

    name = "rfdetr_raw"

    def adapt_outputs(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(outputs) != 2:
            raise ValueError(
                f"{self.name}: expected 2 graph outputs (dets, labels), got {len(outputs)}"
            )
        dets = np.asarray(outputs[0], dtype=np.float32).reshape(-1, 4)
        logits = np.asarray(outputs[1], dtype=np.float32)

        # Per-class sigmoid, not softmax — see module docstring. The background
        # slot is always the last column; RF-DETR's export never reorders it.
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))
        probs = probs[:, :-1]

        # Flat top-k over every (query, class) pair: one query can yield more
        # than one class above threshold.
        flat = probs.ravel()
        k = min(_NUM_SELECT, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k] if k else np.empty(0, dtype=np.int64)
        idx = idx[np.argsort(-flat[idx])]
        scores = flat[idx]
        query_idx, class_ids = np.unravel_index(idx, probs.shape)

        boxes = _cxcywh_to_xyxy(dets[query_idx])
        return boxes, scores.astype(np.float32), class_ids.astype(np.int64)


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
