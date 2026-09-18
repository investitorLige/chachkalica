"""``ArchHandler`` — the one per-architecture seam on the service side.

The generic core (preprocess/postprocess) is arch-agnostic. The only thing that
genuinely varies between archs at inference time is how the raw onnxruntime
output list maps onto the canonical Contract-A 3-tensor
``(boxes_xyxy_input_px, scores, labels)``.

All archs (fasterrcnn/retinanet/yolox/rtdetr/rfdetr/ecdet/dfine) export through our own wrapper, which
emits exactly those three tensors in order, so every handler is currently a
:class:`PassthroughHandler` (identity). The :meth:`adapt_outputs` seam remains
for a future arch whose graph output layout we don't control (e.g. a native
third-party exporter) — it would subclass and override it.
"""

from __future__ import annotations

import numpy as np


class ArchHandler:
    """Base handler: subclass and set ``name``; override ``adapt_outputs`` if the
    graph's raw output layout isn't already ``(boxes, scores, labels)``."""

    name: str = ""

    def adapt_outputs(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def adapt_keypoints(self, outputs: list[np.ndarray]) -> np.ndarray | None:
        """Per-detection joints ``[N, K, 3]`` (``x, y, score``), or ``None``.

        Contract A carries boxes, scores and labels and has nowhere to put a
        keypoint array, so a pose arch's joints would be lost between the graph
        and the caller. This is the second, optional seam that carries them: the
        ``x, y`` are in the **same frame as** :meth:`adapt_outputs`' boxes (the
        graph's input pixels), row ``i`` describes the same detection as box
        ``i``, and the array is left out of the Friendy ``(N, 6)`` tensor
        entirely — only callers that ask for it (see
        ``onnx_infer.postprocess.keypoints_to_normalized``) ever see it, so
        every consumer that reshapes predictions to six columns is untouched.

        ``None`` — the default, and every arch but rtmo — means "this graph has
        no joints", which is different from an empty array (a pose graph that
        found nobody).
        """
        return None


class PassthroughHandler(ArchHandler):
    """Graph outputs are already ``(boxes[N,4], scores[N], labels[N])`` in order."""

    def adapt_outputs(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(outputs) < 3:
            raise ValueError(
                f"{self.name}: expected 3 graph outputs (boxes, scores, labels), "
                f"got {len(outputs)}"
            )
        boxes, scores, labels = outputs[0], outputs[1], outputs[2]
        return (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(scores, dtype=np.float32).reshape(-1),
            np.asarray(labels).reshape(-1).astype(np.int64),
        )
