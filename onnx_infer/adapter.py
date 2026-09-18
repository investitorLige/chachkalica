"""``OnnxAdapter`` — the torch-facing drop-in for the trained-model adapters.

Exposes the same surface ``chachak`` already uses (``predict`` / ``to`` /
``eval``) so it slots into ``load_checkpoint_adapter`` and downstream
(``detector.py``, ``pipeline.py``, ``preview.py``) with no caller changes.

torch is imported lazily and only here, at the boundary: the numpy core stays
torch-free, while ``predict`` returns ``torch.Tensor`` so consumers that call
``.numel()`` / index the ``(N, 6)`` result keep working unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .arch import get_handler
from .meta import ModelMeta
from .postprocess import keypoints_to_normalized, to_friendy
from .preprocess import preprocess
from .session import OnnxModel


class OnnxAdapter:
    def __init__(self, onnx_path: str | Path, meta: ModelMeta, device="cpu") -> None:
        self.meta = meta
        self.name = meta.arch
        self.num_classes = meta.num_classes
        self.score_threshold = meta.score_threshold
        self._handler = get_handler(meta.arch)
        self._model = OnnxModel(onnx_path, meta, device=device)

    def to(self, device) -> "OnnxAdapter":
        self._model.to(device)
        return self

    def eval(self) -> "OnnxAdapter":  # parity with nn.Module-style adapters
        return self

    def predict(self, images, score_threshold: Optional[float] = None,
                with_keypoints: bool = False):
        """List of CHW float [0,1] tensors -> list of Friendy ``(N,6)`` tensors.

        ``with_keypoints`` returns ``(friendy, keypoints)`` pairs instead, the
        second being a ``[M,K,3]`` normalized array row-matched to the first (or
        ``None`` for an arch with no joints) — see ``TrtAdapter.predict``, whose
        contract this mirrors so the two formats of one bundle behave alike.
        """
        import torch

        threshold = self.score_threshold if score_threshold is None else score_threshold
        results = []
        for image in images:
            chw = _to_chw_numpy(image)
            batched, transform = preprocess(chw, self.meta)
            raw = self._model.run(batched)
            boxes, scores, labels = self._handler.adapt_outputs(raw)
            friendy = torch.from_numpy(to_friendy(
                boxes, scores, labels, transform, threshold,
                clip_boxes=self.meta.clip_boxes,
                box_coords=self.meta.box_coords,
            ))
            if not with_keypoints:
                results.append(friendy)
                continue
            joints = self._handler.adapt_keypoints(raw)
            if joints is not None:
                joints = keypoints_to_normalized(
                    joints, scores, transform, threshold,
                    box_coords=self.meta.box_coords,
                )
            results.append((friendy, joints))
        return results


def _to_chw_numpy(image) -> np.ndarray:
    """Coerce a CHW image (torch tensor or ndarray) to a float32 numpy array."""
    if hasattr(image, "detach"):  # torch.Tensor
        image = image.detach().cpu().numpy()
    return np.asarray(image, dtype=np.float32)


def load_onnx_adapter(onnx_path: str | Path, device="cpu") -> tuple[OnnxAdapter, dict]:
    """Build an :class:`OnnxAdapter` + an ``info`` dict shaped like
    ``chachak.infer.load_checkpoint_adapter``'s second return value.
    """
    onnx_path = Path(onnx_path)
    meta = ModelMeta.load(onnx_path.with_suffix(".meta.json"))
    adapter = OnnxAdapter(onnx_path, meta, device=device)
    info = {
        "model_name": meta.arch,
        "num_classes": meta.num_classes,
        "params": {},
        "train_classes": dict(meta.class_map),
    }
    print(
        f"[onnx_infer] Adapter ready: {meta.arch} num_classes={meta.num_classes} "
        f"classes={len(meta.class_map)} device={device}"
    )
    return adapter, info
