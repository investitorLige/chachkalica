"""``TrtAdapter`` — the torch-facing drop-in for the trained-model adapters,
backed by a TensorRT engine.

Same surface as ``onnx_infer/adapter.py::OnnxAdapter`` (``predict`` / ``to`` /
``eval``) so it slots into ``load_checkpoint_adapter`` and everything downstream
with no caller changes, and it decodes the same two frozen contracts, because a
TensorRT engine is just a recompilation of the same ONNX graph.

Where it diverges from ``OnnxAdapter``: **nothing here touches host memory until
the final predictions**. Images arrive as CUDA tensors (person crops are GPU
slices of the frame), are pre-processed on the GPU by
:mod:`trt_infer.preprocess_torch`, handed to the engine by pointer, and decoded on
the GPU by :mod:`trt_infer.postprocess_torch`. ``OnnxAdapter`` cannot do this —
onnxruntime feeds are numpy — which is why the torch executors live in this
package rather than in ``onnx_infer``.

The ``[N,6]`` result is still returned on the **CPU**, matching ``OnnxAdapter`` and
the trained-model adapters. That is deliberate, not an oversight: the tensors are a
few KB, and everything downstream of here (``chachak.boxes.merge_predictions`` and
its hand-rolled class-aware NMS, ``chachak.detector``'s class filter) is a
Python-level loop over tiny tensors that runs faster on the CPU than it would
issuing per-iteration CUDA syncs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from onnx_infer.arch import get_handler
from onnx_infer.meta import ModelMeta
from onnx_infer.postprocess import to_friendy

from .session import TrtModel


class TrtAdapter:
    def __init__(self, engine_path: str | Path, meta: ModelMeta, device="cuda") -> None:
        self.meta = meta
        self.name = meta.arch
        self.num_classes = meta.num_classes
        self.score_threshold = meta.score_threshold
        self._handler = get_handler(meta.arch)
        self._model = TrtModel(engine_path, meta, device=device)

    def to(self, device) -> "TrtAdapter":
        self._model.to(device)
        return self

    def eval(self) -> "TrtAdapter":  # parity with nn.Module-style adapters
        return self

    def predict(self, images, score_threshold: Optional[float] = None):
        """List of CHW float [0,1] tensors -> list of Friendy ``(N,6)`` CPU tensors.

        Images whose preprocessed shape comes out identical (always true when
        ``meta.input.resize_mode`` is ``square``/``letterbox`` — a fixed output
        size regardless of the source image's own size, e.g. this repo's
        person-detector engine) are stacked and submitted to the engine as one
        real ``[B,3,H,W]`` batch instead of one engine call per image. Images that
        preprocess to different shapes (possible for
        ``resize_mode="none"``/``"longest_side"`` fed differently-sized frames)
        still get their own call each — grouping only ever reduces engine calls,
        never changes what each image sees.

        Everything up to the final ``.cpu()`` stays in device memory. The one
        exception is an arch whose handler overrides ``adapt_outputs``: none do
        today, and :func:`~trt_infer.postprocess_torch.adapt_outputs_torch`
        declines rather than guessing, so such an arch takes the numpy path below
        and is merely slow instead of wrong.
        """
        import torch

        from .postprocess_torch import adapt_outputs_torch, to_friendy_torch
        from .preprocess_torch import preprocess_torch

        threshold = self.score_threshold if score_threshold is None else score_threshold
        device = self._model.device
        prepared = [
            preprocess_torch(_as_model_input(image, device), self.meta) for image in images
        ]

        groups: dict = {}
        for index, (batched, _transform) in enumerate(prepared):
            groups.setdefault(tuple(batched.shape[1:]), []).append(index)

        raw_by_index = {}
        for indices in groups.values():
            stacked = torch.cat([prepared[i][0] for i in indices], dim=0)
            triples = self._model.run_torch(stacked)
            for local_i, global_i in enumerate(indices):
                raw_by_index[global_i] = triples[local_i]

        results = []
        for index, (_batched, transform) in enumerate(prepared):
            raw = raw_by_index[index]
            adapted = adapt_outputs_torch(self._handler, raw)
            if adapted is None:
                boxes, scores, labels = self._handler.adapt_outputs(
                    [tensor.detach().cpu().numpy() for tensor in raw]
                )
                friendy = torch.from_numpy(
                    to_friendy(
                        boxes, scores, labels, transform, threshold,
                        clip_boxes=self.meta.clip_boxes,
                        box_coords=self.meta.box_coords,
                    )
                )
            else:
                friendy = to_friendy_torch(
                    *adapted, transform, threshold,
                    clip_boxes=self.meta.clip_boxes,
                    box_coords=self.meta.box_coords,
                )
            results.append(friendy.detach().cpu())
        return results


def _as_model_input(image, device):
    """Coerce a CHW image (torch tensor or ndarray) to a float32 tensor on ``device``.

    A CUDA tensor — the common case, since ``chachak.boxes.crop_image`` slices crops
    straight out of the frame on the GPU — passes through without a copy.
    """
    import torch

    if torch.is_tensor(image):
        return image.to(device=device, dtype=torch.float32)
    # ascontiguousarray first: torch.as_tensor rejects negative strides, which a
    # channel-reversed or flipped numpy view can carry.
    return torch.as_tensor(np.ascontiguousarray(image, dtype=np.float32), device=device)


def load_trt_adapter(engine_path: str | Path, device="cuda") -> tuple[TrtAdapter, dict]:
    """Build a :class:`TrtAdapter` + an ``info`` dict shaped like
    ``chachak.infer.load_checkpoint_adapter``'s second return value.
    """
    engine_path = Path(engine_path)
    meta = ModelMeta.load(engine_path.with_suffix(".meta.json"))
    adapter = TrtAdapter(engine_path, meta, device=device)
    info = {
        "model_name": meta.arch,
        "num_classes": meta.num_classes,
        "params": {},
        "train_classes": dict(meta.class_map),
    }
    print(
        f"[trt_infer] Adapter ready: {meta.arch} num_classes={meta.num_classes} "
        f"classes={len(meta.class_map)} device={device}"
    )
    return adapter, info
