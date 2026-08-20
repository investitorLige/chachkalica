"""Checkpoint loading and low-level adapter inference helpers.

These are the pieces shared by the trained-model path and the detector path:
loading a Friendy checkpoint into an adapter, calling ``adapter.predict`` in a
threshold-tolerant way, and running a long list of images through the adapter in
bounded chunks so the GPU never sees more than ``chunk_size`` images at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

try:
    from ._friendy import build_model
except ImportError:  # run as a flat script
    from _friendy import build_model


def as_class_map(classes: Union[Dict, List, None]) -> Dict[int, str]:
    """Coerce a classes list/mapping into ``{int_id: name}`` (empty if None)."""
    if classes is None:
        return {}
    if isinstance(classes, dict):
        return {int(key): str(value) for key, value in classes.items()}
    return {index: str(name) for index, name in enumerate(classes)}


def load_checkpoint_adapter(checkpoint_path: Union[str, Path], device) -> tuple:
    """Rebuild and load a Friendy checkpoint, ready for inference.

    Mirrors ``friendy_chachkalica/ml/eval_checkpoint.py``: rebuild the adapter from
    the checkpoint's ``model_name``/``num_classes``/``params`` and load the
    weights into ``adapter.model``. Returns ``(adapter, info)`` where ``info``
    carries ``model_name``, ``num_classes``, ``params`` and the training
    ``classes`` (as ``{id: name}``).
    """
    checkpoint_path = Path(checkpoint_path)

    # TensorRT engines are complete inference artifacts rather than torch
    # checkpoints. Their loader reads the matching ``<name>.meta.json``.
    if checkpoint_path.suffix.lower() == ".engine":
        print(f"[chachak] Using TensorRT engine: {checkpoint_path}")
        from trt_infer import load_trt_adapter

        return load_trt_adapter(checkpoint_path, device)

    # Prefer an architecture-free ONNX artifact exported next to the checkpoint
    # (``best.onnx`` + ``best.meta.json``). It runs via onnxruntime without
    # rebuilding the model, so none of the training-arch packages are needed.
    onnx_path = checkpoint_path.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"[chachak] Using ONNX artifact: {onnx_path}")
        from onnx_infer import load_onnx_adapter

        return load_onnx_adapter(onnx_path, device)

    print(f"[chachak] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model_name = state["model_name"]
    model_config = state.get("model_config", {}) or {}
    num_classes = model_config.get("num_classes")
    params = dict(model_config.get("params", {}) or {})

    adapter = build_model(model_name, num_classes=num_classes, **params)
    adapter.to(device)
    adapter.model.load_state_dict(state["model_state_dict"])
    adapter.eval()

    train_classes = as_class_map(state.get("train_dataset", {}).get("classes"))
    info = {
        "model_name": model_name,
        "num_classes": num_classes,
        "params": params,
        "train_classes": train_classes,
    }
    print(
        f"[chachak] Adapter ready: {model_name} num_classes={num_classes} "
        f"classes={len(train_classes)} device={device}"
    )
    return adapter, info


def predict_adapter(
    adapter: Any,
    images: List[torch.Tensor],
    score_threshold: Optional[float] = None,
) -> List[torch.Tensor]:
    """Call ``adapter.predict`` tolerating adapters without a threshold arg.

    Matches ``train.py::_predict_with_config``: some adapters (e.g. RetinaNet)
    define ``predict(self, images)`` with no ``score_threshold``.
    """
    if not images:
        return []
    if score_threshold is None:
        return adapter.predict(images)
    try:
        return adapter.predict(images, score_threshold=score_threshold)
    except TypeError:
        return adapter.predict(images)


def _adapter_max_batch(adapter: Any) -> Optional[int]:
    """Largest batch ``adapter`` can safely be handed in one ``predict`` call.

    Only ``TrtAdapter`` (``trt_infer.adapter.TrtAdapter``) caps this — it stacks
    same-shaped images into one real engine call, and the engine's own built
    profile is the ceiling: ``TrtModel.max_batch`` (read off that profile at load
    time) is how this finds out. That profile is 1 whenever the engine was built
    without asking for wider — the historical default for every arch, and still
    the only option for retinanet/fasterrcnn (see
    ``trt_export.arch.BATCH_AWARE_ARCHS``; their raw-export wrapper hardcodes a
    single image internally, so a wider profile would silently repeat one image's
    detections across the batch, not decode it). Every other adapter (onnx,
    trained torch) has no such ceiling — ``None`` means uncapped. Duck-typed
    rather than an isinstance check so this has no import-time dependency on
    ``trt_infer`` (torch/tensorrt), same reasoning as
    ``chachak.detector._adapter_max_input_hw``.
    """
    max_batch = getattr(getattr(adapter, "_model", None), "max_batch", None)
    return int(max_batch) if isinstance(max_batch, int) else None


def infer_in_chunks(
    adapter: Any,
    images: List[torch.Tensor],
    chunk_size: int,
    score_threshold: Optional[float] = None,
) -> List[torch.Tensor]:
    """Run ``images`` through the adapter ``chunk_size`` at a time, in order.

    ``chunk_size`` is a throughput knob, not a contract the adapter is assumed to
    tolerate — it is clamped to :func:`_adapter_max_batch` when the adapter has
    one, so a caller configured for a wider batch than the engine's own built
    profile supports (e.g. ``people_detect_first`` cropping multiple person boxes
    out of one frame, bound for a bundle built with the default batch-1 profile)
    degrades to one engine call per image instead of the adapter rejecting the
    oversized stack.
    """
    max_batch = _adapter_max_batch(adapter)
    effective_chunk_size = max(1, chunk_size if max_batch is None else min(chunk_size, max_batch))
    results: List[torch.Tensor] = []
    for start in range(0, len(images), effective_chunk_size):
        chunk = images[start : start + effective_chunk_size]
        chunk_results = predict_adapter(adapter, chunk, score_threshold)
        if len(chunk_results) != len(chunk):
            raise RuntimeError(
                "adapter prediction count mismatch: "
                f"images={len(chunk)} predictions={len(chunk_results)}"
            )
        results.extend(chunk_results)
    return results
