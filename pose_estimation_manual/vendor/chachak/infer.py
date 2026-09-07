"""Artifact loading and low-level adapter inference helpers.

These are the pieces shared by the trained-model path and the detector path:
loading an exported artifact into an adapter, calling ``adapter.predict`` in a
threshold-tolerant way, and running a long list of images through the adapter in
bounded chunks so the GPU never sees more than ``chunk_size`` images at once.

Unlike the original in ``chackalica_unified``, this copy loads **exported
artifacts only** (``.engine`` / ``.onnx``). The ``.pt`` branch is gone along with
its ``friendy_chachkalica.registry.build_model`` import: rebuilding a torch
architecture from a checkpoint would drag the whole training stack
(``transformers``, ``rfdetr``, ``ultralytics``, torchvision detection heads) into
a serving container that has no use for any of it. Exporting to an engine is a
step this project requires anyway, so nothing is lost.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


def as_class_map(classes: Union[Dict, List, None]) -> Dict[int, str]:
    """Coerce a classes list/mapping into ``{int_id: name}`` (empty if None)."""
    if classes is None:
        return {}
    if isinstance(classes, dict):
        return {int(key): str(value) for key, value in classes.items()}
    return {index: str(name) for index, name in enumerate(classes)}


def load_checkpoint_adapter(checkpoint_path: Union[str, Path], device) -> tuple:
    """Load an exported artifact into an adapter, ready for inference.

    Returns ``(adapter, info)`` where ``info`` carries ``model_name``,
    ``num_classes``, ``params`` and the training ``classes`` (as ``{id: name}``),
    read from the artifact's ``meta.json`` sidecar.

    Accepts a ``.engine`` (TensorRT, GPU) or a ``.onnx`` (onnxruntime, CPU-capable
    — the escape hatch when the host's TensorRT version can't deserialize an
    engine built elsewhere). A ``.pt`` checkpoint is rejected: see the module
    docstring.
    """
    checkpoint_path = Path(checkpoint_path)
    suffix = checkpoint_path.suffix.lower()

    # TensorRT engines are complete inference artifacts rather than torch
    # checkpoints. Their loader reads the matching ``<name>.meta.json``.
    if suffix == ".engine":
        print(f"[chachak] Using TensorRT engine: {checkpoint_path}")
        from ..trt_infer import load_trt_adapter

        return load_trt_adapter(checkpoint_path, device)

    # Architecture-free ONNX artifact (``best.onnx`` + ``best.meta.json``). Runs
    # via onnxruntime without rebuilding the model, so none of the training-arch
    # packages are needed.
    if suffix == ".onnx":
        print(f"[chachak] Using ONNX artifact: {checkpoint_path}")
        from ..onnx_infer import load_onnx_adapter

        return load_onnx_adapter(checkpoint_path, device)

    # An ONNX sitting next to a path given without a suffix (or with .pt) is
    # still usable — prefer it over failing.
    onnx_path = checkpoint_path.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"[chachak] Using ONNX artifact: {onnx_path}")
        from ..onnx_infer import load_onnx_adapter

        return load_onnx_adapter(onnx_path, device)

    raise ValueError(
        f"Unsupported model artifact: {checkpoint_path}. This runtime loads "
        "exported artifacts only — build a TensorRT '.engine' (preferred) or "
        "export a '.onnx', each with its '.meta.json' sidecar alongside. Torch "
        "'.pt' checkpoints need the training stack, which is deliberately not "
        "installed here."
    )


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


def infer_in_chunks(
    adapter: Any,
    images: List[torch.Tensor],
    chunk_size: int,
    score_threshold: Optional[float] = None,
) -> List[torch.Tensor]:
    """Run ``images`` through the adapter ``chunk_size`` at a time, in order."""
    results: List[torch.Tensor] = []
    for start in range(0, len(images), max(1, chunk_size)):
        chunk = images[start : start + chunk_size]
        chunk_results = predict_adapter(adapter, chunk, score_threshold)
        if len(chunk_results) != len(chunk):
            raise RuntimeError(
                "adapter prediction count mismatch: "
                f"images={len(chunk)} predictions={len(chunk_results)}"
            )
        results.extend(chunk_results)
    return results
