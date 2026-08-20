"""Building a GPU pipeline from a ``PipelineConfig`` and a bundle's artifact paths.

The counterpart to ``chachak.registry.build_pipeline`` plus
``chachak.detector.load_detector``, assembled from :class:`~gpu_infer.engine.AsyncEngine`
instead of the synchronous adapters.

Two small functions are **reimplemented rather than imported**, and the reason is worth
recording: ``chachak.detector`` imports ``chachak.infer``, which imports ``chachak._friendy``,
which appends the repo root to ``sys.path`` and imports the training package. In an exported
bundle that happens to work, because the generated ``friendy_shim.py`` re-exports the handful of
names involved — but relying on that couples this package to a template it does not own, and in a
source checkout it drags in the whole trainer. Both functions are a dozen lines and are pinned
against the originals in ``tests/test_pipelines_cpu.py``, which *may* import ``chachak`` freely
because tests are never vendored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def engine_max_input_hw(engine) -> Optional[Tuple[int, int]]:
    """``(max_h, max_w)`` from an engine's input profile, or ``None`` if unreadable.

    Reimplementation of ``chachak.detector._adapter_max_input_hw``. A TensorRT engine is built
    for a bounded input size, and a 4K frame handed to a 1024-max profile is not a soft failure —
    ``set_input_shape`` is rejected through the logger and the context runs at its previous shape.
    Reading the ceiling lets the caller downscale first.
    """
    try:
        shape = tuple(
            int(dim)
            for dim in engine.model.engine.get_tensor_profile_shape(
                engine.model.input_name, 0
            )[-1]
        )
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    if len(shape) != 4:
        return None
    max_h, max_w = shape[-2:]
    return (max_h, max_w) if max_h > 0 and max_w > 0 else None


def fit_to_input_profile(image, max_input_hw: Optional[Tuple[int, int]]):
    """Aspect-preserving downscale so ``image`` fits ``max_input_hw``. Never upscales.

    Reimplementation of ``chachak.detector._fit_to_input_profile``, operation for operation:
    ``scale = min(1.0, max_h / h, max_w / w)``, a bilinear ``F.interpolate`` with
    ``align_corners=False``, and ``max(1, round(...))`` on each axis. Runs on the device the frame
    is already on.
    """
    import torch.nn.functional as F

    if max_input_hw is None:
        return image
    height, width = int(image.shape[-2]), int(image.shape[-1])
    max_h, max_w = max_input_hw
    scale = min(1.0, max_h / height, max_w / width)
    if scale == 1.0:
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=(max(1, round(height * scale)), max(1, round(width * scale))),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def resolve_person_class_id(
    class_map: Dict,
    *,
    person_class_name: Optional[str] = "person",
    person_class_id: Optional[int] = None,
) -> int:
    """Which class id means "person". Mirrors ``chachak.detector.load_detector``.

    An explicit id wins; otherwise the name is looked up in the detector's own class map; failing
    that it falls back to 0 (COCO's person index) **with a warning**, because a silent fallback
    here means cropping around whatever class 0 happens to be.
    """
    if person_class_id is not None:
        return int(person_class_id)
    normalized = {str(name).lower(): int(cid) for cid, name in (class_map or {}).items()}
    key = str(person_class_name).lower() if person_class_name is not None else None
    if key is not None and key in normalized:
        return normalized[key]
    print(
        f"[gpu_infer] person class {person_class_name!r} not found in detector classes "
        f"{dict(class_map or {})}; defaulting to class id 0"
    )
    return 0


class LoadedEngine:
    """An :class:`AsyncEngine` plus the meta and derived facts the pipeline needs."""

    def __init__(self, engine, meta) -> None:
        self.engine = engine
        self.meta = meta
        self.max_input_hw = engine_max_input_hw(engine)
        self.class_map = {int(cid): str(name) for cid, name in (meta.class_map or {}).items()}

    @property
    def arch(self) -> str:
        return self.meta.arch

    @property
    def batchable(self) -> bool:
        """Whether more than one region can be submitted per execution.

        Read off the engine's own profile rather than the arch name: whether an engine was built
        with a batch profile wider than 1 is a property of that build (see
        ``trt_export.arch.BATCH_AWARE_ARCHS`` for which archs it can even be asked for), not of
        the string in its meta.
        """
        return (self.engine.max_batch or 1) > 1


def load_engine(engine_path, device="cuda", *, warmup: bool = True) -> LoadedEngine:
    """Load one ``.engine`` plus its ``.meta.json`` sidecar."""
    from onnx_infer.meta import ModelMeta

    from .engine import AsyncEngine

    engine_path = Path(engine_path)
    meta = ModelMeta.load(engine_path.with_suffix(".meta.json"))
    loaded = LoadedEngine(AsyncEngine(engine_path, meta, device=device, warmup=warmup), meta)
    print(
        f"[gpu_infer] {engine_path.name}: arch={meta.arch} classes={len(loaded.class_map)} "
        f"max_batch={loaded.engine.max_batch} efficientnms={loaded.engine.efficientnms}"
        + (
            f" input_profile_max={loaded.max_input_hw[0]}x{loaded.max_input_hw[1]}"
            if loaded.max_input_hw
            else ""
        )
    )
    return loaded


def refuse_non_engine(paths: Dict[str, Any]) -> None:
    """Fail loudly if any artifact is not a TensorRT engine.

    This package has no ONNX path by design — ``onnx_infer`` feeds numpy and cannot stay in
    device memory, which is the entire premise here. Better to say so than to quietly run the
    ordinary pipeline and report a speedup that is not happening.
    """
    for role, path in paths.items():
        if path is None:
            continue
        if Path(path).suffix.lower() != ".engine":
            raise SystemExit(
                f"[gpu_infer] {role} artifact is {Path(path).name}, not a .engine. This "
                f"runtime is TensorRT-only — onnxruntime feeds numpy, so it cannot keep a "
                f"frame in device memory. Use the bundle's ordinary infer.py for an ONNX "
                f"bundle, or re-export with --format engine."
            )


def build_gpu_pipeline(
    config,
    model_path,
    device="cuda",
    *,
    detector_path=None,
    extra_model_paths=(),
    options=None,
    eval_classes: Optional[Dict] = None,
):
    """Assemble the pipeline ``config.pipeline`` names. Mirrors ``chachak.registry``'s dispatch.

    Returns a :class:`~gpu_infer.pipeline.GpuPipeline`. ``eval_classes``, when given, is the
    class space every model's predictions get remapped onto — the bundle's ``classes`` block,
    which is what makes a multi-model bundle's outputs comparable.
    """
    from .config import GpuOptions
    from .pipeline import build as build_pipeline

    options = options or GpuOptions()
    refuse_non_engine({"model": model_path, "detector": detector_path})
    refuse_non_engine({f"extra_{i}": p for i, p in enumerate(extra_model_paths)})

    model = load_engine(model_path, device)
    detector = load_engine(detector_path, device) if detector_path else None
    extras = [load_engine(path, device) for path in extra_model_paths]
    return build_pipeline(
        config,
        model,
        detector=detector,
        extras=extras,
        options=options,
        eval_classes=eval_classes,
        device=device,
    )
