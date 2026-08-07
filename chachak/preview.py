"""Synchronous single-image inference for the interactive model preview.

The Django admin's "Preview model on selected dataset" viewer flips through a
dataset one image at a time and needs the model's predictions on *that* image
right now — not a batch eval job. These helpers run one image through either a
chachak pipeline or the raw adapter and return plain box dicts (normalized
center-xywh, ready to draw on a browser canvas). Class names come from the
checkpoint's training class space, exactly like the batch path.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

try:
    from .infer import predict_adapter
except ImportError:  # run as a flat script
    from infer import predict_adapter


def _load_image_tensor(image_path: Union[str, Path], device) -> torch.Tensor:
    """Load an image into the CHW float[0,1] RGB tensor the adapters expect.

    Matches ``friendy_chachkalica/data.py`` so preview inference sees pixels
    identical to training/eval — but the permute/scale run *after* the transfer,
    so the copy carries uint8 instead of float32. A quarter of the bytes (25 MB
    rather than 100 MB for a 4K frame) and the per-pixel divide lands on the GPU
    instead of costing ~50 ms of host time.

    Bit-identical to doing it host-side, but only because the divisor is a tensor:
    uint8 -> float32 is exact (every uint8 is representable), and dividing by a
    tensor selects the correctly-rounded division kernel. Dividing by the plain
    literal ``255.0`` would not be — on CUDA torch lowers ``tensor / python_scalar``
    into a multiply by the reciprocal, and 1/255 is not representable in float32, so
    a handful of pixels come back one ULP off and detections can shift with them.
    ``trt_infer.postprocess_torch._divisors`` exists for the same reason.

    The returned tensor keeps the same non-contiguous CHW view of an HWC buffer the
    host-side version produced, since ``.float()`` preserves the input's strides.
    """
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    # .copy(): PIL exposes a read-only buffer, and torch.from_numpy needs a
    # writable one.
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).to(device)
    scale = torch.tensor(255.0, dtype=torch.float32, device=tensor.device)
    return tensor.permute(2, 0, 1).float() / scale


def _to_box_dicts(
    predictions: torch.Tensor, class_map: Optional[Dict[int, str]]
) -> List[Dict[str, Any]]:
    """Turn a Friendy ``(N, 6)`` tensor into JSON-friendly box dicts.

    Columns are ``[x_center, y_center, width, height, confidence, class_id]``,
    all normalized to the frame (see ``friendy_chachkalica/formats.py``).
    """
    class_map = class_map or {}
    boxes: List[Dict[str, Any]] = []
    for row in predictions.detach().cpu().tolist():
        cx, cy, w, h, conf, class_id = row[:6]
        class_id = int(class_id)
        boxes.append(
            {
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "confidence": conf,
                "class_id": class_id,
                "class_name": class_map.get(class_id, str(class_id)),
            }
        )
    return boxes


def predict_one(
    pipeline,
    info: Dict[str, Any],
    image_path: Union[str, Path],
    device,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run one image through a built chachak ``pipeline`` and return box dicts.

    ``process_batch`` ignores its ``targets`` argument for inference (every
    concrete pipeline uses only the images), so a single empty target is passed.
    """
    image = _load_image_tensor(image_path, device)
    predictions = pipeline.process_batch([image], [{}])[0]
    if score_threshold is not None and predictions.numel():
        predictions = predictions[predictions[:, 4] >= score_threshold]
    return _to_box_dicts(predictions, info.get("train_classes"))


def predict_one_raw(
    adapter,
    info: Dict[str, Any],
    image_path: Union[str, Path],
    device,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run one image directly through the raw ``adapter`` (no pipeline)."""
    image = _load_image_tensor(image_path, device)
    predictions = predict_adapter(adapter, [image], score_threshold)[0]
    return _to_box_dicts(predictions, info.get("train_classes"))
