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
    identical to training/eval.
    """
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


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
