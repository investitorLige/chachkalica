"""People detector: a Friendy checkpoint filtered to the person class.

The detector is just another trained detector checkpoint (any registry
architecture). :class:`Detector` wraps it to return, per input image, only the
person detections above a score threshold — still in Friendy ``(N, 6)`` form
normalized to each input image (a full frame or a tile).
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .infer import infer_in_chunks, load_checkpoint_adapter


class Detector:
    """Wraps a detector adapter and keeps only person boxes above threshold."""

    def __init__(
        self,
        adapter,
        person_class_id: int,
        score_threshold: float = 0.5,
        batch_size: int = 4,
    ) -> None:
        self.adapter = adapter
        self.person_class_id = int(person_class_id)
        self.score_threshold = float(score_threshold)
        self.batch_size = int(batch_size)
        self.max_input_hw = _adapter_max_input_hw(adapter)
        if self.max_input_hw is not None:
            print(
                "[detector] TensorRT input profile max="
                f"{self.max_input_hw[0]}x{self.max_input_hw[1]}; "
                "oversized frames will be aspect-preserving resized before "
                "person detection"
            )

    def predict(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """Return per-image person predictions ``(N, 6)`` (normalized per input)."""
        raw = infer_in_chunks(
            self.adapter,
            [_fit_to_input_profile(image, self.max_input_hw) for image in images],
            self.batch_size, self.score_threshold
        )
        filtered = []
        for preds in raw:
            if preds.numel() == 0:
                filtered.append(preds.reshape(-1, 6))
                continue
            mask = (preds[:, 5].to(torch.int64) == self.person_class_id) & (
                preds[:, 4] >= self.score_threshold
            )
            filtered.append(preds[mask])
        return filtered



def _adapter_max_input_hw(adapter) -> Optional[Tuple[int, int]]:
    model = getattr(adapter, "_model", None)
    engine, input_name = getattr(model, "engine", None), getattr(model, "input_name", None)
    if engine is None or input_name is None:
        return None
    try:
        max_shape = tuple(int(value) for value in engine.get_tensor_profile_shape(input_name, 0)[-1])
    except (AttributeError, TypeError, ValueError):
        return None
    max_h, max_w = max_shape[-2:]
    return (max_h, max_w) if len(max_shape) == 4 and max_h > 0 and max_w > 0 else None


def _fit_to_input_profile(image: torch.Tensor, max_input_hw: Optional[Tuple[int, int]]) -> torch.Tensor:
    if max_input_hw is None:
        return image
    height, width = (int(image.shape[-2]), int(image.shape[-1]))
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

def load_detector(
    checkpoint_path: Union[str, Path],
    device,
    *,
    person_class_name: Optional[str] = "person",
    person_class_id: Optional[int] = None,
    score_threshold: float = 0.5,
    batch_size: int = 4,
) -> Detector:
    """Load a detector checkpoint and resolve which class id is 'person'.

    An explicit ``person_class_id`` wins. Otherwise the id is looked up by name
    in the checkpoint's training classes; if the name is absent, it falls back to
    id 0 (COCO's person index) with a warning.
    """
    adapter, info = load_checkpoint_adapter(checkpoint_path, device)

    resolved_id = person_class_id
    if resolved_id is None:
        classes = info.get("train_classes") or {}
        name_to_id = {str(name).lower(): int(cid) for cid, name in classes.items()}
        key = str(person_class_name).lower() if person_class_name is not None else None
        if key is not None and key in name_to_id:
            resolved_id = name_to_id[key]
        else:
            resolved_id = 0
            print(
                f"[detector] person class '{person_class_name}' not found in "
                f"checkpoint classes {classes}; defaulting to class id 0"
            )

    print(
        f"[detector] Ready: person_class_id={resolved_id} "
        f"score_threshold={score_threshold}"
    )
    return Detector(
        adapter,
        person_class_id=resolved_id,
        score_threshold=score_threshold,
        batch_size=batch_size,
    )
