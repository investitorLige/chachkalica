from typing import Sequence

import torch


FRIENDY_PREDICTION_COLUMNS = (
    "x_center",
    "y_center",
    "width",
    "height",
    "confidence",
    "class_id",
)


def xywhn_to_xyxy(
    box: Sequence[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Convert one normalized YOLO xywh box to absolute xyxy."""
    x_center, y_center, width, height = box
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return [x1, y1, x2, y2]


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert absolute xyxy boxes to absolute xywh boxes."""
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    return torch.stack([x_center, y_center, width, height], dim=1)


def xyxy_to_xywhn(
    boxes: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Convert absolute xyxy boxes to normalized xywh boxes."""
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    xywh = xyxy_to_xywh(boxes)
    normalizer = boxes.new_tensor(
        [image_width, image_height, image_width, image_height]
    )
    return xywh / normalizer


def clip_xyxy(
    boxes: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Clip absolute xyxy boxes to image bounds."""
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_height)
    return boxes


def xyxy_prediction_to_friendy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Build Friendy prediction tensors from absolute xyxy model outputs."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, len(FRIENDY_PREDICTION_COLUMNS)))

    xywhn = xyxy_to_xywhn(
        boxes,
        image_width=image_width,
        image_height=image_height,
    )
    return torch.cat(
        [
            xywhn,
            scores.reshape(-1, 1).to(dtype=boxes.dtype),
            labels.reshape(-1, 1).to(dtype=boxes.dtype),
        ],
        dim=1,
    )
