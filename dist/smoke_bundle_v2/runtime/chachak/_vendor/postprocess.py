from typing import Optional

import torch


def apply_class_aware_nms(
    prediction: torch.Tensor,
    iou_threshold: Optional[float],
) -> torch.Tensor:
    """Apply class-aware NMS to a Friendy-format prediction tensor."""
    if iou_threshold is None:
        return prediction
    if prediction is None or prediction.numel() == 0:
        return prediction

    threshold = float(iou_threshold)
    if threshold <= 0 or threshold > 1:
        raise ValueError("nms_threshold must be in (0, 1] or null")

    prediction = prediction.reshape(-1, 6)
    boxes = _xywhn_to_xyxy(prediction[:, :4])
    scores = prediction[:, 4]
    labels = prediction[:, 5].long()
    keep = _batched_nms(boxes, scores, labels, threshold)
    return prediction[keep]


def class_aware_nms_keep(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Keep-indices of class-aware NMS over xyxy boxes.

    Used by metrics to suppress duplicates at the operating point without going
    through the Friendy (N, 6) tensor format. Indices are returned sorted
    ascending so indexing with them preserves the caller's ordering.
    """
    keep = _batched_nms(boxes, scores, labels, float(iou_threshold))
    return torch.sort(keep).values


def _xywhn_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    x_center, y_center, width, height = boxes.unbind(dim=1)
    return torch.stack(
        [
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2,
        ],
        dim=1,
    )


def _batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    keep_parts = []
    for label in torch.unique(labels):
        class_mask = labels == label
        class_indices = class_mask.nonzero(as_tuple=False).flatten()
        class_keep = _nms(boxes[class_mask], scores[class_mask], iou_threshold)
        keep_parts.append(class_indices[class_keep])

    if not keep_parts:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    keep = torch.cat(keep_parts)
    return keep[torch.argsort(scores[keep], descending=True, stable=True)]


def _nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True, stable=True)
    keep = []
    while int(order.numel()) > 0:
        current = order[0]
        keep.append(current)
        if int(order.numel()) == 1:
            break

        remaining = order[1:]
        ious = _box_iou_one_to_many(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]

    return torch.stack(keep).to(dtype=torch.long)


def _box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_one = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_many = (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )
    union = area_one + area_many - inter
    return inter / union.clamp(min=torch.finfo(boxes.dtype).eps)
