"""Re-map a full-frame detection target into a sub-window's local coordinates.

Both train-time transforms crop a full frame into smaller windows — tiling
(``tiling.tile_batch``) into a grid of tiles, person-cropping
(``cropping.crop_batch``) into detector-derived person crops — and then need the
ground-truth boxes expressed in each window's local frame. That box bookkeeping
is identical for both (clip to the window, drop under-visible slivers, translate
to window-local coords, rebuild ``area``/``orig_size``/``iscrowd``), so it lives
here once. Only the window *source* differs, which is the caller's concern.

A window is an axis-aligned box ``[x0, y0, x0 + win_w, y0 + win_h]`` in the
original frame's pixel coordinates.
"""

from typing import Any, Dict, Optional

import torch


def empty_window_target(
    target: Dict[str, Any],
    win_w: int,
    win_h: int,
) -> Dict[str, Any]:
    """Copy target metadata and replace annotations with a valid empty target.

    Used for genuine background windows so the detector learns from background
    instead of seeing positive windows only.
    """
    new_target = dict(target)
    boxes = target.get("boxes")
    labels = target.get("labels")
    new_target["boxes"] = (
        boxes.new_zeros((0, 4)) if boxes is not None
        else torch.zeros((0, 4), dtype=torch.float32)
    )
    new_target["labels"] = (
        labels.new_zeros((0,), dtype=labels.dtype) if labels is not None
        else torch.zeros((0,), dtype=torch.int64)
    )
    new_target["area"] = new_target["boxes"].new_zeros((0,))
    if "iscrowd" in target and target["iscrowd"] is not None:
        new_target["iscrowd"] = target["iscrowd"].new_zeros((0,))
    orig_size = target.get("orig_size")
    if orig_size is not None:
        new_target["orig_size"] = torch.tensor(
            [win_h, win_w], dtype=orig_size.dtype, device=orig_size.device
        )
    return new_target


def remap_target_to_window(
    target: Dict[str, Any],
    x0: int,
    y0: int,
    win_w: int,
    win_h: int,
    min_visible_fraction: float,
) -> Optional[Dict[str, Any]]:
    """Re-map a full-frame target's boxes into one window's local coordinates.

    Returns a new target dict whose ``boxes`` (absolute xyxy) and ``labels`` are
    restricted and clipped to the window ``[x0, y0, x0 + win_w, y0 + win_h]``, or
    ``None`` when the window contains only a below-threshold object sliver (it
    would be wrong to train that partial object as background, so the window is
    dropped). A genuinely empty window returns an empty target so the detector
    learns from background instead of seeing positive windows only.

    A box is kept only if at least ``min_visible_fraction`` of its original area
    falls inside the window — dropping slivers clipped at a window edge that
    carry too little of the object to be a useful positive.
    """
    boxes = target.get("boxes")
    labels = target.get("labels")
    if boxes is None or boxes.numel() == 0:
        return empty_window_target(target, win_w, win_h)

    boxes = boxes.float()
    x1 = torch.clamp(boxes[:, 0], min=x0, max=x0 + win_w)
    y1 = torch.clamp(boxes[:, 1], min=y0, max=y0 + win_h)
    x2 = torch.clamp(boxes[:, 2], min=x0, max=x0 + win_w)
    y2 = torch.clamp(boxes[:, 3], min=y0, max=y0 + win_h)

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    orig_w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    orig_h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    orig_area = (orig_w * orig_h).clamp(min=1e-6)

    keep = (inter_area / orig_area) >= min_visible_fraction
    if not bool(keep.any()):
        # Ignore a window containing only a clipped object sliver: treating it as
        # background would explicitly train against a real partial object. True
        # background windows remain useful hard negatives and get an empty target.
        if bool((inter_area > 0).any()):
            return None
        return empty_window_target(target, win_w, win_h)

    # Translate the clipped boxes into window-local coords.
    local = torch.stack(
        [x1 - x0, y1 - y0, x2 - x0, y2 - y0], dim=1
    )[keep]

    new_target = dict(target)
    new_target["boxes"] = local
    if labels is not None:
        new_target["labels"] = labels[keep]
    new_target["area"] = (local[:, 2] - local[:, 0]) * (local[:, 3] - local[:, 1])
    orig_size = target.get("orig_size")
    if orig_size is not None:
        new_target["orig_size"] = torch.tensor(
            [win_h, win_w], dtype=orig_size.dtype, device=orig_size.device
        )
    if "iscrowd" in target and target["iscrowd"] is not None:
        new_target["iscrowd"] = target["iscrowd"][keep]
    return new_target
