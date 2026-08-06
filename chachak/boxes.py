"""Geometry for tiling, cropping, and remapping detections into full frames.

A *tile* and a *crop* are the same thing: a sub-region of the frame at an
``(x1, y1)`` pixel offset with its own local width/height. A detector run on that
sub-region returns boxes normalized to the sub-region, and
:func:`remap_local_preds_to_frame` lifts them back into full-frame normalized
coordinates. That single remap serves the tiling pipeline, the person-crop
pipeline, and the step that lifts detector person boxes from tile space to frame
space, so all coordinate math lives here.

Predictions are Friendy ``(N, 6)`` tensors: ``[x_center, y_center, width,
height, confidence, class_id]`` with the box normalized to its own image.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from ._friendy import clip_xyxy, xyxy_to_xywhn
except ImportError:  # run as a flat script
    from _friendy import clip_xyxy, xyxy_to_xywhn


def _xywhn_to_xyxy_tensor(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert an ``(N, 4)`` normalized center-xywh tensor to absolute xyxy."""
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    cx, cy, w, h = boxes.unbind(dim=1)
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return torch.stack([x1, y1, x2, y2], dim=1)


def xywhn_preds_to_xyxy(preds: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Absolute xyxy pixel boxes ``(M, 4)`` from Friendy predictions ``(N, 6)``."""
    if preds.numel() == 0:
        return preds.new_zeros((0, 4))
    return _xywhn_to_xyxy_tensor(preds[:, :4], width, height)


def _tile_starts(total: int, size: int, stride: int) -> List[int]:
    """Sliding-window start offsets covering ``[0, total)``.

    Unlike a flush-to-border scheme, the last window is allowed to run partially
    off the far edge — we stop as soon as a tile reaches ``total`` and leave the
    caller to clamp that final tile to whatever pixels remain.
    """
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + size >= total:
            break
        start += stride
    return starts


def _fixed_tile_starts(total: int, size: int, stride: int) -> List[int]:
    """Starts for full-size windows, with the final one shifted flush to the edge."""
    if total <= size:
        return [0]
    starts = list(range(0, total - size + 1, stride))
    final = total - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def tile_frame_pixels(
    image: torch.Tensor,
    tile_size_px: int,
    overlap: float,
) -> List[Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]]:
    """Split a CHW frame into fixed square source-pixel tiles.

    Windows remain ``tile_size_px`` square and the last window is shifted flush
    to the far edge whenever the frame is large enough. A frame dimension smaller
    than the requested size is zero-padded on the right/bottom; pixels are never
    resized. Returned local sizes describe the padded square canvas so prediction
    remapping and normalized training targets use the same coordinate frame.
    """
    tile_size_px = int(tile_size_px)
    if tile_size_px <= 0:
        raise ValueError(f"tile_size_px must be greater than 0, got {tile_size_px}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    channels, height, width = image.shape
    stride = max(1, int(round(tile_size_px * (1.0 - overlap))))
    tiles = []
    for y in _fixed_tile_starts(height, tile_size_px, stride):
        for x in _fixed_tile_starts(width, tile_size_px, stride):
            content_w = min(tile_size_px, width - x)
            content_h = min(tile_size_px, height - y)
            crop = image[:, y : y + content_h, x : x + content_w]
            if content_w == tile_size_px and content_h == tile_size_px:
                tile = crop
            else:
                tile = image.new_zeros((channels, tile_size_px, tile_size_px))
                tile[:, :content_h, :content_w] = crop
            tiles.append((tile, (x, y), (tile_size_px, tile_size_px)))
    return tiles


def tile_frame(
    image: torch.Tensor,
    tile_w_frac: float,
    tile_h_frac: float,
    overlap: float,
) -> List[Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]]:
    """Split a CHW frame into overlapping tiles sized as a fraction of the frame.

    ``tile_w_frac`` / ``tile_h_frac`` are fractions in ``(0, 1]`` of the frame's
    width / height, so tiling is scale-invariant across differently-sized images.
    Returns a list of ``(tile_chw, (x1, y1), (tile_w, tile_h))``; tiles are views
    into ``image``. ``overlap`` is a fraction in ``[0, 1)``; the stride is
    ``round(tile * (1 - overlap))``. Tiles that would overflow the right/bottom
    edge are clamped to the pixels that remain, so the last column/row is a
    partial tile rather than a full one shifted flush to the border.
    """
    if not 0.0 < tile_w_frac <= 1.0 or not 0.0 < tile_h_frac <= 1.0:
        raise ValueError(
            f"tile fractions must be in (0, 1], got ({tile_w_frac}, {tile_h_frac})"
        )
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    _, height, width = image.shape
    tile_w = min(width, max(1, int(round(width * tile_w_frac))))
    tile_h = min(height, max(1, int(round(height * tile_h_frac))))
    stride_x = max(1, int(round(tile_w * (1.0 - overlap))))
    stride_y = max(1, int(round(tile_h * (1.0 - overlap))))

    tiles = []
    for y in _tile_starts(height, tile_h, stride_y):
        for x in _tile_starts(width, tile_w, stride_x):
            w = min(tile_w, width - x)
            h = min(tile_h, height - y)
            tile = image[:, y : y + h, x : x + w]
            tiles.append((tile, (x, y), (w, h)))
    return tiles


def expand_box(
    box_xyxy: Sequence[float],
    ratio: float,
    frame_w: int,
    frame_h: int,
) -> List[float]:
    """Pad an xyxy box outward by ``ratio`` (fraction of side), clipped to frame."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    if ratio > 0:
        dw = (x2 - x1) * ratio / 2.0
        dh = (y2 - y1) * ratio / 2.0
        x1, y1, x2, y2 = x1 - dw, y1 - dh, x2 + dw, y2 + dh
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(frame_w), x2)
    y2 = min(float(frame_h), y2)
    return [x1, y1, x2, y2]


def crop_image(
    image: torch.Tensor,
    box_xyxy: Sequence[float],
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """Crop a CHW frame to an xyxy box.

    Returns ``(crop_chw, (x1, y1), (crop_w, crop_h))`` with integer pixel bounds
    rounded and clamped inside the frame, mirroring :func:`tile_frame`'s shape.
    """
    _, height, width = image.shape
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    ix1 = int(max(0, min(round(x1), width - 1)))
    iy1 = int(max(0, min(round(y1), height - 1)))
    ix2 = int(max(ix1 + 1, min(round(x2), width)))
    iy2 = int(max(iy1 + 1, min(round(y2), height)))
    crop = image[:, iy1:iy2, ix1:ix2]
    return crop, (ix1, iy1), (ix2 - ix1, iy2 - iy1)


def grow_box_to_min_size(
    box_xyxy: Sequence[float],
    min_size: float,
    frame_w: int,
    frame_h: int,
) -> List[float]:
    """Grow an xyxy box outward (same center) so neither side is below ``min_size``.

    Prefers real frame pixels over padding: a short side is widened using
    neighboring image content, shifting back inside ``[0, frame_w/h]`` if the
    centered window would run off an edge. Only a frame itself narrower or
    shorter than ``min_size`` leaves a side short — the caller then zero-pads
    the crop (mirroring :func:`tile_frame_pixels`) to make up the difference.
    """
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)

    def _grow(lo, hi, total):
        if hi - lo >= min_size:
            return lo, hi
        center = (lo + hi) / 2.0
        half = min(min_size, float(total)) / 2.0
        lo, hi = center - half, center + half
        if lo < 0:
            hi -= lo
            lo = 0.0
        if hi > total:
            lo -= hi - total
            hi = float(total)
        return max(0.0, lo), hi

    x1, x2 = _grow(x1, x2, frame_w)
    y1, y2 = _grow(y1, y2, frame_h)
    return [x1, y1, x2, y2]


def upscale_crop_to_min_size(crop: torch.Tensor, min_size: int) -> torch.Tensor:
    """Bilinearly upscale a CHW crop until neither side is under ``min_size``.

    Only engages when the source frame itself is narrower/shorter than
    ``min_size`` (:func:`grow_box_to_min_size` already used all available real
    pixels). Interpolates rather than zero-pads: every output pixel stays real
    (if blurred) content instead of introducing a blank region that a
    downstream model's own resize/letterbox would then stretch in as if it
    were real detail — and it keeps this floor's behavior independent of
    whatever padding convention a given pretrained model happens to use.

    Both axes are scaled by one factor, so the subject keeps its proportions;
    reaching the floor per-axis instead would squash a 10x20 crop into a
    square. Overshooting on the long side is free — ``min_size`` is a floor,
    not a target.

    Returns the resized tensor only: the crop still covers the same region of
    the source frame, and callers must keep remapping its predictions and
    targets with that original window size (normalized box coordinates are
    invariant under this resize, absolute ones are not).
    """
    _, height, width = crop.shape
    if height >= min_size and width >= min_size:
        return crop
    factor = max(min_size / height, min_size / width)
    target_h = max(min_size, int(round(height * factor)))
    target_w = max(min_size, int(round(width * factor)))
    return F.interpolate(
        crop.unsqueeze(0).float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def remap_local_preds_to_frame(
    preds: torch.Tensor,
    offset_xy: Tuple[int, int],
    local_w: int,
    local_h: int,
    frame_w: int,
    frame_h: int,
) -> torch.Tensor:
    """Lift local-region predictions ``(N, 6)`` into full-frame coordinates.

    ``preds`` boxes are normalized to a region of size ``local_w x local_h`` at
    pixel offset ``offset_xy``. Returns ``(N, 6)`` normalized to the full frame,
    carrying confidence and class_id unchanged.
    """
    if preds.numel() == 0:
        return preds.reshape(-1, 6)
    boxes = _xywhn_to_xyxy_tensor(preds[:, :4], local_w, local_h)
    offset = preds.new_tensor(
        [offset_xy[0], offset_xy[1], offset_xy[0], offset_xy[1]]
    )
    boxes = boxes + offset
    boxes = clip_xyxy(boxes, frame_w, frame_h)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    preds = preds[valid]
    if boxes.numel() == 0:
        return preds.reshape(-1, 6)
    xywhn = xyxy_to_xywhn(boxes, frame_w, frame_h)
    return torch.cat([xywhn, preds[:, 4:6]], dim=1)


def _class_aware_overlap_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """NMS that also catches near-containment duplicates from overlapping tiles."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)

    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break

        rest = order[1:]
        same_class = classes[rest] == classes[current]

        xx1 = torch.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = torch.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = torch.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = torch.minimum(boxes[current, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)

        union = areas[current] + areas[rest] - inter
        iou = inter / union.clamp(min=torch.finfo(boxes.dtype).eps)
        smaller_area = torch.minimum(areas[current], areas[rest])
        contained_overlap = inter / smaller_area.clamp(
            min=torch.finfo(boxes.dtype).eps
        )

        duplicate = same_class & (
            (iou >= threshold) | (contained_overlap >= threshold)
        )
        order = rest[~duplicate]

    return torch.stack(keep).to(dtype=torch.long)


def merge_predictions(
    preds_list: Sequence[torch.Tensor],
    frame_w: int,
    frame_h: int,
    nms_iou: float,
) -> torch.Tensor:
    """Concatenate per-region predictions and de-duplicate with per-class NMS.

    All inputs are full-frame-normalized ``(N, 6)`` tensors (as produced by
    :func:`remap_local_preds_to_frame`). Returns a single ``(M, 6)`` tensor.

    Deliberately has no size floor. It used to take a ``min_box_size`` that
    dropped boxes below a pixel threshold, and the person-crop pipeline passed
    ``detector.min_box_size`` into it — which silently discarded nearly every
    real detection, since that floor sizes *person crops going in* (hundreds of
    px) while these are the model's own item-level boxes coming out (a helmet is
    tens of px). The two are different scales and must never share a threshold,
    so the parameter is gone rather than merely unused: see
    ``pipeline.Pipeline._crop_infer_remap``.
    """
    preds = [p for p in preds_list if p is not None and p.numel() > 0]
    if not preds:
        # Take the device from whatever we were handed — the inputs can all be empty
        # and still be CUDA tensors. Returning a bare CPU tensor here would hand a
        # caller working on the GPU a silently device-mismatched result.
        device = next((p.device for p in preds_list if p is not None), None)
        return torch.zeros((0, 6)) if device is None else torch.zeros((0, 6), device=device)
    preds = torch.cat(preds, dim=0)

    boxes = _xywhn_to_xyxy_tensor(preds[:, :4], frame_w, frame_h)
    scores = preds[:, 4]
    classes = preds[:, 5].to(torch.int64)
    keep = _class_aware_overlap_nms(boxes, scores, classes, nms_iou)
    return preds[keep]
