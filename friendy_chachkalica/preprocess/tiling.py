"""Tile a training batch for tiling pipelines (batch_detect).

When an experiment attaches a tiling pipeline, the model is trained on the same
tiled representation it is validated/tested on: each full-frame image is split
into overlapping tiles and its ground-truth boxes are re-mapped into each tile's
local coordinates, so every tile becomes an independent training sample with its
own targets. Loss is then computed per tile inside the adapter's ``training_step``
(the standard, differentiable tiled-training approach).

Image geometry reuses ``chachak.boxes.tile_frame``; re-tiling the *targets* lives
here because inference never needs it (the eval pipeline predicts per tile and
merges detections back to the frame — it doesn't split labels).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from .window_targets import remap_target_to_window
except ImportError:  # run as a flat script
    from window_targets import remap_target_to_window

# chachak default tiling knobs (mirror chachak.config.TilingConfig) used when a
# knob is left blank on the experiment.
_DEFAULT_TILE_WIDTH_PCT = 50.0
_DEFAULT_TILE_HEIGHT_PCT = 50.0
_DEFAULT_OVERLAP = 0.2

# A re-tiled box is kept only if at least this fraction of its original area
# falls inside the tile — drops slivers clipped at a tile seam that carry too
# little of the object to be a useful positive.
_MIN_VISIBLE_FRACTION = 0.1


def _ensure_chachak_importable() -> None:
    import sys

    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_tiling(tiling: Any) -> Tuple[Optional[int], float, float, float]:
    """Return (fixed_px, width_frac, height_frac, overlap), with defaults."""
    tile_size_px = getattr(tiling, "tile_size_px", None)
    width_pct = getattr(tiling, "tile_width_pct", None) or _DEFAULT_TILE_WIDTH_PCT
    height_pct = getattr(tiling, "tile_height_pct", None) or _DEFAULT_TILE_HEIGHT_PCT
    overlap = getattr(tiling, "overlap", None)
    if overlap is None:
        overlap = _DEFAULT_OVERLAP
    return tile_size_px, width_pct / 100.0, height_pct / 100.0, float(overlap)


def _retile_target(
    target: Dict[str, Any],
    x0: int,
    y0: int,
    tile_w: int,
    tile_h: int,
) -> Optional[Dict[str, Any]]:
    """Re-map a full-frame target's boxes into one tile's local coordinates.

    Thin wrapper over :func:`window_targets.remap_target_to_window` with the
    tiling visibility floor; see that function for the exact semantics (clip to
    the tile, drop under-visible slivers, empty tiles become background).
    """
    return remap_target_to_window(
        target, x0, y0, tile_w, tile_h, _MIN_VISIBLE_FRACTION
    )


def tile_batch(
    images: List[torch.Tensor],
    targets: List[Dict[str, Any]],
    tiling: Any,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """Expand a batch of full frames into per-tile ``(image, target)`` samples.

    Each frame is split by :func:`chachak.boxes.tile_frame`; its targets are
    re-mapped per tile by :func:`_retile_target`. True background tiles are
    retained; tiles containing only tiny clipped object slivers are ignored.
    Returns flat lists suitable for feeding to an adapter's
    ``training_step`` (optionally re-chunked into micro-batches by the caller).
    """
    _ensure_chachak_importable()
    try:
        from chachak.boxes import tile_frame, tile_frame_pixels
    except ImportError:
        from boxes import tile_frame, tile_frame_pixels  # pragma: no cover

    tile_size_px, width_frac, height_frac, overlap = _resolve_tiling(tiling)

    tile_images: List[torch.Tensor] = []
    tile_targets: List[Dict[str, Any]] = []
    for image, target in zip(images, targets):
        tiles = (
            tile_frame_pixels(image, tile_size_px, overlap)
            if tile_size_px is not None
            else tile_frame(image, width_frac, height_frac, overlap)
        )
        for tile, (x0, y0), (tile_w, tile_h) in tiles:
            new_target = _retile_target(target, x0, y0, tile_w, tile_h)
            if new_target is None:
                continue
            tile_images.append(tile)
            tile_targets.append(new_target)
    return tile_images, tile_targets
