"""Device-side tile extraction for the ``batch_detect`` and ``batch_people`` pipelines.

Twin of ``chachak.boxes.tile_frame`` / ``tile_frame_pixels``, split into a **host-side plan**
and a **device-side extraction**, the same division ``onnx_infer.preprocess.plan_preprocess``
makes for resize geometry.

The plan is pure integer arithmetic over the frame's dimensions — no device data is involved,
so computing it on the host costs nothing, keeps it exact, and lets it be cached per distinct
frame size. Only :func:`extract_tiles` touches pixels, and every tile it produces is a view or
a pad of a frame that is already in device memory.

Why the two modes batch differently
-----------------------------------

``tile_frame_pixels`` produces tiles that are **all** ``tile_size_px`` square: the reference
itself zero-pads any tile short of a full square (``boxes.py:100-104``) and reports the padded
square as the local size. So its whole grid is one uniform stack and submits as a single engine
batch.

``tile_frame`` does **not** pad. Its last row and column are *clamped* to whatever pixels
remain (``boxes.py:141-143``) and it reports that clamped extent as the local size. Those tiles
therefore have different shapes, and they must keep them: preprocessing letterboxes a tile to
the model canvas, so a clamped ``(w, h)`` tile and a zero-padded ``(tile_w, tile_h)`` one
resolve to **different scale factors and different pixels**. Padding the ragged edge to make one
tidy batch would silently change what the model sees at every frame border.

So this module groups tiles by shape and leaves batching to the caller, exactly as
``trt_infer.adapter.TrtAdapter.predict`` already does with its own ``groups`` dict
(``adapter.py:85-87``). For a typical grid that still means one large interior batch plus a
couple of small edge batches — most of the benefit, none of the divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple


def tile_starts(total: int, size: int, stride: int) -> List[int]:
    """Sliding-window starts covering ``[0, total)``. Twin of ``boxes._tile_starts``.

    The last window is allowed to run partially off the far edge; the caller clamps it. This
    is *not* the flush-to-border scheme — see :func:`fixed_tile_starts` for that one.
    """
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + size >= total:
            break
        start += stride
    return starts


def fixed_tile_starts(total: int, size: int, stride: int) -> List[int]:
    """Starts for full-size windows, final one flush to the edge. Twin of
    ``boxes._fixed_tile_starts``."""
    if total <= size:
        return [0]
    starts = list(range(0, total - size + 1, stride))
    final = total - size
    if starts[-1] != final:
        starts.append(final)
    return starts


@dataclass(frozen=True)
class TileGrid:
    """Where every tile of one frame sits, and how big it is.

    ``offsets`` and ``local_sizes`` are parallel to each other and to whatever
    :func:`extract_tiles` returns, in the same row-major order the scalar originals emit, so a
    caller can zip them against per-tile predictions without re-deriving anything.

    ``local_sizes`` is the extent predictions are remapped by. For ``padded`` grids that is the
    padded square, not the content — matching ``tile_frame_pixels``, which reports
    ``(tile_size_px, tile_size_px)`` even for a tile whose real content is smaller. Getting this
    backwards would remap every edge tile's detections against the wrong denominator.
    """

    offsets: Tuple[Tuple[int, int], ...]
    local_sizes: Tuple[Tuple[int, int], ...]
    #: Content extent actually copied out of the frame. Equals ``local_sizes`` except on a
    #: padded grid's edge tiles, where the content is smaller than the canvas.
    content_sizes: Tuple[Tuple[int, int], ...]
    #: True for ``tile_frame_pixels`` (zero-pad short tiles to a square canvas), False for
    #: ``tile_frame`` (clamp and keep the ragged shape).
    padded: bool

    def __len__(self) -> int:
        return len(self.offsets)

    def groups_by_shape(self) -> Dict[Tuple[int, int], List[int]]:
        """Tile indices bucketed by ``(height, width)`` of the tensor they produce.

        Same-shaped tiles can be stacked into one real engine batch. On a padded grid this is
        a single bucket; on a clamped grid it is the interior plus one bucket per edge shape.
        """
        groups: Dict[Tuple[int, int], List[int]] = {}
        for index, (width, height) in enumerate(self.local_sizes if self.padded else self.content_sizes):
            groups.setdefault((height, width), []).append(index)
        return groups


@lru_cache(maxsize=64)
def plan_tiles_pixels(
    frame_h: int, frame_w: int, tile_size_px: int, overlap: float
) -> TileGrid:
    """Grid for fixed square source-pixel tiles. Twin of ``boxes.tile_frame_pixels``.

    Cached: a video or an image directory hits a handful of distinct frame sizes, and the grid
    for each is the same every frame.
    """
    tile_size_px = int(tile_size_px)
    if tile_size_px <= 0:
        raise ValueError(f"tile_size_px must be greater than 0, got {tile_size_px}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    stride = max(1, int(round(tile_size_px * (1.0 - overlap))))
    offsets, local_sizes, content_sizes = [], [], []
    for y in fixed_tile_starts(frame_h, tile_size_px, stride):
        for x in fixed_tile_starts(frame_w, tile_size_px, stride):
            offsets.append((x, y))
            local_sizes.append((tile_size_px, tile_size_px))
            content_sizes.append(
                (min(tile_size_px, frame_w - x), min(tile_size_px, frame_h - y))
            )
    return TileGrid(
        offsets=tuple(offsets),
        local_sizes=tuple(local_sizes),
        content_sizes=tuple(content_sizes),
        padded=True,
    )


@lru_cache(maxsize=64)
def plan_tiles_fractional(
    frame_h: int, frame_w: int, tile_w_frac: float, tile_h_frac: float, overlap: float
) -> TileGrid:
    """Grid for tiles sized as a fraction of the frame. Twin of ``boxes.tile_frame``.

    Tiles at the right and bottom edges are clamped, not padded, and their clamped extent is
    the local size — see the module docstring on why that difference is load-bearing.
    """
    if not 0.0 < tile_w_frac <= 1.0 or not 0.0 < tile_h_frac <= 1.0:
        raise ValueError(
            f"tile fractions must be in (0, 1], got ({tile_w_frac}, {tile_h_frac})"
        )
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    tile_w = min(frame_w, max(1, int(round(frame_w * tile_w_frac))))
    tile_h = min(frame_h, max(1, int(round(frame_h * tile_h_frac))))
    stride_x = max(1, int(round(tile_w * (1.0 - overlap))))
    stride_y = max(1, int(round(tile_h * (1.0 - overlap))))

    offsets, local_sizes = [], []
    for y in tile_starts(frame_h, tile_h, stride_y):
        for x in tile_starts(frame_w, tile_w, stride_x):
            width = min(tile_w, frame_w - x)
            height = min(tile_h, frame_h - y)
            offsets.append((x, y))
            local_sizes.append((width, height))
    return TileGrid(
        offsets=tuple(offsets),
        local_sizes=tuple(local_sizes),
        content_sizes=tuple(local_sizes),
        padded=False,
    )


def plan_tiles(frame_h: int, frame_w: int, tiling) -> TileGrid:
    """Dispatch on a ``chachak.config.TilingConfig`` the way ``_tile_infer`` does.

    ``tile_size_px`` wins when set, otherwise the percentage pair — mirroring
    ``chachak.pipeline._tile_infer:74-84`` so a config cannot mean one thing here and another
    there.
    """
    if tiling.tile_size_px is not None:
        return plan_tiles_pixels(frame_h, frame_w, tiling.tile_size_px, tiling.overlap)
    return plan_tiles_fractional(
        frame_h,
        frame_w,
        tiling.tile_width_pct / 100.0,
        tiling.tile_height_pct / 100.0,
        tiling.overlap,
    )


def extract_tiles(image, grid: TileGrid) -> List["object"]:
    """Cut ``grid``'s tiles out of a CHW frame, on whatever device the frame is on.

    A full-size tile is a **view** — no pixels are copied, exactly as the scalar originals
    return views. A short tile on a padded grid gets a fresh zero canvas with the content
    written into its top-left corner, matching ``boxes.py:103-104``; ``new_zeros`` keeps the
    frame's dtype and device, so the pad value is a real zero in the frame's own scale rather
    than whatever a cross-dtype cast would produce.
    """
    channels = int(image.shape[0])
    tiles = []
    for (x, y), (canvas_w, canvas_h), (content_w, content_h) in zip(
        grid.offsets, grid.local_sizes, grid.content_sizes
    ):
        crop = image[:, y : y + content_h, x : x + content_w]
        if not grid.padded or (content_w == canvas_w and content_h == canvas_h):
            tiles.append(crop)
            continue
        tile = image.new_zeros((channels, canvas_h, canvas_w))
        tile[:, :content_h, :content_w] = crop
        tiles.append(tile)
    return tiles


def offsets_tensor(grid: TileGrid, like) -> "object":
    """``[T, 2]`` int64 tile offsets on ``like``'s device, for the batched remap.

    The tail lifts every tile's detections into frame coordinates in one shot, which needs the
    offsets as a tensor rather than the tuple-of-tuples the grid carries for host-side zipping.
    """
    import torch

    return torch.tensor(grid.offsets, dtype=torch.int64, device=like.device).reshape(-1, 2)


def local_sizes_tensor(grid: TileGrid, like) -> "object":
    """``[T, 2]`` int64 ``(width, height)`` local extents on ``like``'s device."""
    import torch

    return torch.tensor(
        grid.local_sizes, dtype=torch.int64, device=like.device
    ).reshape(-1, 2)


def expected_tile_count(frame_h: int, frame_w: int, tiling) -> int:
    """How many tiles ``tiling`` yields for this frame, without building them.

    Used to size the padded tail buffers before any pixels move.
    """
    return len(plan_tiles(frame_h, frame_w, tiling))


def describe(grid: TileGrid) -> str:
    """One line for the operator log: grid size, shape buckets, padding mode."""
    buckets = grid.groups_by_shape()
    shapes = ", ".join(
        f"{h}x{w}x{len(idx)}" for (h, w), idx in sorted(buckets.items())
    )
    mode = "padded" if grid.padded else "clamped"
    return f"{len(grid)} tiles ({mode}), {len(buckets)} shape group(s): {shapes}"
