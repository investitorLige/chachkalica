"""``gpu_infer.geometry`` against the scalar originals it replaces.

Run from the repo root::

    python -m unittest gpu_infer.tests.test_geometry

Every assertion here is **exact** — ``assertEqual`` on ``.tolist()``, never ``allclose``. That
is the whole point of the module under test: the originals compute in python float (float64)
and feed integer pixel slices, so a tolerance would hide precisely the bug class these tests
exist to catch. A half-ULP difference in ``expand_box`` can move a crop boundary by a whole
pixel, and a pixel of crop shift changes what the model sees.

CPU-only and CUDA-free: the functions under test are device-agnostic torch, which is what makes
the bulk of this package testable on any machine.
"""

import random
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from chachak.boxes import (  # noqa: E402
    crop_image,
    expand_box,
    grow_box_to_min_size,
    _xywhn_to_xyxy_tensor,
)
from chachak._friendy import clip_xyxy, xyxy_to_xywh, xyxy_to_xywhn  # noqa: E402
from gpu_infer import geometry  # noqa: E402


def _boxes(count, frame_w, frame_h, *, seed):
    """Random xyxy boxes, some deliberately degenerate or off-frame."""
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        x1 = rng.uniform(-40.0, frame_w + 40.0)
        y1 = rng.uniform(-40.0, frame_h + 40.0)
        rows.append([x1, y1, x1 + rng.uniform(0.0, 300.0), y1 + rng.uniform(0.0, 300.0)])
    return torch.tensor(rows, dtype=torch.float32)


class ConversionParityTest(unittest.TestCase):
    """The ``formats``/``boxes`` conversions, batched, must be bit-identical."""

    FRAME = (1920, 1080)

    def setUp(self):
        self.boxes = _boxes(64, *self.FRAME, seed=11)

    def test_clip_matches_the_scalar_original(self):
        expected = clip_xyxy(self.boxes, *self.FRAME)
        actual = geometry.clip_xyxy_batched(self.boxes, *self.FRAME)
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_xyxy_to_xywh_matches(self):
        expected = xyxy_to_xywh(self.boxes)
        actual = geometry.xyxy_to_xywh_batched(self.boxes)
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_xyxy_to_xywhn_matches(self):
        """The tensor divisor is load-bearing; a scalar one diverges by 1-2 ULP on CUDA."""
        expected = xyxy_to_xywhn(self.boxes, *self.FRAME)
        actual = geometry.xyxy_to_xywhn_batched(self.boxes, *self.FRAME)
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_xywhn_to_xyxy_matches(self):
        normalized = xyxy_to_xywhn(self.boxes, *self.FRAME)
        expected = _xywhn_to_xyxy_tensor(normalized, *self.FRAME)
        actual = geometry.xywhn_to_xyxy_batched(normalized, *self.FRAME)
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_conversions_round_trip_through_a_leading_batch_axis(self):
        """The originals only accept ``[N, 4]``; these must handle ``[B, N, 4]`` too.

        The tail carries ``[crops, detections, 4]``, so a leading axis is the normal case,
        not an edge one.
        """
        stacked = torch.stack([self.boxes, self.boxes.flip(0)])
        clipped = geometry.clip_xyxy_batched(stacked, *self.FRAME)
        self.assertEqual(list(clipped.shape), [2, 64, 4])
        self.assertEqual(
            clipped[0].tolist(), clip_xyxy(self.boxes, *self.FRAME).tolist()
        )
        self.assertEqual(
            clipped[1].tolist(), clip_xyxy(self.boxes.flip(0), *self.FRAME).tolist()
        )

    def test_empty_input_keeps_its_shape(self):
        empty = torch.zeros((0, 4))
        for fn in (
            geometry.clip_xyxy_batched,
            geometry.xyxy_to_xywhn_batched,
            geometry.xywhn_to_xyxy_batched,
        ):
            self.assertEqual(list(fn(empty, *self.FRAME).shape), [0, 4], fn.__name__)
        self.assertEqual(list(geometry.xyxy_to_xywh_batched(empty).shape), [0, 4])


class PerRowBoundsTest(unittest.TestCase):
    """Bounds as a tensor — the thing the scalar originals cannot express."""

    def test_each_row_clips_against_its_own_frame(self):
        boxes = torch.tensor(
            [[10.0, 10.0, 500.0, 400.0], [10.0, 10.0, 500.0, 400.0]]
        )
        widths = torch.tensor([100.0, 480.0])
        heights = torch.tensor([80.0, 360.0])
        actual = geometry.clip_xyxy_batched(boxes, widths, heights)
        # Row 0 clipped against 100x80, row 1 against 480x360 — one call, two frames.
        self.assertEqual(
            actual.tolist(), [[10.0, 10.0, 100.0, 80.0], [10.0, 10.0, 480.0, 360.0]]
        )

    def test_per_row_bounds_agree_with_looping_the_original(self):
        extents = [(1920, 1080), (640, 480), (3840, 2160), (100, 100)]
        boxes = _boxes(len(extents), 3840, 2160, seed=5)
        widths = torch.tensor([float(w) for w, _ in extents])
        heights = torch.tensor([float(h) for _, h in extents])
        actual = geometry.xyxy_to_xywhn_batched(boxes, widths, heights)
        for index, (frame_w, frame_h) in enumerate(extents):
            expected = xyxy_to_xywhn(boxes[index : index + 1], frame_w, frame_h)
            self.assertEqual(
                actual[index].tolist(), expected[0].tolist(), f"row {index}"
            )


class ExpandAndGrowTest(unittest.TestCase):
    """``expand_box`` / ``grow_box_to_min_size`` are float64 in the original."""

    FRAME = (1920, 1080)

    def _loop_expand(self, boxes, ratio, frame):
        return [expand_box(row, ratio, *frame) for row in boxes]

    def test_expand_matches_across_ratios(self):
        boxes = _boxes(80, *self.FRAME, seed=17)
        for ratio in (0.0, 0.05, 0.25, 1.0, 2.5):
            expected = self._loop_expand(boxes, ratio, self.FRAME)
            actual = geometry.expand_boxes(boxes, ratio, *self.FRAME)
            self.assertEqual(actual.dtype, torch.float64, "must widen to float64")
            self.assertEqual(actual.tolist(), expected, f"ratio={ratio}")

    def test_expand_is_a_noop_at_ratio_zero_beyond_the_clip(self):
        """``ratio == 0`` skips the padding branch but still clips — same as the original."""
        boxes = torch.tensor([[-5.0, -5.0, 2000.0, 1200.0]])
        self.assertEqual(
            geometry.expand_boxes(boxes, 0.0, *self.FRAME).tolist(),
            [expand_box(boxes[0], 0.0, *self.FRAME)],
        )

    def test_grow_matches_across_floors(self):
        boxes = _boxes(80, *self.FRAME, seed=23)
        for min_size in (1.0, 32.0, 128.0, 640.0):
            expected = [
                grow_box_to_min_size(row, min_size, *self.FRAME) for row in boxes
            ]
            actual = geometry.grow_boxes_to_min_size(boxes, min_size, *self.FRAME)
            self.assertEqual(actual.tolist(), expected, f"min_size={min_size}")

    def test_grow_when_the_frame_is_smaller_than_the_floor(self):
        """Both sequential corrections fire here — the case a single masked update breaks.

        A 64x48 frame with a 200px floor cannot satisfy the floor on either axis, so every
        box gets re-centred, shifted off the near edge, then shifted back off the far one.
        """
        frame = (64, 48)
        boxes = _boxes(40, *frame, seed=29)
        expected = [grow_box_to_min_size(row, 200.0, *frame) for row in boxes]
        actual = geometry.grow_boxes_to_min_size(boxes, 200.0, *frame)
        self.assertEqual(actual.tolist(), expected)

    def test_grow_leaves_boxes_already_above_the_floor_untouched(self):
        boxes = torch.tensor([[100.0, 100.0, 500.0, 500.0]])
        actual = geometry.grow_boxes_to_min_size(boxes, 64.0, *self.FRAME)
        self.assertEqual(actual.tolist(), [[100.0, 100.0, 500.0, 500.0]])

    def test_expand_then_grow_composes_like_crop_regions_does(self):
        """The real call order in ``Pipeline.crop_regions``: expand, then grow."""
        boxes = _boxes(60, *self.FRAME, seed=31)
        expected = []
        for row in boxes:
            grown = grow_box_to_min_size(
                expand_box(row, 0.15, *self.FRAME), 96.0, *self.FRAME
            )
            expected.append(grown)
        actual = geometry.grow_boxes_to_min_size(
            geometry.expand_boxes(boxes, 0.15, *self.FRAME), 96.0, *self.FRAME
        )
        self.assertEqual(actual.tolist(), expected)


class CropBoundsTest(unittest.TestCase):
    """The four integers ``crop_image`` slices with."""

    FRAME = (1920, 1080)

    def _loop_bounds(self, boxes, frame):
        """``crop_image``'s bounds, obtained the only way it exposes them."""
        image = torch.zeros((3, frame[1], frame[0]))
        rows = []
        for box in boxes:
            _, (ix1, iy1), (crop_w, crop_h) = crop_image(image, box)
            rows.append([ix1, iy1, ix1 + crop_w, iy1 + crop_h])
        return rows

    def test_bounds_match_the_original(self):
        boxes = _boxes(120, *self.FRAME, seed=37)
        expected = self._loop_bounds(boxes, self.FRAME)
        actual = geometry.crop_bounds(boxes, *self.FRAME)
        self.assertEqual(actual.dtype, torch.int64)
        self.assertEqual(actual.tolist(), expected)

    def test_bounds_match_after_expand_and_grow(self):
        """The composed path, which is what the pipeline actually runs."""
        boxes = _boxes(120, *self.FRAME, seed=41)
        expanded = [
            grow_box_to_min_size(expand_box(row, 0.2, *self.FRAME), 64.0, *self.FRAME)
            for row in boxes
        ]
        expected = self._loop_bounds(
            torch.tensor(expanded, dtype=torch.float64), self.FRAME
        )
        actual = geometry.crop_bounds(
            geometry.grow_boxes_to_min_size(
                geometry.expand_boxes(boxes, 0.2, *self.FRAME), 64.0, *self.FRAME
            ),
            *self.FRAME,
        )
        self.assertEqual(actual.tolist(), expected)

    def test_half_way_coordinates_round_half_to_even_like_python(self):
        """``torch.round`` and python ``round()`` both go half-to-even. Pin it.

        If either ever changed to half-away-from-zero, crops would shift a pixel on exactly
        the coordinates most likely to appear in synthetic tests and least likely to be
        noticed in real ones.
        """
        boxes = torch.tensor(
            [[0.5, 1.5, 2.5, 3.5], [4.5, 5.5, 6.5, 7.5], [-0.5, 8.5, 10.5, 11.5]]
        )
        self.assertEqual(
            geometry.crop_bounds(boxes, *self.FRAME).tolist(),
            self._loop_bounds(boxes, self.FRAME),
        )

    def test_degenerate_boxes_still_yield_one_pixel(self):
        """``ix2`` is floored at ``ix1 + 1``, so a zero-area box is never zero-area."""
        boxes = torch.tensor([[10.0, 10.0, 10.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
        actual = geometry.crop_bounds(boxes, *self.FRAME)
        widths, heights = geometry.crop_extents(actual)
        self.assertEqual(widths.tolist(), [1, 1])
        self.assertEqual(heights.tolist(), [1, 1])
        self.assertEqual(actual.tolist(), self._loop_bounds(boxes, self.FRAME))

    def test_boxes_at_and_past_the_far_edge(self):
        frame_w, frame_h = self.FRAME
        boxes = torch.tensor(
            [
                [float(frame_w) - 0.4, float(frame_h) - 0.4, 5000.0, 5000.0],
                [float(frame_w), float(frame_h), float(frame_w), float(frame_h)],
                [-100.0, -100.0, -50.0, -50.0],
            ]
        )
        self.assertEqual(
            geometry.crop_bounds(boxes, *self.FRAME).tolist(),
            self._loop_bounds(boxes, self.FRAME),
        )

    def test_extents_stay_in_frame_pixels(self):
        boxes = _boxes(20, *self.FRAME, seed=43)
        bounds = geometry.crop_bounds(boxes, *self.FRAME)
        widths, heights = geometry.crop_extents(bounds)
        self.assertEqual(widths.tolist(), (bounds[:, 2] - bounds[:, 0]).tolist())
        self.assertEqual(heights.tolist(), (bounds[:, 3] - bounds[:, 1]).tolist())

    def test_empty_input(self):
        actual = geometry.crop_bounds(torch.zeros((0, 4)), *self.FRAME)
        self.assertEqual(list(actual.shape), [0, 4])
        self.assertEqual(actual.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
