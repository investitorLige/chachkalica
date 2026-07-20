"""Unit tests for the pure geometry in boxes.py (no model, no GPU).

Run from the chachak directory:  python -m unittest tests.test_boxes
"""

import sys
import unittest
from pathlib import Path

# Import boxes as a flat module (and let it reach sibling friendy) without
# triggering the full chachak package __init__.
_CHACHAK_DIR = Path(__file__).resolve().parent.parent
if str(_CHACHAK_DIR) not in sys.path:
    sys.path.insert(0, str(_CHACHAK_DIR))

import torch  # noqa: E402

import boxes  # noqa: E402


class TileFrameTest(unittest.TestCase):
    def test_fixed_pixel_tiles_shift_last_window_flush_to_edge(self):
        image = torch.zeros((3, 100, 100))
        tiles = boxes.tile_frame_pixels(image, 40, overlap=0.0)

        starts = sorted({offset for _, offset, _ in tiles})
        self.assertIn((60, 60), starts)
        self.assertTrue(all(tuple(tile.shape) == (3, 40, 40) for tile, _, _ in tiles))
        self.assertTrue(all(size == (40, 40) for _, _, size in tiles))

    def test_fixed_pixel_tile_pads_only_when_frame_is_smaller(self):
        image = torch.ones((3, 20, 30))
        [(tile, offset, size)] = boxes.tile_frame_pixels(image, 40, overlap=0.2)

        self.assertEqual(offset, (0, 0))
        self.assertEqual(size, (40, 40))
        self.assertTrue(torch.equal(tile[:, :20, :30], image))
        self.assertEqual(float(tile[:, 20:, :].sum()), 0.0)
        self.assertEqual(float(tile[:, :, 30:].sum()), 0.0)

    def test_tiles_cover_whole_frame_including_edges(self):
        image = torch.zeros((3, 100, 100))
        # 40% of 100 -> 40px tiles.
        tiles = boxes.tile_frame(image, 0.4, 0.4, overlap=0.25)

        covered = torch.zeros((100, 100), dtype=torch.bool)
        for _, (x, y), (w, h) in tiles:
            self.assertLessEqual(x + w, 100)
            self.assertLessEqual(y + h, 100)
            covered[y : y + h, x : x + w] = True
        self.assertTrue(bool(covered.all()), "tiles must cover every pixel")

    def test_edge_tiles_are_clamped_partial_not_flush(self):
        image = torch.zeros((3, 100, 100))
        # 40px tiles, stride 40 -> starts [0, 40, 80]; the 80 tile runs partial (w=20).
        tiles = boxes.tile_frame(image, 0.4, 0.4, overlap=0.0)
        self.assertEqual(len(tiles), 9)
        widths = sorted({w for _, _, (w, _) in tiles})
        self.assertEqual(widths, [20, 40])

    def test_tile_full_size_is_single_full_tile(self):
        image = torch.zeros((3, 50, 80))
        tiles = boxes.tile_frame(image, 1.0, 1.0, overlap=0.2)
        self.assertEqual(len(tiles), 1)
        _, (x, y), (w, h) = tiles[0]
        self.assertEqual((x, y, w, h), (0, 0, 80, 50))

    def test_rejects_out_of_range_overlap(self):
        image = torch.zeros((3, 100, 100))
        with self.assertRaises(ValueError):
            boxes.tile_frame(image, 0.4, 0.4, overlap=1.0)

    def test_rejects_out_of_range_fraction(self):
        image = torch.zeros((3, 100, 100))
        with self.assertRaises(ValueError):
            boxes.tile_frame(image, 0.0, 0.4, overlap=0.2)
        with self.assertRaises(ValueError):
            boxes.tile_frame(image, 0.4, 1.5, overlap=0.2)


class RemapTest(unittest.TestCase):
    def test_prediction_fully_inside_padding_is_dropped(self):
        preds = torch.tensor([[0.9, 0.9, 0.1, 0.1, 0.9, 0.0]])
        remapped = boxes.remap_local_preds_to_frame(
            preds, (0, 0), 64, 64, 40, 40
        )
        self.assertEqual(tuple(remapped.shape), (0, 6))

    def test_center_box_round_trips_to_full_frame(self):
        frame_w, frame_h = 640, 480
        offset = (100, 50)
        local_w, local_h = 200, 100
        # A box centered in the local region covering half its extent.
        preds = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.9, 1.0]])

        remapped = boxes.remap_local_preds_to_frame(
            preds, offset, local_w, local_h, frame_w, frame_h
        )

        # Local xyxy [50,25,150,75] + offset (100,50) -> frame xyxy [150,75,250,125]
        # center (200,100), size (100,50).
        expected = torch.tensor(
            [[200 / 640, 100 / 480, 100 / 640, 50 / 480, 0.9, 1.0]]
        )
        self.assertTrue(torch.allclose(remapped, expected, atol=1e-6))

    def test_empty_predictions_stay_empty(self):
        remapped = boxes.remap_local_preds_to_frame(
            torch.zeros((0, 6)), (0, 0), 10, 10, 100, 100
        )
        self.assertEqual(tuple(remapped.shape), (0, 6))


class MergeTest(unittest.TestCase):
    def test_duplicate_boxes_collapse_to_one(self):
        box = [0.5, 0.5, 0.2, 0.2, 0.9, 0.0]
        a = torch.tensor([box])
        b = torch.tensor([[*box[:4], 0.8, 0.0]])  # same box, lower score
        merged = boxes.merge_predictions([a, b], 100, 100, nms_iou=0.5)
        self.assertEqual(merged.shape[0], 1)
        self.assertAlmostEqual(float(merged[0, 4]), 0.9)  # higher score kept

    def test_different_classes_are_not_merged(self):
        a = torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.9, 0.0]])
        b = torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.9, 1.0]])
        merged = boxes.merge_predictions([a, b], 100, 100, nms_iou=0.5)
        self.assertEqual(merged.shape[0], 2)

    def test_contained_box_collapses_even_when_iou_is_low(self):
        large = torch.tensor([[0.5, 0.5, 0.6, 0.6, 0.7, 0.0]])
        small = torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.95, 0.0]])

        merged = boxes.merge_predictions([large, small], 100, 100, nms_iou=0.5)

        self.assertEqual(merged.shape[0], 1)
        self.assertAlmostEqual(float(merged[0, 4]), 0.95)
        self.assertTrue(torch.allclose(merged[0, :4], small[0, :4]))

    def test_min_box_size_filters_tiny_boxes(self):
        big = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.9, 0.0]])   # 50x50 px
        tiny = torch.tensor([[0.1, 0.1, 0.01, 0.01, 0.9, 0.0]])  # 1x1 px
        merged = boxes.merge_predictions(
            [big, tiny], 100, 100, nms_iou=0.5, min_box_size=5.0
        )
        self.assertEqual(merged.shape[0], 1)

    def test_all_empty_returns_empty(self):
        merged = boxes.merge_predictions([torch.zeros((0, 6))], 100, 100, nms_iou=0.5)
        self.assertEqual(tuple(merged.shape), (0, 6))


class CropAndExpandTest(unittest.TestCase):
    def test_crop_returns_offset_and_size(self):
        image = torch.arange(3 * 100 * 100, dtype=torch.float32).reshape(3, 100, 100)
        crop, offset, size = boxes.crop_image(image, [10.4, 20.6, 60.5, 80.2])
        self.assertEqual(offset, (10, 21))
        self.assertEqual(size, (crop.shape[2], crop.shape[1]))
        self.assertEqual(size, (50, 59))

    def test_expand_box_pads_and_clips_to_frame(self):
        # 100-wide box expanded by 0.5 -> +25 each side, clipped at 0 and frame_w.
        expanded = boxes.expand_box([10, 10, 110, 110], ratio=0.5, frame_w=120, frame_h=200)
        self.assertEqual(expanded[0], 0.0)         # 10 - 25 -> clipped to 0
        self.assertEqual(expanded[2], 120.0)       # 110 + 25 = 135 -> clipped to 120
        self.assertEqual(expanded[3], 135.0)       # 110 + 25, within height


if __name__ == "__main__":
    unittest.main()
