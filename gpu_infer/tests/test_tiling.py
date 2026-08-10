"""``gpu_infer.tiling`` against ``chachak.boxes``'s tile grids.

    python -m unittest gpu_infer.tests.test_tiling

Exact assertions throughout: the tile pixels must be *equal*, not close, and the reported local
sizes must match tuple for tuple. The local size is the denominator every tile's detections are
remapped by, so an off-by-one there mislocates every box on that tile.

The load-bearing case is :class:`ClampedGridIsNotPaddedTest` — it pins the difference between
the two modes that makes shape-grouping necessary instead of pad-to-uniform batching.
"""

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from chachak.boxes import tile_frame, tile_frame_pixels  # noqa: E402
from chachak.config import TilingConfig  # noqa: E402
from gpu_infer import tiling  # noqa: E402


def _frame(height, width):
    """A frame whose every pixel is unique, so a misplaced tile cannot pass by luck."""
    count = 3 * height * width
    return torch.arange(count, dtype=torch.float32).reshape(3, height, width)


class StartsParityTest(unittest.TestCase):
    """The two start schemes are genuinely different; keep them that way."""

    CASES = [
        (1080, 540, 432), (1080, 540, 540), (100, 100, 100), (100, 640, 512),
        (1920, 960, 768), (37, 16, 13), (1, 1, 1), (4096, 640, 512),
    ]

    def test_sliding_starts_match(self):
        from chachak.boxes import _tile_starts

        for total, size, stride in self.CASES:
            self.assertEqual(
                tiling.tile_starts(total, size, stride),
                _tile_starts(total, size, stride),
                f"{total}/{size}/{stride}",
            )

    def test_fixed_starts_match(self):
        from chachak.boxes import _fixed_tile_starts

        for total, size, stride in self.CASES:
            self.assertEqual(
                tiling.fixed_tile_starts(total, size, stride),
                _fixed_tile_starts(total, size, stride),
                f"{total}/{size}/{stride}",
            )

    def test_the_two_schemes_disagree_where_they_should(self):
        """Sliding runs off the far edge; fixed shifts the last window flush.

        A negative test, so a future 'simplification' that collapses them fails here rather
        than silently changing every edge tile in one of the two modes.
        """
        self.assertEqual(tiling.tile_starts(100, 64, 51), [0, 51])
        self.assertEqual(tiling.fixed_tile_starts(100, 64, 51), [0, 36])


class PixelGridParityTest(unittest.TestCase):
    """``tile_frame_pixels`` — uniform square tiles, zero-padded at the edges."""

    CASES = [
        (1080, 1920, 640, 0.2), (1080, 1920, 512, 0.0), (480, 640, 640, 0.2),
        (100, 100, 256, 0.5), (720, 1280, 384, 0.25), (2160, 3840, 960, 0.1),
    ]

    def test_grid_and_pixels_match(self):
        for frame_h, frame_w, size, overlap in self.CASES:
            with self.subTest(frame=(frame_h, frame_w), size=size, overlap=overlap):
                image = _frame(frame_h, frame_w)
                expected = tile_frame_pixels(image, size, overlap)
                grid = tiling.plan_tiles_pixels(frame_h, frame_w, size, overlap)
                tiles = tiling.extract_tiles(image, grid)

                self.assertEqual(len(tiles), len(expected))
                self.assertEqual(list(grid.offsets), [off for _, off, _ in expected])
                self.assertEqual(
                    list(grid.local_sizes), [local for _, _, local in expected]
                )
                for index, (want_tile, _, _) in enumerate(expected):
                    self.assertTrue(
                        torch.equal(tiles[index], want_tile), f"tile {index}"
                    )

    def test_every_tile_is_the_same_shape_so_the_grid_is_one_batch(self):
        """The reference pads these itself, which is what makes full batching safe here."""
        grid = tiling.plan_tiles_pixels(1080, 1920, 640, 0.2)
        self.assertEqual(len(grid.groups_by_shape()), 1)
        self.assertEqual(list(grid.groups_by_shape()), [(640, 640)])

    def test_short_edge_tiles_are_zero_padded_not_clamped(self):
        image = _frame(100, 100)
        grid = tiling.plan_tiles_pixels(100, 100, 256, 0.0)
        tile = tiling.extract_tiles(image, grid)[0]
        self.assertEqual(list(tile.shape), [3, 256, 256])
        self.assertTrue(torch.equal(tile[:, :100, :100], image))
        self.assertEqual(float(tile[:, 100:, :].abs().sum()), 0.0)

    def test_rejects_the_same_arguments_the_original_rejects(self):
        for size, overlap in ((0, 0.2), (-1, 0.2), (640, 1.0), (640, -0.1)):
            with self.assertRaises(ValueError):
                tiling.plan_tiles_pixels(1080, 1920, size, overlap)


class FractionalGridParityTest(unittest.TestCase):
    """``tile_frame`` — fractional tiles, clamped at the edges."""

    CASES = [
        (1080, 1920, 0.5, 0.5, 0.2), (1080, 1920, 0.5, 0.5, 0.0),
        (1080, 1920, 1.0, 1.0, 0.0), (720, 1280, 0.33, 0.4, 0.15),
        (100, 100, 0.5, 0.5, 0.5), (481, 641, 0.5, 0.5, 0.2),
    ]

    def test_grid_and_pixels_match(self):
        for frame_h, frame_w, w_frac, h_frac, overlap in self.CASES:
            with self.subTest(frame=(frame_h, frame_w), fracs=(w_frac, h_frac)):
                image = _frame(frame_h, frame_w)
                expected = tile_frame(image, w_frac, h_frac, overlap)
                grid = tiling.plan_tiles_fractional(
                    frame_h, frame_w, w_frac, h_frac, overlap
                )
                tiles = tiling.extract_tiles(image, grid)

                self.assertEqual(len(tiles), len(expected))
                self.assertEqual(list(grid.offsets), [off for _, off, _ in expected])
                self.assertEqual(
                    list(grid.local_sizes), [local for _, _, local in expected]
                )
                for index, (want_tile, _, _) in enumerate(expected):
                    self.assertTrue(
                        torch.equal(tiles[index], want_tile), f"tile {index}"
                    )

    def test_rejects_the_same_arguments_the_original_rejects(self):
        for w_frac, h_frac, overlap in (
            (0.0, 0.5, 0.2), (1.5, 0.5, 0.2), (0.5, 0.0, 0.2), (0.5, 0.5, 1.0),
        ):
            with self.assertRaises(ValueError):
                tiling.plan_tiles_fractional(1080, 1920, w_frac, h_frac, overlap)


class ClampedGridIsNotPaddedTest(unittest.TestCase):
    """Why edge tiles are grouped by shape instead of padded into one batch.

    ``tile_frame`` clamps its ragged edge and reports the clamped extent as the local size.
    Zero-padding those tiles to a uniform canvas would hand the model a different image —
    preprocessing letterboxes to the model canvas, so a clamped ``(w, h)`` and a padded
    ``(tile_w, tile_h)`` resolve to different scale factors. These tests pin that, so nobody
    'tidies' the ragged batch later.
    """

    # 481x641 with 0.5 fractions and 0.2 overlap leaves genuinely ragged edges.
    GRID = (481, 641, 0.5, 0.5, 0.2)

    def test_the_grid_really_has_more_than_one_shape(self):
        grid = tiling.plan_tiles_fractional(*self.GRID)
        self.assertGreater(
            len(grid.groups_by_shape()), 1, "test frame chosen to be ragged"
        )

    def test_edge_tiles_keep_their_clamped_shape(self):
        image = _frame(self.GRID[0], self.GRID[1])
        grid = tiling.plan_tiles_fractional(*self.GRID)
        tiles = tiling.extract_tiles(image, grid)
        interior = grid.local_sizes[0]
        ragged = [
            index for index, size in enumerate(grid.local_sizes) if size != interior
        ]
        self.assertTrue(ragged, "expected at least one clamped tile")
        for index in ragged:
            width, height = grid.local_sizes[index]
            self.assertEqual(list(tiles[index].shape), [3, height, width])

    def test_content_and_local_sizes_agree_on_a_clamped_grid(self):
        """Nothing is padded, so the two must be identical here — unlike a padded grid."""
        grid = tiling.plan_tiles_fractional(*self.GRID)
        self.assertEqual(grid.content_sizes, grid.local_sizes)
        self.assertFalse(grid.padded)

    def test_a_padded_grid_reports_the_canvas_as_the_local_size(self):
        """The inverse: content shrinks at the edge but the remap denominator does not."""
        grid = tiling.plan_tiles_pixels(100, 100, 256, 0.0)
        self.assertEqual(grid.local_sizes, ((256, 256),))
        self.assertEqual(grid.content_sizes, ((100, 100),))
        self.assertNotEqual(grid.content_sizes, grid.local_sizes)

    def test_shape_groups_cover_every_tile_exactly_once(self):
        grid = tiling.plan_tiles_fractional(*self.GRID)
        covered = sorted(
            index for indices in grid.groups_by_shape().values() for index in indices
        )
        self.assertEqual(covered, list(range(len(grid))))


class DispatchTest(unittest.TestCase):
    """``plan_tiles`` must pick the same branch ``_tile_infer`` does."""

    def test_tile_size_px_wins_when_set(self):
        config = TilingConfig(tile_size_px=640, tile_width_pct=50.0, overlap=0.2)
        grid = tiling.plan_tiles(1080, 1920, config)
        self.assertTrue(grid.padded)
        self.assertEqual(
            grid.offsets, tiling.plan_tiles_pixels(1080, 1920, 640, 0.2).offsets
        )

    def test_percentages_are_used_when_tile_size_px_is_none(self):
        config = TilingConfig(
            tile_size_px=None, tile_width_pct=50.0, tile_height_pct=50.0, overlap=0.2
        )
        grid = tiling.plan_tiles(1080, 1920, config)
        self.assertFalse(grid.padded)
        self.assertEqual(
            grid.offsets,
            tiling.plan_tiles_fractional(1080, 1920, 0.5, 0.5, 0.2).offsets,
        )

    def test_dispatch_matches_the_pipeline_for_both_branches(self):
        """End-to-end against ``_tile_infer``'s own choice of tiler."""
        image = _frame(1080, 1920)
        for config in (
            TilingConfig(tile_size_px=640, overlap=0.2),
            TilingConfig(tile_size_px=None, tile_width_pct=50.0, tile_height_pct=50.0, overlap=0.2),
        ):
            with self.subTest(tile_size_px=config.tile_size_px):
                if config.tile_size_px is not None:
                    expected = tile_frame_pixels(image, config.tile_size_px, config.overlap)
                else:
                    expected = tile_frame(
                        image,
                        config.tile_width_pct / 100.0,
                        config.tile_height_pct / 100.0,
                        config.overlap,
                    )
                grid = tiling.plan_tiles(1080, 1920, config)
                tiles = tiling.extract_tiles(image, grid)
                self.assertEqual(len(tiles), len(expected))
                for index, (want, off, local) in enumerate(expected):
                    self.assertTrue(torch.equal(tiles[index], want), f"tile {index}")
                    self.assertEqual(grid.offsets[index], off)
                    self.assertEqual(grid.local_sizes[index], local)

    def test_expected_tile_count_agrees_with_extraction(self):
        config = TilingConfig(tile_size_px=640, overlap=0.2)
        self.assertEqual(
            tiling.expected_tile_count(1080, 1920, config),
            len(tiling.extract_tiles(_frame(1080, 1920), tiling.plan_tiles(1080, 1920, config))),
        )


class TensorHelpersTest(unittest.TestCase):
    def test_offsets_and_local_sizes_become_tensors(self):
        image = _frame(1080, 1920)
        grid = tiling.plan_tiles_pixels(1080, 1920, 640, 0.2)
        offsets = tiling.offsets_tensor(grid, image)
        sizes = tiling.local_sizes_tensor(grid, image)
        self.assertEqual(list(offsets.shape), [len(grid), 2])
        self.assertEqual(offsets.dtype, torch.int64)
        self.assertEqual([tuple(row) for row in offsets.tolist()], list(grid.offsets))
        self.assertEqual([tuple(row) for row in sizes.tolist()], list(grid.local_sizes))

    def test_full_size_tiles_are_views_not_copies(self):
        """The scalar originals return views; copying every tile would waste real memory."""
        image = _frame(1080, 1920)
        grid = tiling.plan_tiles_fractional(1080, 1920, 0.5, 0.5, 0.0)
        tiles = tiling.extract_tiles(image, grid)
        # A view shares storage with the frame; a copy would not.
        for index, tile in enumerate(tiles):
            self.assertEqual(
                tile.untyped_storage().data_ptr(),
                image.untyped_storage().data_ptr(),
                f"tile {index} was copied",
            )

    def test_describe_reports_the_shape_buckets(self):
        grid = tiling.plan_tiles_fractional(481, 641, 0.5, 0.5, 0.2)
        line = tiling.describe(grid)
        self.assertIn("clamped", line)
        self.assertIn(f"{len(grid)} tiles", line)


if __name__ == "__main__":
    unittest.main()
