"""``gpu_infer.tail`` against the compacting originals it replaces.

    python -m unittest gpu_infer.tests.test_tail

The claim under test is that carrying a padded buffer plus a mask produces **bit-identical**
detections to indexing them out at every step. So every comparison compacts the masked result
at the end and asserts exact equality against the reference's output.

Two properties get their own tests because they are the ones a plausible refactor breaks:
padded rows must not influence valid ones, and the threshold-then-transform ordering must be
interchangeable with transform-then-threshold.
"""

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from chachak.boxes import remap_local_preds_to_frame  # noqa: E402
from onnx_infer.preprocess import Transform  # noqa: E402
from trt_infer.postprocess_torch import to_friendy_torch  # noqa: E402
from gpu_infer import tail  # noqa: E402


def _raw(regions, max_det, *, seed, canvas=640.0):
    """Random engine-shaped output: xyxy in model-input pixels, scores, integer labels."""
    generator = torch.Generator().manual_seed(seed)
    x1 = torch.rand((regions, max_det), generator=generator) * canvas * 0.8
    y1 = torch.rand((regions, max_det), generator=generator) * canvas * 0.8
    boxes = torch.stack(
        [
            x1,
            y1,
            x1 + torch.rand((regions, max_det), generator=generator) * canvas * 0.3,
            y1 + torch.rand((regions, max_det), generator=generator) * canvas * 0.3,
        ],
        dim=-1,
    )
    scores = torch.rand((regions, max_det), generator=generator)
    labels = torch.randint(0, 5, (regions, max_det), generator=generator)
    return boxes, scores, labels


def _transforms(crop_sizes, canvas=640.0):
    """One ``Transform`` per region, letterboxed to a square canvas like the real planner."""
    made = []
    for crop_w, crop_h in crop_sizes:
        scale = canvas / max(crop_w, crop_h)
        made.append(
            Transform(
                scale_x=(max(1, round(crop_w * scale))) / crop_w,
                scale_y=(max(1, round(crop_h * scale))) / crop_h,
                pad_x=0,
                pad_y=0,
                orig_w=crop_w,
                orig_h=crop_h,
            )
        )
    return made


class MaskFromNumDetectionsTest(unittest.TestCase):
    """The replacement for the one host read that genuinely had to happen."""

    def test_mask_matches_slicing_by_count(self):
        counts = torch.tensor([0, 1, 7, 100], dtype=torch.int32)
        mask = tail.mask_from_num_detections(counts, 100)
        self.assertEqual(list(mask.shape), [4, 100])
        for row, count in enumerate(counts.tolist()):
            self.assertEqual(int(mask[row].sum()), count)
            # Exactly the leading `count` positions, which is what det_boxes[i, :n] takes.
            self.assertTrue(bool(mask[row, :count].all()))
            self.assertFalse(bool(mask[row, count:].any()))

    def test_accepts_the_trailing_axis_the_plugin_emits(self):
        """EfficientNMS emits ``num_detections`` as ``[B, 1]``; session.py reshapes it too."""
        flat = tail.mask_from_num_detections(torch.tensor([3, 5]), 8)
        shaped = tail.mask_from_num_detections(torch.tensor([[3], [5]]), 8)
        self.assertTrue(torch.equal(flat, shaped))

    def test_a_zero_score_detection_survives_the_mask(self):
        """Why a score mask cannot replace the count: threshold 0.0 is legal.

        The repo's eval default is 0.001 and callers pass 0.0, so a zero-scored row inside the
        count is a real detection and must not be confused with the plugin's zero padding.
        """
        boxes, scores, labels = _raw(1, 4, seed=3)
        scores = scores.clone()
        scores[0, 0] = 0.0
        valid = tail.mask_from_num_detections(torch.tensor([2]), 4)
        _, updated = tail.decode_regions(
            boxes, scores, labels, valid, tail.transforms_tensor(_transforms([(100, 200)]), boxes.device),
            score_threshold=0.0,
        )
        self.assertTrue(bool(updated[0, 0]), "zero-scored row inside the count is real")
        self.assertEqual(int(updated.sum()), 2)


class DecodeRegionsParityTest(unittest.TestCase):
    """Batched ``to_friendy_torch``."""

    CROPS = [(120, 300), (640, 640), (57, 91), (900, 400)]

    def _reference(self, boxes, scores, labels, transforms, threshold, **kwargs):
        return [
            to_friendy_torch(
                boxes[index], scores[index], labels[index], transforms[index],
                threshold, **kwargs,
            )
            for index in range(boxes.shape[0])
        ]

    def test_matches_to_friendy_torch_row_for_row(self):
        boxes, scores, labels = _raw(len(self.CROPS), 32, seed=101)
        transforms = _transforms(self.CROPS)
        for threshold in (0.0, 0.001, 0.25, 0.5, 0.9):
            with self.subTest(threshold=threshold):
                expected = self._reference(boxes, scores, labels, transforms, threshold)
                dets, valid = tail.decode_regions(
                    boxes, scores, labels,
                    torch.ones((len(self.CROPS), 32), dtype=torch.bool),
                    tail.transforms_tensor(transforms, boxes.device),
                    score_threshold=threshold,
                )
                for index in range(len(self.CROPS)):
                    self.assertEqual(
                        tail.compact(dets[index], valid[index]).tolist(),
                        expected[index].tolist(),
                        f"region {index}",
                    )

    def test_matches_with_clip_boxes(self):
        boxes, scores, labels = _raw(len(self.CROPS), 32, seed=103)
        transforms = _transforms(self.CROPS)
        expected = self._reference(
            boxes, scores, labels, transforms, 0.1, clip_boxes=True
        )
        dets, valid = tail.decode_regions(
            boxes, scores, labels,
            torch.ones((len(self.CROPS), 32), dtype=torch.bool),
            tail.transforms_tensor(transforms, boxes.device),
            score_threshold=0.1, clip_boxes=True,
        )
        for index in range(len(self.CROPS)):
            self.assertEqual(
                tail.compact(dets[index], valid[index]).tolist(),
                expected[index].tolist(),
                f"region {index}",
            )

    def test_matches_for_input_normalized_boxes(self):
        """``box_coords='input_normalized'`` takes the other branch, with norm_* == 1.0."""
        generator = torch.Generator().manual_seed(107)
        boxes = torch.rand((3, 16, 4), generator=generator).sort(dim=-1).values
        scores = torch.rand((3, 16), generator=generator)
        labels = torch.randint(0, 4, (3, 16), generator=generator)
        transforms = _transforms([(120, 300), (640, 640), (57, 91)])
        expected = self._reference(
            boxes, scores, labels, transforms, 0.2, box_coords="input_normalized"
        )
        dets, valid = tail.decode_regions(
            boxes, scores, labels, torch.ones((3, 16), dtype=torch.bool),
            tail.transforms_tensor(transforms, boxes.device, box_coords="input_normalized"),
            score_threshold=0.2, box_coords="input_normalized",
        )
        for index in range(3):
            self.assertEqual(
                tail.compact(dets[index], valid[index]).tolist(),
                expected[index].tolist(),
                f"region {index}",
            )

    def test_padded_rows_cannot_influence_valid_ones(self):
        """Overwrite the masked-off slots with garbage; the kept rows must not move.

        This is the property that makes the whole padded design safe. If any operation here
        reduced across the detection axis, this test would fail.
        """
        boxes, scores, labels = _raw(2, 24, seed=109)
        transforms = tail.transforms_tensor(_transforms([(120, 300), (640, 640)]), boxes.device)
        valid = torch.zeros((2, 24), dtype=torch.bool)
        valid[:, :6] = True

        clean, clean_valid = tail.decode_regions(
            boxes, scores, labels, valid, transforms, score_threshold=0.0
        )
        poisoned_boxes = boxes.clone()
        poisoned_scores = scores.clone()
        poisoned_boxes[:, 6:] = float("nan")
        poisoned_scores[:, 6:] = float("inf")
        dirty, dirty_valid = tail.decode_regions(
            poisoned_boxes, poisoned_scores, labels, valid, transforms, score_threshold=0.0
        )
        self.assertEqual(
            tail.compact(clean, clean_valid).tolist(),
            tail.compact(dirty, dirty_valid).tolist(),
        )

    def test_thresholding_after_the_transform_is_equivalent(self):
        """The reference thresholds first; the masked form must be able to threshold last."""
        boxes, scores, labels = _raw(2, 32, seed=113)
        transforms = _transforms([(120, 300), (640, 640)])
        tensor_transforms = tail.transforms_tensor(transforms, boxes.device)
        late, late_valid = tail.decode_regions(
            boxes, scores, labels, torch.ones((2, 32), dtype=torch.bool),
            tensor_transforms, score_threshold=0.0,
        )
        late_valid = late_valid & (late[..., tail.SCORE_COLUMN] >= 0.4)
        early, early_valid = tail.decode_regions(
            boxes, scores, labels, torch.ones((2, 32), dtype=torch.bool),
            tensor_transforms, score_threshold=0.4,
        )
        self.assertEqual(
            tail.compact(late, late_valid).tolist(),
            tail.compact(early, early_valid).tolist(),
        )


class RemapToFrameParityTest(unittest.TestCase):
    """Batched ``remap_local_preds_to_frame``."""

    FRAME = (1920, 1080)
    REGIONS = [((0, 0), (640, 640)), ((300, 200), (500, 500)), ((1800, 1000), (120, 80))]

    def _preds(self, count, *, seed):
        generator = torch.Generator().manual_seed(seed)
        centres = torch.rand((count, 2), generator=generator)
        sizes = torch.rand((count, 2), generator=generator) * 0.5
        scores = torch.rand((count, 1), generator=generator)
        classes = torch.randint(0, 4, (count, 1), generator=generator).float()
        return torch.cat([centres, sizes, scores, classes], dim=1)

    def test_matches_the_scalar_original(self):
        per_region = [self._preds(12, seed=200 + i) for i in range(len(self.REGIONS))]
        dets = torch.stack(per_region)
        valid = torch.ones(dets.shape[:2], dtype=torch.bool)
        offsets = torch.tensor([off for off, _ in self.REGIONS], dtype=torch.int64)
        sizes = torch.tensor([size for _, size in self.REGIONS], dtype=torch.int64)

        actual, actual_valid = tail.remap_regions_to_frame(
            dets, valid, offsets, sizes, *self.FRAME
        )
        for index, (offset, (local_w, local_h)) in enumerate(self.REGIONS):
            expected = remap_local_preds_to_frame(
                per_region[index], offset, local_w, local_h, *self.FRAME
            )
            self.assertEqual(
                tail.compact(actual[index], actual_valid[index]).tolist(),
                expected.tolist(),
                f"region {index}",
            )

    def test_degenerate_boxes_are_masked_not_indexed_out(self):
        """A box that clips to zero area must be dropped, as ``boxes[valid]`` does."""
        # Centre far off the right edge: after clipping, x2 == x1 == frame_w.
        dets = torch.tensor([[[5.0, 0.5, 0.1, 0.1, 0.9, 0.0], [0.5, 0.5, 0.2, 0.2, 0.8, 1.0]]])
        valid = torch.ones((1, 2), dtype=torch.bool)
        offsets = torch.tensor([[0, 0]], dtype=torch.int64)
        sizes = torch.tensor([[640, 640]], dtype=torch.int64)
        actual, actual_valid = tail.remap_regions_to_frame(
            dets, valid, offsets, sizes, *self.FRAME
        )
        expected = remap_local_preds_to_frame(dets[0], (0, 0), 640, 640, *self.FRAME)
        self.assertEqual(
            tail.compact(actual[0], actual_valid[0]).tolist(), expected.tolist()
        )
        self.assertEqual(int(actual_valid.sum()), expected.shape[0])

    def test_an_already_invalid_row_stays_invalid(self):
        dets = self._preds(8, seed=211).unsqueeze(0)
        valid = torch.zeros((1, 8), dtype=torch.bool)
        valid[0, ::2] = True
        _, updated = tail.remap_regions_to_frame(
            dets, valid, torch.tensor([[0, 0]]), torch.tensor([[640, 640]]), *self.FRAME
        )
        self.assertFalse(bool(updated[0, 1::2].any()), "mask must only ever narrow")


class ClassLutTest(unittest.TestCase):
    """The replacement for ``remap_raw_predictions_to_eval_classes``' per-row ``.item()``."""

    PRED = {0: "hardhat", 1: "vest", 2: "person"}
    EVAL = {0: "vest", 1: "hardhat"}

    def _reference(self, preds):
        from chachak._friendy import remap_raw_predictions_to_eval_classes

        return remap_raw_predictions_to_eval_classes(preds, self.PRED, self.EVAL)

    def _preds(self, class_ids):
        rows = [
            [0.5, 0.5, 0.2, 0.2, 0.9 - 0.01 * index, float(cid)]
            for index, cid in enumerate(class_ids)
        ]
        return torch.tensor(rows)

    def test_matches_the_reference_including_dropped_classes(self):
        preds = self._preds([0, 1, 2, 1, 0])
        lut = tail.build_class_lut(self.PRED, self.EVAL, preds.device)
        dets, valid = tail.apply_class_lut(
            preds.unsqueeze(0), torch.ones((1, 5), dtype=torch.bool), lut
        )
        # 'person' has no eval class, so its row is dropped exactly as the original drops it.
        self.assertEqual(
            tail.compact(dets[0], valid[0]).tolist(), self._reference(preds).tolist()
        )

    def test_out_of_range_ids_are_dropped_not_clamped(self):
        preds = self._preds([0, 9, 1])
        lut = tail.build_class_lut(self.PRED, self.EVAL, preds.device)
        dets, valid = tail.apply_class_lut(
            preds.unsqueeze(0), torch.ones((1, 3), dtype=torch.bool), lut
        )
        self.assertEqual(
            tail.compact(dets[0], valid[0]).tolist(), self._reference(preds).tolist()
        )
        self.assertFalse(bool(valid[0, 1]), "class 9 is not in the prediction map")

    def test_absent_maps_mean_no_remap(self):
        self.assertIsNone(tail.build_class_lut(None, self.EVAL, "cpu"))
        self.assertIsNone(tail.build_class_lut(self.PRED, {}, "cpu"))
        preds = self._preds([0, 1]).unsqueeze(0)
        valid = torch.ones((1, 2), dtype=torch.bool)
        dets, updated = tail.apply_class_lut(preds, valid, None)
        self.assertTrue(torch.equal(dets, preds))
        self.assertTrue(torch.equal(updated, valid))

    def test_class_ids_truncate_toward_zero_like_int(self):
        """``int(row[5].item())`` truncates; ``.to(torch.int64)`` must agree."""
        preds = torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.9, 1.7]])
        lut = tail.build_class_lut(self.PRED, self.EVAL, preds.device)
        dets, valid = tail.apply_class_lut(
            preds.unsqueeze(0), torch.ones((1, 1), dtype=torch.bool), lut
        )
        self.assertEqual(
            tail.compact(dets[0], valid[0]).tolist(), self._reference(preds).tolist()
        )


class FilterClassTest(unittest.TestCase):
    """Batched ``Detector.predict``'s person filter."""

    def test_matches_the_scalar_predicate(self):
        preds = torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.2, 0.90, 0.0],
                [0.5, 0.5, 0.2, 0.2, 0.40, 0.0],
                [0.5, 0.5, 0.2, 0.2, 0.95, 1.0],
            ]
        )
        valid = tail.filter_class(
            preds.unsqueeze(0), torch.ones((1, 3), dtype=torch.bool), 0, 0.5
        )
        expected = preds[
            (preds[:, 5].to(torch.int64) == 0) & (preds[:, 4] >= 0.5)
        ]
        self.assertEqual(tail.compact(preds, valid[0]).tolist(), expected.tolist())


class TrimAndCompactTest(unittest.TestCase):
    def test_no_limit_is_a_passthrough(self):
        """The default must match ``merge_predictions``, which has no cap at all."""
        dets = torch.rand((40, 6))
        valid = torch.ones(40, dtype=torch.bool)
        trimmed, trimmed_valid = tail.topk_trim(dets, valid, None)
        self.assertTrue(torch.equal(trimmed, dets))
        self.assertTrue(torch.equal(trimmed_valid, valid))

    def test_trim_keeps_the_highest_scoring_valid_rows(self):
        dets = torch.zeros((10, 6))
        dets[:, tail.SCORE_COLUMN] = torch.arange(10, dtype=torch.float32) / 10.0
        valid = torch.ones(10, dtype=torch.bool)
        valid[9] = False  # the top score is invalid and must not be kept
        trimmed, trimmed_valid = tail.topk_trim(dets, valid, 3)
        kept = tail.compact(trimmed, trimmed_valid)[:, tail.SCORE_COLUMN]
        # Compare against the same float32 values, not decimal literals: 0.8 as float32
        # widens back to 0.800000011920929 and a literal comparison fails on precision
        # rather than on the selection this test is about.
        expected = dets[[8, 7, 6], tail.SCORE_COLUMN]
        self.assertEqual(sorted(kept.tolist(), reverse=True), expected.tolist())

    def test_invalid_rows_never_displace_valid_ones(self):
        dets = torch.zeros((6, 6))
        dets[:, tail.SCORE_COLUMN] = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        valid = torch.tensor([True, False, True, False, True, False])
        trimmed, trimmed_valid = tail.topk_trim(dets, valid, 3)
        self.assertEqual(int(trimmed_valid.sum()), 3)
        self.assertEqual(
            sorted(tail.compact(trimmed, trimmed_valid)[:, tail.SCORE_COLUMN].tolist()),
            sorted(dets[[0, 2, 4], tail.SCORE_COLUMN].tolist()),
        )

    def test_overflow_count_reports_what_a_trim_would_drop(self):
        valid = torch.zeros(20, dtype=torch.bool)
        valid[:12] = True
        self.assertEqual(int(tail.overflow_count(valid, 8)), 4)
        self.assertEqual(int(tail.overflow_count(valid, 12)), 0)
        self.assertEqual(int(tail.overflow_count(valid, 50)), 0)
        self.assertEqual(int(tail.overflow_count(valid, None)), 0)

    def test_compact_preserves_order(self):
        dets = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        valid = torch.tensor([True, False, True, True, False])
        self.assertEqual(
            tail.compact(dets, valid).tolist(), dets[[0, 2, 3]].tolist()
        )

    def test_flatten_round_trips_shapes(self):
        dets = torch.rand((3, 7, 6))
        valid = torch.ones((3, 7), dtype=torch.bool)
        flat, flat_valid = tail.flatten_regions(dets, valid)
        self.assertEqual(list(flat.shape), [21, 6])
        self.assertEqual(list(flat_valid.shape), [21])
        self.assertTrue(torch.equal(flat.reshape(3, 7, 6), dets))

    def test_empty_detections_stay_on_the_callers_device(self):
        empty = tail.empty_detections(device=torch.device("cpu"))
        self.assertEqual(list(empty.shape), [0, 6])
        self.assertEqual(empty.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
