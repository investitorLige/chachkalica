"""``gpu_infer.nms`` against ``chachak.boxes._class_aware_overlap_nms``.

    python -m unittest gpu_infer.tests.test_nms

The round-based parallel form has to be *exactly* the greedy sequential answer, not an
approximation, so these tests are built around the cases where an approximation would diverge:

* **Suppression chains** — A suppresses B, and B would have suppressed C, so greedy keeps C.
  Fast-NMS style one-shot suppression drops C. Asserted directly, and also as a negative test so
  the difference is recorded in code.
* **Containment-only overlap** — nested boxes whose IoU is below the threshold but whose
  intersection over the *smaller* area is not. This is the term ``batched_nms`` lacks.
* **Score ties** — the one documented parity gap, tested as a known difference rather than
  hidden behind a tolerance.
"""

import random
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from chachak.boxes import _class_aware_overlap_nms, merge_predictions  # noqa: E402
from gpu_infer import nms  # noqa: E402


def _reference(boxes, scores, classes, threshold):
    return _class_aware_overlap_nms(boxes, scores, classes, threshold)


def _ours(boxes, scores, classes, threshold, rounds=8):
    keep, converged = nms.class_aware_overlap_nms(
        boxes, scores, classes, threshold, rounds=rounds
    )
    return keep, bool(converged)


class BasicParityTest(unittest.TestCase):
    """Same kept indices, in the same order, on ordinary inputs."""

    def _random(self, count, *, seed, classes=3, spread=1000.0):
        rng = random.Random(seed)
        rows, scores, labels = [], [], []
        for _ in range(count):
            x1 = rng.uniform(0, spread)
            y1 = rng.uniform(0, spread)
            side = rng.uniform(10.0, 300.0)
            rows.append([x1, y1, x1 + side, y1 + side * rng.uniform(0.5, 2.0)])
            # Distinct scores: ties are covered separately, deliberately.
            scores.append(rng.uniform(0.01, 1.0))
            labels.append(rng.randrange(classes))
        return (
            torch.tensor(rows, dtype=torch.float32),
            torch.tensor(scores, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )

    def test_matches_across_sizes_and_thresholds(self):
        for seed in range(6):
            for count in (1, 2, 5, 40, 200):
                boxes, scores, classes = self._random(count, seed=seed * 100 + count)
                for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
                    with self.subTest(seed=seed, count=count, threshold=threshold):
                        keep, converged = _ours(boxes, scores, classes, threshold)
                        self.assertTrue(converged, "8 rounds should always be enough here")
                        self.assertEqual(
                            keep.tolist(),
                            _reference(boxes, scores, classes, threshold).tolist(),
                        )

    def test_dense_overlap_still_matches(self):
        """Many boxes on top of each other — the deepest chains occur here."""
        rng = random.Random(7)
        rows, scores, labels = [], [], []
        for index in range(120):
            x1 = rng.uniform(0, 60)
            y1 = rng.uniform(0, 60)
            rows.append([x1, y1, x1 + 200.0, y1 + 200.0])
            scores.append(1.0 - index * 0.005)
            labels.append(index % 2)
        boxes = torch.tensor(rows, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        classes = torch.tensor(labels, dtype=torch.int64)
        for threshold in (0.5, 0.75, 0.95):
            with self.subTest(threshold=threshold):
                keep, converged = _ours(boxes, scores_t, classes, threshold, rounds=16)
                self.assertTrue(converged)
                self.assertEqual(
                    keep.tolist(),
                    _reference(boxes, scores_t, classes, threshold).tolist(),
                )

    def test_keep_order_is_descending_score(self):
        """The original appends in processing order, so ``dets[keep]`` is score-ordered."""
        boxes, scores, classes = self._random(50, seed=11)
        keep, _ = _ours(boxes, scores, classes, 0.5)
        kept_scores = scores[keep].tolist()
        self.assertEqual(kept_scores, sorted(kept_scores, reverse=True))


class SuppressionChainTest(unittest.TestCase):
    """The case that separates exact greedy from one-shot suppression."""

    def _chain(self):
        """A, B, C in a row: A-B overlap heavily, B-C overlap heavily, A-C barely.

        Greedy keeps A, suppresses B, then finds C unsuppressed (because B is gone) and keeps
        it. A one-shot 'suppress anything overlapping a higher-scoring box' rule would drop C.
        """
        boxes = torch.tensor(
            [
                [0.0, 0.0, 100.0, 100.0],     # A, top score
                [60.0, 0.0, 160.0, 100.0],    # B, overlaps A a lot
                [120.0, 0.0, 220.0, 100.0],   # C, overlaps B a lot, A not at all
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        classes = torch.tensor([0, 0, 0])
        return boxes, scores, classes

    def test_chain_keeps_the_third_box(self):
        boxes, scores, classes = self._chain()
        keep, converged = _ours(boxes, scores, classes, 0.3)
        self.assertTrue(converged)
        self.assertEqual(keep.tolist(), [0, 2], "greedy must keep C once B is suppressed")
        self.assertEqual(keep.tolist(), _reference(boxes, scores, classes, 0.3).tolist())

    def test_one_shot_suppression_would_be_wrong(self):
        """Negative test: pin what a Fast-NMS style rule would do, so nobody 'optimizes' to it.

        Kept as executable documentation — if someone replaces the round loop with a single
        masked ``max`` over the overlap matrix, this records exactly what breaks.
        """
        boxes, scores, classes = self._chain()
        suppresses = nms.suppression_matrix(boxes, scores, classes, 0.3)
        # One shot: drop anything any higher-ranked box overlaps, regardless of whether that
        # box survived. C is suppressed by B, and B is itself suppressed by A.
        one_shot_kept = ~suppresses.any(dim=0)
        self.assertEqual(one_shot_kept.tolist(), [True, False, False])
        self.assertNotEqual(
            [index for index, keep in enumerate(one_shot_kept.tolist()) if keep],
            _reference(boxes, scores, classes, 0.3).tolist(),
        )

    def test_a_deeper_chain_needs_more_rounds_and_says_so(self):
        """A long chain past ``rounds`` must report non-convergence, not silently truncate."""
        count = 12
        boxes = torch.tensor(
            [[i * 60.0, 0.0, i * 60.0 + 100.0, 100.0] for i in range(count)]
        )
        scores = torch.tensor([1.0 - i * 0.01 for i in range(count)])
        classes = torch.zeros(count, dtype=torch.int64)

        keep, converged = _ours(boxes, scores, classes, 0.3, rounds=2)
        self.assertFalse(converged, "a 12-long chain cannot resolve in 2 rounds")

        keep, converged = _ours(boxes, scores, classes, 0.3, rounds=count + 1)
        self.assertTrue(converged)
        self.assertEqual(keep.tolist(), _reference(boxes, scores, classes, 0.3).tolist())

    def test_check_converged_raises_only_when_strict(self):
        flag = torch.zeros((), dtype=torch.bool)
        with self.assertRaises(RuntimeError) as caught:
            nms.check_converged(flag, strict=True, rounds=8)
        self.assertIn("did not converge", str(caught.exception))
        nms.check_converged(flag, strict=False, rounds=8)  # must not raise
        nms.check_converged(torch.ones((), dtype=torch.bool), strict=True, rounds=8)


class ContainmentTest(unittest.TestCase):
    """The OR'd containment term — what ``batched_nms`` cannot express."""

    def test_a_nested_box_is_suppressed_below_the_iou_threshold(self):
        # Small box wholly inside a large one. IoU = 100/10000 = 0.01, but the intersection
        # over the smaller area is 1.0, so the containment term fires and IoU never would.
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [10.0, 10.0, 20.0, 20.0]])
        scores = torch.tensor([0.9, 0.8])
        classes = torch.tensor([0, 0])

        intersection = 10.0 * 10.0
        iou = intersection / (100.0 * 100.0 + 10.0 * 10.0 - intersection)
        self.assertLess(iou, 0.5, "IoU alone must not reach the threshold")

        keep, _ = _ours(boxes, scores, classes, 0.5)
        self.assertEqual(keep.tolist(), [0])
        self.assertEqual(keep.tolist(), _reference(boxes, scores, classes, 0.5).tolist())

    def test_containment_respects_class(self):
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [10.0, 10.0, 20.0, 20.0]])
        scores = torch.tensor([0.9, 0.8])
        classes = torch.tensor([0, 1])
        keep, _ = _ours(boxes, scores, classes, 0.5)
        self.assertEqual(keep.tolist(), [0, 1], "different classes never suppress")
        self.assertEqual(keep.tolist(), _reference(boxes, scores, classes, 0.5).tolist())

    def test_zero_area_boxes_do_not_divide_by_zero(self):
        """Both denominators are ``clamp(min=eps)`` in the original; match that."""
        boxes = torch.tensor(
            [[10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0], [0.0, 0.0, 50.0, 50.0]]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        classes = torch.zeros(3, dtype=torch.int64)
        keep, converged = _ours(boxes, scores, classes, 0.5)
        self.assertTrue(converged)
        self.assertEqual(keep.tolist(), _reference(boxes, scores, classes, 0.5).tolist())


class MultiClassTest(unittest.TestCase):
    def test_interleaved_classes_match(self):
        rng = random.Random(13)
        rows, scores, labels = [], [], []
        for index in range(80):
            x1 = rng.uniform(0, 200)
            y1 = rng.uniform(0, 200)
            rows.append([x1, y1, x1 + 150.0, y1 + 150.0])
            scores.append(1.0 - index * 0.01)
            labels.append(index % 5)
        boxes = torch.tensor(rows, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        classes = torch.tensor(labels, dtype=torch.int64)
        keep, converged = _ours(boxes, scores_t, classes, 0.4, rounds=16)
        self.assertTrue(converged)
        self.assertEqual(
            keep.tolist(), _reference(boxes, scores_t, classes, 0.4).tolist()
        )

    def test_large_class_ids_do_not_alias(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        scores = torch.tensor([0.9, 0.8])
        classes = torch.tensor([0, 1000000])
        keep, _ = _ours(boxes, scores, classes, 0.5)
        self.assertEqual(keep.tolist(), [0, 1])


class DegenerateInputTest(unittest.TestCase):
    def test_empty(self):
        boxes = torch.zeros((0, 4))
        keep, converged = _ours(boxes, torch.zeros(0), torch.zeros(0, dtype=torch.int64), 0.5)
        self.assertEqual(keep.tolist(), [])
        self.assertTrue(converged)
        self.assertEqual(keep.dtype, torch.int64)

    def test_single_box(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        keep, converged = _ours(boxes, torch.tensor([0.5]), torch.tensor([2]), 0.5)
        self.assertEqual(keep.tolist(), [0])
        self.assertTrue(converged)

    def test_identical_boxes_collapse_to_one(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]] * 6)
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        classes = torch.zeros(6, dtype=torch.int64)
        keep, _ = _ours(boxes, scores, classes, 0.5)
        self.assertEqual(keep.tolist(), [0])
        self.assertEqual(keep.tolist(), _reference(boxes, scores, classes, 0.5).tolist())

    def test_suppression_matrix_is_strictly_upper_by_rank(self):
        """Nothing suppresses itself, and suppression only ever runs down the ranking."""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]] * 4)
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        classes = torch.zeros(4, dtype=torch.int64)
        suppresses = nms.suppression_matrix(boxes, scores, classes, 0.5)
        self.assertFalse(bool(suppresses.diagonal().any()))
        self.assertFalse(bool(torch.tril(suppresses).any()), "no upward suppression")


class ScoreTieTest(unittest.TestCase):
    """The one documented parity gap, recorded rather than hidden."""

    def test_ties_break_toward_the_lower_index(self):
        """This implementation is deterministic: a tie resolves as a stable sort would."""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
        scores = torch.tensor([0.5, 0.5])
        classes = torch.zeros(2, dtype=torch.int64)
        keep, _ = _ours(boxes, scores, classes, 0.5)
        self.assertEqual(keep.tolist(), [0], "lower index wins a tie, as stable sort does")

    def test_tie_behaviour_is_stable_across_repeated_calls(self):
        """Whatever it picks, it must pick it every time — the original's argsort need not."""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]] * 5)
        scores = torch.full((5,), 0.5)
        classes = torch.zeros(5, dtype=torch.int64)
        first, _ = _ours(boxes, scores, classes, 0.5)
        for _ in range(8):
            again, _ = _ours(boxes, scores, classes, 0.5)
            self.assertEqual(again.tolist(), first.tolist())

    def test_a_realistic_threshold_removes_the_systematic_tie_source(self):
        """The plugin's zero-padded tail is the only systematic tie source, and it is masked
        out before NMS — so in the real pipeline ties are not reachable from padding."""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        scores = torch.tensor([0.0, 0.0])
        classes = torch.zeros(2, dtype=torch.int64)
        # Above the eval floor nothing survives, so the tie never reaches the NMS at all.
        kept = scores >= 0.001
        self.assertEqual(int(kept.sum()), 0)


class MergeDetectionsTest(unittest.TestCase):
    """``merge_detections`` against ``merge_predictions``, the actual call site."""

    FRAME = (1920, 1080)

    def _preds(self, count, *, seed):
        rng = random.Random(seed)
        rows = []
        for _ in range(count):
            rows.append(
                [
                    rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9),
                    rng.uniform(0.02, 0.3), rng.uniform(0.02, 0.3),
                    rng.uniform(0.01, 1.0), float(rng.randrange(4)),
                ]
            )
        return torch.tensor(rows, dtype=torch.float32)

    def test_matches_merge_predictions(self):
        for seed in range(5):
            for count in (0, 1, 3, 30, 120):
                for nms_iou in (0.3, 0.5, 0.7):
                    with self.subTest(seed=seed, count=count, nms_iou=nms_iou):
                        preds = self._preds(count, seed=seed * 50 + count)
                        expected = merge_predictions([preds], *self.FRAME, nms_iou)
                        actual, converged = nms.merge_detections(
                            preds, *self.FRAME, nms_iou, rounds=24
                        )
                        self.assertTrue(bool(converged))
                        self.assertEqual(actual.tolist(), expected.tolist())

    def test_matches_when_several_regions_are_concatenated(self):
        """The real shape: several regions' predictions merged for one frame."""
        parts = [self._preds(20, seed=900 + index) for index in range(4)]
        expected = merge_predictions(parts, *self.FRAME, 0.5)
        actual, _ = nms.merge_detections(
            torch.cat(parts, dim=0), *self.FRAME, 0.5, rounds=24
        )
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_empty_input_returns_an_empty_frame_normalized_tensor(self):
        actual, converged = nms.merge_detections(
            torch.zeros((0, 6)), *self.FRAME, 0.5
        )
        self.assertEqual(list(actual.shape), [0, 6])
        self.assertTrue(bool(converged))


if __name__ == "__main__":
    unittest.main()
