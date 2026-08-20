"""``trained_fp16_gate._topk_l1`` — the comparison the FP16 verdict rests on.

The gate builds an fp32 and an fp16 engine from a trained checkpoint and fails when
their confident detections disagree. Everything about that verdict funnels through
this one pure function, so it is worth pinning directly: a gate that reports PASS on
a broken engine is worse than no gate, and this one did exactly that for the single
most important failure mode — the fp16 engine finding *nothing*.
"""

import math
import unittest

from friendy_chachkalica.ml.trt_export.trained_fp16_gate import _topk_l1

# [x0, y0, x1, y1, score, label]
_A = [10.0, 10.0, 50.0, 90.0, 0.91, 0.0]
_B = [200.0, 40.0, 260.0, 150.0, 0.77, 1.0]


class TopKL1Tests(unittest.TestCase):
    def test_identical_sets_score_zero(self):
        self.assertEqual(_topk_l1([_A, _B], [_A, _B]), 0.0)

    def test_worst_matched_detection_wins(self):
        shifted = list(_B)
        shifted[0] += 0.25
        self.assertAlmostEqual(_topk_l1([_A, _B], [_A, shifted]), 0.25, places=5)

    def test_row_order_does_not_matter(self):
        self.assertEqual(_topk_l1([_A, _B], [_B, _A]), 0.0)

    def test_a_confident_detection_with_no_same_label_twin_is_a_hard_miss(self):
        relabelled = list(_B)
        relabelled[5] = 7.0
        self.assertEqual(_topk_l1([_A, _B], [_A, relabelled]), math.inf)

    def test_an_empty_candidate_side_is_the_worst_outcome_not_a_free_pass(self):
        """The regression this test exists for.

        An fp16 engine that returns *no* detections where fp32 returns several is the
        most complete failure possible, but it used to score 0.0 (via a "nothing to
        compare" short-circuit that treated either side being empty the same) and so
        sailed through the gate. Observed for real: an fp16 engine emitted zero
        detections on all 8 gate images while fp32 emitted 2-4 on each, and the gate
        printed ``worst confident L1 = 0 -> PASS``.
        """
        self.assertEqual(_topk_l1([_A, _B], []), math.inf)

    def test_an_empty_reference_side_still_scores_zero(self):
        """The half of the old short-circuit that was right: with no confident fp32
        detection there is genuinely nothing to hold the fp16 engine to."""
        self.assertEqual(_topk_l1([], [_A, _B]), 0.0)


if __name__ == "__main__":
    unittest.main()
