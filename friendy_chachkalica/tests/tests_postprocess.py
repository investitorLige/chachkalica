import unittest

import torch

from friendy_chachkalica.postprocess import apply_class_aware_nms, class_aware_nms_keep


class ClassAwareNmsTests(unittest.TestCase):
    def test_suppresses_same_class_overlap(self):
        prediction = torch.tensor([
            [0.5, 0.5, 0.4, 0.4, 0.9, 0.0],
            [0.52, 0.5, 0.4, 0.4, 0.8, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.7, 0.0],
        ])

        kept = apply_class_aware_nms(prediction, 0.5)

        self.assertEqual(kept.shape[0], 2)
        # Compare via a float32 tensor: 0.9 has no exact float32 value, so a
        # .tolist() == [0.9, ...] comparison fails on representation noise.
        self.assertTrue(torch.equal(kept[:, 4], torch.tensor([0.9, 0.7])))

    def test_keeps_overlapping_different_classes(self):
        prediction = torch.tensor([
            [0.5, 0.5, 0.4, 0.4, 0.9, 0.0],
            [0.5, 0.5, 0.4, 0.4, 0.8, 1.0],
        ])

        kept = apply_class_aware_nms(prediction, 0.5)

        self.assertEqual(kept.shape[0], 2)
        self.assertTrue(torch.equal(kept[:, 4], torch.tensor([0.9, 0.8])))

    def test_keep_indices_preserve_input_order(self):
        # xyxy boxes; the middle (lower-scored duplicate of the first) is
        # suppressed and the surviving indices come back in input order.
        boxes = torch.tensor([
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 10.0, 52.0, 50.0],
            [70.0, 70.0, 90.0, 90.0],
        ])
        scores = torch.tensor([0.8, 0.9, 0.7])
        labels = torch.tensor([0, 0, 0])

        keep = class_aware_nms_keep(boxes, scores, labels, 0.5)

        self.assertEqual(keep.tolist(), [1, 2])  # 1 outscores 0; 2 is disjoint

    def test_null_threshold_disables_nms(self):
        prediction = torch.tensor([
            [0.5, 0.5, 0.4, 0.4, 0.9, 0.0],
            [0.5, 0.5, 0.4, 0.4, 0.8, 0.0],
        ])

        kept = apply_class_aware_nms(prediction, None)

        self.assertTrue(torch.equal(kept, prediction))


if __name__ == "__main__":
    unittest.main()
