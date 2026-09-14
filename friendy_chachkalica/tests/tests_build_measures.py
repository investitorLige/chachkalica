"""Attributing a ground-truth box to the person it belongs to.

Two rules, and the reason there are two is the whole point: a helmet's IoU with
its own wearer is about 0.02, so anything that ranks people by IoU picks the
wrong one -- and looks entirely reasonable while doing it.
"""

import unittest

import numpy as np

from friendy_chachkalica.ml.build_measures import (
    NO_PERSON,
    _match_poses,
    attribute,
)


def box(x0, y0, x1, y1):
    return np.array([x0, y0, x1, y1], dtype=np.float64)


class AttributionTests(unittest.TestCase):
    #: One person standing left, one right; both full height.
    PEOPLE = np.array([[100.0, 100.0, 200.0, 500.0],
                       [300.0, 100.0, 400.0, 500.0]])

    def test_a_helmet_goes_to_the_person_containing_it_not_the_best_iou(self):
        helmet = box(130, 110, 170, 150)

        # Sanity: IoU against the correct wearer is tiny, so an IoU-ranked
        # implementation has nothing to distinguish it from noise.
        intersection = 40.0 * 40.0
        union = 40.0 * 40.0 + 100.0 * 400.0 - intersection
        self.assertLess(intersection / union, 0.05)

        self.assertEqual(attribute(helmet, False, self.PEOPLE), 0)

    def test_a_helmet_over_the_other_person_goes_to_that_one(self):
        self.assertEqual(attribute(box(330, 110, 370, 150), False, self.PEOPLE), 1)

    def test_a_box_inside_nobody_is_unattributed(self):
        self.assertIsNone(attribute(box(600, 600, 640, 640), False, self.PEOPLE))

    def test_a_box_only_half_inside_a_person_is_unattributed(self):
        # 50% contained, below the floor: a box straddling an edge belongs to
        # neither, and guessing is what the floor exists to prevent.
        self.assertIsNone(attribute(box(180, 200, 220, 240), False, self.PEOPLE))

    def test_overlapping_people_hand_the_box_to_the_tighter_one(self):
        people = np.array([[0.0, 0.0, 1000.0, 1000.0],    # a large foreground figure
                           [100.0, 100.0, 200.0, 500.0]])  # the actual wearer
        self.assertEqual(attribute(box(130, 110, 170, 150), False, people), 1)

    def test_a_person_box_is_matched_by_overlap_not_containment(self):
        # A small person standing in front of a large one is fully contained by
        # them; containment would hand them to the wrong body.
        people = np.array([[0.0, 0.0, 1000.0, 1000.0],
                           [100.0, 100.0, 200.0, 500.0]])
        self.assertEqual(attribute(box(100, 100, 200, 500), True, people), 1)

    def test_a_person_box_nothing_overlaps_is_unattributed(self):
        self.assertIsNone(attribute(box(600, 600, 700, 900), True, self.PEOPLE))


class PoseMatchTests(unittest.TestCase):
    def test_each_person_takes_the_posture_of_the_box_over_them(self):
        people = np.array([[100.0, 100.0, 200.0, 500.0],
                           [300.0, 100.0, 400.0, 500.0]])
        pose_boxes = np.array([[305.0, 105.0, 395.0, 495.0],
                               [102.0, 102.0, 198.0, 498.0]])

        poses = _match_poses(people, pose_boxes, ["sitting", "standing"])

        self.assertEqual(poses, ["standing", "sitting"])

    def test_a_person_the_posture_model_missed_gets_no_posture(self):
        people = np.array([[100.0, 100.0, 200.0, 500.0]])
        poses = _match_poses(people, np.array([[800.0, 800.0, 900.0, 900.0]]), ["standing"])

        # Not a default posture: the two models disagreed, and recording a guess
        # would put that disagreement into the data as an observation.
        self.assertEqual(poses, [None])

    def test_no_postures_at_all_leaves_every_person_unposed(self):
        people = np.array([[100.0, 100.0, 200.0, 500.0]])
        self.assertEqual(_match_poses(people, np.zeros((0, 4)), []), [None])

    def test_no_people_is_an_empty_answer_not_an_error(self):
        self.assertEqual(_match_poses(np.zeros((0, 4)), np.zeros((0, 4)), []), [])


class VocabularyTests(unittest.TestCase):
    def test_the_no_person_bucket_keeps_its_exact_spelling(self):
        # This producer and the Django app's tag_analytics both name this bucket
        # and cannot import each other (no torch one way, no Django the other),
        # so the literal is pinned on both sides. A drift here would split one
        # bucket into two that look identical on the page.
        self.assertEqual(NO_PERSON, "(no person)")


if __name__ == "__main__":
    unittest.main()
