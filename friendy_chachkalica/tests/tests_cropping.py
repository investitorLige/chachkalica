import types
import unittest
from pathlib import Path

import torch

from friendy_chachkalica.config import _parse_pipeline
from friendy_chachkalica.preprocess.cropping import crop_batch

from chachak.pipeline import BatchPeoplePipeline, PeopleDetectFirstPipeline


def _target(boxes, labels, size=100):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "orig_size": torch.tensor([size, size], dtype=torch.int64),
    }


def _person_pred(xyxy, frame_w, frame_h, conf=0.9, cls=0):
    """One detector prediction row (cx, cy, w, h, conf, class) normalized."""
    x1, y1, x2, y2 = xyxy
    return [
        (x1 + x2) / 2.0 / frame_w,
        (y1 + y2) / 2.0 / frame_h,
        (x2 - x1) / frame_w,
        (y2 - y1) / frame_h,
        conf,
        float(cls),
    ]


class _StubDetector:
    """Returns a fixed list of (N, 6) xywhn person predictions per input image."""

    def __init__(self, preds_per_image):
        self._preds = preds_per_image

    def predict(self, images):
        # One (N, 6) tensor per image, in input order (mirrors chachak.Detector).
        return [torch.tensor(rows, dtype=torch.float32).reshape(-1, 6) for rows in self._preds]


def _people_pipeline(detector, *, expand_ratio=0.0, min_box_size=0.0, **extra):
    config = types.SimpleNamespace(
        detector=types.SimpleNamespace(
            expand_ratio=expand_ratio, min_box_size=min_box_size, nms_iou=0.5
        ),
        **extra,
    )
    return PeopleDetectFirstPipeline(
        model_adapter=None, device=torch.device("cpu"), config=config, detector=detector
    )


class CropRegionsTests(unittest.TestCase):
    def test_single_person_box_becomes_one_crop_region(self):
        img = torch.zeros(3, 100, 100)
        det = _StubDetector([[_person_pred([50, 50, 100, 100], 100, 100)]])
        regions = _people_pipeline(det).crop_regions([img])

        self.assertEqual(len(regions), 1)
        self.assertEqual(len(regions[0]), 1)
        crop, (x0, y0), (w, h) = regions[0][0]
        self.assertEqual((x0, y0), (50, 50))
        self.assertEqual((w, h), (50, 50))
        self.assertEqual(tuple(crop.shape), (3, 50, 50))

    def test_expand_ratio_grows_the_crop(self):
        img = torch.zeros(3, 100, 100)
        det = _StubDetector([[_person_pred([40, 40, 60, 60], 100, 100)]])
        # 20px-wide box, expand 0.5 -> +5px each side -> [35,35,65,65].
        regions = _people_pipeline(det, expand_ratio=0.5).crop_regions([img])
        _, (x0, y0), (w, h) = regions[0][0]
        self.assertEqual((x0, y0), (35, 35))
        self.assertEqual((w, h), (30, 30))

    def test_no_detections_yields_empty_regions(self):
        img = torch.zeros(3, 100, 100)
        regions = _people_pipeline(_StubDetector([[]])).crop_regions([img])
        self.assertEqual(regions, [[]])


class CropBatchTests(unittest.TestCase):
    def test_gt_inside_crop_translated_to_local_coords(self):
        img = torch.zeros(3, 100, 100)
        det = _StubDetector([[_person_pred([50, 50, 100, 100], 100, 100)]])
        images, targets = crop_batch([img], [_target([[60, 60, 90, 90]], [1])], _people_pipeline(det))

        self.assertEqual(len(images), 1)
        self.assertEqual(tuple(images[0].shape), (3, 50, 50))
        # Crop starts at (50, 50): [60,60,90,90] -> [10,10,40,40].
        self.assertTrue(torch.equal(targets[0]["boxes"], torch.tensor([[10.0, 10.0, 40.0, 40.0]])))
        self.assertTrue(torch.equal(targets[0]["labels"], torch.tensor([1])))
        self.assertTrue(torch.equal(targets[0]["orig_size"], torch.tensor([50, 50])))

    def test_object_outside_all_crops_is_dropped_and_crop_is_background(self):
        img = torch.zeros(3, 100, 100)
        # Person crop in the top-left; the only GT object sits in the bottom-right.
        det = _StubDetector([[_person_pred([0, 0, 50, 50], 100, 100)]])
        images, targets = crop_batch([img], [_target([[60, 60, 90, 90]], [1])], _people_pipeline(det))

        # The crop is kept as a background negative; the object is never seen.
        self.assertEqual(len(images), 1)
        self.assertEqual(int(targets[0]["labels"].numel()), 0)
        self.assertEqual(tuple(targets[0]["boxes"].shape), (0, 4))

    def test_no_detections_contributes_nothing(self):
        img = torch.zeros(3, 100, 100)
        images, targets = crop_batch(
            [img], [_target([[10, 10, 40, 40]], [1])], _people_pipeline(_StubDetector([[]]))
        )
        self.assertEqual(images, [])
        self.assertEqual(targets, [])

    def test_multiple_people_yield_one_sample_each(self):
        img = torch.zeros(3, 100, 100)
        det = _StubDetector(
            [[
                _person_pred([0, 0, 50, 50], 100, 100),
                _person_pred([50, 50, 100, 100], 100, 100),
            ]]
        )
        images, targets = crop_batch(
            [img],
            [_target([[10, 10, 40, 40], [60, 60, 90, 90]], [1, 2])],
            _people_pipeline(det),
        )
        self.assertEqual(len(images), 2)
        # Each crop keeps exactly the object it frames, remapped to [10,10,40,40].
        for target in targets:
            self.assertEqual(int(target["labels"].numel()), 1)
            self.assertTrue(torch.equal(target["boxes"], torch.tensor([[10.0, 10.0, 40.0, 40.0]])))

    def test_sliver_below_min_visible_dropped_but_crop_retained_as_background(self):
        img = torch.zeros(3, 100, 100)
        # Crop is the top-left 50x50; the object is almost entirely to its right,
        # with only a 1px column (<10% of its area) inside the crop.
        det = _StubDetector([[_person_pred([0, 0, 50, 50], 100, 100)]])
        images, targets = crop_batch([img], [_target([[49, 10, 90, 40]], [1])], _people_pipeline(det))
        # The sliver is ignored (not trained as a positive), and because a real
        # partial object overlaps, the window is dropped rather than kept as a
        # (mislabeled) background negative.
        self.assertEqual(images, [])
        self.assertEqual(targets, [])


class UpscaledCropTargetTests(unittest.TestCase):
    """An upscaled crop's boxes must be in the *tensor's* pixel space.

    A crop under ``min_box_size`` whose frame is smaller than the floor is
    bilinearly upscaled, while ``crop_regions`` keeps reporting the region's size in
    frame pixels (deliberately — that's the geometry remapping uses). The training
    targets have to follow the tensor: torchvision archs read box coordinates as
    pixels of the image they're handed, so pre-upscale boxes land in the top-left
    corner at 1/scale of their real size.
    """

    def test_boxes_follow_the_upscale(self):
        img = torch.zeros(3, 100, 100)
        det = _StubDetector([[_person_pred([0, 0, 100, 100], 100, 100)]])
        pipeline = _people_pipeline(det, min_box_size=200.0)

        images, targets = crop_batch(
            [img], [_target([[10, 10, 50, 50]], [1])], pipeline)

        # The whole frame is the region (100x100), upscaled 2x to clear the floor.
        self.assertEqual(tuple(images[0].shape), (3, 200, 200))
        self.assertTrue(torch.equal(
            targets[0]["boxes"], torch.tensor([[20.0, 20.0, 100.0, 100.0]])))
        self.assertTrue(torch.equal(targets[0]["orig_size"], torch.tensor([200, 200])))
        # area is rebuilt in the same space, not left at the pre-upscale value.
        self.assertEqual(float(targets[0]["area"][0]), 80.0 * 80.0)

    def test_normalized_coords_are_unchanged_by_the_rescale(self):
        # What crop_cache writes (YOLO labels normalized by orig_size) and what the
        # pipeline remaps back onto the frame are both normalized, so this fix must
        # leave them exactly where they were.
        img = torch.zeros(3, 100, 100)
        det = _StubDetector([[_person_pred([0, 0, 100, 100], 100, 100)]])

        plain = crop_batch([img], [_target([[10, 10, 50, 50]], [1])],
                           _people_pipeline(det, min_box_size=0.0))[1][0]
        upscaled = crop_batch([img], [_target([[10, 10, 50, 50]], [1])],
                              _people_pipeline(det, min_box_size=200.0))[1][0]

        def normalized(target):
            h, w = target["orig_size"].tolist()
            box = target["boxes"][0]
            return [round(float(v), 6) for v in
                    (box[0] / w, box[1] / h, box[2] / w, box[3] / h)]

        self.assertEqual(normalized(plain), normalized(upscaled))

    def test_a_crop_that_was_not_upscaled_is_untouched(self):
        img = torch.zeros(3, 100, 100)
        det = _StubDetector([[_person_pred([50, 50, 100, 100], 100, 100)]])
        _, targets = crop_batch(
            [img], [_target([[60, 60, 90, 90]], [1])], _people_pipeline(det))

        self.assertTrue(torch.equal(
            targets[0]["boxes"], torch.tensor([[10.0, 10.0, 40.0, 40.0]])))
        self.assertTrue(torch.equal(targets[0]["orig_size"], torch.tensor([50, 50])))


class _StubModelAdapter:
    """Predicts one centered box per crop, normalized to that crop."""

    def predict(self, images, score_threshold=None):
        return [torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.9, 0.0]]) for _ in images]


class ProcessBatchSmokeTests(unittest.TestCase):
    """The inference path still works after the crop_regions refactor."""

    def test_people_detect_first_returns_full_frame_predictions(self):
        img = torch.zeros(3, 100, 100)
        det = _StubDetector([[_person_pred([50, 50, 100, 100], 100, 100)]])
        pipeline = _people_pipeline(
            det, infer_batch_size=4, merge_nms_iou=0.5, map_score_threshold=None, score_threshold=0.0
        )
        pipeline.model_adapter = _StubModelAdapter()

        preds = pipeline.process_batch([img], [{"orig_size": torch.tensor([100, 100])}])
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].shape[1], 6)
        self.assertGreater(preds[0].shape[0], 0)


class BatchPeopleRegionsTests(unittest.TestCase):
    def test_tiled_detection_merges_into_crop_regions(self):
        img = torch.zeros(3, 100, 100)
        # 50% tiles, no overlap -> four 50x50 tiles. The detector fires on the
        # bottom-right tile only, returning a box covering that whole tile.
        preds_per_tile = [[], [], [], [_person_pred([0, 0, 50, 50], 50, 50)]]
        det = _StubDetector(preds_per_tile)
        config = types.SimpleNamespace(
            detector=types.SimpleNamespace(expand_ratio=0.0, min_box_size=0.0, nms_iou=0.5),
            tiling=types.SimpleNamespace(
                tile_width_pct=50.0, tile_height_pct=50.0, overlap=0.0
            ),
        )
        pipeline = BatchPeoplePipeline(
            model_adapter=None, device=torch.device("cpu"), config=config, detector=det
        )
        regions = pipeline.crop_regions([img])
        self.assertEqual(len(regions[0]), 1)
        _, (x0, y0), (w, h) = regions[0][0]
        # Bottom-right tile starts at (50, 50) and the person fills it.
        self.assertEqual((x0, y0), (50, 50))
        self.assertEqual((w, h), (50, 50))


class ParseExpandRatioTests(unittest.TestCase):
    def test_expand_ratio_parsed_from_detector_block(self):
        spec = _parse_pipeline(
            {
                "name": "people_detect_first",
                "detector": {"checkpoint": "/ckpts/person.pt", "expand_ratio": 0.1},
            },
            Path("/tmp"),
        )
        self.assertAlmostEqual(spec.detector_expand_ratio, 0.1)

    def test_expand_ratio_defaults_to_none_when_absent(self):
        spec = _parse_pipeline(
            {"name": "people_detect_first", "detector": {"checkpoint": "/ckpts/person.pt"}},
            Path("/tmp"),
        )
        self.assertIsNone(spec.detector_expand_ratio)

    def test_negative_expand_ratio_rejected(self):
        with self.assertRaises(ValueError):
            _parse_pipeline(
                {
                    "name": "people_detect_first",
                    "detector": {"checkpoint": "/ckpts/person.pt", "expand_ratio": -0.1},
                },
                Path("/tmp"),
            )

    def test_min_box_size_parsed_from_detector_block(self):
        spec = _parse_pipeline(
            {
                "name": "people_detect_first",
                "detector": {"checkpoint": "/ckpts/person.pt", "min_box_size": 224.0},
            },
            Path("/tmp"),
        )
        self.assertAlmostEqual(spec.detector_min_box_size, 224.0)

    def test_min_box_size_defaults_to_none_when_absent(self):
        spec = _parse_pipeline(
            {"name": "people_detect_first", "detector": {"checkpoint": "/ckpts/person.pt"}},
            Path("/tmp"),
        )
        self.assertIsNone(spec.detector_min_box_size)

    def test_negative_min_box_size_rejected(self):
        with self.assertRaises(ValueError):
            _parse_pipeline(
                {
                    "name": "people_detect_first",
                    "detector": {"checkpoint": "/ckpts/person.pt", "min_box_size": -1.0},
                },
                Path("/tmp"),
            )


if __name__ == "__main__":
    unittest.main()
