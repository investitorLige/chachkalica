import unittest
from pathlib import Path

import torch

try:
    from friendy_chachkalica import data
    from friendy_chachkalica.config import DatasetConfig, _parse_augmentation
except ImportError:
    import data
    from config import DatasetConfig, _parse_augmentation


H, W = 120, 200


def _sample(box, label=3):
    """A black image with a white rectangle exactly where ``box`` says.

    Lets a test check that the transformed box still sits on the transformed
    pixels, rather than only that the arithmetic matches a hand-written copy.
    """
    x1, y1, x2, y2 = box
    image = torch.zeros(3, H, W)
    image[:, y1:y2, x1:x2] = 1.0
    target = {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "labels": torch.tensor([label], dtype=torch.int64),
        "iscrowd": torch.zeros(1, dtype=torch.int64),
        "area": torch.tensor([float((x2 - x1) * (y2 - y1))]),
    }
    return image, target


def _empty_sample():
    return torch.rand(3, H, W), {
        "boxes": torch.zeros(0, 4),
        "labels": torch.zeros(0, dtype=torch.int64),
        "iscrowd": torch.zeros(0, dtype=torch.int64),
        "area": torch.zeros(0),
    }


def _pixel_box(image):
    ys, xs = torch.nonzero(image[0] > 0.5, as_tuple=True)
    return [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]


def _scale_crop(image, target):
    return data._random_scale_crop(
        image, target,
        data.TrainAugmentations.SCALE_RANGE,
        data.TrainAugmentations.MIN_BOX_VISIBILITY,
        data.TrainAugmentations.MIN_BOX_SIDE_PX,
    )


class HorizontalFlipTests(unittest.TestCase):
    def test_box_follows_the_flipped_pixels(self):
        image, target = _sample([20, 30, 70, 90])
        image, target = data._horizontal_flip(image, target)
        self.assertEqual(target["boxes"].tolist(), [[130.0, 30.0, 180.0, 90.0]])
        self.assertEqual(_pixel_box(image), [130, 30, 180, 90])
        self.assertEqual(target["labels"].tolist(), [3])
        self.assertEqual(target["area"].tolist(), [3000.0])

    def test_empty_target(self):
        image, target = _empty_sample()
        flipped, target = data._horizontal_flip(image, target)
        self.assertTrue(torch.equal(flipped, torch.flip(image, dims=[2])))
        self.assertEqual(target["boxes"].shape, (0, 4))


class RandomScaleCropTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_box_follows_the_cropped_pixels_and_stays_in_bounds(self):
        # A box in the middle survives any 60-100% window, so every trial must
        # keep it, on the object, inside the image, with per-box fields aligned.
        for _ in range(300):
            image, target = _sample([60, 30, 140, 90])
            out_image, out = _scale_crop(image, target)
            self.assertEqual(out_image.shape, (3, H, W))
            self.assertEqual(len(out["boxes"]), 1)
            box = out["boxes"][0].tolist()
            for got, want in zip(box, _pixel_box(out_image)):
                self.assertLessEqual(abs(got - want), 1.0)
            self.assertTrue(0 <= box[0] < box[2] <= W and 0 <= box[1] < box[3] <= H, box)
            self.assertEqual(out["labels"].tolist(), [3])
            self.assertEqual(len(out["iscrowd"]), 1)
            self.assertAlmostEqual(
                out["area"][0].item(), (box[2] - box[0]) * (box[3] - box[1]), places=2)

    def test_box_cut_off_by_the_crop_is_dropped_with_its_fields(self):
        # A sliver in the top-left corner plus a central box: whenever the crop
        # window starts right of/below the sliver, the sliver must go and the
        # central box must keep its own label.
        dropped = 0
        for _ in range(300):
            image, target = _sample([60, 30, 140, 90], label=1)
            target["boxes"] = torch.tensor([[0, 0, 4, 4], [60, 30, 140, 90]], dtype=torch.float32)
            target["labels"] = torch.tensor([7, 1])
            target["iscrowd"] = torch.zeros(2, dtype=torch.int64)
            target["area"] = torch.tensor([16.0, 4800.0])
            _, out = _scale_crop(image, target)
            n = len(out["boxes"])
            self.assertEqual(len(out["labels"]), n)
            self.assertEqual(len(out["iscrowd"]), n)
            self.assertEqual(len(out["area"]), n)
            self.assertIn(1, out["labels"].tolist())
            if n == 1:
                dropped += 1
                self.assertEqual(out["labels"].tolist(), [1])
        self.assertGreater(dropped, 0)

    def test_empty_target(self):
        image, target = _empty_sample()
        out_image, out = _scale_crop(image, target)
        self.assertEqual(out_image.shape, (3, H, W))
        self.assertEqual(out["boxes"].shape, (0, 4))


class TrainAugmentationsTests(unittest.TestCase):
    def _flip_rate(self, fractions, trials=4000):
        torch.manual_seed(0)
        aug = data.TrainAugmentations(fractions)
        flipped = 0
        for _ in range(trials):
            image, target = _sample([0, 0, 10, 10])
            _, out = aug(image, target)
            flipped += out["boxes"][0, 0].item() != 0
        return flipped / trials

    def test_fraction_is_the_per_sample_probability(self):
        self.assertAlmostEqual(self._flip_rate({"hflip": 0.3}), 0.3, delta=0.03)
        self.assertEqual(self._flip_rate({"hflip": 1.0}), 1.0)

    def test_same_seed_gives_same_augmentations(self):
        aug = data.TrainAugmentations({"hflip": 0.5, "scale_crop": 0.5})

        def run():
            torch.manual_seed(42)
            return [aug(*_sample([60, 30, 140, 90]))[1]["boxes"].tolist() for _ in range(20)]

        self.assertEqual(run(), run())

    def test_unconfigured_or_zero_builds_no_transform(self):
        def config(augmentation):
            return DatasetConfig(name="d", images=Path("."), labels=None, classes={0: "a"},
                                 role="train", augmentation=augmentation)

        self.assertIsNone(data.build_train_augmentations(config({})))
        self.assertIsNone(data.build_train_augmentations(config({"hflip": 0.0})))
        built = data.build_train_augmentations(config({"scale_crop": 0.4}))
        self.assertEqual((built.hflip, built.scale_crop), (0.0, 0.4))


class ParseAugmentationTests(unittest.TestCase):
    def test_valid_block(self):
        self.assertEqual(
            _parse_augmentation({"hflip": 0.5, "scale_crop": "0.3"}, "train[0]", "train"),
            {"hflip": 0.5, "scale_crop": 0.3},
        )
        self.assertEqual(_parse_augmentation(None, "train[0]", "train"), {})
        self.assertEqual(_parse_augmentation({"hflip": 0}, "train[0]", "train"), {})

    def test_rejections(self):
        for value, role in [
            ({"hflip": 0.5}, "val"),
            ({"rotate": 0.5}, "train"),
            ({"hflip": 1.5}, "train"),
            ({"hflip": "lots"}, "train"),
            (["hflip"], "train"),
        ]:
            with self.subTest(value=value, role=role):
                with self.assertRaises(ValueError):
                    _parse_augmentation(value, "train[0]", role)


if __name__ == "__main__":
    unittest.main()
