import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

try:
    from friendy_chachkalica.data import YoloDetectionDataset
except ImportError:
    from data import YoloDetectionDataset


def _write_valid_image(path, size=(16, 12)):
    Image.new("RGB", size, color=(0, 0, 0)).save(path)


class CorruptImageFallbackTests(unittest.TestCase):
    """A truncated/corrupt image on disk must not kill the whole run.

    Regression for a DataLoader worker crash: one bad JPEG in a fine-tune
    dataset (People_v0.1-ft-ft-2, run 55) took down a run that was hours into
    training. __getitem__ now logs the bad path once and substitutes a
    neighboring sample instead of raising.
    """

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.images_dir = self.tmp_dir / "images"
        self.images_dir.mkdir()

    def _make_dataset(self, num_good, bad_index):
        for i in range(num_good):
            _write_valid_image(self.images_dir / f"{i:03d}.jpg")
        # Truncated file: a real image header with the data cut off, same
        # failure shape as the OSError seen in production.
        seed_path = self.tmp_dir / "seed.jpg"
        _write_valid_image(seed_path)
        full_bytes = seed_path.read_bytes()
        bad_path = self.images_dir / f"{bad_index:03d}.jpg"
        bad_path.write_bytes(full_bytes[: max(1, len(full_bytes) // 4)])
        return YoloDetectionDataset(self.images_dir, None, {0: "person"})

    def test_corrupt_image_falls_back_to_neighboring_sample(self):
        dataset = self._make_dataset(num_good=3, bad_index=1)
        self.assertEqual(len(dataset), 3)

        # Index 1 is corrupt; __getitem__ must return a usable sample from a
        # neighboring index instead of raising.
        image_tensor, target = dataset[1]
        self.assertEqual(image_tensor.shape[0], 3)
        self.assertNotEqual(Path(target["image_path"]).name, "001.jpg")

        # The bad path is remembered so repeat epochs don't re-log it.
        self.assertIn(dataset.image_paths[1], dataset._warned_bad_paths)

    def test_all_corrupt_dataset_raises_instead_of_looping_forever(self):
        _write_valid_image(self.tmp_dir / "seed.jpg")
        truncated = (self.tmp_dir / "seed.jpg").read_bytes()[:5]
        for i in range(3):
            (self.images_dir / f"{i:03d}.jpg").write_bytes(truncated)

        dataset = YoloDetectionDataset(self.images_dir, None, {0: "person"})
        with self.assertRaises(OSError):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
