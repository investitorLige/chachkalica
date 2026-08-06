"""Image loading must be pixel-identical wherever the scaling happens.

``_load_image_tensor`` transfers the decoded uint8 buffer and then does the
permute/scale on the target device, rather than scaling on the host and
transferring float32 — a quarter of the bytes, and the divide moves off the CPU.
That is only safe because the result is bit-identical, which is what these assert.

Run from the chachak directory:  python -m unittest tests.test_preview
"""

import sys
import tempfile
import unittest
from pathlib import Path

_CHACHAK_DIR = Path(__file__).resolve().parent.parent
if str(_CHACHAK_DIR) not in sys.path:
    sys.path.insert(0, str(_CHACHAK_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import preview  # noqa: E402


def _write_image(directory, width=37, height=23):
    """A deterministic RGB image covering the full 0..255 range."""
    from PIL import Image

    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    array[0, 0] = (0, 0, 0)
    array[0, 1] = (255, 255, 255)
    path = Path(directory) / "sample.png"  # lossless, so the pixels survive the round trip
    Image.fromarray(array).save(path)
    return path, array


class LoadImageTensorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path, self.array = _write_image(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_matches_the_host_side_formula(self):
        """The pre-change expression, which friendy_chachkalica/data.py still uses."""
        expected = torch.from_numpy(self.array).permute(2, 0, 1).float() / 255.0

        got = preview._load_image_tensor(self.path, "cpu")

        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, torch.float32)
        self.assertTrue(torch.equal(got, expected))

    def test_is_chw_rgb_in_unit_range(self):
        got = preview._load_image_tensor(self.path, "cpu")

        height, width, _ = self.array.shape
        self.assertEqual(tuple(got.shape), (3, height, width))
        self.assertEqual(float(got.min()), 0.0)
        self.assertEqual(float(got.max()), 1.0)
        # Channel order preserved: pixel (0,1) was written pure white, (0,0) black.
        self.assertTrue(torch.equal(got[:, 0, 1], torch.ones(3)))
        self.assertTrue(torch.equal(got[:, 0, 0], torch.zeros(3)))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
    def test_cuda_is_bit_identical_to_cpu(self):
        """uint8 -> float32 is exact and /255.0 is a single correctly-rounded IEEE
        division, so moving the scale to the device cannot change a pixel."""
        on_cpu = preview._load_image_tensor(self.path, "cpu")
        on_cuda = preview._load_image_tensor(self.path, "cuda")

        self.assertTrue(on_cuda.is_cuda)
        self.assertTrue(torch.equal(on_cuda.cpu(), on_cpu))


if __name__ == "__main__":
    unittest.main()
