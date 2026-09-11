"""Image loading must be pixel-identical wherever the scaling happens, and the
per-call timing must stay opt-in.

``_load_image_tensor`` transfers the decoded uint8 buffer and then does the
permute/scale on the target device, rather than scaling on the host and
transferring float32 — a quarter of the bytes, and the divide moves off the CPU.
That is only safe because the result is bit-identical, which is what these assert.

The second group covers ``timings``: the trainer's ``/predict_image`` passes a
dict through to have the stages measured (which is where the admin's
dataset-inference fps comes from), and a caller that passes nothing must get the
same boxes with no synchronization added.

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


class _StubPipeline:
    """Returns one fixed detection, ignoring its input — the timing path is what
    is under test, not a model."""

    def __init__(self):
        self.calls = 0

    def process_batch(self, images, targets):
        self.calls += 1
        # [cx, cy, w, h, confidence, class_id], normalized — Friendy's layout.
        return [torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.9, 0.0]])]


class PredictOneTimingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path, _array = _write_image(self._tmp.name)
        self.info = {"train_classes": {0: "helmet"}}

    def tearDown(self):
        self._tmp.cleanup()

    def test_boxes_are_unchanged_when_no_timings_dict_is_passed(self):
        boxes = preview.predict_one(_StubPipeline(), self.info, self.path, "cpu")

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["class_name"], "helmet")
        self.assertAlmostEqual(boxes[0]["confidence"], 0.9, places=5)

    def test_a_timings_dict_is_filled_with_the_three_stages(self):
        timings = {}

        boxes = preview.predict_one(
            _StubPipeline(), self.info, self.path, "cpu", timings=timings)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(set(timings), {"load_ms", "infer_ms", "format_ms"})
        self.assertTrue(all(value >= 0 for value in timings.values()))

    def test_the_score_threshold_still_filters_inside_the_timed_stage(self):
        timings = {}

        boxes = preview.predict_one(
            _StubPipeline(), self.info, self.path, "cpu",
            score_threshold=0.95, timings=timings)

        self.assertEqual(boxes, [])
        self.assertIn("infer_ms", timings)


if __name__ == "__main__":
    unittest.main()
