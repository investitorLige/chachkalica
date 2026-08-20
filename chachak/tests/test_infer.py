"""Tests for artifact routing in ``load_checkpoint_adapter`` and the batch-safe
chunking ``infer_in_chunks`` gives every adapter."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_CHACHAK_DIR = Path(__file__).resolve().parent.parent
if str(_CHACHAK_DIR) not in sys.path:
    sys.path.insert(0, str(_CHACHAK_DIR))

from infer import infer_in_chunks, load_checkpoint_adapter  # noqa: E402


class TensorRTEngineLoadTest(unittest.TestCase):
    def test_engine_uses_tensor_rt_adapter(self):
        with patch("trt_infer.load_trt_adapter", return_value=("adapter", {})) as load:
            result = load_checkpoint_adapter("/models/people/person.engine", "cuda")

        self.assertEqual(result, ("adapter", {}))
        load.assert_called_once_with(Path("/models/people/person.engine"), "cuda")


class _RecordingAdapter:
    """Stands in for ``TrtAdapter``/``OnnxAdapter``: records the size of every
    chunk ``infer_in_chunks`` actually hands it, returns one prediction per image.
    """

    def __init__(self, max_batch=None):
        # Only a TrtAdapter carries ``_model.max_batch``; leaving this unset (the
        # onnx/torch case) is what proves ``infer_in_chunks`` stays uncapped.
        if max_batch is not None:
            self._model = SimpleNamespace(max_batch=max_batch)
        self.chunk_sizes = []

    def predict(self, images, score_threshold=None):
        self.chunk_sizes.append(len(images))
        return [object() for _ in images]


class InferInChunksBatchCapTest(unittest.TestCase):
    def test_clamps_to_the_adapters_batch_1_profile(self):
        """A batch-1 TensorRT engine — the default profile for every arch, and
        still the only option for retinanet/fasterrcnn (see
        ``trt_export.arch.BATCH_AWARE_ARCHS``) — must never see more than one
        image per call, no matter what chunk_size the caller configured — this
        is what stops `people_detect_first` crashing on any frame with more than
        one person cropped for such a bundle."""
        adapter = _RecordingAdapter(max_batch=1)
        images = list(range(5))

        results = infer_in_chunks(adapter, images, chunk_size=4)

        self.assertEqual(len(results), 5)
        self.assertEqual(adapter.chunk_sizes, [1, 1, 1, 1, 1])

    def test_never_exceeds_the_adapters_wider_profile_either(self):
        adapter = _RecordingAdapter(max_batch=2)
        images = list(range(5))

        infer_in_chunks(adapter, images, chunk_size=4)

        self.assertEqual(adapter.chunk_sizes, [2, 2, 1])

    def test_uncapped_adapter_keeps_the_caller_supplied_chunk_size(self):
        """No ``_model.max_batch`` (onnx/trained-torch adapters) means no cap —
        unchanged behavior from before this clamp existed."""
        adapter = _RecordingAdapter(max_batch=None)
        images = list(range(5))

        infer_in_chunks(adapter, images, chunk_size=4)

        self.assertEqual(adapter.chunk_sizes, [4, 1])

    def test_a_chunk_size_wider_than_the_profile_still_never_crashes(self):
        """Regression case: infer_batch_size/detector.batch_size (default 4) wider
        than a batch-1 bundle's profile used to reach the adapter as one 3-image
        stack and raise; it must now just take 3 calls of 1."""
        adapter = _RecordingAdapter(max_batch=1)

        infer_in_chunks(adapter, list(range(3)), chunk_size=4)

        self.assertEqual(adapter.chunk_sizes, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
