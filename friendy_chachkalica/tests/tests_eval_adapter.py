"""Tests for evaluating something other than a ``.pt``.

``load_eval_adapter`` is the one place that decides *what* a request's
``checkpoint_path`` names — a torch checkpoint to rebuild, or a finished ONNX /
TensorRT artifact to hand to chachak's loader. Both the routing and its refusals
are worth pinning: getting the class space wrong relabels every prediction
silently, and picking the wrong file scores a model nobody asked about.

Nothing here loads a real artifact; the chachak loader is mocked.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from friendy_chachkalica.ml import eval_checkpoint as eval_checkpoint_mod
from friendy_chachkalica.ml.eval_checkpoint import _clamp_batch_size, load_eval_adapter


class StubAdapter:
    def __init__(self, max_batch=None):
        if max_batch is not None:
            self._model = mock.Mock(max_batch=max_batch)

    def predict(self, images, score_threshold=None):
        return [torch.zeros((0, 6)) for _ in images]


class LoadEvalAdapterTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _chachak_loader(self, info):
        """Patch the loader ``load_eval_adapter`` imports for artifact paths."""
        import chachak.infer

        return mock.patch.object(
            chachak.infer, "load_checkpoint_adapter",
            return_value=(StubAdapter(), info))

    def test_an_onnx_goes_through_chachaks_adapter(self):
        artifact = self.tmp / "m.onnx"
        artifact.write_bytes(b"onnx")
        info = {"model_name": "yolox", "num_classes": 2, "params": {},
                "train_classes": {0: "helmet", 1: "vest"}}

        with self._chachak_loader(info) as loader:
            adapter, got = load_eval_adapter(artifact, "cpu")

        loader.assert_called_once()
        self.assertIsInstance(adapter, StubAdapter)
        self.assertEqual(got["train_classes"], {0: "helmet", 1: "vest"})

    def test_an_engine_goes_through_chachaks_adapter_too(self):
        artifact = self.tmp / "m.engine"
        artifact.write_bytes(b"plan")
        info = {"model_name": "yolox", "num_classes": 1, "params": {},
                "train_classes": {0: "helmet"}}

        with self._chachak_loader(info):
            _adapter, got = load_eval_adapter(artifact, "cpu")

        self.assertEqual(got["model_name"], "yolox")

    def test_an_artifact_without_class_names_is_refused(self):
        """Its ids would have to be read as the eval dataset's, which is a guess
        that mislabels every box instead of failing."""
        artifact = self.tmp / "m.onnx"
        artifact.write_bytes(b"onnx")
        info = {"model_name": "yolox", "num_classes": 2, "params": {},
                "train_classes": {}}

        with self._chachak_loader(info):
            with self.assertRaises(ValueError) as caught:
                load_eval_adapter(artifact, "cpu")
        self.assertIn("meta.json", str(caught.exception))

    def test_a_checkpoint_beside_an_export_still_loads_the_checkpoint(self):
        """chachak's loader prefers a sibling ``.onnx`` — right for serving,
        wrong here, where the request named the ``.pt``."""
        checkpoint = self.tmp / "best.pt"
        (self.tmp / "best.onnx").write_bytes(b"onnx")
        torch.save({
            "model_name": "yolox",
            "model_config": {"num_classes": 2, "params": {}},
            "model_state_dict": {},
            "train_dataset": {"classes": ["helmet", "vest"]},
        }, checkpoint)

        built = StubAdapter()
        built.model = mock.Mock()
        built.to = mock.Mock(return_value=built)
        with mock.patch.object(eval_checkpoint_mod, "build_model", return_value=built) as build:
            with self._chachak_loader({}) as chachak_loader:
                adapter, info = load_eval_adapter(checkpoint, "cpu")

        chachak_loader.assert_not_called()
        build.assert_called_once_with("yolox", num_classes=2)
        self.assertIs(adapter, built)
        self.assertEqual(info["train_classes"], {0: "helmet", 1: "vest"})


class ClampBatchSizeTest(unittest.TestCase):
    def test_an_adapter_with_no_ceiling_keeps_the_requested_batch(self):
        self.assertEqual(_clamp_batch_size(StubAdapter(), 4), 4)

    def test_an_engines_profile_narrows_the_batch(self):
        """A batch-1 engine is the historical default for every arch; handing it
        4 stacked images fails the call rather than splitting it."""
        self.assertEqual(_clamp_batch_size(StubAdapter(max_batch=1), 4), 1)

    def test_a_wider_profile_is_left_alone(self):
        self.assertEqual(_clamp_batch_size(StubAdapter(max_batch=8), 4), 4)


if __name__ == "__main__":
    unittest.main()
