import tempfile
import unittest
from pathlib import Path

import torch

from friendy_chachkalica.ml.checkpoint_info import inspect_checkpoint, resolve_trained_size


class ResolveTrainedSizeTests(unittest.TestCase):
    def test_rtdetr_uses_input_max_size_rounded_up_to_the_multiple(self):
        self.assertEqual(
            resolve_trained_size("rtdetr", {"input_max_size": 600, "input_size_multiple": 32}),
            (608, 608),
        )

    def test_rtdetr_falls_back_to_adapter_defaults_when_params_omit_them(self):
        self.assertEqual(resolve_trained_size("rtdetr", {}), (640, 640))

    def test_rtdetr_resize_disabled_has_no_fixed_size(self):
        self.assertIsNone(resolve_trained_size("rtdetr", {"input_max_size": 0}))

    def test_yolox_uses_input_max_size_rounded_up_to_the_multiple(self):
        self.assertEqual(
            resolve_trained_size("yolox", {"input_max_size": 601, "input_size_multiple": 32}),
            (608, 608),
        )

    def test_rfdetr_uses_the_resolution_param(self):
        self.assertEqual(resolve_trained_size("rfdetr", {"resolution": 384}), (384, 384))

    def test_rfdetr_falls_back_to_the_adapter_default(self):
        self.assertEqual(resolve_trained_size("rfdetr", {}), (560, 560))

    def test_fasterrcnn_has_no_single_trained_size(self):
        self.assertIsNone(resolve_trained_size("fasterrcnn", {"min_size": 800, "max_size": 1333}))

    def test_retinanet_has_no_single_trained_size(self):
        self.assertIsNone(resolve_trained_size("retinanet", {}))


class InspectCheckpointTests(unittest.TestCase):
    def _checkpoint(self, tmp: Path, model_name: str, params: dict) -> Path:
        path = tmp / "model.pt"
        torch.save(
            {
                "model_name": model_name,
                "model_config": {"params": params},
                "model_state_dict": {"weight": torch.ones(1)},
            },
            path,
        )
        return path

    def test_reports_arch_and_trained_size_without_exporting_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._checkpoint(Path(tmp), "rfdetr", {"resolution": 512})
            info = inspect_checkpoint(path)
        self.assertEqual(info, {"arch": "rfdetr", "trained_size": [512, 512]})

    def test_variable_size_arch_reports_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._checkpoint(Path(tmp), "fasterrcnn", {})
            info = inspect_checkpoint(path)
        self.assertEqual(info, {"arch": "fasterrcnn", "trained_size": None})

    def test_missing_params_dict_is_treated_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save({"model_name": "rtdetr", "model_state_dict": {}}, path)
            info = inspect_checkpoint(path)
        self.assertEqual(info, {"arch": "rtdetr", "trained_size": [640, 640]})


if __name__ == "__main__":
    unittest.main()
