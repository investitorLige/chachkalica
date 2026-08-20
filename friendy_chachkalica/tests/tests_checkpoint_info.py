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

    def test_dfine_uses_input_max_size_rounded_up_to_the_multiple(self):
        # Same geometry contract as rtdetr (DFineAdapter is a near-verbatim clone).
        self.assertEqual(
            resolve_trained_size("dfine", {"input_max_size": 600, "input_size_multiple": 32}),
            (608, 608),
        )

    def test_dfine_falls_back_to_adapter_defaults_when_params_omit_them(self):
        self.assertEqual(resolve_trained_size("dfine", {}), (640, 640))

    def test_dfine_resize_disabled_has_no_fixed_size(self):
        self.assertIsNone(resolve_trained_size("dfine", {"input_max_size": 0}))

    def test_yolox_uses_input_max_size_rounded_up_to_the_multiple(self):
        self.assertEqual(
            resolve_trained_size("yolox", {"input_max_size": 601, "input_size_multiple": 32}),
            (608, 608),
        )

    def test_rfdetr_uses_the_resolution_param(self):
        self.assertEqual(resolve_trained_size("rfdetr", {"resolution": 384}), (384, 384))

    def test_rfdetr_falls_back_to_the_adapter_default(self):
        self.assertEqual(resolve_trained_size("rfdetr", {}), (560, 560))

    def test_ecdet_uses_input_max_size_rounded_up_to_the_multiple(self):
        self.assertEqual(
            resolve_trained_size("ecdet", {"input_max_size": 700, "input_size_multiple": 32}),
            (704, 704),
        )

    def test_ecdet_falls_back_to_the_native_640(self):
        self.assertEqual(resolve_trained_size("ecdet", {}), (640, 640))

    def test_ecdet_supports_the_high_resolution_variant(self):
        # Upstream documents 1280 as the high-res alternative to the native 640.
        self.assertEqual(resolve_trained_size("ecdet", {"input_max_size": 1280}), (1280, 1280))

    def test_ecdet_resize_disabled_has_no_fixed_size(self):
        self.assertIsNone(resolve_trained_size("ecdet", {"input_max_size": 0}))

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
        self.assertEqual(
            {k: v for k, v in info.items() if k not in ("auto_precision", "auto_cast_backend")},
            {"arch": "rfdetr", "trained_size": [512, 512], "fp16_trusted": True, "batch_aware": True},
        )

    def test_variable_size_arch_reports_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._checkpoint(Path(tmp), "fasterrcnn", {})
            info = inspect_checkpoint(path)
        self.assertEqual(
            {k: v for k, v in info.items() if k not in ("auto_precision", "auto_cast_backend")},
            {"arch": "fasterrcnn", "trained_size": None, "fp16_trusted": True, "batch_aware": False},
        )

    def test_missing_params_dict_is_treated_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save({"model_name": "rtdetr", "model_state_dict": {}}, path)
            info = inspect_checkpoint(path)
        self.assertEqual(
            {k: v for k, v in info.items() if k not in ("auto_precision", "auto_cast_backend")},
            {"arch": "rtdetr", "trained_size": [640, 640], "fp16_trusted": True, "batch_aware": True},
        )

    def test_dfine_reports_arch_trained_size_and_an_untrusted_plain_fp16(self):
        # dfine IS in UNTRUSTED_FP16 (2026-08-18): the trained-checkpoint gate was
        # finally run on real ustc-community COCO weights and its blanket-cast fp16
        # engine fails — scores collapse while boxes stay put. Distinct from
        # adapters/dfine.py's supports_amp=False, which is the training-time AMP
        # question, not this TRT-export-time flag.
        #
        # batch_aware is True now that ``onnx_export/arch/dfine.py`` exists: its
        # wrapper does the top-k per batch row, so an engine really can be built with
        # a wider profile. This assertion read False while the exporter didn't exist.
        #
        # auto_precision is asserted separately below, because it depends on whether
        # nvidia-modelopt is installed on the machine running the test.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._checkpoint(Path(tmp), "dfine", {"input_max_size": 640})
            info = inspect_checkpoint(path)
        self.assertEqual(
            {k: v for k, v in info.items() if k not in ("auto_precision", "auto_cast_backend")},
            {"arch": "dfine", "trained_size": [640, 640], "fp16_trusted": False,
             "batch_aware": True},
        )

    def test_dfine_auto_precision_follows_whether_modelopt_is_installed(self):
        """``auto_precision`` is not ``fp16 if fp16_trusted else fp32``.

        dfine's plain fp16 is floored, but ModelOpt AutoCast's mixed graph passes
        the gate — so "auto" reaches for fp16-via-AutoCast where ModelOpt exists and
        falls back to the fp32 floor where it does not. The export form defaults its
        precision select from this, which is why it is worth pinning.
        """
        from friendy_chachkalica.ml.trt_export.modelopt_cast import modelopt_version

        with tempfile.TemporaryDirectory() as tmp:
            path = self._checkpoint(Path(tmp), "dfine", {"input_max_size": 640})
            info = inspect_checkpoint(path)
        if modelopt_version() is None:
            self.assertEqual(info["auto_precision"], "fp32")
        else:
            self.assertEqual(info["auto_precision"], "fp16")
            self.assertEqual(info["auto_cast_backend"], "autocast")

    def test_reports_fp16_untrusted_for_an_arch_on_the_fp32_floor(self):
        """The TRT export form reads this to default its precision select.

        ``build_engine`` already floors ``precision="auto"`` to fp32 for these, but
        the admin form names a precision explicitly and would otherwise hand the
        operator the fp16 engine that fails the parity gate.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = self._checkpoint(Path(tmp), "ecdet", {"input_max_size": 640})
            info = inspect_checkpoint(path)
        self.assertEqual(
            {k: v for k, v in info.items() if k not in ("auto_precision", "auto_cast_backend")},
            {"arch": "ecdet", "trained_size": [640, 640], "fp16_trusted": False, "batch_aware": True},
        )


if __name__ == "__main__":
    unittest.main()
