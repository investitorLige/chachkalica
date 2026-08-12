import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from friendy_chachkalica.ml.adapters.rfdetr import build_rfdetr
from friendy_chachkalica.ml.adapters.rtdetr import build_rtdetr
from friendy_chachkalica.ml.adapters.yolox import _load_checkpoint
from friendy_chachkalica.config import ModelConfig, load_config
from friendy_chachkalica.ml.train import (
    _load_warm_start_state,
    _read_initial_checkpoint,
    _warm_start_build_params,
)


class WarmStartCheckpointTests(unittest.TestCase):
    def test_config_parses_init_checkpoint_separately_from_adapter_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "experiment.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "warm-start-test",
                        "output_dir": "output",
                        "datasets": {
                            "train": [
                                {
                                    "name": "dataset",
                                    "images": "images",
                                    "labels": "labels",
                                    "classes": ["object"],
                                }
                            ]
                        },
                        "models": [
                            {
                                "name": "yolox",
                                "num_classes": 1,
                                "init_checkpoint": "checkpoints/best.pt",
                                "variant": "yolox-s",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(
            config.models[0].init_checkpoint,
            root / "checkpoints" / "best.pt",
        )
        self.assertNotIn("init_checkpoint", config.models[0].params)
        self.assertEqual(config.models[0].params["variant"], "yolox-s")

    def test_checkpoint_architecture_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(
                {
                    "model_name": "yolox",
                    "model_state_dict": {"weight": torch.ones(1)},
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                _read_initial_checkpoint(path, "retinanet")

    def test_non_friendy_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.pth"
            torch.save({"model": {"weight": torch.ones(1)}}, path)

            with self.assertRaisesRegex(ValueError, "not a Friendy"):
                _read_initial_checkpoint(path, "yolox")

    def test_source_topology_is_reused_without_redownloading_for_yolox(self):
        config = ModelConfig(
            name="yolox",
            num_classes=3,
            params={"score_threshold": 0.2},
        )
        state = {
            "model_config": {
                "params": {
                    "variant": "yolox-m",
                    "weights": "https://example.test/yolox_m.pth",
                    "score_threshold": 0.3,
                }
            }
        }

        params = _warm_start_build_params(config, state)

        self.assertEqual(params["variant"], "yolox-m")
        self.assertEqual(params["score_threshold"], 0.2)
        self.assertIs(params["weights"], False)

    def test_structural_variant_mismatch_is_rejected(self):
        config = ModelConfig(
            name="retinanet",
            num_classes=3,
            params={"variant": "resnet50_fpn"},
        )
        state = {
            "model_config": {
                "params": {"variant": "resnet50_fpn_v2", "weights": True}
            }
        }

        with self.assertRaisesRegex(ValueError, "parameter mismatch"):
            _warm_start_build_params(config, state)

    def test_rtdetr_keeps_source_repository_to_rebuild_topology(self):
        config = ModelConfig(name="rtdetr", num_classes=3, params={})
        state = {
            "model_config": {
                "params": {"weights": "PekingU/rtdetr_v2_r18vd"}
            }
        }

        params = _warm_start_build_params(config, state)

        self.assertEqual(params["weights"], "PekingU/rtdetr_v2_r18vd")

    def test_compatible_backbone_loads_while_changed_head_stays_initialized(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 2),
        )
        original_head_weight = model[1].weight.detach().clone()
        source = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 3),
        )
        with torch.no_grad():
            source[0].weight.fill_(7.0)
            source[0].bias.fill_(7.0)

        _load_warm_start_state(
            model,
            {"model_state_dict": source.state_dict()},
            Path("source.pt"),
        )

        self.assertTrue(torch.equal(model[0].weight, torch.full_like(model[0].weight, 7.0)))
        self.assertTrue(torch.equal(model[0].bias, torch.full_like(model[0].bias, 7.0)))
        self.assertTrue(torch.equal(model[1].weight, original_head_weight))

    def test_low_parameter_coverage_is_rejected(self):
        model = torch.nn.Linear(10, 10)
        state = {"model_state_dict": {"bias": torch.ones(10)}}

        with self.assertRaisesRegex(ValueError, "not sufficiently compatible"):
            _load_warm_start_state(model, state, Path("wrong.pt"))


class NativePretrainedFailureTests(unittest.TestCase):
    def test_yolox_native_loader_redirects_friendy_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save(
                {
                    "model_name": "yolox",
                    "model_state_dict": {"weight": torch.ones(1)},
                },
                path,
            )

            with self.assertRaisesRegex(ValueError, "Select it as 'Your model'"):
                _load_checkpoint(torch.nn.Linear(1, 1), str(path))

    def test_rtdetr_requested_weights_fail_instead_of_falling_back(self):
        class BrokenConfig:
            @classmethod
            def from_pretrained(cls, weights, **kwargs):
                raise OSError("offline")

            def __init__(self, **kwargs):
                pass

        class FakeModel:
            def __init__(self, config):
                self.config = config

        with mock.patch(
            "friendy_chachkalica.ml.adapters.rtdetr._load_transformers_rtdetr",
            return_value=(BrokenConfig, FakeModel, object),
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing to silently train"):
                build_rtdetr(num_classes=2, weights="missing-repository")

    def test_rfdetr_requested_weights_fail_instead_of_falling_back(self):
        class BrokenRFDETR:
            def __init__(self, **kwargs):
                raise OSError("offline")

        fake_module = SimpleNamespace(RFDETRBase=BrokenRFDETR)
        with mock.patch(
            "friendy_chachkalica.ml.adapters.rfdetr._load_rfdetr",
            return_value=(fake_module, object, object),
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing to silently train"):
                build_rfdetr(num_classes=2, variant="base", weights=True)

    def test_ecdet_requested_weights_fail_instead_of_falling_back(self):
        from friendy_chachkalica.ml.adapters.ecdet import build_ecdet

        with self.assertRaisesRegex(RuntimeError, "refusing to silently train"):
            build_ecdet(
                num_classes=2, variant="ecdet-s", weights="/nonexistent/ecdet.pth",
                input_max_size=320,
            )

    def test_ecdet_rejects_a_friendy_checkpoint_as_native_weights(self):
        """A promoted Friendy checkpoint must be selected as 'Your model' so the
        checked warm-start loader runs, not fed to the native loader."""
        from friendy_chachkalica.ml.adapters.ecdet import build_ecdet

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            torch.save({"model_name": "ecdet", "model_state_dict": {}}, path)
            with self.assertRaises(RuntimeError) as ctx:
                build_ecdet(
                    num_classes=2, variant="ecdet-s", weights=str(path), input_max_size=320,
                )
            self.assertIn("Friendy training checkpoint", str(ctx.exception.__cause__))

    def test_ecdet_unknown_variant_is_rejected(self):
        from friendy_chachkalica.ml.adapters.ecdet import build_ecdet

        with self.assertRaisesRegex(ValueError, "Unknown ECDet variant"):
            build_ecdet(num_classes=2, variant="ecdet-xl")


class ECDetWarmStartStructuralParamsTests(unittest.TestCase):
    """ECDet's structural params must include input_max_size, not just variant.

    ECTransformer bakes its anchors from ``eval_spatial_size`` at build time, so
    warm-starting the same weights onto a different canvas is a shape mismatch, not
    a resize — and a missing entry in the table silently disables the check
    (``.get(name, ())``).
    """

    def test_variant_and_input_max_size_are_both_structural(self):
        from friendy_chachkalica.ml.train import _WARM_START_STRUCTURAL_PARAMS

        self.assertEqual(
            set(_WARM_START_STRUCTURAL_PARAMS["ecdet"]), {"variant", "input_max_size"}
        )

    def test_mismatched_canvas_is_refused(self):
        from friendy_chachkalica.config import ModelConfig
        from friendy_chachkalica.ml.train import _warm_start_build_params

        checkpoint = {
            "model_config": {
                "name": "ecdet",
                "num_classes": 3,
                "params": {"variant": "ecdet-s", "input_max_size": 640},
            }
        }
        config = ModelConfig(
            name="ecdet", num_classes=3,
            params={"variant": "ecdet-s", "input_max_size": 1280},
        )
        with self.assertRaisesRegex(ValueError, "input_max_size"):
            _warm_start_build_params(config, checkpoint)

    def test_matching_topology_is_reused_without_redownloading(self):
        from friendy_chachkalica.config import ModelConfig
        from friendy_chachkalica.ml.train import _warm_start_build_params

        checkpoint = {
            "model_config": {
                "name": "ecdet",
                "num_classes": 3,
                "params": {"variant": "ecdet-s", "input_max_size": 640, "weights": True},
            }
        }
        config = ModelConfig(
            name="ecdet", num_classes=3, params={"variant": "ecdet-s"},
        )
        build_params = _warm_start_build_params(config, checkpoint)
        self.assertEqual(build_params["variant"], "ecdet-s")
        self.assertEqual(build_params["input_max_size"], 640)
        # ecdet's topology comes from the variant, so the pretrained download is
        # skipped and the Friendy state is restored on top instead.
        self.assertIs(build_params["weights"], False)


if __name__ == "__main__":
    unittest.main()
