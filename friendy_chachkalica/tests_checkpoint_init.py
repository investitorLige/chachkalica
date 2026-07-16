import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from friendy_chachkalica.adapters.rfdetr import build_rfdetr
from friendy_chachkalica.adapters.rtdetr import build_rtdetr
from friendy_chachkalica.adapters.yolox import _load_checkpoint
from friendy_chachkalica.config import ModelConfig, load_config
from friendy_chachkalica.train import (
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
            "friendy_chachkalica.adapters.rtdetr._load_transformers_rtdetr",
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
            "friendy_chachkalica.adapters.rfdetr._load_rfdetr",
            return_value=(fake_module, object, object),
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing to silently train"):
                build_rfdetr(num_classes=2, variant="base", weights=True)


if __name__ == "__main__":
    unittest.main()
