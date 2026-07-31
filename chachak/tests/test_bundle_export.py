"""Tests for the bundle exporter's contract and its vendored runtime.

Run from the chachak directory:  python -m unittest tests.test_bundle_export

Exporting a real bundle needs a checkpoint and the training stack, so these cover
the two things that break silently instead:

* the round trip through ``pipeline.json`` — every non-path field of a
  ``PipelineConfig`` survives, ``--conf`` lands where the pipeline actually reads
  it, and batch caps are applied;
* the vendored runtime's import closure — every module copied into a bundle must
  be importable with only stdlib + torch present, which is exactly the assumption
  that ``ml/val.py`` quietly violated.
"""

import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

_CHACHAK_DIR = Path(__file__).resolve().parent.parent
if str(_CHACHAK_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_CHACHAK_DIR.parent))
if str(_CHACHAK_DIR) not in sys.path:
    sys.path.insert(0, str(_CHACHAK_DIR))

import yaml  # noqa: E402

from chachak.bundle_export import cli  # noqa: E402  (pulls chachak.pipeline -> torch)
from chachak.bundle_export import manifest as contract  # noqa: E402
from chachak.bundle_export import vendor  # noqa: E402
from config import PipelineConfig, load_pipeline_config  # noqa: E402

REQUEST = {
    "pipeline": "people_detect_first",
    "name": "ppe",
    "model_checkpoint": "model.pt",
    "images": "imgs",
    "labels": "lbls",
    "classes": ["helmet", "vest"],
    "output_dir": "out",
    "device": "cpu",
    "infer_batch_size": 8,
    "score_threshold": 0.42,
    "merge_nms_iou": 0.55,
    "tiling": {"tile_size_px": 512, "overlap": 0.3, "nms_iou": 0.45},
    "detector": {
        "checkpoint": "detector.pt",
        "score_threshold": 0.6,
        "person_class_id": 3,
        "expand_ratio": 0.2,
        "nms_iou": 0.4,
        "min_box_size": 96,
        "batch_size": 2,
    },
}

ARTIFACTS = {
    "onnx": {
        "model": "models/model.onnx",
        "extra_models": [],
        "detector": "models/detector.onnx",
        "max_batch": {"model": None, "detector": None},
    }
}

# `manifest["pipeline"]` (once flat) plus the wrapper keys build_manifest adds
# around the spread-in PipelineConfig fields — everything else in a built
# manifest is expected to be a carried-over PipelineConfig field.
# `score_threshold` is a wrapper key too: it's a top-level duplicate of
# defaults.conf added for mercury_watchroom's pipeline_config.py, which reads
# score_threshold at the top level and never looks under `defaults`.
_WRAPPER_KEYS = {
    "schema_version",
    "bundle",
    "pipeline",
    "artifacts",
    "defaults",
    "score_threshold",
    "provenance",
}


def _config(**overrides):
    body = {**REQUEST, **overrides}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(body, handle)
    return load_pipeline_config(handle.name)


def _manifest(config=None, artifacts=None, default_format="onnx", conf=0.42):
    return contract.build_manifest(
        config or _config(),
        artifacts=artifacts or ARTIFACTS,
        default_format=default_format,
        default_conf=conf,
        provenance={"source_request": "/somewhere/request.yaml"},
        created_utc="2026-01-01T00:00:00Z",
    )


def _paths(detector=True):
    return {
        "model": Path("/bundle/models/model.onnx"),
        "detector": Path("/bundle/models/detector.onnx") if detector else None,
        "extra_models": [],
    }


class ManifestFieldCoverageTest(unittest.TestCase):
    def test_every_non_path_pipeline_field_is_carried(self):
        """A field added to PipelineConfig must not silently vanish from bundles.

        The manifest is flat: every carried PipelineConfig field (other than
        `pipeline` itself, which becomes the top-level string) is a top-level
        manifest key, spread in alongside the wrapper keys build_manifest adds.
        """
        manifest = _manifest()
        carried = set(manifest) - _WRAPPER_KEYS
        promoted = {"score_threshold", "map_score_threshold"}  # -> defaults.conf
        expected = {
            field.name
            for field in fields(PipelineConfig)
            if field.name not in contract._PATH_KEYS
            and field.name not in promoted
            and field.name != "pipeline"
        }
        self.assertEqual(expected, carried)

    def test_pipeline_is_a_plain_string_not_a_nested_object(self):
        """mercury_watchroom's validator requires a top-level string, not an
        object — the whole point of the flat schema."""
        manifest = _manifest()
        self.assertIsInstance(manifest["pipeline"], str)
        self.assertEqual(manifest["pipeline"], "people_detect_first")

    def test_host_paths_are_not_carried(self):
        manifest = _manifest()
        for key in contract._PATH_KEYS:
            self.assertNotIn(key, manifest)
        self.assertNotIn("checkpoint", manifest["detector"])

    def test_tuned_geometry_survives_verbatim(self):
        manifest = _manifest()
        self.assertEqual(manifest["tiling"]["tile_size_px"], 512)
        self.assertEqual(manifest["tiling"]["overlap"], 0.3)
        self.assertEqual(manifest["tiling"]["nms_iou"], 0.45)
        self.assertEqual(manifest["merge_nms_iou"], 0.55)
        self.assertEqual(manifest["detector"]["expand_ratio"], 0.2)
        self.assertEqual(manifest["detector"]["min_box_size"], 96.0)
        self.assertEqual(manifest["detector"]["person_class_id"], 3)

    def test_artifacts_is_flat_not_keyed_by_format(self):
        """mercury_watchroom's Bundle.clean() reads artifacts.model as a flat
        string; artifacts must never be nested under a format key."""
        artifacts = _manifest()["artifacts"]
        self.assertEqual(artifacts["model"], "models/model.onnx")
        self.assertEqual(artifacts["detector"], "models/detector.onnx")
        self.assertNotIn("onnx", artifacts)
        self.assertNotIn("engine", artifacts)

    def test_artifacts_still_carries_extra_models_and_max_batch(self):
        """Not required by mercury_watchroom, but load-bearing for infer.py
        (multi-model chains, and TensorRT batch-1 profiles) — kept as flat keys
        under artifacts rather than nested per-format."""
        artifacts = _manifest()["artifacts"]
        self.assertEqual(artifacts["extra_models"], [])
        self.assertEqual(artifacts["max_batch"], {"model": None, "detector": None})

    def test_defaults_capture_both_thresholds(self):
        defaults = _manifest()["defaults"]
        self.assertEqual(defaults["conf"], 0.42)
        self.assertEqual(defaults["detector_conf"], 0.6)

    def test_default_conf_prefers_map_score_threshold(self):
        """chachak's inference threshold is map_score_threshold when it is set."""
        from pipeline import _inference_score_threshold

        config = _config(score_threshold=0.001, map_score_threshold=0.35)
        self.assertEqual(_inference_score_threshold(config), 0.35)

    def test_build_rejects_a_default_format_it_has_no_artifacts_for(self):
        with self.assertRaises(contract.ManifestError):
            _manifest(default_format="engine")


class ManifestRoundTripTest(unittest.TestCase):
    """The manifest must rebuild a request the real chachak loader accepts."""

    def _reload(self, manifest, **kwargs):
        from config import pipeline_config_from_dict

        request = contract.pipeline_request_from_manifest(
            manifest, _paths(), fmt="onnx",
            images="/data/images", output_dir="/data/out", **kwargs,
        )
        return pipeline_config_from_dict(request, Path("/bundle"))

    def test_round_trip_preserves_the_pipeline(self):
        original = _config()
        reloaded = self._reload(_manifest(original))
        self.assertEqual(reloaded.pipeline, original.pipeline)
        self.assertEqual(reloaded.classes, original.classes)
        self.assertEqual(reloaded.tiling, original.tiling)
        self.assertEqual(reloaded.merge_nms_iou, original.merge_nms_iou)
        self.assertEqual(reloaded.detector.expand_ratio, original.detector.expand_ratio)
        self.assertEqual(reloaded.detector.min_box_size, original.detector.min_box_size)
        self.assertEqual(reloaded.detector.person_class_id, original.detector.person_class_id)

    def test_artifacts_replace_the_host_checkpoints(self):
        reloaded = self._reload(_manifest())
        self.assertEqual(reloaded.model_checkpoint, Path("/bundle/models/model.onnx"))
        self.assertEqual(reloaded.detector.checkpoint, Path("/bundle/models/detector.onnx"))

    def test_no_conf_reproduces_the_exported_thresholds(self):
        reloaded = self._reload(_manifest(conf=0.42))
        self.assertEqual(reloaded.score_threshold, 0.42)
        self.assertEqual(reloaded.detector.score_threshold, 0.6)

    def test_conf_becomes_the_single_model_stage_threshold(self):
        """--conf must survive _inference_score_threshold, which prefers
        map_score_threshold — so that field has to come back unset."""
        from pipeline import _inference_score_threshold

        reloaded = self._reload(_manifest(), conf=0.7)
        self.assertIsNone(reloaded.map_score_threshold)
        self.assertEqual(_inference_score_threshold(reloaded), 0.7)

    def test_conf_does_not_move_the_detector_threshold(self):
        reloaded = self._reload(_manifest(), conf=0.9)
        self.assertEqual(reloaded.detector.score_threshold, 0.6)

    def test_detector_conf_is_separately_overridable(self):
        reloaded = self._reload(_manifest(), detector_conf=0.15)
        self.assertEqual(reloaded.detector.score_threshold, 0.15)

    def test_batch_size_moves_both_stages(self):
        reloaded = self._reload(_manifest(), batch_size=5)
        self.assertEqual(reloaded.infer_batch_size, 5)
        self.assertEqual(reloaded.detector.batch_size, 5)

    def test_batch_is_clamped_to_the_artifact_profile(self):
        """A TensorRT engine built with a batch-1 profile must not be handed a
        stacked batch; clamping is throughput-only, so results are unaffected."""
        capped = {
            "engine": {
                "model": "models/model.engine",
                "extra_models": [],
                "detector": "models/detector.engine",
                "max_batch": {"model": 1, "detector": 2},
            }
        }
        manifest = _manifest(artifacts=capped, default_format="engine")
        request = contract.pipeline_request_from_manifest(
            manifest, _paths(), fmt="engine", images="i", output_dir="o", batch_size=8
        )
        self.assertEqual(request["infer_batch_size"], 1)
        self.assertEqual(request["detector"]["batch_size"], 2)

    def test_uncapped_format_keeps_the_exported_batch(self):
        request = contract.pipeline_request_from_manifest(
            _manifest(), _paths(), fmt="onnx", images="i", output_dir="o"
        )
        self.assertEqual(request["infer_batch_size"], 8)


class FormatSelectionTest(unittest.TestCase):
    """The old per-format ``artifacts`` layout let a bundle carry both an ONNX
    and an engine set and fall back between them at run time. The flat manifest
    keeps only the primary format's paths (see build_manifest's `primary_format`),
    so `resolve_format` now just reports which one that is, and a mismatched
    `--format` request is a hard error rather than a silent fallback — there is
    nothing left to fall back to.
    """

    def _both(self):
        artifacts = {
            "onnx": ARTIFACTS["onnx"],
            "engine": {
                "model": "models/model.engine",
                "extra_models": [],
                "detector": "models/detector.engine",
                "max_batch": {"model": 1, "detector": 1},
            },
        }
        return _manifest(artifacts=artifacts, default_format="engine")

    def test_reports_the_bundles_single_carried_format(self):
        self.assertEqual(contract.resolve_format(self._both()), "engine")
        self.assertEqual(contract.resolve_format(_manifest()), "onnx")

    def test_matching_request_is_accepted(self):
        self.assertEqual(contract.resolve_format(self._both(), "engine"), "engine")
        self.assertEqual(contract.resolve_format(_manifest(), "onnx"), "onnx")

    def test_mismatched_request_now_fails_loudly_instead_of_falling_back(self):
        with self.assertRaises(contract.ManifestError):
            contract.resolve_format(self._both(), "onnx")
        with self.assertRaises(contract.ManifestError):
            contract.resolve_format(_manifest(), "engine")


class RoleFormatSelectionTest(unittest.TestCase):
    """Which format each role's artifact is carried as (``cli._role_formats``).

    Roles are resolved independently. Requiring one format for every role made the
    normal person-crop bundle impossible: the shipped detector is a prebuilt
    ``.engine``, which cannot be turned back into ONNX, so ``--format onnx`` aborted
    with "No single format is available for every role" every time — and the
    operator had no ``.pt`` to point at instead.
    """

    def _exported(self, **roles):
        # Mirrors _export_role's return shape, paths only.
        return {
            role: {"paths": {fmt: Path(f"/b/models/{role}.{fmt}") for fmt in formats}}
            for role, formats in roles.items()
        }

    def test_onnx_request_with_all_onnx_roles(self):
        exported = self._exported(model=["onnx"], detector=["onnx"])
        self.assertEqual(
            cli._role_formats(exported, want_engine=False),
            {"model": "onnx", "detector": "onnx"},
        )

    def test_onnx_request_keeps_a_prebuilt_engine_detector(self):
        exported = self._exported(model=["onnx"], detector=["engine"])
        self.assertEqual(
            cli._role_formats(exported, want_engine=False),
            {"model": "onnx", "detector": "engine"},
        )

    def test_engine_request_prefers_engine_everywhere_it_exists(self):
        exported = self._exported(model=["onnx", "engine"], detector=["onnx", "engine"])
        self.assertEqual(
            cli._role_formats(exported, want_engine=True),
            {"model": "engine", "detector": "engine"},
        )

    def test_engine_request_keeps_an_onnx_only_role(self):
        exported = self._exported(model=["onnx", "engine"], detector=["onnx"])
        self.assertEqual(
            cli._role_formats(exported, want_engine=True),
            {"model": "engine", "detector": "onnx"},
        )

    def test_extra_models_are_resolved_too(self):
        exported = self._exported(
            model=["onnx"], extra_1=["onnx"], detector=["engine"])
        self.assertEqual(
            cli._role_formats(exported, want_engine=False),
            {"model": "onnx", "extra_1": "onnx", "detector": "engine"},
        )

    def test_a_mixed_bundle_reports_the_models_format(self):
        # The manifest carries one flat artifact set and resolve_format reads the
        # model artifact's extension, so a mixed bundle is an "onnx" bundle that
        # happens to run its detector on an engine.
        mixed = _manifest(artifacts={
            "onnx": {
                "model": "models/model.onnx",
                "extra_models": [],
                "detector": "models/detector.engine",
                "max_batch": {"model": None, "detector": 1},
            }
        })
        self.assertEqual(contract.resolve_format(mixed), "onnx")
        self.assertEqual(mixed["artifacts"]["detector"], "models/detector.engine")
        self.assertEqual(contract.batch_caps(mixed)["detector"], 1)


class SchemaGuardTest(unittest.TestCase):
    def _write(self, body):
        path = Path(tempfile.mkdtemp()) / "pipeline.json"
        path.write_text(json.dumps(body))
        return path

    def test_loads_a_current_manifest(self):
        path = self._write(_manifest())
        self.assertEqual(contract.load_manifest(path)["bundle"]["name"], "ppe")

    def test_refuses_a_future_schema_loudly(self):
        path = self._write({**_manifest(), "schema_version": contract.SCHEMA_VERSION + 1})
        with self.assertRaises(contract.ManifestError):
            contract.load_manifest(path)

    def test_refuses_a_missing_file(self):
        with self.assertRaises(contract.ManifestError):
            contract.load_manifest(Path(tempfile.mkdtemp()) / "absent.json")

    def test_reports_a_missing_artifact(self):
        directory = Path(tempfile.mkdtemp())
        (directory / "pipeline.json").write_text(json.dumps(_manifest()))
        with self.assertRaises(contract.ManifestError):
            contract.artifact_paths(_manifest(), directory, "onnx")


class VendoredRuntimeTest(unittest.TestCase):
    """The bundle's whole point is running without the training toolkit."""

    @classmethod
    def setUpClass(cls):
        cls.runtime = Path(tempfile.mkdtemp()) / "runtime"
        cls.entries = vendor.vendor_runtime(cls.runtime, include_trt=False)

    def test_writes_the_expected_top_level_entries(self):
        self.assertEqual(self.entries, ["bundle_manifest.py", "chachak", "onnx_infer"])

    def test_copies_chachak_verbatim(self):
        """Copied, not reimplemented — a bundle can't drift from the pipeline."""
        for name in vendor._CHACHAK_MODULES:
            self.assertEqual(
                (_CHACHAK_DIR / name).read_bytes(),
                (self.runtime / "chachak" / name).read_bytes(),
                f"{name} differs from the source module",
            )

    def test_generates_the_two_modules_that_reach_into_the_trainer(self):
        for name in ("__init__.py", "_friendy.py"):
            body = (self.runtime / "chachak" / name).read_text()
            self.assertIn("DO NOT EDIT", body)
        shim = (self.runtime / "chachak" / "_friendy.py").read_text()
        # It names the training package in prose and error messages; what matters is
        # that every symbol it re-exports comes from the vendored copies.
        self.assertNotIn("import friendy_chachkalica", shim)
        self.assertNotIn("from friendy_chachkalica", shim)
        self.assertIn("from ._vendor", shim)

    def test_the_generated_init_omits_the_batch_eval_entrypoint(self):
        """run.py needs the Friendy dataloader, so it is not bundled — and the
        package __init__ must not import it or every bundle import would fail."""
        self.assertFalse((self.runtime / "chachak" / "run.py").exists())
        self.assertNotIn(
            "from .run import", (self.runtime / "chachak" / "__init__.py").read_text()
        )

    def test_no_vendored_module_imports_the_training_package(self):
        """The failure mode this catches: vendoring a module whose module-level
        imports pull in the trainer, which only shows up at bundle run time."""
        for path in self.runtime.rglob("*.py"):
            self.assertNotIn(
                "from friendy_chachkalica",
                path.read_text(),
                f"{path.relative_to(self.runtime)} imports the training package",
            )

    def test_the_vendored_import_closure_is_complete(self):
        """Import every vendored module the way a bundle does — with only the
        runtime directory on sys.path, so nothing can resolve against the repo."""
        import importlib
        import subprocess

        modules = sorted(
            ".".join(path.relative_to(self.runtime).with_suffix("").parts).replace(
                ".__init__", ""
            )
            for path in self.runtime.rglob("*.py")
        )
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "for name in %r:\n"
            "    __import__(name)\n" % (str(self.runtime), modules)
        )
        # A subprocess, not importlib here: this process already has the training
        # package importable, which would mask exactly the bug under test.
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        del importlib

    def test_excludes_tests_and_caches(self):
        self.assertFalse((self.runtime / "onnx_infer" / "tests").exists())
        self.assertEqual([], list(self.runtime.rglob("__pycache__")))


if __name__ == "__main__":
    unittest.main()
