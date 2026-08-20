"""Tests for the build node's request handling.

Run inside the node image (no GPU needed — nothing here compiles):

    docker run --rm chachkalica-buildnode python -m unittest discover -s buildnode/tests -t .

What matters here is the path rewriting. A build request arrives from another
machine, and every path in it is meaningless on this one; the failure mode if
that goes wrong is either a confusing "file not found" or, worse, silently
resolving against whatever this node happens to have on disk.
"""

import tempfile
import unittest
from pathlib import Path

from buildnode import builds
from buildnode import compile as compile_mod


class MaterializeRequestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = Path(self._tmp.name)
        self.inputs = self.scratch / "inputs"
        self.inputs.mkdir()
        for name in ("model.onnx", "extra_1.onnx"):
            (self.inputs / name).write_bytes(b"x")
        self.model = self.inputs / "model.onnx"

    def _materialize(self, request, detector=None):
        return builds.materialize_request(
            request, inputs=self.inputs, scratch=self.scratch,
            model_path=self.model, detector_path=detector,
        )

    def test_rewrites_model_and_neutralizes_dataset_fields(self):
        out = self._materialize({
            "pipeline": "batch_detect",
            "model_checkpoint": "model.onnx",
            "images": "/data/somewhere/on/the/other/machine",
            "labels": "/data/labels",
            "output_dir": "/data/runs/whatever",
            "classes": ["a"],
        })
        self.assertEqual(out["model_checkpoint"], str(self.model))
        # Nothing here reads a dataset, and the caller's paths don't exist here.
        self.assertEqual(out["images"], ".")
        self.assertIsNone(out["labels"])
        self.assertTrue(out["output_dir"].startswith(str(self.scratch)))

    def test_detector_is_replaced_with_the_resolved_path(self):
        detector = self.scratch / "cache" / "detector.engine"
        detector.parent.mkdir()
        detector.write_bytes(b"plan")
        out = self._materialize(
            {"pipeline": "people_detect_first", "model_checkpoint": "model.onnx",
             # Whatever the caller wrote here is irrelevant; it named a file on
             # ITS disk. The node substitutes what it actually resolved.
             "detector": {"checkpoint": "/app/models/people/best_ckpt.engine",
                          "expand_ratio": 0.1}},
            detector=detector,
        )
        self.assertEqual(out["detector"]["checkpoint"], str(detector))
        self.assertEqual(out["detector"]["expand_ratio"], 0.1)

    def test_detector_needed_but_absent_is_an_error(self):
        with self.assertRaisesRegex(builds.SpecError, "person detector"):
            self._materialize(
                {"pipeline": "people_detect_first", "model_checkpoint": "model.onnx",
                 "detector": {"checkpoint": "whatever"}},
                detector=None,
            )

    def test_extra_checkpoints_resolve_by_basename(self):
        out = self._materialize({
            "pipeline": "batch_detect", "model_checkpoint": "model.onnx",
            "extra_checkpoints": ["extra_1.onnx"],
        })
        self.assertEqual(out["extra_checkpoints"], [str(self.inputs / "extra_1.onnx")])

    def test_extra_checkpoint_that_was_not_uploaded_is_refused(self):
        with self.assertRaisesRegex(builds.SpecError, "not uploaded"):
            self._materialize({
                "pipeline": "batch_detect", "model_checkpoint": "model.onnx",
                "extra_checkpoints": ["never_sent.onnx"],
            })

    def test_a_traversing_extra_checkpoint_is_refused(self):
        """The request is remote input; a path component in it is never legitimate."""
        for hostile in ("../../etc/passwd", "/etc/passwd", "models/model.onnx", ".."):
            with self.subTest(hostile=hostile):
                with self.assertRaises(builds.SpecError):
                    self._materialize({
                        "pipeline": "batch_detect", "model_checkpoint": "model.onnx",
                        "extra_checkpoints": [hostile],
                    })

    def test_a_non_mapping_request_is_refused(self):
        with self.assertRaisesRegex(builds.SpecError, "must be a mapping"):
            self._materialize(["not", "a", "dict"])


class NeedsDetectorTests(unittest.TestCase):
    """The node decides whether to supply a detector from the request alone."""

    def test_person_pipelines(self):
        for pipeline in ("people_detect_first", "batch_people"):
            self.assertTrue(builds._needs_detector({"pipeline": pipeline}))

    def test_plain_pipelines(self):
        self.assertFalse(builds._needs_detector({"pipeline": "batch_detect"}))

    def test_chain_containing_a_person_pipeline(self):
        self.assertTrue(
            builds._needs_detector({"pipeline": "chain",
                                    "chain": ["batch_detect", "batch_people"]}))
        self.assertFalse(
            builds._needs_detector({"pipeline": "chain", "chain": ["batch_detect"]}))


class Fp16RoutingTests(unittest.TestCase):
    """Which fp16 mechanism a remote build uses — the one thing a node can get
    silently wrong.

    An arch whose blanket-cast fp16 is on the safety floor but whose engine is
    rescued by ModelOpt AutoCast (``ARCH_CAST_BACKEND``) must reach
    ``build_engine_from_onnx`` with ``cast_backend="autocast"``, or the builder
    applies its fp16 default — the blanket cast — and this node ships the exact
    engine the floor exists to keep out, labelled "fp16" like any other. The
    admin's TRT export form always sends an explicit precision, so the explicit
    path matters at least as much as ``auto``.

    The real policy table is used deliberately (it is torch-free); only the
    builder, the modelopt probe and the GPU identity are stubbed.
    """

    def setUp(self):
        from friendy_chachkalica.ml.trt_export import arch as arch_policy

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = Path(self._tmp.name)
        self.onnx = self.scratch / "model.onnx"
        self.onnx.write_bytes(b"x")
        (self.scratch / "model.meta.json").write_text('{"arch": "dfine"}')
        self.captured = {}
        self.arch_policy = arch_policy

    def _install_api(self, *, modelopt: str | None):
        def build_engine_from_onnx(onnx_path, engine_path, **kwargs):
            self.captured.update(kwargs)
            Path(engine_path).write_bytes(b"engine")
            return {"precision": kwargs["precision"]}

        api = {
            "build_engine_from_onnx": build_engine_from_onnx,
            "profile_from_meta": lambda meta, **kw: ((640, 640), (640, 640), (640, 640)),
            "get_fp16_op_block": self.arch_policy.get_fp16_op_block,
            "get_fp16_node_block": self.arch_policy.get_fp16_node_block,
            "is_fp16_trusted": self.arch_policy.is_fp16_trusted,
            "get_cast_backend": self.arch_policy.get_cast_backend,
            "resolve_auto_precision": self.arch_policy.resolve_auto_precision,
            "modelopt_version": lambda: modelopt,
            "has_trt_prep": self.arch_policy.has_trt_prep,
        }
        original = compile_mod._trt_export
        compile_mod._trt_export = lambda: api
        self.addCleanup(lambda: setattr(compile_mod, "_trt_export", original))

        from buildnode import gpu as gpu_mod

        original_describe = gpu_mod.describe
        gpu_mod.describe = lambda: {
            "gpu_name": "stub", "compute_capability": None, "driver_version": None,
        }
        self.addCleanup(lambda: setattr(gpu_mod, "describe", original_describe))

    def _compile(self, precision):
        return compile_mod.compile_engine(
            self.onnx, self.scratch / "model.engine", precision=precision,
        )

    def test_explicit_fp16_on_an_autocast_arch_uses_autocast(self):
        self._install_api(modelopt="0.46.0")
        self._compile("fp16")
        self.assertEqual(self.captured["cast_backend"], "autocast")
        self.assertEqual(self.captured["precision"], "fp16")

    def test_explicit_fp16_without_modelopt_is_refused_not_downgraded(self):
        self._install_api(modelopt=None)
        with self.assertRaises(compile_mod.CompileError) as caught:
            self._compile("fp16")
        self.assertIn("nvidia-modelopt", str(caught.exception))
        self.assertNotIn("cast_backend", self.captured)

    def test_auto_without_modelopt_falls_back_to_fp32(self):
        self._install_api(modelopt=None)
        self._compile("auto")
        self.assertEqual(self.captured["precision"], "fp32")

    def test_an_ordinary_fp16_arch_still_uses_the_graph_cast(self):
        (self.scratch / "model.meta.json").write_text('{"arch": "rtdetr"}')
        self._install_api(modelopt=None)
        self._compile("fp16")
        self.assertEqual(self.captured["cast_backend"], "graph_cast")
        self.assertEqual(self.captured["precision"], "fp16")


class TorchFreeTests(unittest.TestCase):
    """The node's whole premise: none of this may need torch.

    Also enforced as a Dockerfile build step, but kept here so it fails in a
    developer's editor rather than only at image build.
    """

    def test_the_build_path_imports_without_torch(self):
        import sys

        import buildnode.builds  # noqa: F401
        import buildnode.compile  # noqa: F401
        import chachak.bundle_export.cli  # noqa: F401

        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
