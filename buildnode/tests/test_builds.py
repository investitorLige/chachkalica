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
