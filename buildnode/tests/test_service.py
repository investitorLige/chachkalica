"""Tests for the node's HTTP surface: auth, spec validation, build lifecycle."""

import json
import os
import unittest

os.environ.setdefault("BUILDNODE_TOKEN", "unit-test-token")

from fastapi.testclient import TestClient  # noqa: E402

from buildnode import auth, service  # noqa: E402

TOKEN = {"Authorization": "Bearer unit-test-token"}


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(service.app)

    def test_health_requires_a_token(self):
        """/health is behind auth too — an open capability probe tells a scanner
        exactly which GPU it just found."""
        self.assertEqual(self.client.get("/health").status_code, 401)

    def test_a_wrong_token_is_rejected(self):
        resp = self.client.get("/health", headers={"Authorization": "Bearer nope"})
        self.assertEqual(resp.status_code, 401)

    def test_a_non_bearer_header_is_rejected(self):
        resp = self.client.get("/health", headers={"Authorization": "unit-test-token"})
        self.assertEqual(resp.status_code, 401)

    def test_the_right_token_is_accepted(self):
        self.assertEqual(self.client.get("/health", headers=TOKEN).status_code, 200)

    def test_a_node_with_no_token_configured_refuses_everything(self):
        """503, not 401: the node is misconfigured, the caller is not at fault."""
        original = os.environ.get("BUILDNODE_TOKEN")
        os.environ["BUILDNODE_TOKEN"] = ""
        try:
            self.assertEqual(auth.configured_token(), "")
            resp = self.client.get("/health", headers=TOKEN)
            self.assertEqual(resp.status_code, 503)
        finally:
            os.environ["BUILDNODE_TOKEN"] = original


class HealthShapeTests(unittest.TestCase):
    def test_reports_the_fields_django_stores(self):
        body = TestClient(service.app).get("/health", headers=TOKEN).json()
        for field in ("status", "node_version", "gpu_name", "tensorrt_version",
                      "busy", "queue_len", "person_detector"):
            self.assertIn(field, body)
        self.assertIn(body["status"], ("ok", "degraded"))


class SpecValidationTests(unittest.TestCase):
    """``_parse_spec`` rejects a bad request before anything is uploaded."""

    def _parse(self, **overrides):
        spec = {"name": "thing", "fmt": "engine", "precision": "fp16",
                "request": {"pipeline": "batch_detect"}}
        spec.update(overrides)
        return service._parse_spec(json.dumps(spec))

    def test_gpu_infer_defaults_to_false(self):
        self.assertFalse(self._parse()["gpu_infer"])

    def test_gpu_infer_survives_the_spec_whitelist(self):
        """``_parse_spec`` rebuilds its return dict by hand, so an un-added key is silently
        dropped -- the node would then accept the request and quietly build a bundle without
        ``infer_gpu.py``. This is the test that catches that regression."""
        self.assertTrue(self._parse(gpu_infer=True)["gpu_infer"])

    def test_a_valid_spec_parses(self):
        parsed = self._parse()
        self.assertEqual(parsed["name"], "thing")
        self.assertEqual(parsed["model_file"], "model.onnx")

    def test_a_name_with_a_path_in_it_is_refused(self):
        """The name becomes a directory inside the scratch dir."""
        for bad in ("../escape", "a/b", "..", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(Exception):
                    self._parse(name=bad)

    def test_an_unknown_format_is_refused(self):
        with self.assertRaises(Exception):
            self._parse(fmt="tensorflow")

    def test_an_unknown_precision_is_refused(self):
        with self.assertRaises(Exception):
            self._parse(precision="int4")

    def test_a_malformed_input_hw_is_refused(self):
        for bad in ([640], [0, 640], [-1, -1], "640x640"):
            with self.subTest(bad=bad):
                with self.assertRaises(Exception):
                    self._parse(input_hw=bad)

    def test_input_hw_is_normalized_to_ints(self):
        self.assertEqual(self._parse(input_hw=["640", "480"])["input_hw"], (640, 480))

    def test_a_request_that_is_not_an_object_is_refused(self):
        with self.assertRaises(Exception):
            self._parse(request="batch_detect")

    def test_non_json_is_refused(self):
        with self.assertRaises(Exception):
            service._parse_spec("{not json")


class UnknownBuildTests(unittest.TestCase):
    def test_polling_an_unknown_build_is_404(self):
        """Django's client turns this into a terminal "the node lost it"."""
        resp = TestClient(service.app).get("/builds/nope", headers=TOKEN)
        self.assertEqual(resp.status_code, 404)

    def test_downloading_an_unknown_build_is_404(self):
        resp = TestClient(service.app).get("/builds/nope/artifact", headers=TOKEN)
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
