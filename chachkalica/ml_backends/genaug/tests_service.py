"""genaug-backend service tests, with the mock editor (no GPU, no weights).

Run inside the genaug-backend image:

    docker compose run --rm -T --no-deps genaug-backend \\
        python -m unittest ml_backends.genaug.tests_service -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

_TMP = tempfile.TemporaryDirectory()
os.environ["GENAUG_DATA_ROOT"] = _TMP.name
os.environ["GENAUG_IDLE_UNLOAD_S"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from ml_backends.genaug import service  # noqa: E402


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(service.app)
        self.root = Path(_TMP.name)
        self.src = self.root / "src.jpg"
        Image.new("RGB", (333, 211), (120, 160, 200)).save(self.src)

    def edit(self, **overrides):
        body = {"image_path": str(self.src), "output_path": str(self.root / "out" / "a.png"),
                "prompt": "night cctv, sensor noise", "seed": 1, "editor": "mock"}
        body.update(overrides)
        return self.client.post("/edit", json=body)

    def test_edit_writes_same_size_output(self):
        resp = self.edit()
        self.assertEqual(resp.status_code, 200, resp.text)
        with Image.open(self.root / "out" / "a.png") as out:
            self.assertEqual(out.size, (333, 211))
        self.assertEqual(resp.json()["editor"]["label"], "mock")
        self.assertEqual(self.client.get("/health").json()["loaded"]["label"], "mock")

    def test_exif_rotation_is_applied_before_editing(self):
        rotated = self.root / "rot.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6  # "rotate 90 CW to display"
        Image.new("RGB", (300, 100), (10, 200, 10)).save(rotated, exif=exif)
        resp = self.edit(image_path=str(rotated), output_path=str(self.root / "out" / "r.png"))
        self.assertEqual(resp.status_code, 200, resp.text)
        with Image.open(self.root / "out" / "r.png") as out:
            self.assertEqual(out.size, (100, 300))

    def test_paths_outside_the_data_root_are_refused(self):
        self.assertEqual(self.edit(output_path="/tmp/elsewhere.png").status_code, 400)
        self.assertEqual(self.edit(image_path="/etc/hostname").status_code, 400)

    def test_missing_weights_name_the_cache_directory(self):
        resp = self.edit(editor="firered")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("fetch_genaug_weights", resp.json()["detail"])

    def test_deterministic_per_seed(self):
        self.edit(output_path=str(self.root / "out" / "s1.png"))
        self.edit(output_path=str(self.root / "out" / "s2.png"))
        a = (self.root / "out" / "s1.png").read_bytes()
        b = (self.root / "out" / "s2.png").read_bytes()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
