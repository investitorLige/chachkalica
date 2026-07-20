"""Tests for load_pipeline_config: parsing, defaults, and validation.

Run from the chachak directory:  python -m unittest tests.test_config
"""

import sys
import tempfile
import unittest
from pathlib import Path

_CHACHAK_DIR = Path(__file__).resolve().parent.parent
if str(_CHACHAK_DIR) not in sys.path:
    sys.path.insert(0, str(_CHACHAK_DIR))

import yaml  # noqa: E402

from config import load_pipeline_config  # noqa: E402


def write_config(body):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(body, tmp)
    tmp.close()
    return tmp.name


BASE = {
    "pipeline": "batch_detect",
    "model_checkpoint": "model.pt",
    "images": "imgs",
    "labels": "lbls",
    "classes": ["a", "b"],
}


class LoadConfigTest(unittest.TestCase):
    def test_minimal_batch_detect_loads_with_defaults(self):
        config = load_pipeline_config(write_config(BASE))
        self.assertEqual(config.pipeline, "batch_detect")
        self.assertEqual(config.classes, {0: "a", 1: "b"})
        self.assertEqual(config.tiling.tile_width_pct, 50.0)
        self.assertEqual(config.tiling.tile_height_pct, 50.0)
        self.assertEqual(config.tiling.overlap, 0.2)
        self.assertEqual(config.infer_batch_size, 4)
        self.assertTrue(config.model_checkpoint.is_absolute())

    def test_fixed_pixel_tile_size_is_parsed(self):
        body = {**BASE, "tiling": {"tile_size_px": 640, "overlap": 0.1}}
        config = load_pipeline_config(write_config(body))
        self.assertEqual(config.tiling.tile_size_px, 640)

    def test_nonpositive_fixed_pixel_tile_size_is_rejected(self):
        body = {**BASE, "tiling": {"tile_size_px": 0}}
        with self.assertRaises(ValueError):
            load_pipeline_config(write_config(body))

    def test_paths_resolve_relative_to_config_file(self):
        config = load_pipeline_config(write_config(BASE))
        # images "imgs" resolves against the temp config's own directory.
        self.assertEqual(config.images.name, "imgs")
        self.assertTrue(config.images.is_absolute())

    def test_classes_mapping_is_accepted(self):
        body = {**BASE, "classes": {0: "cat", 3: "dog"}}
        config = load_pipeline_config(write_config(body))
        self.assertEqual(config.classes, {0: "cat", 3: "dog"})

    def test_unknown_pipeline_rejected(self):
        body = {**BASE, "pipeline": "nope"}
        with self.assertRaises(ValueError):
            load_pipeline_config(write_config(body))

    def test_missing_required_field_rejected(self):
        body = dict(BASE)
        del body["model_checkpoint"]
        with self.assertRaises(ValueError):
            load_pipeline_config(write_config(body))

    def test_bad_overlap_rejected(self):
        body = {**BASE, "tiling": {"overlap": 1.0}}
        with self.assertRaises(ValueError):
            load_pipeline_config(write_config(body))

    def test_people_pipeline_requires_detector_checkpoint(self):
        body = {**BASE, "pipeline": "people_detect_first"}
        with self.assertRaises(ValueError):
            load_pipeline_config(write_config(body))

    def test_people_pipeline_with_detector_ok(self):
        body = {
            **BASE,
            "pipeline": "people_detect_first",
            "detector": {"checkpoint": "det.pt", "person_class_id": 2},
        }
        config = load_pipeline_config(write_config(body))
        self.assertTrue(config.detector.checkpoint.is_absolute())
        self.assertEqual(config.detector.person_class_id, 2)

    def test_chain_requires_nonempty_chain_list(self):
        body = {**BASE, "pipeline": "chain"}
        with self.assertRaises(ValueError):
            load_pipeline_config(write_config(body))

    def test_chain_with_people_member_requires_detector(self):
        body = {**BASE, "pipeline": "chain", "chain": ["batch_detect", "people_detect_first"]}
        with self.assertRaises(ValueError):
            load_pipeline_config(write_config(body))

    def test_chain_valid(self):
        body = {
            **BASE,
            "pipeline": "chain",
            "chain": ["batch_detect", "people_detect_first"],
            "detector": {"checkpoint": "det.pt"},
        }
        config = load_pipeline_config(write_config(body))
        self.assertEqual(config.chain, ["batch_detect", "people_detect_first"])


if __name__ == "__main__":
    unittest.main()
