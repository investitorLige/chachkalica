"""Tests for artifact routing in ``load_checkpoint_adapter``."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_CHACHAK_DIR = Path(__file__).resolve().parent.parent
if str(_CHACHAK_DIR) not in sys.path:
    sys.path.insert(0, str(_CHACHAK_DIR))

from infer import load_checkpoint_adapter  # noqa: E402


class TensorRTEngineLoadTest(unittest.TestCase):
    def test_engine_uses_tensor_rt_adapter(self):
        with patch("trt_infer.load_trt_adapter", return_value=("adapter", {})) as load:
            result = load_checkpoint_adapter("/models/people/person.engine", "cuda")

        self.assertEqual(result, ("adapter", {}))
        load.assert_called_once_with(Path("/models/people/person.engine"), "cuda")


if __name__ == "__main__":
    unittest.main()
