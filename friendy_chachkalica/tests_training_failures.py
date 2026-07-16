import unittest
from pathlib import Path

import torch

from friendy_chachkalica.train import (
    ExperimentTrainingError,
    _raise_if_training_failed,
    _require_finite_loss,
)


class ExperimentFailurePropagationTests(unittest.TestCase):
    def test_successful_results_return_normally(self):
        results = [{"run_index": 0, "run_name": "working-model", "last_epoch": 2}]

        _raise_if_training_failed(results, Path("results.yaml"))

    def test_failed_results_raise_after_preserving_all_results(self):
        results = [
            {"run_index": 0, "run_name": "working-model", "last_epoch": 2},
            {
                "run_index": 1,
                "run_name": "broken-model",
                "error_type": "RuntimeError",
                "error": "CUDA out of memory",
            },
        ]

        with self.assertRaises(ExperimentTrainingError) as caught:
            _raise_if_training_failed(results, Path("output/results.yaml"))

        error = caught.exception
        self.assertIs(error.results, results)
        self.assertEqual(error.failures, [results[1]])
        self.assertEqual(error.results_path, Path("output/results.yaml"))
        self.assertIn("broken-model", str(error))
        self.assertIn("output/results.yaml", str(error))


class FiniteLossTests(unittest.TestCase):
    def test_finite_scalar_loss_is_accepted(self):
        _require_finite_loss(
            torch.tensor(1.25),
            {"classification": torch.tensor(0.5)},
            phase="training",
        )

    def test_nonfinite_loss_identifies_phase_and_components(self):
        with self.assertRaises(FloatingPointError) as caught:
            _require_finite_loss(
                torch.tensor(float("nan")),
                {
                    "classification": torch.tensor(float("nan")),
                    "box": torch.tensor(0.5),
                },
                phase="validation",
            )

        message = str(caught.exception)
        self.assertIn("Non-finite validation loss", message)
        self.assertIn("classification", message)
        self.assertNotIn("box", message)

    def test_non_scalar_loss_is_rejected(self):
        with self.assertRaises(TypeError):
            _require_finite_loss(
                torch.tensor([1.0, 2.0]),
                {},
                phase="training",
            )


if __name__ == "__main__":
    unittest.main()
