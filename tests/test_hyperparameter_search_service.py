import tempfile
import unittest
from pathlib import Path

from synergie.services.hyperparameter_search_service import (
    load_optimized_model_parameters,
    save_optimized_model_parameters,
)


class HyperparameterSearchServiceTests(unittest.TestCase):
    def test_saves_and_loads_best_parameters_per_task_and_architecture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "optimized.json"

            save_optimized_model_parameters(
                "success",
                "tcn",
                {
                    "trial": 3,
                    "best_val_accuracy": 0.91,
                    "parameters": {"filters": 64, "dropout": 0.2, "batch_size": 16},
                },
                path=path,
            )

            loaded = load_optimized_model_parameters("success", "tcn", path=path)

            self.assertEqual(loaded["trial"], 3)
            self.assertEqual(loaded["best_val_accuracy"], 0.91)
            self.assertEqual(loaded["parameters"]["filters"], 64)
            self.assertIsNone(load_optimized_model_parameters("type", "tcn", path=path))
