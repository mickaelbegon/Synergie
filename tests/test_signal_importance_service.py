import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from synergie.services.signal_importance_service import compute_signal_importance


class SignalImportanceServiceTests(unittest.TestCase):
    def test_compute_signal_importance_reports_scalar_features(self):
        dataset = SimpleNamespace(
            temporal_features_test=np.zeros((4, 6, 10), dtype=np.float32),
            scalar_features_test=np.array(
                [[50.0, 1.50], [55.0, 1.55], [60.0, 1.60], [65.0, 1.65]],
                dtype=np.float32,
            ),
            labels_test=np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
        )

        class FakeLoader:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_success_data(self):
                return dataset

        class FakePredictor:
            def predict(self, inputs, verbose=0):
                scalar = inputs["scalar_input"]
                return np.array([[0.9, 0.1] if row[0] < 58 else [0.1, 0.9] for row in scalar], dtype=np.float32)

        with mock.patch("core.model.training.loader.Loader", FakeLoader):
            with mock.patch("core.model.model.load_model", return_value=FakePredictor()):
                result = compute_signal_importance("success", "unused", model_path="fake.keras", repeats=1, temporal_windows=2)

        self.assertEqual([item["label"] for item in result["scalar_importance"]], ["weight", "height"])
        self.assertEqual(result["model_path"], "fake.keras")
