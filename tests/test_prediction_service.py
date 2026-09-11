import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import constants
from synergie.config import JUMP_WINDOW_FRAMES
from synergie.services.prediction_service import PredictionService


class _FakeModel:
    def __init__(self, outputs, *, two_inputs: bool = False):
        self.outputs = outputs
        self.inputs = [object(), object()] if two_inputs else [object()]
        self.received = None

    def predict(self, values, verbose=0):
        self.received = values
        return self.outputs


class PredictionServiceTests(unittest.TestCase):
    def _segment(self):
        return pd.DataFrame(
            {
                column: ["1"] * JUMP_WINDOW_FRAMES
                for column in constants.fields_to_keep
            }
        )

    def test_predict_segments_normalizes_numeric_inputs(self):
        type_model = _FakeModel(np.array([[0.1, 0.9]]), two_inputs=True)
        success_model = _FakeModel(np.array([[0.8, 0.2]]), two_inputs=True)

        predicted_type, predicted_success = PredictionService(type_model, success_model).predict_segments([self._segment()])

        self.assertEqual(predicted_type.tolist(), [1.0])
        self.assertEqual(predicted_success.tolist(), [0.0])
        self.assertEqual(type_model.received["temporal_input"].dtype, np.float32)
        self.assertEqual(type_model.received["scalar_input"].shape, (1, 2))

    def test_prefill_annotation_rows_reads_segments_once(self):
        type_model = _FakeModel(np.array([[0.9, 0.1]]), two_inputs=True)
        success_model = _FakeModel(np.array([[0.1, 0.9]]), two_inputs=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            segment_path = Path(tmpdir) / "segment.csv"
            self._segment().to_csv(segment_path, index=False)
            rows = pd.DataFrame([{"path": segment_path.as_posix(), "type": 8, "success": 2}])

            result = PredictionService(type_model, success_model).prefill_annotation_rows(rows)

            self.assertEqual(result["updated"], 1)
            self.assertEqual(result["rows"].iloc[0]["type"], 0)
            self.assertEqual(result["rows"].iloc[0]["success"], 1)
