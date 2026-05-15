import unittest

import numpy as np
import pandas as pd

import constants
from core.data_treatment.data_generation.modelPredictor import ModelPredictor
from synergie.config import JUMP_WINDOW_FRAMES


class _FakeModel:
    def __init__(self, outputs, *, two_inputs: bool = False):
        self.outputs = outputs
        self.inputs = [object(), object()] if two_inputs else [object()]
        self.received = None

    def predict(self, values):
        self.received = values
        return self.outputs


class ModelPredictorTests(unittest.TestCase):
    def test_predict_uses_numeric_feature_matrix_and_scalar_placeholders(self):
        frame = pd.DataFrame(
            {
                column: ["1"] * JUMP_WINDOW_FRAMES
                for column in constants.fields_to_keep
            }
        )
        type_model = _FakeModel(np.array([[0.1, 0.9]]), two_inputs=True)
        success_model = _FakeModel(np.array([[0.8, 0.2]]), two_inputs=True)

        predicted_type, predicted_success = ModelPredictor(type_model, success_model).predict([frame])

        self.assertEqual(predicted_type.tolist(), [1.0])
        self.assertEqual(predicted_success.tolist(), [0.0])
        self.assertEqual(type_model.received["temporal_input"].dtype, np.float32)
        self.assertEqual(type_model.received["temporal_input"].shape[-1], len(constants.fields_to_keep))
        self.assertEqual(type_model.received["scalar_input"].shape, (1, 2))

    def test_predict_keeps_unknown_labels_for_invalid_length(self):
        short_frame = pd.DataFrame(
            {
                column: [1.0]
                for column in constants.fields_to_keep
            }
        )
        model = _FakeModel(np.empty((0, 2)))

        predicted_type, predicted_success = ModelPredictor(model, model).predict([short_frame])

        self.assertEqual(predicted_type.tolist(), [8.0])
        self.assertEqual(predicted_success.tolist(), [2.0])
