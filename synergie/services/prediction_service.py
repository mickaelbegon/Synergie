from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import constants
from synergie.config import JUMP_WINDOW_FRAMES, SUCCESS_WINDOW_FRAMES, SUCCESS_WINDOW_START, TYPE_WINDOW_FRAMES, TYPE_WINDOW_START
from synergie.services.hdf5_archive_service import hdf5_segment_paths, load_segment_dataframe


class PredictionService:
    """Load jump models once and provide consistent prediction helpers."""

    def __init__(self, type_model, success_model) -> None:
        self.type_model = type_model
        self.success_model = success_model

    @classmethod
    def from_paths(cls, type_model_path: str, success_model_path: str) -> "PredictionService":
        """Load compatible models from disk for inference."""
        from core.model import model

        return cls(
            model.load_model(type_model_path, for_training=False),
            model.load_model(success_model_path, for_training=False),
        )

    @staticmethod
    def numeric_features(dataframe: pd.DataFrame) -> np.ndarray:
        """Return the model temporal channels as finite `float32` values."""
        numeric = dataframe[constants.fields_to_keep].apply(pd.to_numeric, errors="coerce")
        return np.nan_to_num(numeric.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def predict_segments(self, segments: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
        """Predict type and success for full jump windows."""
        valid_indices: list[int] = []
        type_windows = []
        success_windows = []
        predicted_type = np.full(len(segments), 8.0)
        predicted_success = np.full(len(segments), 2.0)

        for index, segment in enumerate(segments):
            features = self.numeric_features(segment)
            if len(features) < JUMP_WINDOW_FRAMES:
                continue
            valid_indices.append(index)
            type_windows.append(features[TYPE_WINDOW_START : TYPE_WINDOW_START + TYPE_WINDOW_FRAMES])
            success_windows.append(features[SUCCESS_WINDOW_START : SUCCESS_WINDOW_START + SUCCESS_WINDOW_FRAMES])

        if not valid_indices:
            return predicted_type, predicted_success

        type_scores = self._predict(self.type_model, np.asarray(type_windows, dtype=np.float32))
        success_scores = self._predict(self.success_model, np.asarray(success_windows, dtype=np.float32))
        for prediction_index, segment_index in enumerate(valid_indices):
            predicted_type[segment_index] = int(np.argmax(type_scores[prediction_index]))
            predicted_success[segment_index] = int(np.argmax(success_scores[prediction_index]))
        return predicted_type, predicted_success

    def prefill_annotation_rows(self, annotation_rows: pd.DataFrame) -> dict:
        """Prefill trainable labels for existing annotation rows."""
        frame = annotation_rows.copy()
        if frame.empty:
            return {"rows": frame, "updated": 0, "skipped": 0}

        segments: list[pd.DataFrame] = []
        valid_indices: list[int] = []
        for index, row in frame.iterrows():
            path = Path(str(row.get("path", "")))
            if not path.exists() and str(path).replace("\\", "/") not in hdf5_segment_paths():
                continue
            segment = load_segment_dataframe(path)
            if len(segment) < JUMP_WINDOW_FRAMES:
                continue
            segments.append(segment)
            valid_indices.append(index)

        if not valid_indices:
            return {"rows": frame, "updated": 0, "skipped": int(len(frame))}

        predicted_type, predicted_success = self.predict_segments(segments)
        for prediction_index, row_index in enumerate(valid_indices):
            frame.at[row_index, "type"] = int(predicted_type[prediction_index])
            frame.at[row_index, "success"] = int(predicted_success[prediction_index])
        return {
            "rows": frame,
            "updated": len(valid_indices),
            "skipped": int(len(frame) - len(valid_indices)),
        }

    def _predict(self, trained_model, temporal_features: np.ndarray):
        """Support both current two-input and historical one-input models."""
        model_inputs = getattr(trained_model, "inputs", None)
        if model_inputs is not None and len(model_inputs) > 1:
            scalar_features = np.zeros((len(temporal_features), 2), dtype=np.float32)
            return trained_model.predict(
                {"temporal_input": temporal_features, "scalar_input": scalar_features},
            )
        return trained_model.predict(temporal_features)
