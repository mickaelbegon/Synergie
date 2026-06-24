import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synergie.services.annotation_plot_service import (
    annotation_gyro_column,
    annotation_segment_available,
    annotation_segment_path,
    annotation_type_window_start_imu_ms,
    prepare_annotation_segment_signal,
)


class AnnotationPlotServiceTests(unittest.TestCase):
    def test_annotation_segment_available_accepts_csv_or_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            segment = Path(tmpdir) / "segment.csv"
            segment.write_text("ms,Gyr_X\n0,0\n", encoding="utf-8")

            self.assertTrue(annotation_segment_available(segment, archive_paths_factory=lambda: set()))
            self.assertTrue(
                annotation_segment_available(
                    "data/pending/segments/session/sensor/jump.csv",
                    archive_paths_factory=lambda: {"data/pending/segments/session/sensor/jump.csv"},
                )
            )
            self.assertFalse(annotation_segment_available(Path(tmpdir) / "missing.csv", archive_paths_factory=lambda: set()))

    def test_prepare_annotation_segment_signal_loads_cleans_and_recomputes(self):
        source = pd.DataFrame({"ms": [0, 1], "Gyr_X": [1.0, 2.0], "Acc_X": [0.0, 0.0]})

        def fake_clean(frame, acceleration_limit_g):
            cleaned = frame.copy()
            cleaned["clean_limit"] = acceleration_limit_g
            return cleaned, {"replaced": 0}

        def fake_recompute(frame, smoothing_sigma, threshold):
            prepared = frame.copy()
            prepared["Gyr_X_smoothed"] = prepared["Gyr_X"] * smoothing_sigma
            prepared["threshold"] = threshold
            return prepared

        result = prepare_annotation_segment_signal(
            {"path": "archive/segment.csv"},
            acceleration_limit_g=32.0,
            smoothing_sigma=3.0,
            threshold=-0.2,
            load_dataframe=lambda _path: source,
            archive_paths_factory=lambda: {"archive/segment.csv"},
            clean_frame=fake_clean,
            recompute_derivatives=fake_recompute,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["cleaning_report"], {"replaced": 0})
        self.assertEqual(result["gyro_column"], "Gyr_X_smoothed")
        self.assertEqual(result["dataframe"]["Gyr_X_smoothed"].tolist(), [3.0, 6.0])
        self.assertEqual(result["dataframe"]["threshold"].tolist(), [-0.2, -0.2])

    def test_prepare_annotation_segment_signal_returns_none_when_missing(self):
        result = prepare_annotation_segment_signal(
            {"path": "missing.csv"},
            acceleration_limit_g=32.0,
            smoothing_sigma=3.0,
            threshold=-0.2,
            archive_paths_factory=lambda: set(),
        )

        self.assertIsNone(result)

    def test_prepare_annotation_segment_signal_returns_none_when_path_missing(self):
        result = prepare_annotation_segment_signal(
            {"sensor_id": "1"},
            acceleration_limit_g=32.0,
            smoothing_sigma=3.0,
            threshold=-0.2,
            archive_paths_factory=lambda: set(),
        )

        self.assertIsNone(result)

    def test_annotation_gyro_column_prefers_smoothed_signal(self):
        self.assertEqual(annotation_gyro_column(pd.DataFrame({"Gyr_X": [1]})), "Gyr_X")
        self.assertEqual(annotation_gyro_column(pd.DataFrame({"Gyr_X": [1], "Gyr_X_smoothed": [1]})), "Gyr_X_smoothed")

    def test_annotation_type_window_start_imu_ms_reads_segment_timeline(self):
        source = pd.DataFrame({"ms": [1000.0, 1008.0, 1016.0], "Gyr_X": [1.0, 2.0, 3.0]})

        result = annotation_type_window_start_imu_ms(
            {"path": "archive/segment.csv"},
            type_window_start=1,
            load_dataframe=lambda _path: source,
            archive_paths_factory=lambda: {"archive/segment.csv"},
        )

        self.assertEqual(result, 1008.0)

    def test_annotation_type_window_start_imu_ms_returns_none_when_unavailable(self):
        self.assertIsNone(
            annotation_type_window_start_imu_ms(
                {"path": "missing.csv"},
                type_window_start=1,
                archive_paths_factory=lambda: set(),
            )
        )
        self.assertIsNone(
            annotation_type_window_start_imu_ms(
                {"sensor_id": "1"},
                type_window_start=1,
                archive_paths_factory=lambda: set(),
            )
        )
        self.assertIsNone(
            annotation_type_window_start_imu_ms(
                {"path": "archive/segment.csv"},
                type_window_start=99,
                load_dataframe=lambda _path: pd.DataFrame({"ms": [1000.0]}),
                archive_paths_factory=lambda: {"archive/segment.csv"},
            )
        )

    def test_annotation_segment_path_handles_missing_and_blank_values(self):
        self.assertIsNone(annotation_segment_path({}))
        self.assertIsNone(annotation_segment_path({"path": ""}))
        self.assertEqual(annotation_segment_path({"path": "segment.csv"}), Path("segment.csv"))


if __name__ == "__main__":
    unittest.main()
