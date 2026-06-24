import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from synergie.services.annotation_generation_service import (
    estimate_sensor_impact_offset_ms,
    process_new_imu_session_for_annotation,
)


class AnnotationGenerationServiceTests(unittest.TestCase):
    def test_estimate_sensor_impact_offset_ms_uses_first_strong_acceleration_change(self):
        ms = list(range(0, 2100, 100))
        frame = pd.DataFrame(
            {
                "ms": ms,
                "Acc_X": [0 if value < 1200 else 10 for value in ms],
                "Acc_Y": [0 for _value in ms],
                "Acc_Z": [0 for _value in ms],
            }
        )

        self.assertEqual(estimate_sensor_impact_offset_ms(frame), 1200.0)

    def test_process_new_imu_session_for_annotation_exports_sorted_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_one = root / "1_D422CD0076F7_20250911_085656.csv"
            raw_two = root / "2_D422CD007712_20250911_085656.csv"
            for path in (raw_one, raw_two):
                path.write_text("ms,Acc_X,Acc_Y,Acc_Z\n0,0,0,0\n10,1,0,0\n", encoding="utf-8")

            jump_late = SimpleNamespace(
                df=pd.DataFrame({"ms": [100], "Gyr_X": [1]}),
                startTimestamp=200.0,
                endTimestamp=250.0,
                rotation=2.0,
            )
            jump_early = SimpleNamespace(
                df=pd.DataFrame({"ms": [50], "Gyr_X": [1]}),
                startTimestamp=100.0,
                endTimestamp=150.0,
                rotation=1.0,
            )

            class FakeTrainingSession:
                def __init__(self, dataframe, sampleTimefineSynchro=0):
                    self.df = dataframe
                    self.jumps = [jump_late] if len(dataframe) == 2 and dataframe.iloc[1]["Acc_X"] == 1 else [jump_early]

            file_metadata = [
                {
                    "path": raw_one,
                    "sensor_id": "1",
                    "device_id": "A",
                    "date_token": "20250911",
                    "time_token": "085656",
                    "recorded_at": pd.Timestamp("2025-09-11T08:56:56").to_pydatetime(),
                },
                {
                    "path": raw_two,
                    "sensor_id": "2",
                    "device_id": "B",
                    "date_token": "20250911",
                    "time_token": "085656",
                    "recorded_at": pd.Timestamp("2025-09-11T08:56:56").to_pydatetime(),
                },
            ]
            with mock.patch("synergie.services.annotation_generation_service.files_for_new_imu_session", return_value=file_metadata):
                with mock.patch("core.data_treatment.data_generation.trainingSession.trainingSession", FakeTrainingSession):
                    result = process_new_imu_session_for_annotation(raw_one, pending_root=root / "pending")

            exported = pd.read_csv(result["annotation_csv"])
            self.assertEqual(result["sensor_count"], 2)
            self.assertEqual(len(exported), 2)
            self.assertEqual(exported["sensor_id"].astype(str).tolist(), ["1", "2"])
            self.assertEqual(result["prediction_status"], "not_requested")
