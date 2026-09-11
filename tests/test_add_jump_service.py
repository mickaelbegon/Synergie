import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from synergie.services.add_jump_service import (
    annotation_index_for_path,
    append_manual_annotation_row,
    build_manual_jump_segment,
    build_manual_annotation_row,
    compute_rotation_between_ms,
    detect_add_jump_candidates,
    format_add_jump_candidate_labels,
    manual_segment_path,
    template_for_sensor,
)


class AddJumpServiceTests(unittest.TestCase):
    def test_detect_add_jump_candidates_returns_indexed_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "sensor.csv"
            raw_path.write_text("ms,Acc_X\n0,0\n", encoding="utf-8")
            jumps = [
                SimpleNamespace(startTimestamp=100.0, endTimestamp=200.0, rotation=1.25),
                SimpleNamespace(startTimestamp=300.0, endTimestamp=450.0, rotation=2.0),
            ]

            class FakeTrainingSession:
                def __init__(self, dataframe, detection_threshold, smoothing_sigma, combination_gap_frames):
                    self.dataframe = dataframe
                    self.detection_threshold = detection_threshold
                    self.smoothing_sigma = smoothing_sigma
                    self.combination_gap_frames = combination_gap_frames
                    self.jumps = jumps

            candidates = detect_add_jump_candidates(
                raw_path,
                detection_threshold=-0.2,
                smoothing_sigma=30,
                combination_gap_frames=120,
                training_session_factory=FakeTrainingSession,
            )

            self.assertEqual([candidate["index"] for candidate in candidates], [1, 2])
            self.assertEqual(candidates[0]["raw_path"], raw_path)
            self.assertEqual(candidates[0]["session"].detection_threshold, -0.2)

    def test_format_add_jump_candidate_labels(self):
        candidates = [
            {"index": 1, "jump": SimpleNamespace(startTimestamp=100.0, endTimestamp=200.0, rotation=1.25)},
        ]

        self.assertEqual(format_add_jump_candidate_labels(candidates), ["001 | start 100 ms | end 200 ms | rotation 1.2"])

    def test_build_manual_jump_segment_keeps_fixed_window_near_takeoff(self):
        frame = pd.DataFrame({"ms": [0, 100, 200, 300, 400, 500]})
        candidate = {
            "session": SimpleNamespace(df=frame),
            "jump": SimpleNamespace(combinate=True),
        }

        segment, start_index = build_manual_jump_segment(
            candidate,
            310.0,
            frames_before_takeoff=2,
            window_frames=4,
        )

        self.assertEqual(start_index, 3)
        self.assertEqual(segment["ms"].tolist(), [100, 200, 300, 400])
        self.assertEqual(segment["Combination"].tolist(), [1, 1, 1, 1])

    def test_compute_rotation_between_ms_integrates_gyr_x(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 500, 1000],
                "SampleTimeFine": [0, 500_000, 1_000_000],
                "Gyr_X": [360.0, 360.0, 0.0],
            }
        )

        self.assertEqual(compute_rotation_between_ms(frame, 0.0, 1000.0), 1.0)

    def test_build_manual_annotation_row_preserves_template_and_manual_metadata(self):
        row = build_manual_annotation_row(
            {"path": "old.csv", "skater": "athlete_a", "impact_offset_ms": 800.0},
            segment_path=Path("segments") / "manual.csv",
            source_file_name="raw_sensor.csv",
            sensor_id="2",
            start_ms=1800.25,
            end_ms=2200.75,
            impact_offset_ms=800.0,
            rotations=1.24,
            takeoff_video_text="00:01.800",
            landing_video_text="00:02.200",
            detection_threshold=-0.2,
            smoothing_sigma=30,
            combination_gap_frames=120,
            format_video_ms=lambda value: f"{value:.0f} ms",
        )

        self.assertEqual(row["path"], "segments/manual.csv")
        self.assertEqual(row["videoTimeStamp"], "1000 ms")
        self.assertEqual(row["skater"], "athlete_a")
        self.assertEqual(row["athlete_id"], "sensor_2")
        self.assertEqual(row["rotations"], 1.2)
        self.assertEqual(row["source_file"], "raw_sensor.csv")
        self.assertEqual(row["sensor_id"], "2")
        self.assertEqual(row["synced_start_ms"], 1000.25)
        self.assertEqual(row["manual_takeoff_video_ms"], "00:01.800")

    def test_append_manual_annotation_row_keeps_timeline_order(self):
        frame = pd.DataFrame(
            [
                {"path": "late.csv", "synced_start_ms": 3000.0, "sensor_id": "2", "start_ms": 3800.0},
                {"path": "early.csv", "synced_start_ms": 1000.0, "sensor_id": "1", "start_ms": 1800.0},
            ]
        )
        row = {"path": "middle.csv", "synced_start_ms": 2000.0, "sensor_id": "1", "start_ms": 2800.0}

        updated = append_manual_annotation_row(frame, row)

        self.assertEqual(updated["path"].tolist(), ["early.csv", "middle.csv", "late.csv"])
        self.assertEqual(annotation_index_for_path(updated, "middle.csv"), 1)

    def test_template_for_sensor_returns_first_matching_row(self):
        frame = pd.DataFrame(
            [
                {"sensor_id": "1", "path": "sensor1_first.csv", "skater": "a"},
                {"sensor_id": "2", "path": "sensor2.csv", "skater": "b"},
                {"sensor_id": "1", "path": "sensor1_second.csv", "skater": "c"},
            ]
        )

        self.assertEqual(template_for_sensor(frame, "1")["path"], "sensor1_first.csv")
        self.assertEqual(template_for_sensor(frame, "3"), {})

    def test_manual_segment_path_uses_template_directory_and_next_row_number(self):
        path = manual_segment_path(
            {"path": "segments/existing.csv"},
            Path("annotations") / "session.csv",
            "2",
            row_count=41,
        )

        self.assertEqual(path.as_posix(), "segments/manual_sensor2_jump042.csv")


if __name__ == "__main__":
    unittest.main()
