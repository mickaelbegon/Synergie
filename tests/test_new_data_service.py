import tempfile
import unittest
from pathlib import Path

from synergie.services.new_data_service import (
    list_new_imu_sessions,
    parse_new_imu_filename,
    suggest_for_annotation_output_path,
)


class NewDataServiceTests(unittest.TestCase):
    def test_parse_new_imu_filename_extracts_metadata(self):
        metadata = parse_new_imu_filename("1_D422CD0076F7_20250911_085656.csv")

        self.assertEqual(metadata["sensor_id"], "1")
        self.assertEqual(metadata["device_id"], "D422CD0076F7")
        self.assertEqual(metadata["recorded_at"].year, 2025)

    def test_list_new_imu_sessions_groups_files_by_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "1_D422CD0076F7_20250911_085656.csv").write_text("x", encoding="utf-8")
            (root / "2_D422CD007712_20250911_085656.csv").write_text("x", encoding="utf-8")

            sessions = list_new_imu_sessions(root=root)

            self.assertEqual(len(sessions), 1)
            self.assertEqual([item["sensor_id"] for item in sessions[0]["files"]], ["1", "2"])

    def test_suggest_for_annotation_output_path_returns_clear_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = suggest_for_annotation_output_path(
                "1_D422CD0076F7_20250911_085656.csv",
                pending_root=tmpdir,
            )

            self.assertEqual(candidate.name, "20250911_085656_for_annotation.csv")
