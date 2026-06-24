import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from synergie.services.data_validation_service import format_data_validation_report, validate_data_files


class DataValidationServiceTests(unittest.TestCase):
    def test_reports_missing_raw_folder_and_pending_missing_segment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pending = root / "data" / "pending"
            pending.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "path": "data/pending/segments/session/sensor1/missing.csv",
                        "sensor_id": "1",
                        "source_file": "1_DEVICE_20250911_085656.csv",
                    }
                ]
            ).to_csv(pending / "20250911_085656_for_annotation.csv", index=False)

            with patch("synergie.services.data_validation_service.list_sessions", return_value=["20250911_085656"]), patch(
                "synergie.services.data_validation_service.session_metadata",
                return_value={"path": "1109/0856", "sample_time_fine_synchro": 0},
            ):
                result = validate_data_files(
                    raw_root=root / "data" / "raw",
                    pending_root=pending,
                    annotated_root=root / "data" / "annotated",
                    hdf5_archive_path=None,
                )

            messages = "\n".join(issue["message"] for issue in result["issues"])
            self.assertIn("no local raw folder", messages)
            self.assertIn("missing segment", messages)
            self.assertIn("source_file(s) not found", messages)
            self.assertGreaterEqual(result["summary"]["errors"], 1)

    def test_accepts_pending_segment_represented_in_hdf5_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "data" / "raw" / "1109" / "0856"
            pending = root / "data" / "pending"
            raw.mkdir(parents=True)
            pending.mkdir(parents=True)
            (raw / "1_DEVICE_20250911_085656.csv").write_text("ms,Gyr_X\n0,0\n", encoding="utf-8")
            archived_path = "data/pending/segments/session/sensor1/jump001.csv"
            pd.DataFrame([{"path": archived_path, "sensor_id": "1", "source_file": "1_DEVICE_20250911_085656.csv"}]).to_csv(
                pending / "20250911_085656_for_annotation.csv",
                index=False,
            )
            archive = root / "data" / "synergie_archive.h5"
            archive.parent.mkdir(parents=True, exist_ok=True)
            import h5py

            with h5py.File(archive, "w") as handle:
                handle.attrs["segment_path_index_json"] = json.dumps({archived_path: "/trials/trial_000000"})

            with patch("synergie.services.data_validation_service.list_sessions", return_value=["20250911_085656"]), patch(
                "synergie.services.data_validation_service.session_metadata",
                return_value={"path": "1109/0856", "sample_time_fine_synchro": 0},
            ):
                result = validate_data_files(
                    raw_root=root / "data" / "raw",
                    pending_root=pending,
                    annotated_root=root / "data" / "annotated",
                    hdf5_archive_path=archive,
                )

            self.assertEqual(result["summary"]["errors"], 0)

    def test_missing_raw_source_is_info_when_segment_is_archived(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "data" / "raw" / "1109" / "0856"
            pending = root / "data" / "pending"
            raw.mkdir(parents=True)
            pending.mkdir(parents=True)
            (raw / "other_session_file.csv").write_text("ms,Gyr_X\n0,0\n", encoding="utf-8")
            archived_path = "data/pending/segments/session/sensor1/jump001.csv"
            pd.DataFrame([{"path": archived_path, "sensor_id": "1", "source_file": "1_DEVICE_20250911_085656.csv"}]).to_csv(
                pending / "20250911_085656_for_annotation.csv",
                index=False,
            )
            archive = root / "data" / "synergie_archive.h5"
            archive.parent.mkdir(parents=True, exist_ok=True)
            import h5py

            with h5py.File(archive, "w") as handle:
                handle.attrs["segment_path_index_json"] = json.dumps({archived_path: "/trials/trial_000000"})

            with patch("synergie.services.data_validation_service.list_sessions", return_value=["20250911_085656"]), patch(
                "synergie.services.data_validation_service.session_metadata",
                return_value={"path": "1109/0856", "sample_time_fine_synchro": 0},
            ):
                result = validate_data_files(
                    raw_root=root / "data" / "raw",
                    pending_root=pending,
                    annotated_root=root / "data" / "annotated",
                    hdf5_archive_path=archive,
                )

            messages_by_severity = {issue["message"]: issue["severity"] for issue in result["issues"]}
            source_message = next(message for message in messages_by_severity if "source_file(s) not found" in message)
            self.assertEqual(messages_by_severity[source_message], "info")
            self.assertEqual(result["summary"]["errors"], 0)
            self.assertEqual(result["summary"]["warnings"], 0)

    def test_formats_empty_report(self):
        report = format_data_validation_report({"summary": {"errors": 0, "warnings": 0, "info": 0}, "issues": []})

        self.assertIn("No obvious data consistency issue", report)


if __name__ == "__main__":
    unittest.main()
