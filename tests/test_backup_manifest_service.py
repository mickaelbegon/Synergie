import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

from synergie.services.backup_manifest_service import build_backup_manifest, write_backup_manifest
from synergie.services.annotation_service import save_annotation_metadata


class BackupManifestServiceTests(unittest.TestCase):
    def test_build_backup_manifest_groups_sessions_pending_and_training_trials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pending = root / "pending"
            annotated = root / "annotated"
            dataset = annotated / "total"
            new = root / "new"
            pending.mkdir()
            dataset.mkdir(parents=True)
            annotation_csv = pending / "20250911_085656_for_annotation.csv"
            pd.DataFrame(
                [
                    {
                        "path": "data/pending/segments/session/sensor1/jump001.csv",
                        "videoTimeStamp": "00:12",
                        "type": 8,
                        "success": 2,
                        "sensor_id": "1",
                        "start_ms": 12000,
                        "end_ms": 12600,
                    }
                ]
            ).to_csv(annotation_csv, index=False)
            save_annotation_metadata(
                annotation_csv,
                {
                    "video_path": "D:/videos/session.mov",
                    "video_directory": "D:/videos",
                    "sensor_sync_offsets_ms": {"1": 42.0},
                },
            )
            pd.DataFrame(
                [
                    {
                        "path": "data/annotated/20250911/0856/jump001.csv",
                        "type": 0,
                        "success": 1,
                        "athlete_id": "sensor_1",
                    }
                ]
            ).to_csv(dataset / "jumplist.csv", index=False)

            with mock.patch("synergie.services.backup_manifest_service.list_sessions", return_value=["1331"]):
                with mock.patch(
                    "synergie.services.backup_manifest_service.session_metadata",
                    return_value={"path": "1109/0856", "sample_time_fine_synchro": 123},
                ):
                    manifest = build_backup_manifest(
                        new_root=new,
                        pending_root=pending,
                        annotated_root=annotated,
                        training_dataset_root=dataset,
                        generated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                    )

            self.assertEqual(manifest["format"], "synergie-backup-manifest-v1")
            self.assertEqual(manifest["summary"]["configured_sessions"], 1)
            self.assertEqual(manifest["summary"]["pending_trials"], 1)
            self.assertEqual(manifest["summary"]["trainable_training_trials"], 1)
            self.assertEqual(manifest["configured_sessions"][0]["session_id"], "1331")
            self.assertEqual(manifest["pending_annotation_files"][0]["metadata"]["sensor_sync_offsets_ms"]["1"], 42.0)
            self.assertEqual(manifest["pending_annotation_files"][0]["trials"][0]["sensor_id"], "1")
            self.assertEqual(manifest["training_dataset"]["trials"][0]["athlete_id"], "sensor_1")

    def test_write_backup_manifest_creates_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backup_manifest.json"
            with mock.patch("synergie.services.backup_manifest_service.build_backup_manifest", return_value={"format": "test"}):
                result = write_backup_manifest(path)

            self.assertEqual(result, path)
            self.assertIn('"format": "test"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
