import json
import tempfile
import unittest
from pathlib import Path

import h5py
import pandas as pd

from synergie.services.hdf5_archive_service import (
    archive_segment_csvs,
    export_hdf5_archive,
    load_segment_dataframe,
    plan_segment_csv_cleanup,
)


class Hdf5ArchiveServiceTests(unittest.TestCase):
    def test_export_hdf5_archive_writes_manifest_and_segment_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment = root / "segment.csv"
            pd.DataFrame(
                {
                    "Euler_X": [0.0, 1.0],
                    "Gyr_X": [10.0, 20.0],
                    "Acc_X": [1.0, 2.0],
                    "Combination": [0, 0],
                }
            ).to_csv(segment, index=False)
            archive = root / "archive.h5"
            manifest = {
                "format": "synergie-backup-manifest-v1",
                "pending_annotation_files": [
                    {
                        "path": "pending.csv",
                        "trials": [{"path": str(segment), "sensor_id": "1", "type": 8}],
                    }
                ],
                "training_dataset": {"trials": []},
            }

            result = export_hdf5_archive(archive, manifest=manifest)

            self.assertEqual(result["trials_indexed"], 1)
            self.assertEqual(result["segments_written"], 1)
            with h5py.File(archive, "r") as handle:
                self.assertEqual(handle.attrs["format"], "synergie-hdf5-archive-v1")
                manifest_payload = json.loads(handle["manifest"][()].decode("utf-8"))
                self.assertEqual(manifest_payload["format"], "synergie-backup-manifest-v1")
                trial = handle["trials/pending/trial_000000"]
                self.assertEqual(json.loads(trial.attrs["metadata_json"])["sensor_id"], "1")
                self.assertEqual(trial["segment"].shape, (2, 4))
                self.assertEqual(json.loads(trial["segment"].attrs["columns_json"]), ["Euler_X", "Gyr_X", "Acc_X", "Combination"])
                path_index = json.loads(handle.attrs["segment_path_index_json"])
                self.assertIn(segment.as_posix(), path_index)

            loaded = load_segment_dataframe(segment, archive)
            self.assertEqual(list(loaded.columns), ["Euler_X", "Gyr_X", "Acc_X", "Combination"])
            self.assertEqual(float(loaded.loc[1, "Gyr_X"]), 20.0)

            cleanup = plan_segment_csv_cleanup(archive)
            self.assertEqual(cleanup["candidate_count"], 1)
            self.assertGreater(cleanup["candidate_bytes"], 0)

    def test_archive_segment_csvs_moves_csv_but_hdf5_loader_still_reads_segment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment = root / "data" / "annotated" / "session" / "jump001.csv"
            segment.parent.mkdir(parents=True)
            pd.DataFrame({"Gyr_X": [10.0, 20.0], "Acc_X": [1.0, 2.0]}).to_csv(segment, index=False)
            archive = root / "archive.h5"
            destination = root / "csv_archive"
            manifest = {
                "format": "synergie-backup-manifest-v1",
                "pending_annotation_files": [{"path": "pending.csv", "trials": [{"path": str(segment)}]}],
                "training_dataset": {"trials": []},
            }
            export_hdf5_archive(archive, manifest=manifest)

            dry_run = archive_segment_csvs(archive, destination_root=destination)
            self.assertFalse(dry_run["applied"])
            self.assertTrue(segment.exists())

            applied = archive_segment_csvs(archive, destination_root=destination, apply=True)
            self.assertEqual(applied["moved_count"], 1)
            self.assertFalse(segment.exists())
            self.assertTrue(Path(applied["moved"][0]["target"]).exists())
            loaded = load_segment_dataframe(segment, archive)
            self.assertEqual(float(loaded.loc[1, "Gyr_X"]), 20.0)

    def test_export_hdf5_archive_keeps_metadata_when_segment_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "archive.h5"
            manifest = {
                "format": "synergie-backup-manifest-v1",
                "pending_annotation_files": [{"path": "pending.csv", "trials": [{"path": "missing.csv"}]}],
                "training_dataset": {"trials": []},
            }

            result = export_hdf5_archive(archive, manifest=manifest)

            self.assertEqual(result["trials_indexed"], 1)
            self.assertEqual(result["segments_missing"], 1)
            with h5py.File(archive, "r") as handle:
                self.assertTrue(bool(handle["trials/pending/trial_000000"].attrs["segment_missing"]))


if __name__ == "__main__":
    unittest.main()
