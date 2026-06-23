import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synergie.services.hdf5_archive_service import export_hdf5_archive
from synergie.services.quality_service import analyze_jump_quality


class QualityServiceTests(unittest.TestCase):
    def test_analyze_jump_quality_flags_missing_file_and_saturation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            segment_ok = dataset_root / "jump_ok.csv"
            segment_sat = dataset_root / "jump_sat.csv"
            segment_ok.write_text(
                "ms,Gyr_X,Acc_X,X_gyr_second_derivative\n0,10,1,0.1\n100,20,2,0.2\n200,30,3,0.3\n",
                encoding="utf-8",
            )
            segment_sat.write_text(
                "ms,Gyr_X,Acc_X,X_gyr_second_derivative\n0,10,1,0.1\n100,2100,2,0.2\n200,25,3,0.3\n",
                encoding="utf-8",
            )
            (dataset_root / "jumplist.csv").write_text(
                "path,type,skater,success,rotations\n"
                f"{segment_ok.as_posix()},0,alice,1,2.0\n"
                f"{segment_sat.as_posix()},0,bob,1,2.0\n"
                f"{(dataset_root / 'missing.csv').as_posix()},0,charlie,1,2.0\n",
                encoding="utf-8",
            )

            analysis = analyze_jump_quality(dataset_root, saturation_threshold=1900.0)

            self.assertEqual(analysis["total_labelled_jumps"], 3)
            self.assertEqual(analysis["suspicious_count"], 2)
            suspicious_reasons = {record["skater"]: record["reasons"] for record in analysis["suspicious_records"]}
            self.assertIn("gyro_saturation", suspicious_reasons["bob"])
            self.assertIn("missing_segment_file", suspicious_reasons["charlie"])

    def test_analyze_jump_quality_reads_archived_hdf5_segment_after_csv_move(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            os.chdir(root)
            try:
                dataset_root = root / "data" / "annotated" / "total"
                segment = root / "data" / "annotated" / "20250911" / "0856" / "jump.csv"
                dataset_root.mkdir(parents=True)
                segment.parent.mkdir(parents=True)
                relative_segment = "data/annotated/20250911/0856/jump.csv"
                pd.DataFrame(
                    {
                        "ms": [0.0, 100.0, 200.0],
                        "Gyr_X": [10.0, 20.0, 30.0],
                        "Acc_X": [1.0, 2.0, 3.0],
                        "X_gyr_second_derivative": [0.1, 0.2, 0.3],
                    }
                ).to_csv(segment, index=False)
                pd.DataFrame(
                    [{"path": relative_segment, "type": 0, "skater": "alice", "success": 1, "rotations": 2.0}]
                ).to_csv(dataset_root / "jumplist.csv", index=False)
                export_hdf5_archive(
                    "data/synergie_archive.h5",
                    manifest={
                        "format": "synergie-backup-manifest-v1",
                        "pending_annotation_files": [],
                        "training_dataset": {"trials": [{"path": relative_segment}]},
                    },
                )
                segment.unlink()

                analysis = analyze_jump_quality(dataset_root)
            finally:
                os.chdir(original_cwd)

        self.assertEqual(analysis["total_labelled_jumps"], 1)
        self.assertFalse(analysis["records"][0]["missing_file"])

    def test_analyze_jump_quality_recomputes_stale_extreme_second_derivative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            segment = dataset_root / "jump.csv"
            pd.DataFrame(
                {
                    "ms": [0.0, 100.0, 200.0, 300.0, 400.0],
                    "Gyr_X": [0.0, 10.0, 20.0, 10.0, 0.0],
                    "Acc_X": [1.0, 2.0, 3.0, 2.0, 1.0],
                    "X_gyr_second_derivative": [0.0, 1e29, -1e29, 0.0, 0.0],
                }
            ).to_csv(segment, index=False)
            pd.DataFrame([{"path": segment.as_posix(), "type": 0, "skater": "alice", "success": 1, "rotations": 2.0}]).to_csv(
                dataset_root / "jumplist.csv",
                index=False,
            )

            analysis = analyze_jump_quality(dataset_root)

        self.assertLess(analysis["records"][0]["max_abs_second_derivative"], 1e6)
