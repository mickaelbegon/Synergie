import tempfile
import unittest
from pathlib import Path

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
