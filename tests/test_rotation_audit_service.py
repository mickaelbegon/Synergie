import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synergie.services.hdf5_archive_service import export_hdf5_archive
from synergie.services.rotation_audit_service import audit_turn_estimation, load_turn_audit_signal


class RotationAuditServiceTests(unittest.TestCase):
    def test_reports_confusion_matrix_and_axel_half_turns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "session_for_annotation.csv").write_text(
                "path,type,turns,rotations,annotation_status,detection_status\n"
                "a.csv,0,2,2.1,completed,detected_jump\n"
                "b.csv,5,2.5,2.2,completed,detected_jump\n"
                "c.csv,0,3,1.8,completed,detected_jump\n"
                "d.csv,0,2,2.0,pending,detected_jump\n",
                encoding="utf-8",
            )

            result = audit_turn_estimation(root)

        self.assertEqual(result["labelled_jumps"], 3)
        self.assertEqual(result["labels"], [2.0, 2.5, 3.0])
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        self.assertEqual(len(result["suspicious_records"]), 1)
        self.assertEqual([item["path"] for item in result["suspicious_records"]], ["c.csv"])
        self.assertEqual(result["rounding_rule_summary"][0]["label"], "round")
        self.assertGreaterEqual(result["contact_offset_summary"][0]["offset_turns"], -0.1)
        self.assertLessEqual(result["contact_offset_summary"][0]["offset_turns"], 0.6)
        self.assertIn("fixed_contact_offset_0.45", [item["label"] for item in result["strategy_summary"]])
        self.assertEqual(
            [item["label"] for item in result["rounding_rule_summary"]],
            ["round", "ceil_minus_0.30", "floor_plus_0.50", "ceil_minus_0.15"],
        )
        self.assertEqual(result["rounding_rule_summary"][0]["exact_accuracy"], 2 / 3)
        self.assertEqual(result["type_summary"][0]["best_rule"], "round")
        self.assertEqual(result["type_summary"][1]["best_rule"], "round")
        self.assertEqual(
            [item["label"] for item in result["strategy_summary"]],
            [
                "current_round",
                "fixed_contact_offset_0.16",
                "fixed_contact_offset_0.21",
                "fixed_contact_offset_0.45",
                "hybrid_non_axel_shift",
                "best_rule_per_type_observed",
            ],
        )
        self.assertEqual(result["strategy_by_skater"], [])

    def test_recomputes_measured_rotation_from_training_jumplist(self):
        import pandas as pd
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            segment = root / "jump.csv"
            segment.write_text("placeholder\n", encoding="utf-8")
            pd.DataFrame([{"path": segment.as_posix(), "type": 0, "rotations": 2.0}]).to_csv(root / "jumplist.csv", index=False)
            with patch(
                "synergie.services.rotation_audit_service._measured_rotation_from_segment",
                return_value=1.8,
            ):
                result = audit_turn_estimation(root)

        self.assertEqual(result["labelled_jumps"], 1)
        self.assertEqual(result["records"][0]["annotated_turns"], 2.0)
        self.assertEqual(result["records"][0]["measured_rotation"], 1.8)

    def test_loads_signal_with_same_rotation_interval_bounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "jump.csv"
            path.write_text(
                "SampleTimeFine,ms,Gyr_X,Acc_X,X_gyr_second_derivative_crossing\n"
                "0,0,0,1,0\n"
                "100000,100,10,2,1\n"
                "200000,200,20,3,1\n"
                "300000,300,30,4,0\n",
                encoding="utf-8",
            )

            signal = load_turn_audit_signal(path)

        self.assertIsNotNone(signal)
        self.assertEqual(signal["takeoff_index"], 0)
        self.assertEqual(signal["landing_index"], 2)

    def test_loads_turn_signal_from_hdf5_after_segment_csv_move(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            os.chdir(root)
            try:
                segment = root / "data" / "annotated" / "20250911" / "0856" / "jump.csv"
                segment.parent.mkdir(parents=True)
                relative_segment = "data/annotated/20250911/0856/jump.csv"
                frame_count = 300
                gyr_x = [0.0] * frame_count
                for index in range(80, 160):
                    gyr_x[index] = 1000.0
                pd.DataFrame(
                    {
                        "SampleTimeFine": [index * 100000 for index in range(frame_count)],
                        "ms": [float(index * 100) for index in range(frame_count)],
                        "Gyr_X": gyr_x,
                        "Acc_X": [1.0] * frame_count,
                    }
                ).to_csv(segment, index=False)
                export_hdf5_archive(
                    "data/synergie_archive.h5",
                    manifest={
                        "format": "synergie-backup-manifest-v1",
                        "pending_annotation_files": [],
                        "training_dataset": {"trials": [{"path": relative_segment}]},
                    },
                )
                segment.unlink()

                signal = load_turn_audit_signal(relative_segment)
            finally:
                os.chdir(original_cwd)

        self.assertIsNotNone(signal)
        self.assertEqual(signal["takeoff_index"], 90)
        self.assertEqual(signal["landing_index"], 150)
