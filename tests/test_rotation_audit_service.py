import tempfile
import unittest
from pathlib import Path

from synergie.services.rotation_audit_service import audit_turn_estimation


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
        self.assertEqual(result["suspicious_records"][0]["path"], "c.csv")
        self.assertEqual(result["rounding_rule_summary"][0]["label"], "round")
        self.assertEqual(
            [item["label"] for item in result["rounding_rule_summary"]],
            ["round", "ceil_minus_0.30", "floor_plus_0.50", "ceil_minus_0.15"],
        )
        self.assertEqual(result["rounding_rule_summary"][0]["exact_accuracy"], 2 / 3)
        self.assertEqual(result["type_summary"][0]["best_rule"], "round")
        self.assertEqual(result["type_summary"][1]["best_rule"], "round")
        self.assertEqual(
            [item["label"] for item in result["strategy_summary"]],
            ["current_round", "hybrid_non_axel_shift", "best_rule_per_type_observed"],
        )
        self.assertEqual(result["strategy_summary"][0]["exact_accuracy"], 2 / 3)
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
