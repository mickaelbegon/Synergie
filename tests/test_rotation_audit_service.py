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
