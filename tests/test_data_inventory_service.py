import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synergie.services.data_inventory_service import build_data_inventory
from synergie.services.workflow_state_service import record_pending_session


class DataInventoryServiceTests(unittest.TestCase):
    def test_builds_inventory_from_new_pending_annotated_and_training_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            new_root = root / "new"
            pending_root = root / "pending"
            annotated_root = root / "annotated"
            dataset_root = annotated_root / "total"
            new_root.mkdir()
            pending_root.mkdir()
            (annotated_root / "20250911" / "0856").mkdir(parents=True)
            dataset_root.mkdir(parents=True)
            (new_root / "1_D422CD0076F7_20250911_085656.csv").write_text("x\n", encoding="utf-8")
            (pending_root / "20250911_085656_for_annotation.csv").write_text("x\n", encoding="utf-8")
            (annotated_root / "20250911" / "0856" / "segment.csv").write_text("x\n", encoding="utf-8")
            pd.DataFrame({"path": ["data/annotated/20250911/0856/segment.csv"]}).to_csv(
                dataset_root / "jumplist.csv",
                index=False,
            )
            record_pending_session(
                session_key="20250911_085656",
                source_files=[new_root / "1_D422CD0076F7_20250911_085656.csv"],
                annotation_csv=pending_root / "20250911_085656_for_annotation.csv",
                raw_destination=root / "raw" / "1109" / "0856",
                prediction_status="predicted",
                root=pending_root,
            )

            rows = build_data_inventory(
                new_root=new_root,
                pending_root=pending_root,
                annotated_root=annotated_root,
                training_dataset_root=dataset_root,
            )

            by_session = {row["session"]: row for row in rows}
            self.assertEqual(by_session["20250911_085656"]["new_files"], 1)
            self.assertEqual(by_session["20250911_085656"]["pending_files"], 1)
            self.assertEqual(by_session["20250911_085656"]["workflow_status"], "pending_annotation")
            self.assertEqual(by_session["20250911/0856"]["annotated_files"], 1)
            self.assertEqual(by_session["20250911/0856"]["training_rows"], 1)
