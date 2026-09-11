import tempfile
import unittest
from pathlib import Path

from synergie.services import workflow_state_service


class WorkflowStateServiceTests(unittest.TestCase):
    def test_records_pending_and_finalized_session_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            entry = workflow_state_service.record_pending_session(
                session_key="20250911_085656",
                source_files=["data/new/a.csv", "data/new/b.csv"],
                annotation_csv="data/pending/session_for_annotation.csv",
                raw_destination="data/raw/1109/0856",
                prediction_status="predicted",
                root=root,
            )

            self.assertEqual(entry["status"], "pending_annotation")
            loaded = workflow_state_service.session_workflow_entry("20250911_085656", root=root)
            self.assertEqual(loaded["prediction_status"], "predicted")

            finalized = workflow_state_service.mark_finalized_session("20250911_085656", root=root)

            self.assertEqual(finalized["status"], "finalized")
