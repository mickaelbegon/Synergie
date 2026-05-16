import tempfile
import unittest
from pathlib import Path

from synergie import session_store
from synergie.services import session_service


class SessionServiceTests(unittest.TestCase):
    def test_list_all_session_csv_files_returns_processing_metadata(self):
        original_file = session_store.SESSIONS_FILE
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_store.SESSIONS_FILE = root / "sessions.json"
            session_store.save_sessions(
                {
                    "a": {"path": "session_a", "sample_time_fine_synchro": 111},
                    "b": {"path": "session_b", "sample_time_fine_synchro": 222},
                }
            )
            for folder, filename in (("session_a", "a.csv"), ("session_b", "b.csv")):
                directory = root / folder
                directory.mkdir()
                (directory / filename).write_text("x\n", encoding="utf-8")

            records = session_service.list_all_session_csv_files(raw_root=root)

            self.assertEqual([record["session_name"] for record in records], ["a", "b"])
            self.assertEqual([record["sample_time_fine_synchro"] for record in records], [111, 222])
        session_store.SESSIONS_FILE = original_file
