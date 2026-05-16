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

    def test_list_session_csv_files_excludes_jumplist_exports(self):
        original_file = session_store.SESSIONS_FILE
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_store.SESSIONS_FILE = root / "sessions.json"
            session_store.save_sessions({"a": {"path": "session_a", "sample_time_fine_synchro": 111}})
            session_dir = root / "session_a"
            session_dir.mkdir()
            imu_csv = session_dir / "1_D422CD0076F7_20250911_085656.csv"
            jumplist_csv = session_dir / "0910_jumplist_partie1.csv"
            imu_csv.write_text("x\n", encoding="utf-8")
            jumplist_csv.write_text("x\n", encoding="utf-8")

            files = session_service.list_session_csv_files("a", raw_root=root)

            self.assertEqual(files, [imu_csv])
        session_store.SESSIONS_FILE = original_file
