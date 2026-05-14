import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from core.database.DatabaseManager import DatabaseManager


class DatabaseManagerTests(unittest.TestCase):
    def test_resolve_credentials_path_prefers_env_var(self):
        temp_root = Path(__file__).resolve().parents[1] / ".tmp" / "test_database_manager"
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            credentials_path = temp_root / DatabaseManager.CREDENTIALS_FILENAME
            credentials_path.write_text('{"type":"service_account"}', encoding="utf-8")

            with patch.dict(os.environ, {"SYNERGIE_FIREBASE_CREDENTIALS": str(credentials_path)}, clear=False):
                resolved = DatabaseManager.resolve_credentials_path()
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(resolved, credentials_path)

    def test_resolve_credentials_path_raises_with_checked_locations(self):
        with patch.dict(
            os.environ,
            {"SYNERGIE_FIREBASE_CREDENTIALS": r"C:\missing\firebase.json"},
            clear=False,
        ):
            with self.assertRaises(FileNotFoundError) as context:
                DatabaseManager.resolve_credentials_path()

        message = str(context.exception)
        self.assertIn(DatabaseManager.CREDENTIALS_FILENAME, message)
        self.assertIn("SYNERGIE_FIREBASE_CREDENTIALS", message)
        self.assertIn(r"C:\missing\firebase.json", message)


if __name__ == "__main__":
    unittest.main()
