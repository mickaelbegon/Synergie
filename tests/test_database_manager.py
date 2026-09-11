import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from core.database.DatabaseManager import DatabaseManager


class DatabaseManagerTests(unittest.TestCase):
    def test_resolve_credentials_path_accepts_rotated_filename_in_config(self):
        temp_root = Path(__file__).resolve().parents[1] / ".tmp" / "test_database_manager_glob"
        config_dir = temp_root / "config"
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        credentials_path = config_dir / "s2m-skating-firebase-adminsdk-3ofmb-0556d6ac57.json"
        credentials_path.write_text('{"type":"service_account"}', encoding="utf-8")

        original_method = DatabaseManager.credential_search_directories
        try:
            DatabaseManager.credential_search_directories = classmethod(lambda cls: [config_dir])  # type: ignore[method-assign]
            resolved = DatabaseManager.resolve_credentials_path()
        finally:
            DatabaseManager.credential_search_directories = original_method  # type: ignore[method-assign]
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(resolved, credentials_path)

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
        empty_root = Path(__file__).resolve().parents[1] / ".tmp" / "test_database_manager_empty"
        if empty_root.exists():
            shutil.rmtree(empty_root, ignore_errors=True)
        empty_root.mkdir(parents=True, exist_ok=True)
        original_method = DatabaseManager.credential_search_directories
        with patch.dict(
            os.environ,
            {"SYNERGIE_FIREBASE_CREDENTIALS": r"C:\missing\firebase.json"},
            clear=False,
        ):
            try:
                DatabaseManager.credential_search_directories = classmethod(lambda cls: [empty_root])  # type: ignore[method-assign]
                with self.assertRaises(FileNotFoundError) as context:
                    DatabaseManager.resolve_credentials_path()
            finally:
                DatabaseManager.credential_search_directories = original_method  # type: ignore[method-assign]
                shutil.rmtree(empty_root, ignore_errors=True)

        message = str(context.exception)
        self.assertIn(DatabaseManager.CREDENTIALS_FILENAME, message)
        self.assertIn(DatabaseManager.CREDENTIALS_GLOB, message)
        self.assertIn("SYNERGIE_FIREBASE_CREDENTIALS", message)
        self.assertIn(r"C:\missing\firebase.json", message)
        self.assertIn("Checked directories", message)


if __name__ == "__main__":
    unittest.main()
