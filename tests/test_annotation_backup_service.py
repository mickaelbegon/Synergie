import json
import tempfile
import unittest
from pathlib import Path

from synergie.services.annotation_backup_service import (
    create_finalization_backup,
    protected_backup_path,
    sha256_file,
    update_finalization_backup,
)


class AnnotationBackupServiceTests(unittest.TestCase):
    def test_backup_copies_protected_file_and_records_immutable_source_digest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "jumplist.csv"
            source.write_text("path,type\na.csv,1\n", encoding="utf-8")

            backup = create_finalization_backup(
                backup_root=root / "backups",
                session_key="20251002_100739",
                protected_files=[("jumplist", source)],
                context={"mode": "dry-run"},
            )

            copied = protected_backup_path(backup, source)
            self.assertIsNotNone(copied)
            self.assertTrue(copied.is_file())
            self.assertEqual(sha256_file(source), sha256_file(copied))
            manifest = json.loads(backup["manifest_path"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(manifest["context"]["mode"], "dry-run")

            source.write_text("path,type\na.csv,8\n", encoding="utf-8")
            self.assertNotEqual(sha256_file(source), sha256_file(copied))
            updated = update_finalization_backup(backup["manifest_path"], status="rolled_back", error="simulated")
            self.assertEqual(updated["status"], "rolled_back")
