import tempfile
import unittest
from pathlib import Path

from synergie.services.training_service import promote_trained_model


class TrainingServiceTests(unittest.TestCase):
    def test_promote_trained_model_replaces_existing_latest_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "run.keras"
            latest = root / "latest.keras"
            source.write_text("new", encoding="utf-8")
            latest.write_text("old", encoding="utf-8")

            promote_trained_model(source, latest)

            self.assertEqual(latest.read_text(encoding="utf-8"), "new")
