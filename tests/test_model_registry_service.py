import tempfile
import unittest
from pathlib import Path

from synergie.services.model_registry_service import (
    format_pretrained_model_label,
    latest_model_path_for_task,
    model_format,
)


class ModelRegistryServiceTests(unittest.TestCase):
    def test_format_pretrained_model_label_includes_accuracy_and_status(self):
        label = format_pretrained_model_label(
            {
                "label": "Demo",
                "architecture": "tcn",
                "performance": {"test_accuracy": 0.91},
                "compatible": True,
                "path_exists": True,
            }
        )

        self.assertIn("acc 0.910", label)
        self.assertIn("compatible", label)

    def test_latest_model_path_for_task_uses_active_aliases(self):
        self.assertTrue(latest_model_path_for_task("type").endswith("checkpoint.keras"))
        self.assertTrue(latest_model_path_for_task("success").endswith("success.keras"))

    def test_model_format_detects_keras_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.keras"
            path.write_text("placeholder", encoding="utf-8")

            self.assertEqual(model_format(path), "keras_v3")
