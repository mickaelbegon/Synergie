import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from synergie.services.csv_processing_service import process_csv_file


class CsvProcessingServiceTests(unittest.TestCase):
    def test_process_csv_file_forwards_selected_model_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "raw.csv"
            output_path = Path(tmpdir) / "out.csv"
            input_path.write_text("a\n1\n", encoding="utf-8")
            with mock.patch("core.data_treatment.data_generation.exporter.export", return_value=pd.DataFrame({"x": [1]})) as export_mock:
                result = process_csv_file(
                    str(input_path),
                    output_path=str(output_path),
                    type_model_path="type.keras",
                    success_model_path="success.keras",
                )

            self.assertEqual(result["path"], output_path)
            self.assertEqual(export_mock.call_args.kwargs["type_model_path"], "type.keras")
            self.assertEqual(export_mock.call_args.kwargs["success_model_path"], "success.keras")
