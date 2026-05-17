import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synergie.services.legacy_import_service import _normalize_legacy_skater_id, import_legacy_jumplist


class LegacyImportServiceTests(unittest.TestCase):
    def test_imports_trainable_legacy_rows_into_total_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_dir = root / "annotated" / "20250901" / "0910"
            dataset_dir = root / "annotated" / "total"
            session_dir.mkdir(parents=True)
            dataset_dir.mkdir(parents=True)
            segment = session_dir / "segment.csv"
            segment.write_text("x\n", encoding="utf-8")
            pd.DataFrame(
                {
                    "path": ["segment.csv"],
                    "type": [0],
                    "sucess": [1],
                    "skater": ["alice"],
                }
            ).to_csv(session_dir / "jumplist.csv", index=False)
            pd.DataFrame({"path": [], "type": [], "success": [], "skater": []}).to_csv(
                dataset_dir / "jumplist.csv",
                index=False,
            )

            result = import_legacy_jumplist(session_dir / "jumplist.csv", dataset_path=dataset_dir)

            merged = pd.read_csv(dataset_dir / "jumplist.csv")
            self.assertEqual(result["rows_added"], 1)
            self.assertEqual(len(merged), 1)
            self.assertTrue(merged.loc[0, "path"].endswith("segment.csv"))

    def test_normalizes_session_prefixed_skater_ids(self):
        self.assertEqual(_normalize_legacy_skater_id("20250901_0910_10"), "10")
        self.assertEqual(_normalize_legacy_skater_id("sensor_10"), "sensor_10")
