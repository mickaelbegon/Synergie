import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synergie.services.annotation_source_service import find_annotation_raw_source_path


class AnnotationSourceServiceTests(unittest.TestCase):
    def test_find_annotation_raw_source_path_uses_direct_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "sensor.csv"
            source.write_text("ms\n0\n", encoding="utf-8")
            frame = pd.DataFrame([{"sensor_id": "1", "source_file": str(source)}])

            self.assertEqual(find_annotation_raw_source_path(frame, "1"), source)

    def test_find_annotation_raw_source_path_searches_roots_by_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "raw"
            source = root / "session" / "sensor.csv"
            source.parent.mkdir(parents=True)
            source.write_text("ms\n0\n", encoding="utf-8")
            frame = pd.DataFrame([{"sensor_id": "1", "source_file": "sensor.csv"}])

            self.assertEqual(find_annotation_raw_source_path(frame, "1", search_roots=[root]), source)

    def test_find_annotation_raw_source_path_returns_none_when_missing(self):
        frame = pd.DataFrame([{"sensor_id": "1", "source_file": "missing.csv"}])

        self.assertIsNone(find_annotation_raw_source_path(frame, "2", search_roots=[]))


if __name__ == "__main__":
    unittest.main()
