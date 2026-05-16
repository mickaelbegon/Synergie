import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synergie.services.annotation_finalization_service import finalize_annotation_file
from synergie.services.training_dataset_service import load_training_dataset_state


class AnnotationFinalizationServiceTests(unittest.TestCase):
    def test_finalize_annotation_file_archives_moves_and_merges_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "data" / "annotated" / "total"
            annotated_root = root / "data" / "annotated"
            pending_root = root / "data" / "pending"
            segment_dir = pending_root / "segments" / "session" / "sensor1"
            dataset_root.mkdir(parents=True)
            segment_dir.mkdir(parents=True)
            (dataset_root / "jumplist.csv").write_text("path,type,success,skater\nexisting.csv,0,1,alice\n", encoding="utf-8")
            segment = segment_dir / "jump001.csv"
            segment.write_text("ms,Gyr_X\n0,0\n", encoding="utf-8")
            annotation_file = pending_root / "20250911_085656_for_annotation.csv"
            pd.DataFrame(
                [
                    {
                        "path": segment.as_posix(),
                        "type": 1,
                        "success": 1,
                        "skater": "sensor_1",
                        "athlete_id": "bob",
                        "annotation_status": "annotated",
                        "sensor_id": "1",
                        "session_key": "20250911_085656",
                    }
                ]
            ).to_csv(annotation_file, index=False)

            result = finalize_annotation_file(annotation_file, dataset_path=dataset_root, annotated_root=annotated_root)

            merged = pd.read_csv(dataset_root / "jumplist.csv")
            self.assertEqual(result["rows_added"], 1)
            self.assertEqual(merged.iloc[1]["skater"], "bob")
            self.assertTrue(load_training_dataset_state(dataset_root)["retraining_recommended"])
