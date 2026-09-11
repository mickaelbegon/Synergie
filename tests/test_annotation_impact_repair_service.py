import unittest
from pathlib import Path

import pandas as pd

from synergie.services.annotation_impact_repair_service import redetect_annotation_sync_impacts


class AnnotationImpactRepairServiceTests(unittest.TestCase):
    def test_redetect_annotation_sync_impacts_updates_offsets_and_synced_times(self):
        frame = pd.DataFrame(
            [
                {"sensor_id": "1", "impact_offset_ms": 8.0, "start_ms": 12400.0, "end_ms": 13000.0},
                {"sensor_id": "2", "impact_offset_ms": 8.0, "start_ms": 22400.0, "end_ms": 23000.0},
            ]
        )

        result = redetect_annotation_sync_impacts(
            frame,
            raw_path_for_sensor=lambda sensor_id: Path(f"sensor_{sensor_id}.csv"),
            estimate_impact_ms=lambda path: 12000.0 if path.name == "sensor_1.csv" else 22000.0,
        )

        repaired = result["frame"]
        self.assertEqual(result["skipped"], [])
        self.assertEqual([item["sensor_id"] for item in result["updated"]], ["1", "2"])
        self.assertEqual(repaired["impact_offset_ms"].tolist(), [12000.0, 22000.0])
        self.assertEqual(repaired["synced_start_ms"].tolist(), [400.0, 400.0])
        self.assertEqual(repaired["synced_end_ms"].tolist(), [1000.0, 1000.0])

    def test_redetect_annotation_sync_impacts_reports_missing_raw(self):
        frame = pd.DataFrame([{"sensor_id": "1", "impact_offset_ms": 8.0, "start_ms": 12400.0}])

        result = redetect_annotation_sync_impacts(
            frame,
            raw_path_for_sensor=lambda _sensor_id: None,
            estimate_impact_ms=lambda _path: 12000.0,
        )

        self.assertEqual(result["updated"], [])
        self.assertIn("raw CSV not found", result["skipped"][0])


if __name__ == "__main__":
    unittest.main()
