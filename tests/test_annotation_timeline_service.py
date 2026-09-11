import unittest

import pandas as pd

from synergie.services.annotation_timeline_service import build_annotation_timeline_items


class AnnotationTimelineServiceTests(unittest.TestCase):
    def test_build_annotation_timeline_includes_sync_before_jump_at_same_time(self):
        frame = pd.DataFrame(
            [
                {
                    "sensor_id": "1",
                    "impact_offset_ms": 800.0,
                    "start_ms": 1800.0,
                    "athlete_id": "sensor_1",
                    "detection_status": "detected_jump",
                }
            ]
        )
        metadata = {"sensor_sync_offsets_ms": {"1": 1000.0}, "sensor_sync_sources": {"1": {"method": "impact"}}}

        items = build_annotation_timeline_items(
            frame,
            metadata,
            row_video_time_ms=lambda row, _sensor_id: float(row["start_ms"]),
            imu_to_video_ms=lambda imu_ms, _sensor_id: float(imu_ms) + 1000.0,
            format_ms=lambda value: f"{value:.0f}ms",
        )

        self.assertEqual([item["kind"] for item in items], ["sync_impact", "jump"])
        self.assertIn("impact", items[0]["label"])
        self.assertIn("001 | 1800ms | sensor_1 | detected_jump", items[1]["label"])

    def test_build_annotation_timeline_uses_block_sync_source_label(self):
        frame = pd.DataFrame([{"sensor_id": "2", "impact_offset_ms": 12000.0, "start_ms": 13000.0}])
        metadata = {"block_sync_offset_ms": 400.0, "block_sync_source": {"method": "block_impact"}}

        items = build_annotation_timeline_items(
            frame,
            metadata,
            row_video_time_ms=lambda row, _sensor_id: float(row["start_ms"]) + 400.0,
            imu_to_video_ms=lambda imu_ms, _sensor_id: float(imu_ms) + 400.0,
            format_ms=lambda value: f"{value:.0f}ms",
        )

        self.assertEqual(items[0]["kind"], "sync_impact")
        self.assertIn("block_impact", items[0]["label"])

    def test_build_annotation_timeline_flags_impacts_mapping_before_video_start(self):
        frame = pd.DataFrame([{"sensor_id": "2", "impact_offset_ms": 8.0, "start_ms": 797676.0}])
        metadata = {"block_sync_offset_ms": -1616.0, "block_sync_source": {"method": "block_impact", "sensor_id": "1"}}

        items = build_annotation_timeline_items(
            frame,
            metadata,
            row_video_time_ms=lambda row, _sensor_id: max(float(row["start_ms"]) - 1616.0, 0.0),
            imu_to_video_ms=lambda imu_ms, _sensor_id: max(float(imu_ms) - 1616.0, 0.0),
            format_ms=lambda value: f"{value:.0f}ms",
        )

        self.assertEqual(items[0]["kind"], "sync_impact")
        self.assertIn("SYNC | 0ms", items[0]["label"])
        self.assertIn("maps before video start (-1608 ms)", items[0]["label"])

    def test_build_annotation_timeline_adds_prediction_label(self):
        frame = pd.DataFrame(
            [
                {
                    "sensor_id": "1",
                    "start_ms": 1000.0,
                    "athlete_id": "sensor_1",
                    "detection_status": "detected_jump",
                    "prediction_source": "model",
                    "type": 1,
                    "success": 0,
                }
            ]
        )

        items = build_annotation_timeline_items(
            frame,
            {},
            row_video_time_ms=lambda row, _sensor_id: float(row["start_ms"]),
            imu_to_video_ms=lambda imu_ms, _sensor_id: float(imu_ms),
            format_ms=lambda value: f"{value:.0f}ms",
        )

        self.assertIn("model: Flip / success 0", items[0]["label"])


if __name__ == "__main__":
    unittest.main()
