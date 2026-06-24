import unittest

from synergie.services.annotation_sync_service import (
    annotation_imu_to_video_ms,
    annotation_row_video_time_ms,
    annotation_sync_source_summary,
    annotation_sync_summary,
    annotation_video_to_imu_ms,
    block_offset_from_impact,
    block_offset_from_jump,
)


class AnnotationSyncServiceTests(unittest.TestCase):
    def test_block_sync_takes_precedence_for_jump_video_time(self):
        metadata = {"block_sync_offset_ms": 800.0, "sensor_sync_offsets_ms": {"2": 100.0}}
        row = {"start_ms": "2500", "synced_start_ms": "100"}

        self.assertEqual(annotation_row_video_time_ms(metadata, row, "2"), 3300.0)
        self.assertEqual(annotation_imu_to_video_ms(metadata, 1200.0, "2"), 2000.0)
        self.assertEqual(annotation_video_to_imu_ms(metadata, 2000.0, "2"), 1200.0)

    def test_sensor_sync_is_fallback_without_block_sync(self):
        metadata = {"sensor_sync_offsets_ms": {"2": 100.0}}
        row = {"synced_start_ms": "2500"}

        self.assertEqual(annotation_row_video_time_ms(metadata, row, "2"), 2600.0)
        self.assertEqual(annotation_imu_to_video_ms(metadata, 1200.0, "2"), 1300.0)
        self.assertEqual(annotation_video_to_imu_ms(metadata, 1800.0, "2", impact_offset_ms=800.0), 2500.0)

    def test_sync_summaries_describe_block_source(self):
        metadata = {
            "block_sync_offset_ms": 400.0,
            "block_sync_source": {
                "method": "block_impact",
                "sensor_id": "1",
                "imu_impact_ms": 8000.0,
                "video_ms": 12400.0,
            },
        }

        self.assertEqual(annotation_sync_summary(metadata, ["1", "2"], "1"), "Block sync active for all sensors")
        self.assertIn(
            "block sync by impact sensor 1",
            annotation_sync_source_summary(metadata, "1", format_ms=lambda value: f"{value:.0f} ms"),
        )

    def test_block_offset_helpers(self):
        self.assertEqual(block_offset_from_jump(5000.0, {"start_ms": "4200"}), 800.0)
        self.assertEqual(block_offset_from_impact(12400.0, 8000.0), 4400.0)


if __name__ == "__main__":
    unittest.main()
