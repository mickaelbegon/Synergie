import unittest

from synergie.services.annotation_sync_service import (
    annotation_imu_to_video_ms,
    annotation_review_video_target_ms,
    annotation_row_video_time_ms,
    annotation_sync_context_text,
    annotation_sync_source_summary,
    annotation_sync_summary,
    annotation_video_to_imu_ms,
    block_offset_from_impact,
    block_offset_from_jump,
    sync_impact_diagnostic_text,
    sync_impact_selection_text,
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

    def test_review_video_target_prefers_type_window_start_when_available(self):
        metadata = {"block_sync_offset_ms": 800.0, "sensor_sync_offsets_ms": {"2": 100.0}}
        row = {"start_ms": "2500", "synced_start_ms": "100"}

        self.assertEqual(
            annotation_review_video_target_ms(metadata, row, "2", type_window_start_imu_ms=2100.0),
            2900.0,
        )

    def test_review_video_target_falls_back_to_jump_start(self):
        metadata = {"sensor_sync_offsets_ms": {"2": 100.0}}
        row = {"synced_start_ms": "2500"}

        self.assertEqual(
            annotation_review_video_target_ms(metadata, row, "2", type_window_start_imu_ms=None),
            2600.0,
        )

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

    def test_sync_context_text_handles_empty_and_row_contexts(self):
        format_ms = lambda value: f"{value:.0f} ms"

        self.assertEqual(
            annotation_sync_context_text({}, None, sensor_id=None, sensor_ids=[], format_ms=format_ms),
            "No sync offset saved for current sensor.",
        )
        self.assertEqual(
            annotation_sync_context_text({}, {"start_ms": 1000}, sensor_id=None, sensor_ids=[], format_ms=format_ms),
            "No sensor ID available for this entry.",
        )
        self.assertIn(
            "Block sync active for all sensors | offset +400 ms",
            annotation_sync_context_text(
                {"block_sync_offset_ms": 400.0, "block_sync_source": {"method": "block_jump", "video_ms": 1400.0}},
                None,
                sensor_id="1",
                sensor_ids=["1"],
                format_ms=format_ms,
            ),
        )
        self.assertIn(
            "jump at 1500 ms in video",
            annotation_sync_context_text(
                {"sensor_sync_offsets_ms": {"2": 500.0}},
                {"synced_start_ms": 1000.0},
                sensor_id="2",
                sensor_ids=["1", "2"],
                format_ms=format_ms,
            ),
        )

    def test_block_offset_helpers(self):
        self.assertEqual(block_offset_from_jump(5000.0, {"start_ms": "4200"}), 800.0)
        self.assertEqual(block_offset_from_impact(12400.0, 8000.0), 4400.0)

    def test_sync_impact_text_helpers(self):
        format_ms = lambda value: f"{value / 1000:.3f}s"

        self.assertEqual(
            sync_impact_selection_text(
                {"sensor_id": "2", "impact_ms": 8000.0, "video_ms": 12400.0},
                format_ms=format_ms,
            ),
            "Sync impact for sensor 2: IMU 8000 ms (8.000s) | video 12.400s",
        )
        self.assertIn(
            "orange line at 8000 ms",
            sync_impact_diagnostic_text("2", 8000.0, format_ms=format_ms),
        )


if __name__ == "__main__":
    unittest.main()
