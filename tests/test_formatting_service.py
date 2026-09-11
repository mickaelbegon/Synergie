import unittest

from synergie.services.formatting_service import (
    format_annotation_video_info,
    format_file_size,
    format_video_cache_cleared_status,
    format_video_cache_button,
    format_video_ms,
    format_video_preparation_message,
    parse_video_ms,
)


class FormattingServiceTests(unittest.TestCase):
    def test_format_video_ms_uses_hours_when_needed(self):
        self.assertEqual(format_video_ms(12_408), "00:12.408")
        self.assertEqual(format_video_ms(3_661_234), "1:01:01.234")

    def test_parse_video_ms_supports_ui_formats(self):
        self.assertEqual(parse_video_ms("00:12.408"), 12_408.0)
        self.assertEqual(parse_video_ms("1:01:01.234"), 3_661_234.0)
        self.assertEqual(parse_video_ms("01:02"), 62_000.0)
        self.assertIsNone(parse_video_ms(""))

    def test_format_file_size_uses_binary_units(self):
        self.assertEqual(format_file_size(512), "512 B")
        self.assertEqual(format_file_size(1536), "1.5 KB")
        self.assertEqual(format_file_size(2 * 1024 * 1024), "2.0 MB")

    def test_formats_annotation_video_cache_and_info_labels(self):
        self.assertEqual(format_video_cache_button(2 * 1024 * 1024), "Clear cache (2.0 MB)")
        self.assertEqual(
            format_annotation_video_info(
                video_name="session.mov",
                frame_count=120,
                duration_ms=12_408,
                cache_note=" | optimized playback",
            ),
            "session.mov | 120 frames | 00:12.408 | optimized playback",
        )

    def test_formats_annotation_video_preparation_and_cache_status(self):
        self.assertIn(
            "1/2 Caching local copy (2.0 MB)",
            format_video_preparation_message(
                {"copy_needed": True, "size_bytes": 2 * 1024 * 1024},
                {"proxy_needed": True},
            ),
        )
        self.assertEqual(
            format_video_cache_cleared_status({"removed_files": 2, "freed_bytes": 1536}),
            "Video cache cleared: 2 file(s), 1.5 KB freed.",
        )


if __name__ == "__main__":
    unittest.main()
