import unittest

from synergie.services.formatting_service import format_file_size, format_video_ms, parse_video_ms


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


if __name__ == "__main__":
    unittest.main()
