import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from synergie.services import video_service


class VideoServiceTests(unittest.TestCase):
    def test_annotation_reference_datetime_prefers_recorded_at_column(self):
        result = video_service.annotation_reference_datetime(
            Path("20250911_085656_for_annotation.csv"),
            annotation_rows={"recorded_at": ["2025-09-11T08:56:58", "2025-09-11T08:56:56"]},
        )

        self.assertEqual(result, datetime(2025, 9, 11, 8, 56, 56))

    def test_find_matching_videos_ranks_closest_timestamp_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            annotation_csv = root / "20250911_085656_for_annotation.csv"
            annotation_csv.write_text("path,type\n", encoding="utf-8")
            videos = root / "videos"
            videos.mkdir()
            best = videos / "session_20250911_085700.mp4"
            later = videos / "session_20250911_090500.mp4"
            for path in (best, later):
                path.write_text("video", encoding="utf-8")

            result = video_service.find_matching_videos(annotation_csv, videos, recursive=True, limit=2)

            self.assertEqual([item["path"] for item in result["matches"]], [best, later])

    def test_read_video_metadata_prefers_filename_over_filesystem_dates(self):
        video_path = Path("D:/202510131207.MOV")
        fake_stat = SimpleNamespace(
            st_ctime=datetime(2026, 5, 9, 14, 56, 30).timestamp(),
            st_mtime=datetime(2025, 10, 13, 12, 7, 39).timestamp(),
        )
        with mock.patch("synergie.services.video_service.Path.stat", return_value=fake_stat):
            with mock.patch("synergie.services.video_service.read_video_creation_time_with_ffprobe", return_value=None):
                metadata = video_service.read_video_metadata(video_path)

        self.assertEqual(metadata["recorded_at_source"], "filename")
        self.assertEqual(metadata["recorded_at"], datetime(2025, 10, 13, 12, 7, 0))

    def test_cached_video_path_keeps_local_workspace_drive_video_in_place(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "local.mov"
            video_path.write_bytes(b"video")

            result = video_service.cached_video_path(video_path)

        self.assertFalse(result["from_cache"])
        self.assertEqual(result["path"], video_path.resolve())

    def test_video_cache_size_and_clear_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "video_cache"
            cache_root.mkdir()
            (cache_root / "a.mov").write_bytes(b"12345")
            nested = cache_root / "nested"
            nested.mkdir()
            (nested / "b.mp4").write_bytes(b"123")

            self.assertEqual(video_service.video_cache_size_bytes(cache_root), 8)

            result = video_service.clear_video_cache(cache_root)

            self.assertEqual(result["removed_files"], 2)
            self.assertEqual(result["freed_bytes"], 8)
            self.assertEqual(video_service.video_cache_size_bytes(cache_root), 0)

    def test_clear_video_cache_handles_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "missing"

            result = video_service.clear_video_cache(cache_root)

            self.assertEqual(result["removed_files"], 0)
            self.assertEqual(result["freed_bytes"], 0)
