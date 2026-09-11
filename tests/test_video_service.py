import tempfile
import unittest
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
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

    def test_should_cache_mac_external_volume_from_local_workspace(self):
        source = PurePosixPath("/Volumes/ExternalDrive/20250929.MOV")
        workspace = PurePosixPath("/Users/micka/Documents/GIT/Synergie_Data")

        self.assertTrue(video_service.should_cache_video(source, workspace))

    def test_should_not_cache_mac_video_on_same_external_volume_as_workspace(self):
        source = PurePosixPath("/Volumes/ExternalDrive/videos/20250929.MOV")
        workspace = PurePosixPath("/Volumes/ExternalDrive/Synergie_Data")

        self.assertFalse(video_service.should_cache_video(source, workspace))

    def test_should_cache_different_windows_drive(self):
        source = PureWindowsPath("D:/videos/20250929.MOV")
        workspace = PureWindowsPath("C:/Users/micka/Documents/GIT/Synergie_Data")

        self.assertTrue(video_service.should_cache_video(source, workspace))

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

    def test_cached_video_path_falls_back_to_source_when_copy_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "external.mov"
            source.write_bytes(b"video")
            cache_root = root / "cache"
            with mock.patch("synergie.services.video_service.should_cache_video", return_value=True):
                with mock.patch("synergie.services.video_service.shutil.copy2", side_effect=OSError("Device not configured")):
                    result = video_service.cached_video_path(source, cache_root=cache_root)

            self.assertFalse(result["from_cache"])
            self.assertTrue(result["cache_failed"])
            self.assertEqual(result["path"], source.resolve())

    def test_cached_video_path_uses_existing_valid_cache_without_copying(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "external.mov"
            source.write_bytes(b"video")
            stat = source.stat()
            fingerprint = video_service.hashlib.sha1(f"{source.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8")).hexdigest()[:16]
            cache_root = root / "cache"
            cache_root.mkdir()
            cached = cache_root / f"external-{fingerprint}.mov"
            cached.write_bytes(b"video")

            with mock.patch("synergie.services.video_service.should_cache_video", return_value=True):
                with mock.patch("synergie.services.video_service.shutil.copy2") as copy_mock:
                    result = video_service.cached_video_path(source, cache_root=cache_root)

            self.assertTrue(result["from_cache"])
            self.assertEqual(result["path"], cached.resolve())
            copy_mock.assert_not_called()

    def test_video_cache_status_reports_copy_needed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "external.mov"
            source.write_bytes(b"video")
            with mock.patch("synergie.services.video_service.should_cache_video", return_value=True):
                status = video_service.video_cache_status(source, cache_root=root / "cache")

            self.assertTrue(status["should_cache"])
            self.assertTrue(status["copy_needed"])
            self.assertEqual(status["size_bytes"], 5)

    def test_video_cache_status_reports_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "external.mov"
            source.write_bytes(b"video")
            with mock.patch("synergie.services.video_service.should_cache_video", return_value=True):
                first_status = video_service.video_cache_status(source, cache_root=root / "cache")
                first_status["cache_path"].parent.mkdir()
                first_status["cache_path"].write_bytes(b"video")
                second_status = video_service.video_cache_status(source, cache_root=root / "cache")

            self.assertFalse(second_status["copy_needed"])

    def test_video_proxy_status_reports_missing_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "external.mov"
            source.write_bytes(b"video")
            with mock.patch("synergie.services.video_service.shutil.which", return_value=None):
                status = video_service.video_proxy_status(source, cache_root=Path(tmpdir) / "cache")

            self.assertFalse(status["can_create_proxy"])
            self.assertFalse(status["proxy_needed"])

    def test_video_proxy_status_can_be_disabled_by_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "external.mov"
            source.write_bytes(b"video")
            with mock.patch.dict("os.environ", {"SYNERGIE_DISABLE_VIDEO_PROXY": "1"}):
                with mock.patch("synergie.services.video_service.shutil.which", return_value="ffmpeg"):
                    status = video_service.video_proxy_status(source, cache_root=Path(tmpdir) / "cache")

            self.assertFalse(status["can_create_proxy"])
            self.assertFalse(status["proxy_needed"])
            self.assertIsNone(status["ffmpeg_path"])

    def test_optimized_playback_video_path_uses_proxy_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "external.mov"
            source.write_bytes(b"video")
            cache_root = root / "cache"

            def fake_create_proxy(_input_path, proxy_path):
                proxy_path.parent.mkdir(parents=True, exist_ok=True)
                proxy_path.write_bytes(b"proxy")

            with mock.patch("synergie.services.video_service.shutil.which", return_value="ffmpeg"):
                with mock.patch("synergie.services.video_service.should_cache_video", return_value=False):
                    with mock.patch("synergie.services.video_service._create_video_proxy", side_effect=fake_create_proxy):
                        result = video_service.optimized_playback_video_path(source, cache_root=cache_root)

            self.assertTrue(result["from_proxy"])
            self.assertTrue(result["proxy_created"])
            self.assertEqual(Path(result["path"]).suffix, ".mp4")

    def test_optimized_playback_video_path_falls_back_when_proxy_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "external.mov"
            source.write_bytes(b"video")
            with mock.patch("synergie.services.video_service.shutil.which", return_value="ffmpeg"):
                with mock.patch("synergie.services.video_service.should_cache_video", return_value=False):
                    with mock.patch("synergie.services.video_service._create_video_proxy", side_effect=OSError("ffmpeg failed")):
                        result = video_service.optimized_playback_video_path(source, cache_root=root / "cache")

            self.assertFalse(result["from_proxy"])
            self.assertTrue(result["proxy_failed"])
            self.assertEqual(result["path"], source.resolve())
