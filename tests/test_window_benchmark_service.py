import unittest
from pathlib import Path

from synergie.services.window_benchmark_service import benchmark_temporal_offsets, benchmark_temporal_windows
from synergie.services.segment_reexport_service import _matching_raw_path, ensure_post_takeoff_context, ensure_pre_takeoff_context


class WindowBenchmarkServiceTests(unittest.TestCase):
    def test_requires_no_explicit_windows_to_define_defaults(self):
        self.assertTrue(callable(benchmark_temporal_windows))

    def test_rejects_negative_offsets_without_longer_segments(self):
        with self.assertRaisesRegex(ValueError, "re-exported with more pre-takeoff context"):
            benchmark_temporal_offsets("type", "unused", "inceptiontime", offsets=[-40], auto_reexport=False)

    def test_matching_raw_path_supports_legacy_segment_names(self):
        self.assertEqual(
            _matching_raw_path(Path("data/annotated/0406/0927/0927_3_2816.csv")),
            Path("data/raw/0406/0927/3_D422CD0077D1_20240604_092734.csv"),
        )

    def test_reexport_context_accepts_already_sufficient_segments(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "jumplist.csv").write_text("path\nsegment.csv\n", encoding="utf-8")
            with patch(
                "synergie.services.segment_reexport_service.reexport_segment_with_context",
                return_value=False,
            ):
                result = ensure_pre_takeoff_context(dataset, 180)

        self.assertEqual(result, {"reexported": 0, "already_sufficient": 1, "skipped": 0})

    def test_post_context_accepts_already_sufficient_segments(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        with TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            dataset.mkdir()
            (dataset / "jumplist.csv").write_text("path\nsegment.csv\n", encoding="utf-8")
            with patch(
                "synergie.services.segment_reexport_service.reexport_segment_with_post_context",
                return_value=False,
            ):
                result = ensure_post_takeoff_context(dataset, 220)

        self.assertEqual(result, {"reexported": 0, "already_sufficient": 1, "skipped": 0})
