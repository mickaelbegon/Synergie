import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import constants
from synergie.services.training_cache_service import load_or_build_training_cache, training_cache_path


class TrainingCacheServiceTests(unittest.TestCase):
    def _build_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        dataset.mkdir()
        segment = dataset / "segment.csv"
        pd.DataFrame({column: range(300) for column in constants.fields_to_keep}).to_csv(segment, index=False)
        (dataset / "jumplist.csv").write_text(
            f"path,type,success,skater\n{segment.as_posix()},0,1,alice\n",
            encoding="utf-8",
        )
        return dataset

    def test_builds_and_reuses_numeric_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self._build_dataset(Path(temp_dir))

            built = load_or_build_training_cache(
                dataset,
                type_window_start=0,
                type_window_frames=240,
                success_window_start=160,
                success_window_frames=140,
            )
            original_read_csv = pd.read_csv

            def guarded_read_csv(path, *args, **kwargs):
                if Path(path).name == "segment.csv":
                    raise AssertionError("cache should avoid segment CSV reads")
                return original_read_csv(path, *args, **kwargs)

            with patch("pandas.read_csv", side_effect=guarded_read_csv):
                reused = load_or_build_training_cache(
                    dataset,
                    type_window_start=0,
                    type_window_frames=240,
                    success_window_start=160,
                    success_window_frames=140,
                )
            cache_exists = training_cache_path(dataset).exists()

        self.assertFalse(built["from_cache"])
        self.assertTrue(cache_exists)
        self.assertTrue(reused["from_cache"])
        self.assertEqual(tuple(reused["type_windows"].shape), (1, 240, len(constants.fields_to_keep)))
        self.assertEqual(tuple(reused["success_windows"].shape), (1, 140, len(constants.fields_to_keep)))

    def test_rebuilds_when_segment_file_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self._build_dataset(Path(temp_dir))
            first = load_or_build_training_cache(
                dataset,
                type_window_start=0,
                type_window_frames=240,
                success_window_start=160,
                success_window_frames=140,
            )
            segment = dataset / "segment.csv"
            segment.write_text(segment.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            second = load_or_build_training_cache(
                dataset,
                type_window_start=0,
                type_window_frames=240,
                success_window_start=160,
                success_window_frames=140,
            )

        self.assertFalse(first["from_cache"])
        self.assertFalse(second["from_cache"])

    def test_can_skip_incomplete_windows_for_benchmarks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self._build_dataset(Path(temp_dir))
            result = load_or_build_training_cache(
                dataset,
                type_window_start=0,
                type_window_frames=240,
                success_window_start=200,
                success_window_frames=140,
                skip_incomplete_windows=True,
            )

        expected_path = str(dataset / "segment.csv").replace("\\", "/")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["metadata"]["skipped_paths"], [expected_path])
