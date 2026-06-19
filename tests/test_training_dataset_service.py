import tempfile
import unittest
from pathlib import Path

from synergie.services.training_dataset_service import (
    describe_training_dataset,
    exclude_training_dataset_row,
    find_training_dataset_duplicates,
    load_training_dataset_rows,
)


class TrainingDatasetServiceTests(unittest.TestCase):
    def test_describe_training_dataset_reports_balance_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            (dataset_root / "jumplist.csv").write_text(
                "path,type,success,skater\n"
                "a.csv,0,1,alice\n"
                "a.csv,0,1,alice\n"
                "b.csv,1,0,bob\n"
                "ignored.csv,8,2,charlie\n",
                encoding="utf-8",
            )

            description = describe_training_dataset("success", str(dataset_root), augment_mirror=False)

            self.assertEqual(description["base_samples"], 3)
            self.assertEqual(description["class_counts"], {"0": 1, "1": 2})
            self.assertTrue(description["has_duplicates"])

    def test_find_training_dataset_duplicates_reports_repeated_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            (dataset_root / "jumplist.csv").write_text(
                "path,type,success,skater\n"
                "a.csv,0,1,alice\n"
                "a.csv,0,1,alice\n"
                "b.csv,1,0,bob\n",
                encoding="utf-8",
            )

            duplicates = find_training_dataset_duplicates(dataset_root)

            self.assertEqual(duplicates["duplicate_paths"], ["a.csv"])
            self.assertEqual(duplicates["duplicate_rows"], 2)

    def test_load_training_dataset_rows_skips_excluded_rows_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            (dataset_root / "jumplist.csv").write_text(
                "path,type,success,skater\n"
                "a.csv,0,1,alice\n"
                "b.csv,8,2,bob\n"
                "c.csv,1,0,charlie\n",
                encoding="utf-8",
            )

            rows = load_training_dataset_rows(dataset_root)

            self.assertEqual([row["path"] for row in rows], ["a.csv", "c.csv"])
            self.assertEqual([row["row_index"] for row in rows], [0, 2])

    def test_exclude_training_dataset_row_marks_row_non_trainable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            jumplist_path = dataset_root / "jumplist.csv"
            jumplist_path.write_text(
                "path,type,success,skater\n"
                "a.csv,0,1,alice\n"
                "b.csv,1,0,bob\n",
                encoding="utf-8",
            )

            updated = exclude_training_dataset_row(dataset_root, 1, excluded_reason="weird_signal")
            rows = load_training_dataset_rows(dataset_root)

            self.assertEqual(updated["type"], 8)
            self.assertEqual(updated["success"], 2)
            self.assertEqual(updated["excluded_reason"], "weird_signal")
            self.assertEqual([row["path"] for row in rows], ["a.csv"])
