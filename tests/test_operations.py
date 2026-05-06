import tempfile
import unittest
from pathlib import Path

from synergie import operations


class OperationsTests(unittest.TestCase):
    def test_list_session_csv_files_returns_matching_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "2009" / "1331"
            session_path.mkdir(parents=True)
            csv_path = session_path / "sample.csv"
            txt_path = session_path / "ignore.txt"
            csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
            txt_path.write_text("x", encoding="utf-8")

            files = operations.list_session_csv_files("1331", raw_root=tmpdir)

            self.assertEqual(files, [csv_path])

    def test_list_session_csv_files_returns_empty_list_for_missing_session_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = operations.list_session_csv_files("1331", raw_root=tmpdir)
            self.assertEqual(files, [])

    def test_list_directory_files_returns_all_files_in_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "sample.csv"
            txt_path = root / "notes.txt"
            nested_dir = root / "nested"
            nested_dir.mkdir()
            csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
            txt_path.write_text("hello", encoding="utf-8")
            (nested_dir / "ignore.csv").write_text("x", encoding="utf-8")

            files = operations.list_directory_files(root)

            self.assertEqual(files, [txt_path, csv_path])

    def test_parse_training_id_from_csv_path_extracts_identifier(self):
        csv_path = Path("data/raw/2009/1331/123_training42.csv")
        self.assertEqual(operations.parse_training_id_from_csv_path(csv_path), "training42")

    def test_parse_training_id_from_csv_path_returns_none_for_invalid_name(self):
        csv_path = Path("data/raw/2009/1331/training42.csv")
        self.assertIsNone(operations.parse_training_id_from_csv_path(csv_path))

    def test_session_synchro_reads_known_session_metadata(self):
        self.assertEqual(
            operations.session_synchro("1331"),
            operations.session_metadata("1331")["sample_time_fine_synchro"],
        )

    def test_describe_file_returns_basic_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "example.csv"
            file_path.write_text("a,b\n1,2\n", encoding="utf-8")

            description = operations.describe_file(file_path)

            self.assertEqual(description["path"], file_path)
            self.assertEqual(description["name"], "example.csv")
            self.assertEqual(description["suffix"], ".csv")
            self.assertGreater(description["size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
