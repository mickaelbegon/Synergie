import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from synergie import operations
from synergie import pretrained_models
from synergie import session_store


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

    def test_next_jumplist_output_path_returns_first_free_increment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "jumplist_partie1.csv").write_text("x", encoding="utf-8")
            (root / "jumplist_partie2.csv").write_text("x", encoding="utf-8")

            candidate = operations.next_jumplist_output_path(root)

            self.assertEqual(candidate, root / "jumplist_partie3.csv")

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

    def test_add_session_persists_new_session_in_json_store(self):
        original_file = session_store.SESSIONS_FILE
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file = Path(tmpdir) / "sessions.json"
            session_store.SESSIONS_FILE = temp_file
            session_store.save_sessions({"1331": {"path": "2009/1331", "sample_time_fine_synchro": 965369596}})

            metadata = operations.add_session("9999", "custom/9999", 123456)

            self.assertEqual(metadata["path"], "custom/9999")
            self.assertEqual(metadata["sample_time_fine_synchro"], 123456)
            sessions = session_store.load_sessions()
            self.assertIn("9999", sessions)
        session_store.SESSIONS_FILE = original_file

    def test_describe_training_dataset_returns_class_distribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            (dataset_root / "jumplist.csv").write_text(
                "type,success,skater\n"
                "0,1,alice\n"
                "1,0,bob\n"
                "1,1,bob\n"
                "8,1,skip_me\n"
                "2,2,skip_me_too\n",
                encoding="utf-8",
            )

            stats = operations.describe_training_dataset("type", dataset_root)

            self.assertEqual(stats["task"], "type")
            self.assertEqual(stats["base_samples"], 3)
            self.assertEqual(stats["effective_samples"], 6)
            self.assertEqual(stats["unique_skaters"], 2)
            self.assertEqual(stats["class_counts"], {"0": 1, "1": 2})
            self.assertTrue(stats["augment_mirror"])

    def test_describe_training_dataset_accepts_float_encoded_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            (dataset_root / "jumplist.csv").write_text(
                "type,success,skater\n"
                "1.0,1.0,alice\n"
                "0.0,0.0,bob\n",
                encoding="utf-8",
            )

            stats = operations.describe_training_dataset("success", dataset_root)

            self.assertEqual(stats["base_samples"], 2)
            self.assertEqual(stats["class_counts"], {"0": 1, "1": 1})

    def test_list_pretrained_training_models_filters_compatible_entries(self):
        original_file = pretrained_models.PRETRAINED_MODELS_FILE
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pretrained_models.json"
            model_dir = Path(tmpdir) / "existing_model"
            model_dir.mkdir()
            config_path.write_text(
                "["
                "{\"id\":\"ok\",\"label\":\"Compatible\",\"task\":\"type\",\"architecture\":\"inceptiontime\",\"path\":\""
                + str(model_dir).replace("\\", "\\\\")
                + "\",\"compatible\":true,\"notes\":\"ok\",\"performance\":{\"test_accuracy\":0.9}},"
                "{\"id\":\"bad\",\"label\":\"Broken\",\"task\":\"type\",\"architecture\":\"transformer\",\"path\":\"missing\",\"compatible\":false,\"notes\":\"broken\",\"performance\":null}"
                "]",
                encoding="utf-8",
            )
            pretrained_models.PRETRAINED_MODELS_FILE = config_path

            models = operations.list_pretrained_training_models(task="type", compatible_only=True)

            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["id"], "ok")
            self.assertIn("acc 0.900", operations.format_pretrained_model_label(models[0]))
        pretrained_models.PRETRAINED_MODELS_FILE = original_file

    def test_register_trained_model_persists_unique_entry(self):
        original_file = pretrained_models.PRETRAINED_MODELS_FILE
        original_root = pretrained_models.MODEL_ARCHIVE_ROOT
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pretrained_models.json"
            archive_root = Path(tmpdir) / "archive"
            model_path = archive_root / "type" / "type-inceptiontime-20260507-120000"
            model_path.mkdir(parents=True)
            pretrained_models.PRETRAINED_MODELS_FILE = config_path
            pretrained_models.MODEL_ARCHIVE_ROOT = archive_root

            model_id = pretrained_models.build_trained_model_id(
                "type",
                "inceptiontime",
                trained_at=datetime(2026, 5, 7, 12, 0, 0),
            )
            self.assertEqual(model_id, "type-inceptiontime-20260507-120000")

            entry = pretrained_models.register_trained_model(
                model_id=model_id,
                label="Type inceptiontime type-inceptiontime-20260507-120000",
                task="type",
                architecture="inceptiontime",
                path=str(model_path).replace("\\", "/"),
                dataset="data/annotated/total",
                performance={"test_accuracy": 0.91, "test_samples": 120},
                notes="test",
            )

            self.assertTrue(entry["path_exists"])
            persisted = pretrained_models.load_pretrained_models()
            self.assertEqual(len(persisted), 1)
            self.assertEqual(persisted[0]["id"], model_id)
            self.assertEqual(persisted[0]["performance"]["test_accuracy"], 0.91)
        pretrained_models.PRETRAINED_MODELS_FILE = original_file
        pretrained_models.MODEL_ARCHIVE_ROOT = original_root


if __name__ == "__main__":
    unittest.main()
