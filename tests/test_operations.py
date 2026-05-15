import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from synergie import operations
from synergie import pretrained_models
from synergie import session_store

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


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

    def test_parse_new_imu_filename_extracts_metadata(self):
        metadata = operations.parse_new_imu_filename("1_D422CD0076F7_20250911_085656.csv")

        self.assertEqual(metadata["sensor_id"], "1")
        self.assertEqual(metadata["device_id"], "D422CD0076F7")
        self.assertEqual(metadata["date_token"], "20250911")
        self.assertEqual(metadata["time_token"], "085656")
        self.assertEqual(metadata["recorded_at"].year, 2025)

    def test_suggest_for_annotation_output_path_uses_clear_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = operations.suggest_for_annotation_output_path(
                "1_D422CD0076F7_20250911_085656.csv",
                pending_root=tmpdir,
            )

            self.assertEqual(candidate.name, "20250911_085656_for_annotation.csv")

    def test_process_csv_file_forwards_selected_model_paths(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "raw.csv"
            output_path = Path(tmpdir) / "out.csv"
            input_path.write_text("a\n1\n", encoding="utf-8")
            with mock.patch("core.data_treatment.data_generation.exporter.export", return_value=pd.DataFrame({"x": [1]})) as export_mock:
                operations.process_csv_file(
                    str(input_path),
                    output_path=str(output_path),
                    type_model_path="type.keras",
                    success_model_path="success.keras",
                )

            export_mock.assert_called_once()
            self.assertEqual(export_mock.call_args.kwargs["type_model_path"], "type.keras")
            self.assertEqual(export_mock.call_args.kwargs["success_model_path"], "success.keras")

    def test_list_new_imu_files_skips_hidden_and_done_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "29092025" / "1002"
            nested.mkdir(parents=True)
            (nested / "1_D422CD0076F7_20250929_100230.csv").write_text("x", encoding="utf-8")
            (nested / "._1_D422CD0076F7_20250929_100230.csv").write_text("x", encoding="utf-8")
            done_dir = root / "done"
            done_dir.mkdir()
            (done_dir / "2_D422CD007712_20250929_100230.csv").write_text("x", encoding="utf-8")

            files = operations.list_new_imu_files(root=root)

            self.assertEqual(len(files), 1)
            self.assertEqual(files[0]["sensor_id"], "1")

    def test_annotation_turn_options_keep_simple_radios_for_axel(self):
        self.assertEqual(operations.annotation_turn_options("Axel"), ["1", "2", "3", "4"])
        self.assertEqual(operations.annotation_turn_options(5), ["1", "2", "3", "4"])
        self.assertEqual(operations.annotation_turn_options("Loop"), ["1", "2", "3", "4"])
        self.assertEqual(operations.annotation_turn_options("toe_loop"), ["1", "2", "3", "4"])

    def test_annotation_turn_value_helpers_convert_axel_half_turns_on_save(self):
        self.assertEqual(operations.annotation_turn_value_for_storage("axel", "2"), "2.5")
        self.assertEqual(operations.annotation_turn_value_for_storage("loop", "2"), "2")
        self.assertEqual(operations.annotation_turn_value_for_ui("axel", "3.5"), "3")
        self.assertEqual(operations.annotation_turn_value_for_ui("loop", "3"), "3")

    def test_annotation_review_status_maps_to_backend_columns(self):
        self.assertEqual(
            operations.annotation_review_status_to_backend("not_seen_on_video"),
            {"video_status": "not_seen_on_video", "detection_status": "detected_jump"},
        )
        self.assertEqual(
            operations.annotation_review_status_to_backend("not_a_jump"),
            {"video_status": "visible", "detection_status": "not_a_jump"},
        )
        self.assertEqual(
            operations.annotation_review_status_from_row({"video_status": "visible", "detection_status": "manual_missing_jump"}),
            "manual_missing_jump",
        )
        self.assertEqual(
            operations.annotation_review_status_from_row({"video_status": "not_seen_on_video", "detection_status": "detected_jump"}),
            "not_seen_on_video",
        )

    def test_summarize_annotation_progress_counts_explicit_pending_status(self):
        import pandas as pd

        rows = pd.DataFrame(
            [
                {"annotation_status": "pending", "type": 8, "success": 2},
                {"annotation_status": "annotated", "type": 0, "success": 1},
                {"annotation_status": "annotated", "type": 8, "success": 2},
            ]
        )

        self.assertEqual(
            operations.summarize_annotation_progress(rows),
            {"total": 3, "pending": 1, "completed": 2},
        )

    def test_summarize_pending_annotation_files_aggregates_all_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a_for_annotation.csv").write_text(
                "annotation_status,type,success\npending,8,2\nannotated,0,1\n",
                encoding="utf-8",
            )
            (root / "b_for_annotation.csv").write_text(
                "annotation_status,type,success\npending,8,2\npending,8,2\n",
                encoding="utf-8",
            )

            summary = operations.summarize_pending_annotation_files(root)

            self.assertEqual(summary["pending"], 3)
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["total"], 4)
            self.assertEqual(len(summary["files"]), 2)

    def test_annotate_combination_flags_marks_close_jumps_per_sensor(self):
        import pandas as pd

        rows = pd.DataFrame(
            [
                {"sensor_id": "1", "synced_start_ms": 1000.0},
                {"sensor_id": "1", "synced_start_ms": 2200.0},
                {"sensor_id": "1", "synced_start_ms": 5000.0},
                {"sensor_id": "2", "synced_start_ms": 2100.0},
            ]
        )

        result = operations.annotate_combination_flags(rows)

        self.assertEqual(result["combination"].tolist(), [True, True, False, False])

    def test_suggest_turns_from_rotation_bounds_result(self):
        self.assertEqual(operations.suggest_turns_from_rotation(0.6), "1")
        self.assertEqual(operations.suggest_turns_from_rotation(2.3), "2")
        self.assertEqual(operations.suggest_turns_from_rotation(5.1), "4")

    def test_analyze_detection_review_labels_collects_false_positive_and_negative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a_for_annotation.csv").write_text(
                "sensor_id,path,start_ms,synced_start_ms,video_status,detection_status\n"
                "1,a.csv,10,10,visible,not_a_jump\n"
                "2,b.csv,20,20,visible,manual_missing_jump\n"
                "3,c.csv,30,30,visible,detected_jump\n",
                encoding="utf-8",
            )

            analysis = operations.analyze_detection_review_labels(root)

            self.assertEqual(analysis["false_positive_count"], 1)
            self.assertEqual(analysis["false_negative_count"], 1)
            self.assertEqual(analysis["reviewed_detected"], 1)

    @unittest.skipUnless(HAS_NUMPY, "numpy is not available")
    def test_optimize_detection_parameters_returns_ranked_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            positive = root / "positive.csv"
            negative = root / "negative.csv"
            positive.write_text("Gyr_X\n0\n0\n100\n-100\n0\n", encoding="utf-8")
            negative.write_text("Gyr_X\n0\n0\n0\n0\n0\n", encoding="utf-8")
            (root / "a_for_annotation.csv").write_text(
                "path,video_status,detection_status\n"
                f"{positive.as_posix()},visible,detected_jump\n"
                f"{negative.as_posix()},visible,not_a_jump\n",
                encoding="utf-8",
            )

            result = operations.optimize_detection_parameters(root, thresholds=[-0.01], smoothing_sigmas=[1])

            self.assertEqual(result["reviewed_segments"], 2)
            self.assertEqual(len(result["results"]), 1)
            self.assertIsNotNone(result["best"])

    def test_sample_hyperparameter_trials_is_reproducible_and_bounded(self):
        first = operations.sample_hyperparameter_trials("success", "tcn", 4, random_seed=7)
        second = operations.sample_hyperparameter_trials("success", "tcn", 4, random_seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertLessEqual(len(first), len(operations.hyperparameter_search_space("success", "tcn")))

    def test_latest_model_path_for_task_uses_active_aliases(self):
        self.assertEqual(operations.latest_model_path_for_task("type"), "core/model/saved_models/checkpoint.keras")
        self.assertEqual(operations.latest_model_path_for_task("success"), "core/model/saved_models/success.keras")

    @unittest.skipUnless(HAS_NUMPY, "numpy is not available")
    def test_analyze_jump_quality_flags_missing_file_and_saturation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            segment_ok = dataset_root / "jump_ok.csv"
            segment_sat = dataset_root / "jump_sat.csv"
            segment_ok.write_text(
                "ms,Gyr_X,Acc_X,X_gyr_second_derivative\n0,10,1,0.1\n100,20,2,0.2\n200,30,3,0.3\n",
                encoding="utf-8",
            )
            segment_sat.write_text(
                "ms,Gyr_X,Acc_X,X_gyr_second_derivative\n0,10,1,0.1\n100,2100,2,0.2\n200,25,3,0.3\n",
                encoding="utf-8",
            )
            (dataset_root / "jumplist.csv").write_text(
                "path,type,skater,success,rotations\n"
                f"{segment_ok.as_posix()},0,alice,1,2.0\n"
                f"{segment_sat.as_posix()},0,bob,1,2.0\n"
                f"{(dataset_root / 'missing.csv').as_posix()},0,charlie,1,2.0\n",
                encoding="utf-8",
            )

            analysis = operations.analyze_jump_quality(dataset_root, saturation_threshold=1900.0)

            self.assertEqual(analysis["total_labelled_jumps"], 3)
            self.assertEqual(analysis["suspicious_count"], 2)
            suspicious_reasons = {record["skater"]: record["reasons"] for record in analysis["suspicious_records"]}
            self.assertIn("gyro_saturation", suspicious_reasons["bob"])
            self.assertIn("missing_segment_file", suspicious_reasons["charlie"])

    def test_list_new_imu_sessions_groups_files_by_shared_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "1_D422CD0076F7_20250911_085656.csv").write_text("x", encoding="utf-8")
            (root / "2_D422CD007712_20250911_085656.csv").write_text("x", encoding="utf-8")
            (root / "3_D422CD0077D1_20250911_090000.csv").write_text("x", encoding="utf-8")

            sessions = operations.list_new_imu_sessions(root=root)

            self.assertEqual(len(sessions), 2)
            self.assertEqual(sessions[0]["session_key"], "20250911_085656")
            self.assertEqual(len(sessions[0]["files"]), 2)
            self.assertEqual([item["sensor_id"] for item in sessions[0]["files"]], ["1", "2"])

    def test_annotation_metadata_persists_video_path_and_sensor_offsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_csv = Path(tmpdir) / "20250911_085656_for_annotation.csv"
            annotation_csv.write_text("path,type\n", encoding="utf-8")

            operations.set_annotation_video_path(annotation_csv, Path(tmpdir) / "session.mp4")
            operations.set_annotation_video_directory(annotation_csv, Path(tmpdir) / "videos")
            operations.set_annotation_sensor_sync_offset(annotation_csv, "2", 183.25)

            metadata = operations.load_annotation_metadata(annotation_csv)

            self.assertTrue(operations.annotation_metadata_path(annotation_csv).exists())
            self.assertTrue(metadata["video_path"].endswith("session.mp4"))
            self.assertTrue(metadata["video_directory"].endswith("videos"))
            self.assertEqual(metadata["sensor_sync_offsets_ms"]["2"], 183.25)
            self.assertEqual(operations.get_annotation_sensor_sync_offset(metadata, "2"), 183.25)

    def test_compute_annotation_jump_video_time_ms_applies_sensor_offset(self):
        row = {"synced_start_ms": "1520.5"}

        result = operations.compute_annotation_jump_video_time_ms(row, sensor_sync_offset_ms=180)

        self.assertEqual(result, 1700.5)

    def test_annotation_reference_datetime_prefers_recorded_at_column(self):
        annotation_csv = Path("20250911_085656_for_annotation.csv")

        result = operations.annotation_reference_datetime(
            annotation_csv,
            annotation_rows={"recorded_at": ["2025-09-11T08:56:58", "2025-09-11T08:56:56"]},
        )

        self.assertEqual(result, datetime(2025, 9, 11, 8, 56, 56))

    def test_find_matching_videos_ranks_filename_timestamp_closest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            annotation_csv = root / "20250911_085656_for_annotation.csv"
            annotation_csv.write_text("path,type\n", encoding="utf-8")
            videos = root / "videos"
            videos.mkdir()
            best = videos / "session_20250911_085700.mp4"
            later = videos / "session_20250911_090500.mp4"
            earlier = videos / "session_20250911_084000.mp4"
            for path in (best, later, earlier):
                path.write_text("video", encoding="utf-8")

            result = operations.find_matching_videos(annotation_csv, videos, recursive=True, limit=3)

            self.assertEqual(result["reference_datetime"], datetime(2025, 9, 11, 8, 56, 56))
            self.assertEqual([item["path"] for item in result["matches"]], [best, later, earlier])
            self.assertEqual(result["matches"][0]["recorded_at_source"], "filename")

    def test_parse_datetime_from_text_supports_compact_minute_precision_video_names(self):
        result = operations._parse_datetime_from_text("202510131207")

        self.assertEqual(result, datetime(2025, 10, 13, 12, 7, 0))

    def test_read_video_metadata_prefers_modified_over_created_when_no_better_source_exists(self):
        video_path = Path("D:/202510131207.MOV")
        fake_stat = SimpleNamespace(
            st_ctime=datetime(2026, 5, 9, 14, 56, 30).timestamp(),
            st_mtime=datetime(2025, 10, 13, 12, 7, 39).timestamp(),
        )
        with mock.patch("synergie.operations.Path.stat", return_value=fake_stat):
            with mock.patch("synergie.operations._read_video_creation_time_with_ffprobe", return_value=None):
                with mock.patch("synergie.operations._parse_datetime_from_text", return_value=None):
                    metadata = operations.read_video_metadata(video_path)

        self.assertEqual(metadata["recorded_at_source"], "filesystem.modified")
        self.assertEqual(metadata["recorded_at"], datetime(2025, 10, 13, 12, 7, 39))

    def test_read_video_metadata_prefers_filename_over_filesystem_dates_for_copied_videos(self):
        video_path = Path("D:/202510131207.MOV")
        fake_stat = SimpleNamespace(
            st_ctime=datetime(2026, 5, 9, 14, 56, 30).timestamp(),
            st_mtime=datetime(2025, 10, 13, 12, 7, 39).timestamp(),
        )
        with mock.patch("synergie.operations.Path.stat", return_value=fake_stat):
            with mock.patch("synergie.operations._read_video_creation_time_with_ffprobe", return_value=None):
                metadata = operations.read_video_metadata(video_path)

        self.assertEqual(metadata["recorded_at_source"], "filename")
        self.assertEqual(metadata["recorded_at"], datetime(2025, 10, 13, 12, 7, 0))

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
            self.assertEqual(stats["recommended_class_weight"], {"0": 1.5, "1": 0.75})
            self.assertFalse(stats["stratified_split_possible"])
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
            self.assertFalse(stats["has_duplicates"])

    def test_describe_training_dataset_reports_stratified_split_for_balanced_success_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            (dataset_root / "jumplist.csv").write_text(
                "type,success,skater\n"
                "0,0,alice\n"
                "0,0,alice\n"
                "1,1,bob\n"
                "1,1,bob\n"
                "2,1,charlie\n"
                "2,0,charlie\n",
                encoding="utf-8",
            )

            stats = operations.describe_training_dataset("success", dataset_root)

            self.assertEqual(stats["class_counts"], {"0": 3, "1": 3})
            self.assertEqual(stats["recommended_class_weight"], {"0": 1.0, "1": 1.0})
            self.assertTrue(stats["stratified_split_possible"])

    def test_find_training_dataset_duplicates_reports_repeated_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            (dataset_root / "jumplist.csv").write_text(
                "path,type,success,skater\n"
                "jump_a.csv,0,1,a\n"
                "jump_a.csv,0,1,a\n"
                "jump_b.csv,1,0,b\n",
                encoding="utf-8",
            )

            duplicates = operations.find_training_dataset_duplicates(dataset_root)

            self.assertTrue(duplicates["has_duplicates"])
            self.assertEqual(duplicates["duplicate_rows"], 2)
            self.assertEqual(duplicates["duplicate_paths"], ["jump_a.csv"])

    def test_finalize_annotation_file_archives_moves_and_merges_trainable_rows(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "data" / "annotated" / "total"
            annotated_root = root / "data" / "annotated"
            pending_root = root / "data" / "pending"
            segment_dir = pending_root / "segments" / "session" / "sensor1"
            dataset_root.mkdir(parents=True)
            segment_dir.mkdir(parents=True)
            (dataset_root / "jumplist.csv").write_text(
                "path,type,success,skater\nexisting.csv,0,1,alice\n",
                encoding="utf-8",
            )
            segment = segment_dir / "jump001.csv"
            segment.write_text("ms,Gyr_X\n0,0\n", encoding="utf-8")
            annotation_file = pending_root / "20250911_085656_for_annotation.csv"
            pd.DataFrame(
                [
                    {
                        "path": segment.as_posix(),
                        "type": 1,
                        "success": 1,
                        "skater": "sensor_1",
                        "athlete_id": "bob",
                        "annotation_status": "annotated",
                        "sensor_id": "1",
                        "session_key": "20250911_085656",
                    }
                ]
            ).to_csv(annotation_file, index=False)

            result = operations.finalize_annotation_file(
                annotation_file,
                dataset_path=dataset_root,
                annotated_root=annotated_root,
            )

            merged = pd.read_csv(dataset_root / "jumplist.csv")
            self.assertEqual(result["rows_added"], 1)
            self.assertTrue(result["archive_path"].exists())
            self.assertEqual(len(merged), 2)
            self.assertEqual(merged.iloc[1]["skater"], "bob")
            self.assertTrue(Path(merged.iloc[1]["path"]).exists())
            self.assertFalse(annotation_file.exists())
            self.assertTrue(result["completed_annotation_path"].exists())
            self.assertTrue(operations.load_training_dataset_state(dataset_root)["retraining_recommended"])

    def test_finalize_annotation_file_refuses_pending_rows(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "total"
            dataset_root.mkdir()
            (dataset_root / "jumplist.csv").write_text("path,type,success,skater\n", encoding="utf-8")
            annotation_file = root / "pending_for_annotation.csv"
            pd.DataFrame(
                [{"path": "jump.csv", "type": 0, "success": 1, "annotation_status": "pending"}]
            ).to_csv(annotation_file, index=False)

            with self.assertRaisesRegex(ValueError, "still pending"):
                operations.finalize_annotation_file(annotation_file, dataset_path=dataset_root, annotated_root=root / "annotated")

    def test_list_pretrained_training_models_filters_compatible_entries(self):
        original_file = pretrained_models.PRETRAINED_MODELS_FILE
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pretrained_models.json"
            model_dir = Path(tmpdir) / "existing_model.keras"
            model_dir.write_text("placeholder", encoding="utf-8")
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

    def test_list_pretrained_training_models_orders_most_recent_first(self):
        original_file = pretrained_models.PRETRAINED_MODELS_FILE
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "pretrained_models.json"
            older_model = Path(tmpdir) / "older.keras"
            newer_model = Path(tmpdir) / "newer.keras"
            older_model.write_text("older", encoding="utf-8")
            newer_model.write_text("newer", encoding="utf-8")
            config_path.write_text(
                "["
                "{\"id\":\"success-tcn-20260506-235959\",\"label\":\"Older\",\"task\":\"success\",\"architecture\":\"tcn\",\"path\":\""
                + str(older_model).replace("\\", "\\\\")
                + "\",\"compatible\":true,\"notes\":\"older\",\"performance\":{\"test_accuracy\":0.8}},"
                "{\"id\":\"success-tcn-20260507-071710\",\"label\":\"Newer\",\"task\":\"success\",\"architecture\":\"tcn\",\"path\":\""
                + str(newer_model).replace("\\", "\\\\")
                + "\",\"compatible\":true,\"notes\":\"newer\",\"performance\":{\"test_accuracy\":0.85}}"
                "]",
                encoding="utf-8",
            )
            pretrained_models.PRETRAINED_MODELS_FILE = config_path

            models = operations.list_pretrained_training_models(task="success", compatible_only=True)

            self.assertEqual([model["label"] for model in models], ["Newer", "Older"])
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
