import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synergie.services.detection_tuning_service import (
    analyze_detection_review_labels,
    audit_detection_regressions,
    load_optimized_detection_parameters,
    optimize_detection_parameters,
    save_optimized_detection_parameters,
    write_detection_regression_audit,
)
from synergie.services.hdf5_archive_service import export_hdf5_archive


class DetectionTuningServiceTests(unittest.TestCase):
    def test_audits_issue_categories_without_changing_annotation_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "session_for_annotation.csv"
            content = (
                "sensor_id,path,video_status,detection_status,detection_issue_category,rotation_direction\n"
                "1,a.csv,visible,not_a_jump,twizzle,left\n"
                "2,b.csv,visible,manual_missing_jump,,gauche\n"
                "3,c.csv,visible,not_a_jump_or_drill,doublon,right\n"
                "4,d.csv,visible,detected_jump,,right\n"
            )
            source.write_text(content, encoding="utf-8")

            result = audit_detection_regressions(root)

            self.assertEqual(source.read_text(encoding="utf-8"), content)
            self.assertEqual(result["false_positive_count"], 2)
            self.assertEqual(result["false_negative_count"], 1)
            self.assertEqual(result["reviewed_detected"], 1)
            self.assertEqual(result["records"][0]["category"], "twizzle")
            self.assertEqual(result["records"][0]["rotation_direction"], "left")
            self.assertEqual(result["records"][1]["category"], "unclassified_false_negative")
            self.assertEqual(result["records"][2]["category"], "duplicate_detection")
            self.assertEqual(len(result["source_files"][0]["sha256"]), 64)

    def test_writes_audit_only_to_explicit_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "session_for_annotation.csv"
            source.write_text(
                "video_status,detection_status\nvisible,not_a_jump\n",
                encoding="utf-8",
            )
            output = root / "reports" / "regression.json"

            written = write_detection_regression_audit(root, output)

            self.assertEqual(written, output)
            self.assertTrue(output.exists())
            self.assertEqual(source.read_text(encoding="utf-8"), "video_status,detection_status\nvisible,not_a_jump\n")

    def test_ignores_pending_rows_when_annotation_status_is_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "session_for_annotation.csv").write_text(
                "annotation_status,video_status,detection_status\n"
                "pending,visible,detected_jump\n"
                "annotated,visible,detected_jump\n",
                encoding="utf-8",
            )

            result = audit_detection_regressions(root)

            self.assertEqual(result["reviewed_detected"], 1)
            self.assertEqual(result["total_reviewed"], 1)

    def test_collects_reviewed_false_positive_and_negative_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a_for_annotation.csv").write_text(
                "sensor_id,path,start_ms,synced_start_ms,video_status,detection_status\n"
                "1,a.csv,10,10,visible,not_a_jump\n"
                "2,b.csv,20,20,visible,weird_signal\n"
                "3,c.csv,30,30,visible,manual_missing_jump\n"
                "4,d.csv,40,40,visible,detected_jump\n"
                "5,e.csv,50,50,visible,jump_drill\n"
                "6,f.csv,60,60,visible,not_a_jump_or_drill\n",
                encoding="utf-8",
            )

            analysis = analyze_detection_review_labels(root)

            self.assertEqual(analysis["false_positive_count"], 4)
            self.assertEqual(analysis["false_negative_count"], 1)
            self.assertEqual(analysis["reviewed_detected"], 1)

    def test_ranks_detection_parameter_results(self):
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

            result = optimize_detection_parameters(root, thresholds=[-0.01], smoothing_sigmas=[1])

            self.assertEqual(result["reviewed_segments"], 2)
            self.assertEqual(len(result["results"]), 1)
            self.assertIsNotNone(result["best"])

    def test_optimizer_uses_only_reviewed_unique_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment = root / "segment.csv"
            segment.write_text("Gyr_X\n0\n0\n100\n-100\n0\n", encoding="utf-8")
            (root / "a_for_annotation.csv").write_text(
                "path,annotation_status,video_status,detection_status\n"
                f"{segment.as_posix()},annotated,visible,detected_jump\n"
                f"{segment.as_posix()},pending,visible,detected_jump\n",
                encoding="utf-8",
            )

            result = optimize_detection_parameters(root, thresholds=[-0.01], smoothing_sigmas=[1])

            self.assertEqual(result["reviewed_segments"], 1)

    def test_optimizer_rejects_conflicting_reviewed_duplicate_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment = root / "segment.csv"
            segment.write_text("Gyr_X\n0\n0\n100\n-100\n0\n", encoding="utf-8")
            (root / "a_for_annotation.csv").write_text(
                "path,annotation_status,video_status,detection_status\n"
                f"{segment.as_posix()},annotated,visible,detected_jump\n"
                f"{segment.as_posix()},annotated,visible,not_a_jump\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Conflicting reviewed detection labels"):
                optimize_detection_parameters(root, thresholds=[-0.01], smoothing_sigmas=[1])

    def test_optimizer_can_compare_both_derivative_polarities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment = root / "segment.csv"
            segment.write_text("Gyr_X\n0\n0\n100\n-100\n0\n", encoding="utf-8")
            (root / "a_for_annotation.csv").write_text(
                "path,annotation_status,video_status,detection_status\n"
                f"{segment.as_posix()},annotated,visible,detected_jump\n",
                encoding="utf-8",
            )

            result = optimize_detection_parameters(
                root,
                thresholds=[-0.01],
                smoothing_sigmas=[1],
                derivative_polarities=[-1, 1],
            )

            self.assertEqual({item["derivative_polarity"] for item in result["results"]}, {-1, 1})

    def test_optimizer_cleans_impossible_gyro_spikes_like_production_detector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment = root / "spike.csv"
            segment.write_text("Gyr_X\n0\n0\n1000000000\n0\n0\n", encoding="utf-8")
            (root / "a_for_annotation.csv").write_text(
                "path,annotation_status,video_status,detection_status\n"
                f"{segment.as_posix()},annotated,visible,weird_signal\n",
                encoding="utf-8",
            )

            result = optimize_detection_parameters(root, thresholds=[-0.01], smoothing_sigmas=[1])

            self.assertEqual(result["best"]["true_negative"], 1)
            self.assertEqual(result["best"]["false_positive"], 0)

    def test_optimizes_detection_parameters_from_hdf5_after_segment_csv_move(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            os.chdir(root)
            try:
                pending_root = root / "data" / "pending"
                segment = root / "data" / "annotated" / "20250911" / "0856" / "jump.csv"
                pending_root.mkdir(parents=True)
                segment.parent.mkdir(parents=True)
                relative_segment = "data/annotated/20250911/0856/jump.csv"
                pd.DataFrame({"Gyr_X": [0.0, 0.0, 100.0, -100.0, 0.0]}).to_csv(segment, index=False)
                (pending_root / "session_for_annotation.csv").write_text(
                    "path,video_status,detection_status\n"
                    f"{relative_segment},visible,detected_jump\n",
                    encoding="utf-8",
                )
                export_hdf5_archive(
                    "data/synergie_archive.h5",
                    manifest={
                        "format": "synergie-backup-manifest-v1",
                        "pending_annotation_files": [{"path": "data/pending/session_for_annotation.csv", "trials": [{"path": relative_segment}]}],
                        "training_dataset": {"trials": []},
                    },
                )
                segment.unlink()

                result = optimize_detection_parameters(pending_root, thresholds=[-0.01], smoothing_sigmas=[1])
            finally:
                os.chdir(original_cwd)

        self.assertEqual(result["reviewed_segments"], 1)
        self.assertEqual(len(result["results"]), 1)
        self.assertIsNotNone(result["best"])

    def test_saves_and_loads_optimized_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "optimized.json"
            save_optimized_detection_parameters(
                {
                    "threshold": -0.2,
                    "smoothing_sigma": 20,
                    "balanced_error": 0.1,
                    "false_positive": 1,
                    "false_negative": 2,
                },
                path=path,
            )

            loaded = load_optimized_detection_parameters(path)

            self.assertEqual(loaded["threshold"], -0.2)
            self.assertEqual(loaded["smoothing_sigma"], 20.0)
