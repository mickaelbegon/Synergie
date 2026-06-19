import tempfile
import unittest
from pathlib import Path

from synergie.services.detection_tuning_service import (
    analyze_detection_review_labels,
    load_optimized_detection_parameters,
    optimize_detection_parameters,
    save_optimized_detection_parameters,
)


class DetectionTuningServiceTests(unittest.TestCase):
    def test_collects_reviewed_false_positive_and_negative_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a_for_annotation.csv").write_text(
                "sensor_id,path,start_ms,synced_start_ms,video_status,detection_status\n"
                "1,a.csv,10,10,visible,not_a_jump\n"
                "2,b.csv,20,20,visible,weird_signal\n"
                "3,c.csv,30,30,visible,manual_missing_jump\n"
                "4,d.csv,40,40,visible,detected_jump\n",
                encoding="utf-8",
            )

            analysis = analyze_detection_review_labels(root)

            self.assertEqual(analysis["false_positive_count"], 2)
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
