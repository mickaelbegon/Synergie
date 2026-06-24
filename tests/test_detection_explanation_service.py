import unittest

import pandas as pd

from synergie.services.detection_explanation_service import (
    angular_velocity_peak_context,
    detection_threshold_context,
    nearest_interval,
    threshold_intervals,
)


class DetectionExplanationServiceTests(unittest.TestCase):
    def test_threshold_intervals_groups_contiguous_active_samples(self):
        intervals = threshold_intervals([0, 10, 20, 30, 40], [False, True, True, False, True])

        self.assertEqual(intervals, [(10.0, 20.0), (40.0, 40.0)])
        self.assertEqual(nearest_interval(intervals, 12.0), (10.0, 20.0))
        self.assertEqual(nearest_interval(intervals, 35.0), (40.0, 40.0))

    def test_detection_threshold_context_explains_selected_interval(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 100, 200, 300],
                "X_gyr_second_derivative": [0.0, -0.3, -0.25, 0.0],
                "Gyr_X_smoothed": [10.0, 200.0, 100.0, 20.0],
            }
        )

        context = detection_threshold_context(
            frame,
            {"start_ms": 110.0},
            threshold=-0.2,
            plot_scale=100.0,
        )

        self.assertIn("Why detected", context["diagnostic"])
        self.assertEqual(context["selected_interval"], (100.0, 200.0))
        self.assertEqual(context["min_ms"], 100.0)
        self.assertEqual(context["warning"], "")

    def test_detection_threshold_context_warns_when_threshold_far_from_peak(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 100, 500, 600],
                "X_gyr_second_derivative": [0.0, 0.0, -0.3, 0.0],
                "Gyr_X_smoothed": [500.0, 50.0, 30.0, 20.0],
            }
        )

        context = detection_threshold_context(frame, {"start_ms": 500.0}, threshold=-0.2, plot_scale=100.0)

        self.assertIn("Biomech check", context["warning"])

    def test_detection_threshold_context_flags_no_interval_after_cleaning(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 100, 200],
                "X_gyr_second_derivative": [-0.1, -0.09, -0.08],
                "Gyr_X": [10.0, 20.0, 30.0],
            }
        )

        context = detection_threshold_context(frame, {"start_ms": 100.0}, threshold=-0.2, plot_scale=100.0)

        self.assertIn("No under-threshold interval remains", context["diagnostic"])
        self.assertIn("old/noisy false positive", context["warning"])

    def test_angular_velocity_peak_context_uses_absolute_peak(self):
        frame = pd.DataFrame({"ms": [0, 100], "Gyr_X": [100.0, -300.0]})

        text, warning = angular_velocity_peak_context(frame, 100.0)

        self.assertIn("negative", text)
        self.assertEqual(warning, "")


if __name__ == "__main__":
    unittest.main()
