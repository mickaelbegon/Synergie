import unittest
from types import SimpleNamespace

import pandas as pd

from synergie.services.inspect_signal_service import (
    inspect_axis_labels,
    inspect_detection_threshold_spec,
    inspect_drag_zoom_selection,
    inspect_jump_center_markers,
    inspect_jump_center_ms,
    inspect_jump_has_gyro_saturation,
    inspect_jump_legend_specs,
    inspect_jump_window_bounds_ms,
    inspect_jump_zoom_view,
    inspect_selected_jump_title,
    inspect_signal_series_specs,
    inspect_text,
    inspect_zoom_range_title,
    inspect_zoom_range_view,
)


class InspectSignalServiceTests(unittest.TestCase):
    def test_inspect_axis_labels_are_standardized(self):
        self.assertEqual(
            inspect_axis_labels(),
            {"x": "ms", "left_y": "Gyroscope", "right_y": "2nd derivative"},
        )

    def test_inspect_jump_legend_specs_include_optional_markers(self):
        base_specs = inspect_jump_legend_specs()
        labels = [spec["label"] for spec in base_specs]

        self.assertEqual(labels, ["Type window", "Success window", "Gyro saturation", "Sync impact"])
        self.assertIn("Detected jump center", [spec["label"] for spec in inspect_jump_legend_specs(include_center=True)])
        self.assertIn("Selected jump", [spec["label"] for spec in inspect_jump_legend_specs(include_selected=True)])

    def test_inspect_signal_series_specs_target_expected_axes_and_columns(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 1],
                "Gyr_X_unfiltered": [10, 20],
                "Gyr_X_smoothed": [11, 21],
                "X_gyr_second_derivative": [-0.1, -0.2],
            }
        )

        specs = inspect_signal_series_specs(frame)

        self.assertEqual([spec["axis"] for spec in specs], ["left", "left", "right"])
        self.assertEqual([spec["style"]["label"] for spec in specs], ["Gyr_X raw", "Gyr_X smoothed", "2nd derivative"])
        self.assertEqual(specs[0]["y"].tolist(), [10, 20])
        self.assertEqual(specs[2]["style"]["color"], "crimson")

    def test_inspect_detection_threshold_spec_is_stable(self):
        self.assertEqual(
            inspect_detection_threshold_spec(-0.2),
            {"y": -0.2, "style": {"color": "crimson", "linestyle": "--", "label": "Detection threshold"}},
        )

    def test_inspect_text_helpers_format_titles(self):
        self.assertEqual(inspect_text("overview_title"), "Signals used to localise jumps")
        self.assertEqual(inspect_selected_jump_title(3), "Zoom on selected jump #3")
        self.assertEqual(inspect_selected_jump_title(3, has_gyro_saturation=True), "Zoom on selected jump #3 | Gyro saturated")
        self.assertEqual(inspect_zoom_range_title(100.4, 250.6), "Zoom on selected range: 100-251 ms")

    def test_inspect_jump_center_markers_prepare_overview_points(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 100, 200, 300, 400],
                "Gyr_X_smoothed": [0, 10, 20, 30, 40],
                "Gyr_X_unfiltered": [0, 10, 20, 30, 40],
            }
        )
        jumps = [
            SimpleNamespace(start=1, end=3, df=[1, 2], df_success=[1]),
            SimpleNamespace(start=4, end=4, df=[1], df_success=[1]),
        ]

        markers = inspect_jump_center_markers(
            frame,
            jumps,
            segment_frames_before_takeoff=1,
            type_window_start=0,
            type_window_frames=2,
        )

        self.assertEqual(markers[0], {"index": 0, "center_ms": 200.0, "center_y": 20.0})
        self.assertEqual(markers[1], {"index": 1, "center_ms": 400.0, "center_y": 40.0})

    def test_inspect_jump_window_bounds_use_session_timeline(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 100, 200, 300, 400, 500],
                "Gyr_X_unfiltered": [0, 0, 0, 0, 0, 0],
            }
        )
        jump = SimpleNamespace(start=3, end=5, df=[1, 2, 3], df_success=[1, 2])

        bounds = inspect_jump_window_bounds_ms(
            frame,
            jump,
            segment_frames_before_takeoff=2,
            type_window_start=1,
            type_window_frames=3,
        )

        self.assertEqual(bounds["type"], (200.0, 400.0))
        self.assertEqual(bounds["success"], (300.0, 400.0))
        self.assertEqual(bounds["detected"], (300.0, 500.0))

    def test_inspect_jump_center_ms_uses_detected_window(self):
        frame = pd.DataFrame({"ms": [0, 100, 200, 300], "Gyr_X_unfiltered": [0, 0, 0, 0]})
        jump = SimpleNamespace(start=1, end=3, df=[1, 2], df_success=[1])

        center = inspect_jump_center_ms(
            frame,
            jump,
            segment_frames_before_takeoff=10,
            type_window_start=0,
            type_window_frames=2,
        )

        self.assertEqual(center, 200.0)

    def test_inspect_jump_has_gyro_saturation_checks_local_window(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 100, 200, 300, 400],
                "Gyr_X_unfiltered": [0, 10, 2000, 20, 0],
            }
        )
        jump = SimpleNamespace(start=2, end=3, df=[1, 2], df_success=[1])

        self.assertTrue(
            inspect_jump_has_gyro_saturation(
                frame,
                jump,
                segment_frames_before_takeoff=1,
                saturation_threshold=1500.0,
            )
        )
        self.assertFalse(
            inspect_jump_has_gyro_saturation(
                frame,
                jump,
                segment_frames_before_takeoff=1,
                saturation_threshold=2500.0,
            )
        )

    def test_inspect_jump_zoom_view_returns_padded_slice_and_limits(self):
        frame = pd.DataFrame({"ms": [0, 100, 200, 300, 400], "Gyr_X_unfiltered": [0, 0, 0, 0, 0]})
        jump = SimpleNamespace(start=2, end=3, df=[1], df_success=[1])

        result = inspect_jump_zoom_view(frame, jump, padding_frames=1)

        self.assertEqual(result["view_df"]["ms"].tolist(), [100, 200, 300, 400])
        self.assertEqual(result["x_min_ms"], 100.0)
        self.assertEqual(result["x_max_ms"], 400.0)

    def test_inspect_zoom_range_view_sorts_drag_bounds(self):
        frame = pd.DataFrame({"ms": [0, 100, 200, 300], "Gyr_X_unfiltered": [0, 0, 0, 0]})

        result = inspect_zoom_range_view(frame, 250.0, 50.0)

        self.assertEqual(result["view_df"]["ms"].tolist(), [100, 200])
        self.assertEqual(result["x_min_ms"], 50.0)
        self.assertEqual(result["x_max_ms"], 250.0)

    def test_inspect_drag_zoom_selection_rejects_empty_or_tiny_drags(self):
        self.assertFalse(inspect_drag_zoom_selection(100.0, None)["accepted"])
        self.assertFalse(inspect_drag_zoom_selection(100.0, 103.0, min_width_ms=5.0)["accepted"])

    def test_inspect_drag_zoom_selection_sorts_range_and_formats_status(self):
        result = inspect_drag_zoom_selection(250.0, 50.0, min_width_ms=5.0)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["range_ms"], (50.0, 250.0))
        self.assertEqual(result["status"], "Inspect IMU zoom: 50-250 ms")


if __name__ == "__main__":
    unittest.main()
