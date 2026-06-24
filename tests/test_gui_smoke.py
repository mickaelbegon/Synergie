import unittest


class GuiSmokeTests(unittest.TestCase):
    def test_tool_gui_imports_without_starting_tk_mainloop(self):
        from synergie.tool_gui import SynergieToolsApp

        self.assertTrue(hasattr(SynergieToolsApp, "ANNOTATION_SHORTCUTS"))
        self.assertEqual(SynergieToolsApp.ANNOTATION_SHORTCUTS["t"], "toe_loop")

    def test_operations_exports_gui_service_helpers(self):
        from synergie import operations

        required_helpers = [
            "AnnotationVideoPlayer",
            "ANNOTATION_JUMP_TYPE_SHORTCUTS",
            "annotation_segment_available",
            "annotation_segment_path",
            "annotation_review_video_target_ms",
            "annotation_sync_context_text",
            "annotation_type_window_start_imu_ms",
            "annotation_ui_values_from_row",
            "build_annotation_timeline_items",
            "build_manual_annotation_row",
            "data_validation_action_summary",
            "data_validation_status_text",
            "detection_threshold_context",
            "format_annotation_video_info",
            "format_video_cache_cleared_status",
            "format_video_cache_button",
            "format_video_ms",
            "format_video_preparation_message",
            "inspect_axis_labels",
            "inspect_detection_threshold_spec",
            "inspect_drag_zoom_selection",
            "inspect_jump_center_markers",
            "inspect_jump_center_ms",
            "inspect_jump_has_gyro_saturation",
            "inspect_jump_legend_specs",
            "inspect_jump_window_bounds_ms",
            "inspect_jump_zoom_view",
            "inspect_selected_jump_title",
            "inspect_signal_series_specs",
            "inspect_text",
            "inspect_zoom_range_title",
            "inspect_zoom_range_view",
            "parse_video_ms",
            "prepare_annotation_segment_signal",
            "prepare_sync_impact_signal",
            "redetect_annotation_sync_impacts",
            "resolve_annotation_shortcut",
            "save_annotation_file_values",
            "save_annotation_values",
            "sync_impact_diagnostic_text",
            "sync_impact_selection_text",
            "sync_impact_signal_window",
        ]

        for helper in required_helpers:
            with self.subTest(helper=helper):
                self.assertTrue(hasattr(operations, helper))


if __name__ == "__main__":
    unittest.main()
