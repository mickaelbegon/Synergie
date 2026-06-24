import unittest

from synergie.services.annotation_shortcut_service import (
    ANNOTATION_SHORTCUTS_HELP_TEXT,
    resolve_annotation_shortcut,
)


class AnnotationShortcutServiceTests(unittest.TestCase):
    def test_resolves_video_and_save_shortcuts(self):
        self.assertEqual(resolve_annotation_shortcut("space"), {"action": "play_pause"})
        self.assertEqual(resolve_annotation_shortcut("left"), {"action": "seek_frame", "delta": -1})
        self.assertEqual(resolve_annotation_shortcut("right"), {"action": "seek_frame", "delta": 1})
        self.assertEqual(resolve_annotation_shortcut("s", ctrl=True), {"action": "save"})
        self.assertEqual(resolve_annotation_shortcut("s", ctrl=False), {"action": "jump_type", "value": "salchow"})

    def test_resolves_annotation_field_shortcuts(self):
        self.assertEqual(resolve_annotation_shortcut("t"), {"action": "jump_type", "value": "toe_loop"})
        self.assertEqual(
            resolve_annotation_shortcut("x"),
            {"action": "review_status", "value": "weird_signal", "status": "weird signal / bad bounds"},
        )
        self.assertEqual(resolve_annotation_shortcut("2", turn_options=["1", "2"]), {"action": "turns", "value": "2"})
        self.assertIsNone(resolve_annotation_shortcut("4", turn_options=["1", "2"]))
        self.assertEqual(resolve_annotation_shortcut("r"), {"action": "success", "value": "1"})

    def test_help_text_mentions_core_shortcuts(self):
        self.assertIn("t/f/z/s/a", ANNOTATION_SHORTCUTS_HELP_TEXT)
        self.assertIn("Ctrl+S", ANNOTATION_SHORTCUTS_HELP_TEXT)


if __name__ == "__main__":
    unittest.main()
