import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class FakeWindow:
    def __init__(self):
        self.icon_args = None

    def wm_iconphoto(self, *args):
        self.icon_args = args


def _fake_pil_modules(open_result="icon"):
    pil = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")
    image_tk = types.ModuleType("PIL.ImageTk")

    image.open = lambda path: open_result
    image_tk.PhotoImage = lambda icon: f"photo:{icon}"
    pil.Image = image
    pil.ImageTk = image_tk
    return {"PIL": pil, "PIL.Image": image, "PIL.ImageTk": image_tk}


class WindowUtilsTests(unittest.TestCase):
    def test_bundled_path_uses_pyinstaller_meipass_when_available(self):
        module = importlib.import_module("front.window_utils")

        with patch.object(module.sys, "_MEIPASS", r"C:\bundle", create=True):
            self.assertEqual(module.bundled_path("img/logo.png"), Path(r"C:\bundle") / "img/logo.png")

    def test_set_synergie_icon_sets_photo_and_keeps_reference(self):
        module = importlib.import_module("front.window_utils")
        window = FakeWindow()

        with patch.dict(sys.modules, _fake_pil_modules()):
            module.set_synergie_icon(window)

        self.assertEqual(window.icon_args, (False, "photo:icon"))
        self.assertEqual(window._synergie_icon, "photo:icon")

    def test_set_synergie_icon_ignores_missing_icon_file(self):
        module = importlib.import_module("front.window_utils")
        window = FakeWindow()
        pil_modules = _fake_pil_modules()
        pil_modules["PIL.Image"].open = lambda path: (_ for _ in ()).throw(FileNotFoundError())

        with patch.dict(sys.modules, pil_modules):
            module.set_synergie_icon(window)

        self.assertIsNone(window.icon_args)
        self.assertFalse(hasattr(window, "_synergie_icon"))


if __name__ == "__main__":
    unittest.main()
