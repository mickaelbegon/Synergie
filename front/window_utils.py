from pathlib import Path
import sys


def bundled_path(relative_path: str) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", "."))
    return base_path / relative_path


def set_synergie_icon(window) -> None:
    try:
        from PIL import Image, ImageTk
    except ImportError:
        return
    try:
        icon = Image.open(bundled_path("img/Logo_s2mJUMP_RGB.png"))
    except (FileNotFoundError, OSError):
        return
    photo = ImageTk.PhotoImage(icon)
    window.wm_iconphoto(False, photo)
    window._synergie_icon = photo
