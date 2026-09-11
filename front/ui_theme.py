from tkinter.font import BOLD, Font

import ttkbootstrap as ttkb


PRIMARY = "#106b5f"
SURFACE = "#f7faf9"
MUTED = "#5f6f6b"
DANGER = "#b42318"
WARNING = "#b54708"
SUCCESS = "#047857"


def setup_styles(root) -> None:
    """Configure the shared Tk styles used by the DOT workflow."""
    style = ttkb.Style()
    default_font = Font(root, size=11)
    title_font = Font(root, size=24, weight=BOLD)
    section_font = Font(root, size=15, weight=BOLD)
    button_font = Font(root, size=13, weight=BOLD)
    small_font = Font(root, size=10)

    style.configure("Synergie.TFrame", background=SURFACE)
    style.configure("Panel.TFrame", background="white", borderwidth=1, relief="solid")
    style.configure("PanelContent.TFrame", background="white")
    style.configure("Title.TLabel", font=title_font, foreground="#12312d", background=SURFACE)
    style.configure("Section.TLabel", font=section_font, foreground="#12312d", background=SURFACE)
    style.configure("Body.TLabel", font=default_font, foreground="#203b36", background=SURFACE)
    style.configure("Muted.TLabel", font=small_font, foreground=MUTED, background=SURFACE)
    style.configure("PanelTitle.TLabel", font=section_font, foreground="#12312d", background="white")
    style.configure("PanelBody.TLabel", font=default_font, foreground="#203b36", background="white")
    style.configure("PanelMuted.TLabel", font=small_font, foreground=MUTED, background="white")
    style.configure("SuccessBadge.TLabel", font=small_font, foreground="white", background=SUCCESS)
    style.configure("WarningBadge.TLabel", font=small_font, foreground="white", background=WARNING)
    style.configure("DangerBadge.TLabel", font=small_font, foreground="white", background=DANGER)
    style.configure("NeutralBadge.TLabel", font=small_font, foreground="white", background=MUTED)
    style.configure("home.TButton", font=button_font, padding=(18, 12))
    style.configure("my.TButton", font=button_font, padding=(14, 10))
