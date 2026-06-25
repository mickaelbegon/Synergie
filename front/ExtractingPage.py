import threading
import time
from tkinter import TclError

import ttkbootstrap as ttkb

from front.ui_theme import setup_styles


class ExtractingPage:
    def __init__(self, deviceTag: str, estimatedTime, event: threading.Event) -> None:
        self.deviceTag = deviceTag
        self.event = event
        self.window = ttkb.Toplevel(title="Extraction", size=(720, 260), topmost=True)
        setup_styles(self.window)
        self.window.place_window_center()
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)

        frame = ttkb.Frame(self.window, style="Synergie.TFrame", padding=28)
        frame.grid_columnconfigure(0, weight=1)
        self.text = ttkb.StringVar(
            frame,
            value=f"Extraction du capteur {self.deviceTag}",
        )
        ttkb.Label(frame, textvariable=self.text, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttkb.Label(
            frame,
            text="Ne déconnectez pas ce capteur avant la fin de l'extraction.",
            style="Body.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(8, 18))
        self.max_val = max(1, 60 * estimatedTime)
        self.progressExtract = ttkb.Progressbar(
            frame,
            value=0,
            maximum=self.max_val,
            style="success.Striped.Horizontal.TProgressbar",
            mode="determinate",
        )
        self.progressExtract.start(1000)
        self.progressExtract.grid(row=2, column=0, sticky="we")
        frame.grid(row=0, column=0, sticky="nsew")
        self.checkFinish()

    def checkFinish(self):
        try:
            self.checkProgressBar()
        except TclError:
            pass
        if self.event.is_set():
            self.text.set("Extraction terminée")
            time.sleep(1)
            self.window.destroy()
            return
        self.window.after(1000, self.checkFinish)

    def checkProgressBar(self):
        if self.progressExtract["value"] >= self.max_val - 1:
            self.progressExtract.stop()
            self.progressExtract["value"] = self.max_val
