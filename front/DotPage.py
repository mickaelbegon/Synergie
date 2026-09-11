from math import ceil
from typing import List

import ttkbootstrap as ttkb

from core.utils.DotDevice import DotDevice
from front.DotFrame import DotFrame


class DotPage(ttkb.Frame):
    def __init__(self, parent, dotsconnected: List[DotDevice], **kwargs) -> None:
        super().__init__(parent, style="Synergie.TFrame", **kwargs)
        self.parent = parent
        self.dotsFrames: List[DotFrame] = []
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame = ttkb.Frame(self, style="Synergie.TFrame")
        columns = min(4, max(1, len(dotsconnected)))
        rows = max(1, ceil(len(dotsconnected) / columns))
        for row in range(rows):
            self.frame.grid_rowconfigure(row, weight=0, pad=18)
        for column in range(columns):
            self.frame.grid_columnconfigure(column, weight=1, pad=18, uniform="dots")

        for index, device in enumerate(dotsconnected):
            newdot = DotFrame(self.frame, device)
            self.dotsFrames.append(newdot)
            newdot.grid(row=index // columns, column=index % columns, sticky="new", padx=8, pady=8)
        self.frame.grid(row=0, column=0, sticky="new", padx=24, pady=10)

    def updatePage(self):
        for dotFrame in self.dotsFrames:
            dotFrame.updateDot()
