from datetime import datetime

import ttkbootstrap as ttkb

from core.utils.DotDevice import DotDevice


class DotFrame(ttkb.Frame):
    def __init__(self, parent, device: DotDevice, **kwargs) -> None:
        super().__init__(parent, style="Panel.TFrame", padding=16, **kwargs)
        self.parent = parent
        self.device = device
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.imageLabel = ttkb.Label(self, image=self._display_image(), background="white")
        self.imageLabel.grid(row=0, column=0, rowspan=4, sticky="nw", padx=(0, 14))

        self.titleLabel = ttkb.Label(self, text=f"Capteur {self.device.deviceTagName}", style="PanelTitle.TLabel")
        self.titleLabel.grid(row=0, column=1, sticky="w")
        self.statusLabel = ttkb.Label(self, text="", style="NeutralBadge.TLabel", padding=(8, 4))
        self.statusLabel.grid(row=1, column=1, sticky="w", pady=(6, 8))

        self.recordingLabel = ttkb.Label(self, text="", style="PanelBody.TLabel", wraplength=260)
        self.recordingLabel.grid(row=2, column=1, sticky="we")

        self.batteryText = ttkb.StringVar(self, value="")
        ttkb.Label(self, textvariable=self.batteryText, style="PanelMuted.TLabel").grid(
            row=3,
            column=1,
            sticky="w",
            pady=(8, 0),
        )
        self.batteryBar = ttkb.Progressbar(self, maximum=100, value=0, mode="determinate")
        self.batteryBar.grid(row=4, column=0, columnspan=2, sticky="we", pady=(10, 8))

        self.recordsLabel = ttkb.Label(self, text="", style="PanelMuted.TLabel")
        self.recordsLabel.grid(row=5, column=0, columnspan=2, sticky="w")
        self.pluggedLabel = ttkb.Label(self, text="", style="PanelMuted.TLabel")
        self.pluggedLabel.grid(row=6, column=0, columnspan=2, sticky="w", pady=(3, 0))
        self.updateDot()

    def updateDot(self):
        self.imageLabel.configure(image=self._display_image())
        status_text, status_style = self._status()
        self.statusLabel.configure(text=status_text, style=status_style)
        self.batteryBar.configure(value=max(0, min(100, int(self.device.batteryLevel or 0))))
        charge_text = "en charge" if getattr(self.device, "isBatteryCharging", False) else "sur batterie"
        self.batteryText.set(f"Batterie {self.device.batteryLevel}% - {charge_text}")
        self.pluggedLabel.configure(text="USB branché" if self.device.isPlugged else "USB débranché")

        recording = "en cours" if self.device.isRecording else str(self.device.recordingCount)
        self.recordsLabel.configure(text=f"Enregistrements stockés : {recording}")
        self.recordingLabel.configure(text=self._recording_message())

    def _display_image(self):
        if self.device.isPlugged:
            return getattr(self.device, "imageActive", self.device.currentImage)
        return getattr(self.device, "imageInactive", self.device.currentImage)

    def _status(self) -> tuple[str, str]:
        if self.device.isRecording:
            return ("Enregistrement", "SuccessBadge.TLabel")
        if self.device.recordingCount > 0:
            return ("À exporter", "WarningBadge.TLabel")
        if not self.device.isPlugged:
            return ("Débranché", "NeutralBadge.TLabel")
        return ("Prêt", "SuccessBadge.TLabel")

    def _recording_message(self) -> str:
        if self.device.isRecording:
            duration = datetime.now().timestamp() - self.device.timingRecord
            displayTime = "{:02d}:{:02d}".format(int(duration // 60), int(duration % 60))
            return f"Enregistrement en cours depuis {displayTime}"
        if self.device.recordingCount > 0:
            return "Données présentes. Rebranchez puis exportez quand vous êtes prêt."
        return "Aucun enregistrement actif."
