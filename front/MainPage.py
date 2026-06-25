import threading
from typing import List
import webbrowser

import ttkbootstrap as ttkb

from core.database.DatabaseManager import DatabaseManager
from core.utils.DotDevice import DotDevice
from core.utils.DotManager import DotManager
from front.DotPage import DotPage
from front.ExtractingPage import ExtractingPage
from front.ui_theme import setup_styles


class MainPage:
    def __init__(
        self,
        dotsConnected: List[DotDevice],
        dot_manager: DotManager,
        db_manager: DatabaseManager,
        root: ttkb.Window = None,
    ) -> None:
        self.root = root
        self.dotsConnected = dotsConnected
        self.dot_manager = dot_manager
        self.db_manager = db_manager
        setup_styles(self.root)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.frame = ttkb.Frame(root, style="Synergie.TFrame", padding=40)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self.waitingFrame = ttkb.Frame(self.frame, style="Synergie.TFrame")
        self.waitingFrame.grid_columnconfigure(0, weight=1)
        self.waitingStatusVar = ttkb.StringVar(self.waitingFrame, value="Initialisation de la connexion aux capteurs")
        ttkb.Label(self.waitingFrame, text="Connexion des capteurs DOT", style="Title.TLabel").grid(row=0, column=0)
        ttkb.Label(
            self.waitingFrame,
            text="Gardez les capteurs branchés pendant la détection USB et Bluetooth.",
            style="Body.TLabel",
        ).grid(row=1, column=0, pady=(10, 20))
        self.waitingStatusLabel = ttkb.Label(self.waitingFrame, textvariable=self.waitingStatusVar, style="Muted.TLabel")
        self.waitingStatusLabel.grid(row=2, column=0, pady=(0, 18))
        waitingProgress = ttkb.Progressbar(self.waitingFrame, mode="indeterminate", length=360)
        waitingProgress.grid(row=3, column=0, sticky="we")
        waitingProgress.start(10)
        self.waitingFrame.grid(row=0, column=0)
        self.frame.grid(sticky="nswe")

    def set_waiting_status(self, message: str):
        if hasattr(self, "waitingStatusVar") and message:
            self.waitingStatusVar.set(message)

    def make_dot_page(self):
        self.frame.destroy()
        self.frame = ttkb.Frame(self.root, style="Synergie.TFrame", padding=28)
        self.frame.grid_rowconfigure(0, weight=0)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_rowconfigure(2, weight=0)
        self.frame.grid_columnconfigure(0, weight=1)

        self._make_header()

        self.dotPage = DotPage(self.frame, self.dotsConnected)
        self.dotPage.grid(row=1, column=0, sticky="nsew")

        self.make_export_button()
        self.frame.grid(sticky="nswe")
        self.run_periodic_background_func()

    def _make_header(self):
        header = ttkb.Frame(self.frame, style="Synergie.TFrame")
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)
        ttkb.Label(header, text="Capteurs DOT", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttkb.Label(
            header,
            text="Débranchez un capteur prêt pour démarrer. Rebranchez un capteur en enregistrement pour arrêter ou exporter.",
            style="Body.TLabel",
            wraplength=760,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttkb.Button(
            header,
            text="Ouvrir la visualisation",
            style="home.TButton",
            command=lambda: webbrowser.open("https://synergie-qc.streamlit.app/"),
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=(24, 0))
        header.grid(row=0, column=0, sticky="we", pady=(0, 18))

    def make_export_button(self):
        self.estimatedTime = self.dot_manager.getExportEstimatedTime()
        self.exportFrame = ttkb.Frame(self.frame, style="Panel.TFrame", padding=18)
        self.exportFrame.grid_columnconfigure(0, weight=1)
        self.exportFrame.grid_columnconfigure(1, weight=0)
        ttkb.Label(
            self.exportFrame,
            text=f"Export global · environ {round(self.estimatedTime, 0)} min",
            style="PanelTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttkb.Label(
            self.exportFrame,
            text="Exporte tous les capteurs branchés qui ont des données stockées.",
            style="PanelMuted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttkb.Button(
            self.exportFrame,
            text="Exporter",
            style="home.TButton",
            command=self.export_all_dots,
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=(18, 0))
        self.saveFile = ttkb.Checkbutton(self.exportFrame, text="Inclure les données complètes pour la recherche")
        self.saveFile.state(["!alternate"])
        self.saveFile.grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.exportFrame.grid(row=2, column=0, sticky="we", pady=(18, 0))

    def export_all_dots(self):
        for device in self.dotsConnected:
            if (not device.isRecording) and device.isPlugged and device.recordingCount > 0:
                extractEvent = threading.Event()
                saveFile = self.saveFile.instate(["selected"])
                threading.Thread(target=device.exportData, args=([saveFile, extractEvent]), daemon=True).start()
                ExtractingPage(device.deviceTagName, self.estimatedTime, extractEvent)

    def run_periodic_background_func(self):
        self.dotPage.updatePage()
        self.root.after(1000, self.run_periodic_background_func)
