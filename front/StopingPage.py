import sys
import threading
import time
from tkinter import TclError

from PIL import Image, ImageTk
import ttkbootstrap as ttkb

from core.database.DatabaseManager import DatabaseManager
from core.utils.DotDevice import DotDevice
from front.ui_theme import setup_styles


class StopingPage:
    def __init__(self, device: DotDevice, db_manager: DatabaseManager, on_close=None) -> None:
        self.device = device
        self.db_manager = db_manager
        self.on_close = on_close
        self.deviceTag = self.device.deviceTagName
        self.window = ttkb.Toplevel(title="Arrêter un enregistrement", size=(820, 420), topmost=True)
        setup_styles(self.window)
        self.window.place_window_center()
        try:
            ico = Image.open(f"{sys._MEIPASS}/img/Logo_s2mJUMP_RGB.png")
        except (AttributeError, FileNotFoundError, OSError):
            ico = Image.open("img/Logo_s2mJUMP_RGB.png")
        photo = ImageTk.PhotoImage(ico)
        self.window.wm_iconphoto(False, photo)
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        self.frame = ttkb.Frame(self.window, style="Synergie.TFrame", padding=28)
        self.frame.grid_columnconfigure(0, weight=1)
        ttkb.Label(self.frame, text=f"Capteur {self.deviceTag}", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttkb.Label(
            self.frame,
            text="L'enregistrement est prêt à être arrêté. Vous pouvez extraire maintenant ou garder les données stockées.",
            style="Body.TLabel",
            wraplength=700,
        ).grid(row=1, column=0, sticky="w", pady=(8, 20))

        actions = ttkb.Frame(self.frame, style="Synergie.TFrame")
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        ttkb.Button(actions, text="Arrêter seulement", style="my.TButton", command=self.stopRecord).grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10),
        )
        self.estimatedTime = self.device.getExportEstimatedTime()
        ttkb.Button(
            actions,
            text=f"Arrêter et extraire · {round(self.estimatedTime, 0)} min",
            style="home.TButton",
            command=self.stopRecordAndExtract,
        ).grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        actions.grid(row=2, column=0, sticky="we")

        self.saveFile = ttkb.Checkbutton(self.frame, text="Inclure les données complètes pour la recherche")
        self.saveFile.state(["!alternate"])
        self.saveFile.grid(row=3, column=0, sticky="w", pady=(18, 0))
        self.frame.grid(sticky="nsew")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def stopRecord(self):
        recordStopped = self.device.stopRecord()
        self.frame.destroy()
        self.frame = ttkb.Frame(self.window, style="Synergie.TFrame", padding=28)
        message = (
            f"Enregistrement arrêté sur le capteur {self.deviceTag}"
            if recordStopped
            else "Erreur durant l'arrêt. Impossible d'arrêter l'enregistrement."
        )
        ttkb.Label(self.frame, text=message, style="Title.TLabel", wraplength=700).grid()
        self.frame.grid(sticky="nsew")
        self.window.update()
        time.sleep(1)
        self.close()

    def stopRecordAndExtract(self):
        recordStopped = self.device.stopRecord()
        if recordStopped:
            self.device.currentImage = self.device.imageInactive
        saveFile = self.saveFile.instate(["selected"])
        self.frame.destroy()
        self.frame = ttkb.Frame(self.window, style="Synergie.TFrame", padding=28)
        self.frame.grid_columnconfigure(0, weight=1)
        message = (
            f"Enregistrement arrêté sur le capteur {self.deviceTag}"
            if recordStopped
            else "Erreur durant l'arrêt. Impossible d'arrêter l'enregistrement."
        )
        self.text = ttkb.StringVar(self.frame, value=message)
        self.label = ttkb.Label(self.frame, textvariable=self.text, style="Title.TLabel", wraplength=720)
        self.label.grid(row=0, column=0, sticky="w", pady=(0, 18))
        ttkb.Label(
            self.frame,
            text="Ne déconnectez pas ce capteur avant la fin de l'extraction.",
            style="Body.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 18))
        self.max_val = max(1, 60 * self.estimatedTime)
        self.progressExtract = ttkb.Progressbar(
            self.frame,
            value=0,
            maximum=self.max_val,
            style="success.Striped.Horizontal.TProgressbar",
            mode="determinate",
        )
        self.progressExtract.start(1000)
        self.progressExtract.grid(row=2, column=0, sticky="we")
        self.frame.grid(sticky="nsew")
        self.window.update()
        self.extractEvent = threading.Event()
        self.checkFinish()
        if recordStopped:
            threading.Thread(target=self.device.exportData, args=([saveFile, self.extractEvent]), daemon=True).start()
        else:
            self.extractEvent.set()

    def checkFinish(self):
        try:
            self.checkProgressBar()
        except TclError:
            pass
        self.window.update()
        if self.extractEvent.is_set():
            self.text.set("Extraction terminée")
            time.sleep(1)
            self.close()
            return
        self.window.after(1000, self.checkFinish)

    def checkProgressBar(self):
        if self.progressExtract["value"] >= self.max_val - 1:
            self.progressExtract.stop()
            self.progressExtract["value"] = self.max_val

    def close(self):
        if self.on_close is not None:
            self.on_close()
            self.on_close = None
        self.window.destroy()
