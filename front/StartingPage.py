import time
from math import ceil
from tkinter import VERTICAL

import ttkbootstrap as ttkb

from core.database.DatabaseManager import DatabaseManager, TrainingData
from core.utils.DotDevice import DotDevice
from front.ui_theme import setup_styles
from front.window_utils import set_synergie_icon
from synergie.sensor_assignment_history import assignment_label, record_assignment, sort_skaters_for_sensor


class StartingPage:
    def __init__(self, device: DotDevice, db_manager: DatabaseManager, userConnected: str, on_close=None) -> None:
        self.device = device
        self.db_manager = db_manager
        self.on_close = on_close
        self.deviceTag = self.device.deviceTagName
        self.skaters = sort_skaters_for_sensor(
            self.db_manager.getAllSkaterFromCoach(userConnected),
            self.deviceTag,
        )

        self.window = ttkb.Toplevel(title="Démarrer un enregistrement", size=(1180, 620), topmost=True)
        setup_styles(self.window)
        self.window.place_window_center()
        set_synergie_icon(self.window)
        self.window.grid_rowconfigure(0, weight=0)
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=0)

        header = ttkb.Frame(self.window, style="Synergie.TFrame", padding=(28, 24, 28, 12))
        header.grid_columnconfigure(0, weight=1)
        ttkb.Label(header, text=f"Démarrer le capteur {self.deviceTag}", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttkb.Label(
            header,
            text=(
                "Choisissez l'athlète associé à ce capteur pour créer l'entraînement. "
                "Les athlètes déjà associés à ce capteur apparaissent en premier."
            ),
            style="Body.TLabel",
            wraplength=900,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        header.grid(row=0, column=0, columnspan=2, sticky="we")

        self.canvas = ttkb.Canvas(self.window, highlightthickness=0)
        self.frame = ttkb.Frame(self.canvas, style="Synergie.TFrame", padding=(18, 10, 18, 24))
        columns = 4
        for column in range(columns):
            self.frame.grid_columnconfigure(column, weight=1, uniform="skaters")

        button_width = ceil((280 - 24) / 11)
        for index, skater in enumerate(self.skaters):
            history_label = assignment_label(self.deviceTag, skater.skater_id)
            button_text = f"\n{skater.skater_name}\n{history_label}\n" if history_label else f"\n{skater.skater_name}\n"
            button = ttkb.Button(
                self.frame,
                text=button_text,
                style="my.TButton",
                width=button_width,
                command=(lambda x=skater.skater_id, y=skater.skater_name: self.startRecord(x, y)),
            )
            button.grid(row=index // columns, column=index % columns, sticky="nsew", padx=10, pady=10)

        self.frame.bind("<Enter>", self._bound_to_mousewheel)
        self.frame.bind("<Leave>", self._unbound_to_mousewheel)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        scroll = ttkb.Scrollbar(self.window, orient=VERTICAL, command=self.canvas.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.bind("<Configure>", self._resize_canvas)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def startRecord(self, skaterId: str, skaterName: str):
        deviceId = self.device.deviceId
        new_training = TrainingData(0, skaterId, 0, deviceId, [])
        self.db_manager.set_current_record(deviceId, self.db_manager.save_training_data(new_training))
        recordStarted = self.device.startRecord()
        if recordStarted:
            record_assignment(self.deviceTag, skaterId, skaterName)
        self.canvas.destroy()
        self.frame = ttkb.Frame(self.window, style="Synergie.TFrame", padding=28)
        message = (
            f"Enregistrement démarré sur le capteur {self.deviceTag} pour {skaterName}"
            if recordStarted
            else "Erreur durant le lancement. Impossible de démarrer l'enregistrement."
        )
        ttkb.Label(self.frame, text=message, style="Title.TLabel", wraplength=820).grid()
        self.frame.grid(row=1, column=0, sticky="nsew")
        self.window.update()
        time.sleep(1)
        self.close()

    def close(self):
        if self.on_close is not None:
            self.on_close()
            self.on_close = None
        self.window.destroy()

    def _resize_canvas(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _bound_to_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
