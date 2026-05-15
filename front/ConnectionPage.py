from tkinter.font import BOLD, Font
import json
from pathlib import Path

import ttkbootstrap as ttkb
from core.database.DatabaseManager import DatabaseManager

class ConnectionPage:
    def __init__(self, root : ttkb.Window, dbManager : DatabaseManager) -> None:
        self.root = root
        self.dbManager = dbManager
        self.userConnected = ""
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.frame = ttkb.Frame(self.root)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_rowconfigure(2, weight=1)
        self.frame.grid_rowconfigure(3, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        labelFont = Font(self.root, size=15, weight=BOLD)
        self.label = ttkb.Label(self.frame, text="Veuillez entrer votre adresse mail", font=labelFont)
        self.label.grid(row=0, column=0, sticky="s")
        self.accountVar = ttkb.StringVar(self.frame, value=self._default_email())
        self.entry = ttkb.Entry(self.frame, textvariable=self.accountVar)
        self.entry.grid(row=1, column=0)
        self.entry.bind("<Return>", self._on_submit)
        buttonStyle = ttkb.Style()
        buttonStyle.configure('home.TButton', font=Font(self.frame, size=20, weight=BOLD))
        self.button = ttkb.Button(self.frame, text="Se connecter", style="home.TButton", command=self.register)
        self.button.grid(row=2, column=0, sticky="n")
        self.errorVar = ttkb.StringVar(self.frame, value="")
        self.errorLabel = ttkb.Label(self.frame, textvariable=self.errorVar, font=labelFont)
        self.errorLabel.grid(row=3, column=0)
        self.frame.grid(sticky="nswe")
    
    def register(self):
        userFound = self.dbManager.findUserByEmail(self.accountVar.get())
        if userFound != []:
            x = userFound[0]
            if x.get("role") == "COACH":
                print("Connecté")
                self.userConnected = x.id
            else:
                self.errorVar.set("Erreur : vous avez besoin d'un compte entraîneur")
        else:
            self.errorVar.set("Erreur : cet utilisateur n'existe pas")

    def _on_submit(self, _event=None):
        self.register()

    def _default_email(self) -> str:
        config_path = Path(__file__).resolve().parents[1] / "config" / "local_debug.json"
        if config_path.is_file():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    value = payload.get("default_coach_email")
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            except (OSError, json.JSONDecodeError):
                pass
        return ""
