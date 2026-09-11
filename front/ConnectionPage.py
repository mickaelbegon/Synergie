import json
from pathlib import Path

import ttkbootstrap as ttkb

from core.database.DatabaseManager import DatabaseManager
from front.ui_theme import setup_styles


class ConnectionPage:
    def __init__(self, root: ttkb.Window, dbManager: DatabaseManager) -> None:
        self.root = root
        self.dbManager = dbManager
        self.userConnected = ""
        setup_styles(self.root)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.frame = ttkb.Frame(self.root, style="Synergie.TFrame", padding=40)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=0)
        self.frame.grid_rowconfigure(2, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        header = ttkb.Frame(self.frame, style="Synergie.TFrame")
        ttkb.Label(header, text="Connexion entraîneur", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttkb.Label(
            header,
            text="Identifiez-vous pour associer les enregistrements DOT aux athlètes du bon groupe.",
            style="Body.TLabel",
            wraplength=540,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        header.grid(row=0, column=0, sticky="s", pady=(0, 24))

        form = ttkb.Frame(self.frame, style="Panel.TFrame", padding=28)
        form.grid_columnconfigure(0, weight=1)
        ttkb.Label(form, text="Adresse courriel", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.accountVar = ttkb.StringVar(self.frame, value=self._default_email())
        self.entry = ttkb.Entry(form, textvariable=self.accountVar, width=42)
        self.entry.grid(row=1, column=0, sticky="we", pady=(8, 16))
        self.entry.bind("<Return>", self._on_submit)

        self.button = ttkb.Button(form, text="Se connecter", style="home.TButton", command=self.register)
        self.button.grid(row=2, column=0, sticky="we")
        self.errorVar = ttkb.StringVar(self.frame, value="")
        self.errorVar.trace_add("write", self._sync_error_label)
        self.errorLabel = ttkb.Label(form, textvariable=self.errorVar, style="DangerBadge.TLabel", padding=(10, 6))
        self.errorLabel.grid(row=3, column=0, sticky="we", pady=(14, 0))
        self.errorLabel.grid_remove()
        form.grid(row=1, column=0, sticky="n", ipadx=20, ipady=10)
        self.frame.grid(sticky="nswe")

    def register(self):
        userFound = self.dbManager.findUserByEmail(self.accountVar.get().strip())
        if userFound != []:
            user = userFound[0]
            if user.get("role") == "COACH":
                print("Connecté")
                self.errorVar.set("")
                self.userConnected = user.id
            else:
                self.errorVar.set("Erreur : vous avez besoin d'un compte entraîneur")
        else:
            self.errorVar.set("Erreur : cet utilisateur n'existe pas")

    def _sync_error_label(self, *_args):
        if self.errorVar.get():
            self.errorLabel.grid()
        else:
            self.errorLabel.grid_remove()

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
