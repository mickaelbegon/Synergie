import logging
from tkinter.font import BOLD, Font
import ttkbootstrap as ttkb
from tkinter import messagebox
from core.database.DatabaseManager import DatabaseManager
from core.utils.internet_check import check_internet_connection, has_internet_connection

class ConnectionPage:
    """
    The ConnectionPage class represents a login interface where the user enters their email address
    to connect. This page checks if the user is a coach and, if so, allows them to access the rest
    of the application.
    """

    def __init__(self, root: ttkb.Window, dbManager: DatabaseManager) -> None:
        self.root = root
        self.dbManager = dbManager
        self.userConnected = ""  # Will store the user ID if the user is successfully connected.

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler("connection_page.log"),
                logging.StreamHandler()
            ]
        )

        # Configure the main window grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create a frame to hold all ConnectionPage widgets.
        self.frame = ttkb.Frame(self.root)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_rowconfigure(2, weight=1)
        self.frame.grid_rowconfigure(3, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # A bold font for labels.
        labelFont = Font(self.root, size=15, weight=BOLD)

        # Prompt label instructing user to enter their email.
        self.label = ttkb.Label(
            self.frame,
            text="Veuillez entrer votre adresse mail",
            font=labelFont
        )
        self.label.grid(row=0, column=0, sticky="s", pady=(10, 5))

        # Entry field for user email input.
        self.accountVar = ttkb.StringVar(self.frame, value="")
        self.entry = ttkb.Entry(self.frame, textvariable=self.accountVar, width=30)
        self.entry.grid(row=1, column=0, pady=5)
        self.entry.focus_set()  # Set focus to the entry widget

        # Bind the ENTER key to the register method
        self.entry.bind('<Return>', self.register)

        # Bind the key press event to clear error messages when user starts typing.
        self.entry.bind('<Key>', self.clear_error)

        # Style configuration for the "Se connecter" (Connect) button.
        buttonStyle = ttkb.Style()
        buttonStyle.configure('home.TButton', font=Font(self.frame, size=12, weight=BOLD))

        # Button to trigger the login/verification process.
        self.button = ttkb.Button(
            self.frame,
            text="Se connecter",
            style="home.TButton",
            command=self.register
        )
        self.button.grid(row=2, column=0, sticky="n", pady=10)

        # Label to show error messages
        self.errorVar = ttkb.StringVar(self.frame, value="")
        self.errorLabel = ttkb.Label(
            self.frame,
            textvariable=self.errorVar,
            font=labelFont,
            foreground="red"
        )
        self.errorLabel.grid(row=3, column=0, pady=(5, 10))

        # Place the frame in the main window.
        self.frame.grid(sticky="nswe")

        # Check for internet connectivity upon initialization.
        self.perform_internet_check()

    def clear_error(self, event):
        self.errorVar.set("")

    def perform_internet_check(self):
        """
        Checks if the computer is connected to the internet.
        If not, offers a retry/cancel dialog.
        """
        if not check_internet_connection():
            retry = messagebox.askretrycancel(
                "Connexion Internet",
                "Aucune connexion Internet détectée. Veuillez vérifier votre réseau."
            )
            if retry:
                # User chose to retry the connection check.
                self.perform_internet_check()
            else:
                # User chose to cancel; disable the connection button.
                self.button.config(state='disabled')
                self.errorVar.set("Connexion internet requise pour se connecter.")
        else:
            # Internet is connected; enable the connection button.
            self.button.config(state='normal')

    def register(self, event=None):
        """
        Attempt to find and authenticate the user by their email.
        Checks internet connectivity before proceeding.
        """
        try:
            self.button.config(state='disabled')
            self.logger.info("Register button clicked.")
        except tk.TclError as e:
            self.logger.error(f"Error disabling the button: {e}")
            return

        try:
            # Re-check internet connectivity before attempting to connect to Firebase.
            if not has_internet_connection():
                messagebox.showerror(
                    "Connexion Internet",
                    "Aucune connexion Internet détectée. Veuillez vérifier votre réseau."
                )
                self.perform_internet_check()
                return

            # Attempt to find the user by email.
            user_email = self.accountVar.get().strip()
            if not user_email:
                self.errorVar.set("Veuillez entrer une adresse email valide.")
                self.logger.warning("Empty email entered.")
                return

            self.logger.info(f"Attempting to find user with email: {user_email}")
            userFound = self.dbManager.findUserByEmail(user_email)

            if userFound:
                user_doc = userFound[0]
                user_role = user_doc.get("role")
                self.logger.info(f"User found: {user_doc.id}, Role: {user_role}")
                if user_role == "COACH":
                    self.logger.info("User is a coach. Access granted.")
                    self.userConnected = user_doc.id
                    self.frame.destroy()
                else:
                    self.errorVar.set("Erreur : vous avez besoin d'un compte entraîneur.")
                    self.logger.warning("User is not a coach.")
            else:
                self.errorVar.set("Erreur : cet utilisateur n'existe pas.")
                self.logger.warning("No user found with the provided email.")
        except Exception as e:
            self.logger.error(f"Exception occurred during registration: {e}")
            self.errorVar.set("Une erreur inattendue est survenue.")
        finally:
            if hasattr(self, 'button') and self.button.winfo_exists():
                try:
                    self.button.config(state='normal')
                    self.logger.debug("Button re-enabled after registration attempt.")
                except tk.TclError as e:
                    self.logger.error(f"Error re-enabling the button: {e}")
