import socket
import logging
from tkinter.font import BOLD, Font
import ttkbootstrap as ttkb
from tkinter import messagebox
from core.database.DatabaseManager import DatabaseManager


class ConnectionPage:
    """
    The ConnectionPage class represents a login interface where the user enters their email address
    to connect. This page checks if the user is a coach and, if so, allows them to access the rest
    of the application. It also verifies internet connectivity before allowing any database operations.
    """

    def __init__(self, root: ttkb.Window, dbManager: DatabaseManager) -> None:
        """
        Initialize the ConnectionPage.

        This constructor sets up the UI components for the connection page and ensures that
        the application is connected to the internet before allowing the user to attempt
        to connect to the Firebase database.

        Args:
            root (ttkb.Window): The main application window.
            dbManager (DatabaseManager): An instance of the DatabaseManager for database operations.
        """
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

        # Configure the main window grid to properly contain the frame.
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

        # Bind the ENTER key to the register method when the entry widget is focused.
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

        # Label to show error messages when user is not found or not a coach.
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
        self.check_internet_connection()

    def clear_error(self, event):
        """
        Clears the error message when the user starts typing a new email.

        Args:
            event: The event object triggered by key press.
        """
        self.errorVar.set("")

    def check_internet_connection(self):
        """
        Check if the computer is connected to the internet.

        If there is no internet connection, display a dialog with Retry and Cancel options.
        The user can choose to retry the connection check or cancel, which will disable the
        connection functionality.
        """
        if not self._has_internet_connection():
            retry = messagebox.askretrycancel(
                "Connexion Internet",
                "Aucune connexion Internet détectée. Veuillez vérifier votre réseau."
            )
            if retry:
                # User chose to retry the connection check.
                self.check_internet_connection()
            else:
                # User chose to cancel; disable the connection button.
                self.button.config(state='disabled')
                self.errorVar.set("Connexion internet requise pour se connecter.")
        else:
            # Internet is connected; enable the connection button.
            self.button.config(state='normal')

    def _has_internet_connection(self) -> bool:
        """
        Check if the computer is connected to the internet.

        This method tries to create a socket connection to a well-known host (Google DNS).
        If it succeeds, it assumes the internet is accessible.

        Returns:
            bool: True if internet connection is available, False otherwise.
        """
        try:
            socket.setdefaulttimeout(3)
            # Connect to Google's DNS server to check for internet connectivity.
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
            return True
        except socket.error:
            return False

    def register(self, event=None):
        """
        Attempt to find and authenticate the user by their email.

        This method first checks for internet connectivity. If connected, it proceeds to
        search for the user in the database. If the user exists and has a "COACH" role,
        the login is successful. Otherwise, an appropriate error message is displayed.

        Args:
            event: The event object (optional, used when called via key binding).
        """
        # Disable the button to prevent multiple clicks.
        try:
            self.button.config(state='disabled')
            self.logger.info("Register button clicked.")
        except tk.TclError as e:
            self.logger.error(f"Error disabling the button: {e}")
            return  # Exit if button cannot be disabled

        try:
            # Re-check internet connectivity before attempting to connect to Firebase.
            if not self._has_internet_connection():
                messagebox.showerror(
                    "Connexion Internet",
                    "Aucune connexion Internet détectée. Veuillez vérifier votre réseau."
                )
                self.check_internet_connection()
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
                # User exists in the database, check their role.
                user_doc = userFound[0]
                user_role = user_doc.get("role")
                self.logger.info(f"User found: {user_doc.id}, Role: {user_role}")
                if user_role == "COACH":
                    # User is a coach, grant access.
                    self.logger.info("User is a coach. Access granted.")
                    self.userConnected = user_doc.id
                    # Proceed to the next step in the application, e.g., closing the connection page.
                    self.frame.destroy()
                else:
                    # User is not a coach, show error message.
                    self.errorVar.set("Erreur : vous avez besoin d'un compte entraîneur.")
                    self.logger.warning("User is not a coach.")
            else:
                # No matching user found, show error message.
                self.errorVar.set("Erreur : cet utilisateur n'existe pas.")
                self.logger.warning("No user found with the provided email.")
        except Exception as e:
            self.logger.error(f"Exception occurred during registration: {e}")
            self.errorVar.set("Une erreur inattendue est survenue.")
        finally:
            # Re-enable the button after processing, only if it still exists.
            if hasattr(self, 'button') and self.button.winfo_exists():
                try:
                    self.button.config(state='normal')
                    self.logger.debug("Button re-enabled after registration attempt.")
                except tk.TclError as e:
                    self.logger.error(f"Error re-enabling the button: {e}")
