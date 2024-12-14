import time
import sys
from tkinter import VERTICAL
from PIL import Image, ImageTk
from tkinter.font import BOLD, Font
from math import ceil
import ttkbootstrap as ttkb

from core.utils.DotDevice import DotDevice
from core.database.DatabaseManager import DatabaseManager, TrainingData


class StartingPage:
    """
    A class that creates a tkinter Toplevel window used for confirming and starting
    a training data recording session for a given device and a selected skater.

    This class provides a user interface that displays all the skaters associated
    with a particular coach. The coach can then select a skater and start recording
    training data for that skater on the specified device.
    """

    def __init__(self, device: DotDevice, db_manager: DatabaseManager, userConnected: str) -> None:
        """
        Initialize the StartingPage window.

        Args:
            device (DotDevice): The device instance for which the recording will be initiated.
            db_manager (DatabaseManager): The database manager instance for retrieving skaters
                                          and handling training data operations.
            userConnected (str): The username or identifier of the connected coach.

        The constructor sets up the Toplevel window, applies a logo icon (if available),
        and creates a scrollable interface listing all skaters. Each skater is represented
        by a button which, when clicked, triggers the start of a recording.
        """
        self.device = device
        self.db_manager = db_manager
        self.deviceTag = self.device.deviceTagName

        # Retrieve all skaters associated with the connected coach from the database.
        self.skaters = self.db_manager.getAllSkaterFromCoach(userConnected)

        # Create a top-level window for confirmation and align it at the center of the screen.
        self.window = ttkb.Toplevel(title="Confirmation", size=(1400, 400), topmost=True)
        self.window.place_window_center()

        # Try to load the logo from the PyInstaller bundle (if frozen) or from the local directory.
        try:
            ico = Image.open(f'{sys._MEIPASS}/img/Logo_s2mJUMP_RGB.png')  # Used if running from a PyInstaller build.
        except:
            ico = Image.open('img/Logo_s2mJUMP_RGB.png')  # Fallback for development environment.
        photo = ImageTk.PhotoImage(ico)
        self.window.wm_iconphoto(False, photo)

        # Configure the layout of the window.
        self.window.grid_rowconfigure(0, weight=0)  # Top row for title/label
        self.window.grid_rowconfigure(1, weight=1)  # Bottom row for skaters list
        self.window.grid_columnconfigure(0, weight=1, pad=20)  # Main content column
        self.window.grid_columnconfigure(1, weight=0)  # Scrollbar column

        # Primary label indicating the device for which the recording can be started.
        self.label = ttkb.Label(
            self.window,
            text=f"Lancer un enregistrement sur le capteur {self.deviceTag}",
            font=Font(self.window, size=20, weight=BOLD)
        )
        self.label.grid(row=0, column=0, columnspan=2, pady=20)

        # Create a canvas to enable scrolling if the list of skaters exceeds the vertical space.
        self.canvas = ttkb.Canvas(self.window)
        self.canvas.grid_rowconfigure(0, weight=1)
        self.canvas.grid_columnconfigure(0, weight=1)

        # Frame that will hold all the skater buttons.
        self.frame = ttkb.Frame(self.canvas)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        # Evenly distribute space across 5 columns for the skater buttons.
        for col in range(5):
            self.frame.grid_columnconfigure(col, weight=1)

        # Define a style for the skater buttons to ensure uniform appearance.
        buttonStyle = ttkb.Style()
        buttonStyle.configure('my.TButton', font=Font(self.frame, size=12, weight=BOLD))

        # Dynamically create buttons for each skater.
        # Each button, when clicked, calls startRecord with the skater's ID and name.
        for i, skater in enumerate(self.skaters):
            button = ttkb.Button(
                self.frame,
                text=f"\n{skater.skater_name}\n",  # Additional newlines for visual padding.
                style="my.TButton",
                width=ceil((250 - 24) / 11),  # Adjusting width to fit text comfortably.
                command=(lambda x=skater.skater_id, y=skater.skater_name: self.startRecord(x, y))
            )
            # Place the button in a grid, distributing them in a 5-column layout.
            button.grid(row=i // 5 + 1, column=i % 5, padx=10, pady=10)

        # Bind mouse enter/leave events to enable or disable mousewheel scrolling on the frame.
        self.frame.bind('<Enter>', self._bound_to_mousewheel)
        self.frame.bind('<Leave>', self._unbound_to_mousewheel)

        self.frame.grid(row=0, column=0, sticky="nswe")

        # Add a vertical scrollbar to the window for the canvas.
        scroll = ttkb.Scrollbar(self.window, orient=VERTICAL, command=self.canvas.yview)
        scroll.grid(row=1, column=1, sticky="ns")

        # Configure canvas scrolling through the scrollbar.
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.bind(
            '<Configure>',
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Place the frame inside the canvas. The anchor is "center" to align content properly.
        self.canvas.create_window((0, 0), window=self.frame, anchor="center")
        self.canvas.grid(row=1, column=0, sticky="nswe", padx=10)

        self.window.grid()

    def startRecord(self, skaterId: str, skaterName: str):
        """
        Start the recording process for a selected skater on the given device.

        This method:
        - Creates a new training data record in the database linked to the current device and skater.
        - Initiates the actual recording process on the device.
        - Updates the UI to provide feedback (e.g., success or error message).
        - Closes the confirmation window after a brief pause.

        Args:
            skaterId (str): The unique identifier of the selected skater.
            skaterName (str): The name of the selected skater.
        """
        deviceId = self.device.deviceId

        # Create a new training data entry in the database and mark it as the current record.
        new_training = TrainingData(0, skaterId, 0, deviceId, [])
        self.db_manager.set_current_record(deviceId, self.db_manager.save_training_data(new_training))

        # Attempt to start the record on the actual physical device.
        recordStarted = self.device.startRecord()

        # Clean up the initial UI components: remove the canvas and the main label.
        self.canvas.destroy()
        self.label.destroy()

        # Create a new frame to display the recording status message.
        self.frame = ttkb.Frame(self.window)
        if recordStarted:
            message = f"Enregistrement commencé sur le capteur {self.deviceTag} pour {skaterName}"
        else:
            message = "Erreur durant le lancement, impossible de lancer l'enregistrement"

        label = ttkb.Label(self.frame, text=message, font=Font(self.window, size=20, weight=BOLD))
        label.grid()
        self.frame.grid(row=1, column=0)

        # Update the UI to ensure the message is visible, then wait for a short time before closing.
        self.window.update()
        time.sleep(1)

        # Clean up and close the confirmation window.
        self.canvas.destroy()
        self.window.destroy()

    def _bound_to_mousewheel(self, event):
        """
        Bind the mousewheel scrolling event to the canvas when the mouse is over the frame.

        By binding <MouseWheel> events to the entire application (via bind_all),
        the user can scroll the list of skaters by hovering the mouse over the frame.
        """
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        """
        Unbind the mousewheel scrolling event when the mouse leaves the frame.

        Once the mouse leaves the frame, the user should not be able to scroll
        by using the mousewheel, avoiding unwanted scrolling behavior.
        """
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        """
        Handle mousewheel scrolling within the canvas.

        This method translates mousewheel events into vertical scrolling of the canvas.
        The 'event.delta' value gives the scroll direction and magnitude.
        Using 'yview_scroll' we move the view by a certain number of "units".

        Args:
            event: The tkinter mousewheel event containing the scroll direction and magnitude.
        """
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
