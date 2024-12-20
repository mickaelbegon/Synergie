import time
import tkinter as tk
from PIL import Image, ImageTk
import ttkbootstrap as ttkb
from tkinter import messagebox
import threading
import sys

from front.ConnectionPage import ConnectionPage
from core.database.DatabaseManager import DatabaseManager
from core.utils.DotManager import DotManager
from front.StartingPage import StartingPage
from front.StopingPage import StopingPage
from front.MainPage import MainPage

class App:
    """
    The main application class that orchestrates the connection process,
    device (dot) management, and transitions between different pages (ConnectionPage,
    MainPage, StartingPage, and StoppingPage).

    This class initializes the database and dot managers, checks user
    connection status, manages dot connections, and launches appropriate UI pages.
    """

    def __init__(self, root: ttkb.Window):
        """
        Initialize the application with the given root window.

        Args:
            root (ttkb.Window): The main application window (tkinter-based).
        """
        # Initialize the database manager and the dot manager.
        self.db_manager = DatabaseManager()
        self.dot_manager = DotManager(self.db_manager)

        # Store the root window for later use in the application.
        self.root = root

        # Launch the ConnectionPage to prompt user login.
        self.connectionPage = ConnectionPage(self.root, self.db_manager)

        # Check periodically if the user has logged in.
        self.checkConnection()

    def checkConnection(self):
        """
        Periodically check if the user is connected. If the user is connected,
        destroy the connection frame and launch the main page. Otherwise,
        schedule another check.
        """
        if self.connectionPage.userConnected != "":
            # User is connected, store the user id and move to the main page.
            self.userConnected = self.connectionPage.userConnected
            self.connectionPage.frame.destroy()
            self.root.update()
            self.launchMainPage()
        else:
            # User not connected yet, recheck after 100 ms.
            self.root.after(100, self.checkConnection)

    def launchMainPage(self):
        """
        Launch the MainPage once the user is connected. This sets up
        an asynchronous initialization process to detect and connect dots.
        """
        # Initialize main page with empty list of dots (will be updated later).
        self.mainPage = MainPage([], self.dot_manager, self.db_manager, self.root)

        # Create an event to signal when initialization is complete.
        self.initialEvent = threading.Event()

        # Run the initialization in a separate thread to avoid blocking the UI.
        threading.Thread(target=self.initialize, args=([self.initialEvent]), daemon=True).start()

        # Check periodically if initialization is done.
        self.checkInit()

    def checkInit(self):
        """
        Once the initialization (first device connection attempt) is complete,
        update the main page with the connected devices and display them.
        """
        if self.initialEvent.is_set():
            # Initialization is done, update dots in the MainPage.
            self.mainPage.dotsConnected = self.dot_manager.get_devices()
            self.mainPage.make_dot_page()
            # Clear the event for potential future use if needed.
            self.initialEvent.clear()
        else:
            # Not initialized yet, check again after 100 ms.
            self.root.after(100, self.checkInit)

    def initialize(self, initialEvent: threading.Event):
        """
        Attempt the first connection to all necessary devices (dots).
        If any device is not connected, prompt the user to retry until all
        are connected or the user cancels. Once successful, start a separate thread
        to monitor device connections/disconnections (USB events).

        Args:
            initialEvent (threading.Event): An event to set once initialization is done.
        """
        (check, unconnectedDevice) = self.dot_manager.first_connection()
        # If not all devices are connected, prompt user to reconnect missing devices.
        while not check:
            deviceMessage = f"{unconnectedDevice[0]}"
            for deviceTag in unconnectedDevice[1:]:
                deviceMessage += f", {deviceTag}"
            # Ask the user to retry the connection for missing devices.
            retry = messagebox.askretrycancel("Connexion", f"Veuillez reconnecter les capteurs {deviceMessage}")
            if not retry:
                # User chose to cancel; exit the initialization loop.
                return
            (check, unconnectedDevice) = self.dot_manager.first_connection()

        # Once all devices are connected, set the event to signal completion.
        initialEvent.set()

        # Now that all devices are connected, check if any are currently recording.
        devices = self.dot_manager.get_devices()
        for device in devices:
            # Assuming `device.is_recording` is a boolean attribute or a method
            if device.is_recording:
                # Stop the device from recording
                stopped = self.dot_manager.stop_recording(device)
                if stopped:
                    self.logger.info(f"Stopped recording on device {device.deviceId}")
                else:
                    self.logger.warning(f"Failed to stop recording on device {device.deviceId}")


        # Start a separate thread to monitor device connections/disconnections.
        usb_detection_thread = threading.Thread(
            target=self.checkUsbDots,
            args=([self.startStopping, self.startStarting]),
            daemon=True
        )
        usb_detection_thread.start()

    def checkUsbDots(self, callbackStop, callbackStart):
        """
        Continuously monitor device connection status. If a device is connected
        while recording, stop it and notify the user. If a device is disconnected while
        not recording, start it.

        Args:
            callbackStop (function): Function to call when a recording device reconnects.
            callbackStart (function): Function to call when a non-recording device disconnects.
        """
        while True:
            # Check the status of devices.
            checkUsb = self.dot_manager.check_devices()
            lastConnected = checkUsb[0]    # Devices that got connected since last check.
            lastDisconnected = checkUsb[1] # Devices that got disconnected since last check.

            # If any devices have been connected, check if they need to be stopped.
            if lastConnected:
                print("Connexion")
                for device in lastConnected:
                    # If device is currently recording or has pending records, stop it.
                    if device.is_recording or device.recordingCount > 0:
                        callbackStop(device)

            # If any devices have been disconnected, check if we should start them.
            if lastDisconnected:
                print("Deconnexion")
                for device in lastDisconnected:
                    # If device is not currently recording, start it.
                    if not device.is_recording:
                        callbackStart(device)

            # Wait a short time before checking again.
            time.sleep(0.2)

    def handle_device_recording(self, device):
        """
        Handle stopping the recording on a device and notify the user.

        This method stops the recording on the given device and displays a message
        to inform the user that the recording has been stopped.

        Args:
            device: The device instance that is currently recording and needs to be stopped.
        """
        # Stop the recording on the device.
        self.dot_manager.stop_recording(device)

        # Notify the user that the recording has been stopped.
        messagebox.showinfo(
            "Arrêt de l'enregistrement",
            f"L'enregistrement sur le capteur {device.deviceTagName} a été arrêté."
        )

        # Optionally, you can also launch the StoppingPage or perform other UI updates here.
        self.startStopping(device)

    def startStopping(self, device):
        """
        Launch the StopingPage for a given device to stop recording.

        Args:
            device: The device instance that needs to be stopped.
        """
        StopingPage(device, self.db_manager)

    def startStarting(self, device):
        """
        Launch the StartingPage for a given device to start a new recording.

        Args:
            device: The device instance to start recording.
        """
        StartingPage(device, self.db_manager, self.userConnected)


# Entry point of the application.
if __name__ == "__main__":
    # Initialize the main application window.
    root = ttkb.Window(title="Synergie", themename="minty")

    # Create an instance of the App class with the root window.
    myapp = App(root)

    # Maximize the window to the screen size.
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.geometry(f"{width}x{height}")

    # Try to load the application icon from the PyInstaller bundle or from a local path.
    try:
        ico = Image.open(f'{sys._MEIPASS}/img/Logo_s2mJUMP_RGB.png')
    except:
        ico = Image.open('img/Logo_s2mJUMP_RGB.png')

    photo = ImageTk.PhotoImage(ico)
    root.wm_iconphoto(False, photo)

    # Start the main event loop.
    root.mainloop()
