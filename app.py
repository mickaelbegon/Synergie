import synergie.runtime  # noqa: F401
import logging
import sys
import time
from PIL import Image, ImageTk
import ttkbootstrap as ttkb
from tkinter import messagebox
import threading

from front.ConnectionPage import ConnectionPage
from core.database.DatabaseManager import *

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

class App:

    def __init__(self, root : ttkb.Window):
        self.db_manager = DatabaseManager()
        self.root = root
        self.dot_manager = None

        try:
            from core.utils.DotManager import DotManager
            self.dot_manager = DotManager(self.db_manager)
        except Exception as exc:
            messagebox.showerror(
                "Synergie",
                "Le SDK Movella DOT n'est pas disponible dans cet environnement.\n"
                f"Détail: {exc}",
            )
            raise

        self.connectionPage = ConnectionPage(self.root, self.db_manager)
        self.checkConnection()

    def checkConnection(self):
        if self.connectionPage.userConnected != "":
            self.userConnected = self.connectionPage.userConnected
            self.connectionPage.frame.destroy()
            self.root.update()
            self.launchMainPage()
        else:
            self.root.after(100, self.checkConnection)

    def launchMainPage(self):
        from front.MainPage import MainPage
        self.mainPage = MainPage([], self.dot_manager, self.db_manager, self.root)
        self.initialEvent = threading.Event()
        self.initializationCancelled = False
        threading.Thread(target=self.initialize, args=([self.initialEvent]), daemon=True).start()
        self.checkInit()
    
    def checkInit(self):
        if self.initialEvent.is_set():
            if self.initializationCancelled:
                self.initialEvent.clear()
                return
            self.mainPage.dotsConnected = self.dot_manager.getDevices()
            self.mainPage.make_dot_page()
            self.initialEvent.clear()
        else:
            self.root.after(100, self.checkInit)

    def initialize(self, initialEvent : threading.Event):
        (check, unconnectedDevice) = self.dot_manager.firstConnection()
        while not check:
            deviceMessage = ", ".join(unconnectedDevice)
            if deviceMessage:
                retry_message = f"Veuillez reconnecter les capteurs {deviceMessage}"
            elif self.dot_manager.lastError:
                retry_message = self.dot_manager.lastError
            else:
                retry_message = "La connexion aux capteurs a echoue. Veuillez reessayer."
            should_retry = self._ask_retry_cancel(
                "Connexion",
                retry_message,
            )
            if not should_retry:
                _logger.warning("Sensor initialization cancelled by user.")
                self.initializationCancelled = True
                initialEvent.set()
                return
            (check, unconnectedDevice) = self.dot_manager.firstConnection()

        initialEvent.set()

        usb_detection_thread = threading.Thread(target=self.checkUsbDots, args=([self.startStopping, self.startStarting]))
        usb_detection_thread.daemon = True
        usb_detection_thread.start()

    def checkUsbDots(self, callbackStop, callbackStart):
        while True:
            checkUsb = self.dot_manager.checkDevices()
            lastConnected = checkUsb[0]
            lastDisconnected = checkUsb[1]
            if lastConnected:
                _logger.info("USB connection detected")
                for device in lastConnected:
                    if device.isRecording or device.recordingCount > 0 :
                        self.root.after(0, lambda selected_device=device: callbackStop(selected_device))
            if lastDisconnected:
                _logger.info("USB disconnection detected")
                for device in lastDisconnected:
                    if not device.isRecording:
                        self.root.after(0, lambda selected_device=device: callbackStart(selected_device))
            time.sleep(0.2)

    def startStopping(self, device):
        from front.StopingPage import StopingPage
        StopingPage(device, self.db_manager)
    
    def startStarting(self, device):
        from front.StartingPage import StartingPage
        StartingPage(device, self.db_manager, self.userConnected)

    def _ask_retry_cancel(self, title: str, message: str) -> bool:
        decision = {"retry": False}
        event = threading.Event()

        def prompt():
            decision["retry"] = messagebox.askretrycancel(title, message)
            event.set()

        self.root.after(0, prompt)
        event.wait()
        return bool(decision["retry"])

def main():
    root = ttkb.Window(title="Synergie", themename="minty")
    App(root)
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.geometry("%dx%d" % (width, height))
    try :
        ico = Image.open(f'{sys._MEIPASS}/img/Logo_s2mJUMP_RGB.png')
    except:
        ico = Image.open(f'img/Logo_s2mJUMP_RGB.png')
    photo = ImageTk.PhotoImage(ico)
    root.wm_iconphoto(False, photo)
    root.mainloop()


if __name__ == "__main__":
    main()
