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
from core.utils.sensor_diagnostics import probe_movella_usb_detection

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
        self.initializationCancelled = False
        self.root.after(100, self.initialize)

    def initialize(self):
        probe = probe_movella_usb_detection(timeout_seconds=20)
        if not probe.ok:
            if probe.timed_out:
                retry_message = (
                    "Le scan USB du SDK Movella ne repond pas.\n\n"
                    "Windows voit probablement des ports COM, mais le SDK reste bloque pendant la detection.\n"
                    "Fermez les autres applications Movella/Xsens, puis debranchez et rebranchez les capteurs USB."
                )
            else:
                retry_message = (
                    "Le diagnostic USB Movella a echoue.\n\n"
                    f"Detail: {probe.error or probe.output or 'aucun detail disponible'}"
                )
            if self._ask_retry_cancel("Connexion capteurs", retry_message):
                self.root.after(100, self.initialize)
            else:
                self.initializationCancelled = True
            return

        if probe.detected_count == 0:
            if self._ask_retry_cancel(
                "Connexion capteurs",
                "Aucun capteur Movella DOT detecte en USB.\n\n"
                "Branchez les capteurs en USB, attendez quelques secondes, puis reessayez.",
            ):
                self.root.after(100, self.initialize)
            else:
                self.initializationCancelled = True
            return

        self.mainPage.set_waiting_status(f"{probe.detected_count} capteur(s) USB detecte(s): {', '.join(probe.ports or [])}")
        (check, unconnectedDevice) = self.dot_manager.firstConnection()
        self.mainPage.set_waiting_status(self.dot_manager.statusMessage or self.dot_manager.lastError)
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
                return
            (check, unconnectedDevice) = self.dot_manager.firstConnection()
            self.mainPage.set_waiting_status(self.dot_manager.statusMessage or self.dot_manager.lastError)

        self.mainPage.dotsConnected = self.dot_manager.getDevices()
        self.mainPage.make_dot_page()

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
        return bool(messagebox.askretrycancel(title, message))

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
