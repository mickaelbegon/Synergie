import os
import logging

import numpy as np
from core.utils.DotDevice import DotDevice
from core.database.DatabaseManager import DatabaseManager
from core.utils.xdpchandler import *
from core.utils.device_support import is_valid_bluetooth_address
import asyncio
from typing import List
if os.name == 'nt':
    try:
        from winrt.windows.devices import radios
    except ImportError:
        radios = None

_logger = logging.getLogger(__name__)

async def bluetooth_power(turn_on):
    if radios is None:
        raise RuntimeError(
            "Windows Bluetooth control module 'winrt' is not available. "
            "Install the WinRT dependencies or leave Bluetooth enabled manually."
        )
    all_radios = await radios.Radio.get_radios_async()
    for this_radio in all_radios:
        if this_radio.kind == radios.RadioKind.BLUETOOTH:
            if turn_on:
                result = await this_radio.set_state_async(radios.RadioState.ON)
            else:
                result = await this_radio.set_state_async(radios.RadioState.OFF)

class DotManager:
    """
    Class pour gérer la première connexion aux capteurs
    """
    def __init__(self, db_manager : DatabaseManager) -> None:
        self.db_manager = db_manager
        self.error = False
        self.devices : List[DotDevice] = []
        self.previousConnected : List[DotDevice] = []
        self.lastError = ""
        self.statusMessage = ""

    def firstConnection(self) -> tuple[bool, List[str]]:
        """
        Première connexion aux capteurs, pour cela on désactive d'abord le bluetooth pour se connecter en USB aux capteurs,
        puis on réactive le bluetooth pour détecter les possibles connections bluetooth.
        On lie ces deux connexions grâce au deviceId et on créé des DotDevice qui englobe ces deux connexions pour un capteur.
        Il y a aussi une vérification lors de l'initialisation que les connexions bluetooth correspondent aux connexions USB disponible
        """
        self.devices = []
        self.previousConnected = []
        self.lastError = ""
        self.statusMessage = "Preparing first sensor connection"
        check = True
        self._set_bluetooth_power(False)
        xdpcHandler = XdpcHandler()
        if not xdpcHandler.initialize():
            self.lastError = "Unable to initialize the Movella DOT connection manager."
            xdpcHandler.cleanup()
            return (False, [])
        self.statusMessage = "Detecting USB-connected DOT sensors"
        xdpcHandler.detectUsbDevices()
        detected_usb_ports = xdpcHandler.detectedDots()
        if not detected_usb_ports:
            self.lastError = (
                "Aucun capteur Movella DOT detecte en USB. "
                "Au demarrage, les capteurs doivent etre branches en USB "
                "(pas seulement allumes en Bluetooth)."
            )
            xdpcHandler.cleanup()
            return (False, [])
        self.portInfoUsb = {}
        retries = 10
        while len(xdpcHandler.connectedUsbDots()) < len(xdpcHandler.detectedDots()) and retries > 0:
            self.statusMessage = (
                f"Opening USB connections ({len(xdpcHandler.connectedUsbDots())}/"
                f"{len(xdpcHandler.detectedDots())})"
            )
            xdpcHandler.connectDots()
            retries -= 1
            time.sleep(0.2)
        if len(xdpcHandler.connectedUsbDots()) < len(xdpcHandler.detectedDots()):
            self.lastError = (
                f"Seulement {len(xdpcHandler.connectedUsbDots())} capteur(s) USB sur "
                f"{len(xdpcHandler.detectedDots())} ont pu etre ouverts. "
                "Verifiez les cables USB et rebranchez les capteurs."
            )
        expected_bluetooth_addresses = []
        for device in xdpcHandler.connectedUsbDots():
            self.portInfoUsb[str(device.deviceId())] = device.portInfo()
            if hasattr(device, "bluetoothAddress"):
                bluetooth_address = device.bluetoothAddress()
                if is_valid_bluetooth_address(bluetooth_address):
                    expected_bluetooth_addresses.append(bluetooth_address)
        xdpcHandler.cleanup()
        self._set_bluetooth_power(True)
        xdpcHandler = XdpcHandler()
        if not xdpcHandler.initialize():
            self.lastError = "Unable to initialize the Movella DOT Bluetooth scanner."
            xdpcHandler.cleanup()
            return (False, [])
        scan_attempts = 3
        while scan_attempts > 0:
            self.statusMessage = "Scanning Bluetooth advertisements from DOT sensors"
            xdpcHandler.scanForDots(white_list=expected_bluetooth_addresses)
            found_addresses = [port_info.bluetoothAddress() for port_info in xdpcHandler.detectedDots()]
            if not expected_bluetooth_addresses or all(address in found_addresses for address in expected_bluetooth_addresses):
                break
            scan_attempts -= 1
            time.sleep(1)
        self.portInfoBt = xdpcHandler.detectedDots()
        xdpcHandler.cleanup()
        if self.portInfoUsb and not self.portInfoBt:
            self.lastError = (
                "Les capteurs ont ete vus en USB mais pas retrouves en Bluetooth. "
                "Laissez le Bluetooth Windows active et attendez quelques secondes, "
                "puis reessayez."
            )
            return (False, [])

        unconnectedDevice = []

        for portInfoBt in self.portInfoBt:
            if not is_valid_bluetooth_address(portInfoBt.bluetoothAddress()):
                continue
            device = self.db_manager.get_dot_from_bluetooth(portInfoBt.bluetoothAddress())
            if device is None :
                _logger.info("Adding a new device")
                deviceId = self.connectNewDevice(portInfoBt)
            else:
                deviceId = device.id
            portInfoUsb = self.portInfoUsb.get(deviceId, None)
            if portInfoUsb is not None:
                try:
                    self.devices.append(DotDevice(portInfoUsb, portInfoBt, self.db_manager))
                except Exception as exc:
                    self.lastError = str(exc)
                    _logger.error(f"Unable to initialize device {deviceId}: {exc}")
                    if device is not None:
                        unconnectedDevice.append(device.get("tag_name"))
                    else:
                        unconnectedDevice.append(deviceId)
                    check = False
            else:
                missing_name = device.get('tag_name') if device is not None else deviceId
                _logger.warning(f"Please plug sensor {missing_name}")
                unconnectedDevice.append(missing_name)
                check = False

        self.previousConnected = self.devices
        self.statusMessage = f"{len(self.devices)} sensor(s) ready"
        return (check, unconnectedDevice)
    
    def checkDevices(self) -> tuple[List[DotDevice], List[DotDevice]]:
        """
        Détection des capteurs connectés en USB afin de capter un branchement ou un débranchement
        """
        connected : List[DotDevice] = []
        for device in self.devices:
            if device.isBatteryCharging:
                connected.append(device)

        lastConnected = []
        lastDisconnected = []
        if len(self.previousConnected) > len(connected):
            for device in self.previousConnected:
                if device not in connected:
                    device.closeUsb()
                    lastDisconnected.append(device)
        elif len(self.previousConnected) < len(connected):
            for device in connected:
                if device not in self.previousConnected:
                    if device.openUsb():
                        lastConnected.append(device)
        else:
            pass

        self.previousConnected = connected
        return(lastConnected,lastDisconnected)

    def getExportEstimatedTime(self):
        """
        Estimation du temps d'extraction pour tous les capteurs en même temps
        """
        estimatedTime = [0]
        for device in self.devices:
            estimatedTime.append(device.getExportEstimatedTime())
        return np.max(estimatedTime)

    def getDevices(self):
        return self.devices
    
    def connectNewDevice(self, portInfoBt : XsPortInfo):
        """
        Ajoute un capteur à la base de données
        """
        manager = XsDotConnectionManager()
        checkDevice = False
        retries = 5
        while not checkDevice and retries > 0:
            manager.closePort(portInfoBt)
            if not manager.openPort(portInfoBt):
                _logger.warning(f"Connection to Device {portInfoBt.bluetoothAddress()} failed")
                checkDevice = False
            else:
                device : XsDotDevice = manager.device(portInfoBt.deviceId())
                if device is None:
                    checkDevice = False
                else:
                    time.sleep(1)
                    checkDevice = (device.deviceTagName() != '') and (device.batteryLevel() != 0)
            retries -= 1
            if not checkDevice:
                time.sleep(0.2)
        if not checkDevice:
            raise ConnectionError(f"Unable to connect new bluetooth device {portInfoBt.bluetoothAddress()}")
        self.db_manager.save_dot_data(str(device.deviceId()), device.bluetoothAddress(), device.deviceTagName())
        manager.closePort(portInfoBt)
        return str(device.deviceId())

    def _set_bluetooth_power(self, turn_on: bool):
        try:
            if os.name == 'nt':
                asyncio.run(bluetooth_power(turn_on))
            elif os.name == 'posix':
                os.system('rfkill unblock bluetooth' if turn_on else 'rfkill block bluetooth')
        except Exception as exc:
            self.lastError = str(exc)
            _logger.warning(f"Unable to toggle bluetooth power automatically: {exc}")
