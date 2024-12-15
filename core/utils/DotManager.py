import os
import time
import asyncio
import numpy as np
from typing import List, Tuple
from core.utils.DotDevice import DotDevice
from core.database.DatabaseManager import DatabaseManager
from core.utils.xdpchandler import XdpcHandler  # Ensure XdpcHandler is correctly imported

if os.name == 'nt':
    from winrt.windows.devices import radios

import logging

# Import XsPortInfo explicitly from movelladot_pc_sdk
from movelladot_pc_sdk.movelladot_pc_sdk_py39_64 import XsPortInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DotManager:
    """
    Class to manage the initial connection to sensors.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.error = False
        self.devices: List[DotDevice] = []
        self.previousConnected: List[DotDevice] = []
        logger.info("DotManager initialized.")

    async def bluetooth_power(self, turn_on: bool):
        """
        Asynchronously turn Bluetooth radios on or off.

        Args:
            turn_on (bool): True to turn on Bluetooth, False to turn it off.
        """
        all_radios = await radios.Radio.get_radios_async()
        for this_radio in all_radios:
            if this_radio.kind == radios.RadioKind.BLUETOOTH:
                if turn_on:
                    result = await this_radio.set_state_async(radios.RadioState.ON)
                    logger.info("Bluetooth turned ON.")
                else:
                    result = await this_radio.set_state_async(radios.RadioState.OFF)
                    logger.info("Bluetooth turned OFF.")

    def firstConnection(self) -> Tuple[bool, List[str]]:
        """
        Initial connection to sensors. Disables Bluetooth to connect via USB first,
        then re-enables Bluetooth to detect possible Bluetooth connections.
        Links USB and Bluetooth connections using deviceId and creates DotDevice instances
        that encompass both connections for each sensor. Also verifies that Bluetooth
        connections correspond to available USB connections during initialization.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean indicating success and
            a list of unconnected device tag names.
        """
        self.devices = []
        self.previousConnected = []
        check = True

        # Disable Bluetooth
        if os.name == 'nt':
            asyncio.run(self.bluetooth_power(False))
        elif os.name == 'posix':
            os.system('rfkill block bluetooth')
            logger.info("Bluetooth blocked using rfkill.")
        else:
            logger.warning("Unsupported OS for Bluetooth power control.")

        # Initialize USB connection handler
        xdpcHandler = XdpcHandler()
        if not xdpcHandler.initialize():
            logger.error("Failed to initialize XdpcHandler for USB.")
            xdpcHandler.cleanup()
            self.error = True
            return (False, ["Initialization failed"])

        xdpcHandler.detectUsbDevices()
        self.portInfoUsb = {}
        while len(xdpcHandler.connectedUsbDots()) < len(xdpcHandler.detectedDots()):
            xdpcHandler.connectDots()
        for device in xdpcHandler.connectedUsbDots():
            self.portInfoUsb[str(device.deviceId())] = device.portInfo()
        xdpcHandler.cleanup()
        logger.info(f"Connected USB devices: {list(self.portInfoUsb.keys())}")

        # Re-enable Bluetooth
        if os.name == 'nt':
            asyncio.run(self.bluetooth_power(True))
        elif os.name == 'posix':
            os.system('rfkill unblock bluetooth')
            logger.info("Bluetooth unblocked using rfkill.")
        else:
            logger.warning("Unsupported OS for Bluetooth power control.")

        # Initialize Bluetooth connection handler
        xdpcHandler = XdpcHandler()
        if not xdpcHandler.initialize():
            logger.error("Failed to initialize XdpcHandler for Bluetooth.")
            xdpcHandler.cleanup()
            self.error = True
            return (False, ["Initialization failed"])
        xdpcHandler.scanForDots()
        self.portInfoBt = xdpcHandler.detectedDots()
        xdpcHandler.cleanup()
        logger.info(f"Detected Bluetooth devices: {[bt.bluetoothAddress() for bt in self.portInfoBt]}")  # Fixed Line

        unconnectedDevice = []

        # Link USB and Bluetooth devices
        for portInfoBt in self.portInfoBt:
            device = self.db_manager.get_dot_from_bluetooth(portInfoBt.bluetoothAddress())
            if device is None:
                logger.info("Adding a new device.")
                deviceId = self.connectNewDevice(portInfoBt)
            else:
                deviceId = device.id
                logger.info(f"Found existing device with ID: {deviceId}")

            portInfoUsb = self.portInfoUsb.get(deviceId, None)
            if portInfoUsb is not None:
                dot_device = DotDevice(portInfoUsb, portInfoBt, self.db_manager)
                self.devices.append(dot_device)
                logger.info(f"DotDevice created for device ID: {deviceId}")
            else:
                tag_name = device.get('tag_name') if device else "Unknown"
                logger.warning(f"Please plug sensor {tag_name}.")
                unconnectedDevice.append(tag_name)
                time.sleep(5)
                check = False

        self.previousConnected = self.devices.copy()
        logger.info(f"Total connected devices: {len(self.devices)}")
        return (check, unconnectedDevice)

    def checkDevices(self) -> Tuple[List[DotDevice], List[DotDevice]]:
        """
        Detects USB-connected sensors to capture any new connections or disconnections.

        Returns:
            Tuple[List[DotDevice], List[DotDevice]]: Lists of last connected and last disconnected devices.
        """
        connected: List[DotDevice] = []
        for device in self.devices:
            if device.btDevice.isCharging():
                connected.append(device)

        lastConnected = []
        lastDisconnected = []
        if len(self.previousConnected) > len(connected):
            for device in self.previousConnected:
                if device not in connected:
                    device.closeUsb()
                    lastDisconnected.append(device)
                    logger.info(f"Device {device.deviceId} disconnected.")
        elif len(self.previousConnected) < len(connected):
            for device in connected:
                if device not in self.previousConnected:
                    device.openUsb()
                    lastConnected.append(device)
                    logger.info(f"Device {device.deviceId} connected.")
        else:
            logger.info("No changes in device connections.")

        self.previousConnected = connected.copy()
        return (lastConnected, lastDisconnected)

    def getExportEstimatedTime(self) -> float:
        """
        Estimates the extraction time for all sensors simultaneously.

        Returns:
            float: The maximum estimated time among all devices.
        """
        estimatedTimes = [0]
        for device in self.devices:
            estimatedTimes.append(device.getExportEstimatedTime())
        max_time = np.max(estimatedTimes)
        logger.info(f"Estimated export time: {max_time} seconds.")
        return max_time

    def getDevices(self) -> List[DotDevice]:
        """
        Retrieves the list of managed DotDevice instances.

        Returns:
            List[DotDevice]: The list of devices.
        """
        return self.devices.copy()

    def connectNewDevice(self, portInfoBt: XsPortInfo) -> str:
        """
        Adds a new sensor to the database.

        Args:
            portInfoBt (XsPortInfo): The Bluetooth port information of the device.

        Returns:
            str: The device ID of the newly connected device.
        """
        manager = XsDotConnectionManager()
        checkDevice = False
        while not checkDevice:
            manager.closePort(portInfoBt)
            if not manager.openPort(portInfoBt):
                logger.error(f"Connection to Device {portInfoBt.bluetoothAddress()} failed")
                checkDevice = False
            else:
                device: XsDotDevice = manager.device(portInfoBt.deviceId())
                if device is None:
                    logger.warning("Bluetooth device not found after opening port.")
                    checkDevice = False
                else:
                    time.sleep(1)
                    checkDevice = (device.deviceTagName() != '') and (device.batteryLevel() != 0)
                    if checkDevice:
                        logger.info(f"Connected to new device: {device.deviceTagName()} with ID: {device.deviceId()}")
                    else:
                        logger.warning("Device initialization incomplete.")

        self.db_manager.save_dot_data(str(device.deviceId()), device.bluetoothAddress(), device.deviceTagName())
        manager.closePort(portInfoBt)
        return str(device.deviceId())

    def stop_recording(self, dot_device: DotDevice) -> bool:
        """
        Stops recording on the specified DotDevice.

        Args:
            dot_device (DotDevice): The DotDevice instance on which to stop recording.

        Returns:
            bool: True if recording was successfully stopped, False otherwise.
        """
        if dot_device in self.devices:
            try:
                success = dot_device.stopRecord()
                if success:
                    logger.info(f"Recording stopped on device {dot_device.deviceId}.")
                else:
                    logger.error(f"Failed to stop recording on device {dot_device.deviceId}.")
                return success
            except AttributeError as e:
                logger.error(f"DotDevice {dot_device.deviceId} does not have a stopRecord method: {e}")
                return False
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred while stopping recording on device {dot_device.deviceId}: {e}")
                return False
        else:
            logger.error(f"Device {dot_device.deviceId} is not managed by DotManager.")
            return False

    def stop_all_recordings(self):
        """
        Stops recording on all managed DotDevice instances.
        """
        for device in self.devices:
            self.stop_recording(device)
        logger.info("All recordings have been stopped.")
