import logging
import os
import time
import asyncio
if os.name == 'nt':
    from winrt.windows.devices import radios
from typing import List, Tuple, Optional
import numpy as np
from movelladot_pc_sdk.movelladot_pc_sdk_py39_64 import XsPortInfo
# Import other necessary classes from the SDK as needed
from movelladot_pc_sdk.movelladot_pc_sdk_py39_64 import (
    XsDotDevice,
    XsDotUsbDevice,
    XsDotConnectionManager,
    XsDotCallback,
    XsDataPacket,
)
from core.database.DatabaseManager import DatabaseManager
from core.utils.xdpchandler import XdpcHandler
from core.utils.DotDevice import DotDevice  # Ensure this is the updated DotDevice class


# Configure logging with timestamps and levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("dot_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DotManager:
    """
    Class to manage the initial connection to sensors.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """
        Initialize the DotManager.

        Args:
            db_manager (DatabaseManager): An instance of the DatabaseManager for database operations.
        """
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
        try:
            all_radios = await radios.Radio.get_radios_async()
            for this_radio in all_radios:
                if this_radio.kind == radios.RadioKind.BLUETOOTH:
                    if turn_on:
                        result = await this_radio.set_state_async(radios.RadioState.ON)
                        logger.info("Bluetooth turned ON.")
                    else:
                        result = await this_radio.set_state_async(radios.RadioState.OFF)
                        logger.info("Bluetooth turned OFF.")
        except Exception as e:
            logger.error(f"Exception in bluetooth_power: {e}")

    def first_connection(self) -> Tuple[bool, List[str]]:
        """
        Initial connection to sensors.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean indicating success and
            a list of unconnected device tag names.
        """
        self.devices = []
        self.previousConnected = []
        check = True

        # Disable Bluetooth
        if os.name == 'nt':
            try:
                asyncio.run(self.bluetooth_power(False))
            except Exception as e:
                logger.error(f"Failed to disable Bluetooth: {e}")
                check = False
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
        self.port_info_usb = {}
        while len(xdpcHandler.connectedUsbDots()) < len(xdpcHandler.detectedDots()):
            xdpcHandler.connectDots()
        for device in xdpcHandler.connectedUsbDots():
            self.port_info_usb[str(device.deviceId())] = device.portInfo()
        xdpcHandler.cleanup()
        logger.info(f"Connected USB devices: {list(self.port_info_usb.keys())}")

        # Re-enable Bluetooth
        if os.name == 'nt':
            try:
                asyncio.run(self.bluetooth_power(True))
            except Exception as e:
                logger.error(f"Failed to enable Bluetooth: {e}")
                check = False
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
        self.port_info_bt = xdpcHandler.detectedDots()
        xdpcHandler.cleanup()
        logger.info(f"Detected Bluetooth devices: {[bt.bluetoothAddress() for bt in self.port_info_bt]}")

        unconnected_device = []

        # Link USB and Bluetooth devices
        for port_info_bt in self.port_info_bt:
            device = self.db_manager.get_dot_from_bluetooth(port_info_bt.bluetoothAddress())
            if device is None:
                logger.info("Adding a new device.")
                device_id = self.connect_new_device(port_info_bt)
            else:
                device_id = device.id
                logger.info(f"Found existing device with ID: {device_id}")

            port_info_usb = self.port_info_usb.get(device_id, None)
            if port_info_usb is not None:
                try:
                    dot_device = DotDevice(port_info_usb, port_info_bt, self.db_manager)
                    self.devices.append(dot_device)
                    logger.info(f"DotDevice created for device ID: {device_id}")
                except Exception as e:
                    logger.error(f"Error creating DotDevice for device ID {device_id}: {e}")
                    unconnected_device.append("Initialization error")
                    check = False
            else:
                tag_name = device.get('tag_name') if device else "Unknown"
                logger.warning(f"Please plug sensor {tag_name}.")
                unconnected_device.append(tag_name)
                time.sleep(5)
                check = False

        self.previousConnected = self.devices #.copy()
        logger.info(f"Total connected devices: {len(self.devices)}")
        return (check, unconnected_device)

    def connect_new_device(self, port_info_bt: XsPortInfo) -> Optional[str]:
        """
        Adds a new sensor to the database.

        Args:
            port_info_bt (XsPortInfo): The Bluetooth port information of the device.

        Returns:
            Optional[str]: The device ID of the newly connected device, or None if failed.
        """
        try:
            manager = XsDotConnectionManager()
            checkDevice = False
            device_id = None
            while not checkDevice:
                manager.closePort(port_info_bt)
                if not manager.openPort(port_info_bt):
                    logger.error(f"Connection to Device {port_info_bt.bluetoothAddress()} failed")
                    checkDevice = False
                else:
                    device = manager.device(port_info_bt.deviceId())
                    if device is None:
                        logger.warning("Bluetooth device not found after opening port.")
                        checkDevice = False
                    else:
                        time.sleep(1)
                        checkDevice = (device.deviceTagName() != '') and (device.batteryLevel() != 0)
                        if checkDevice:
                            logger.info(f"Connected to new device: {device.deviceTagName()} with ID: {device.deviceId()}")
                            device_id = str(device.deviceId())
                        else:
                            logger.warning("Device initialization incomplete.")

            if device_id:
                try:
                    self.db_manager.save_dot_data(device_id, device.bluetoothAddress(), device.deviceTagName())
                    logger.info(f"Device {device_id} data saved to database.")
                except Exception as e:
                    logger.error(f"Failed to save device data to database: {e}")
                finally:
                    manager.closePort(port_info_bt)
                return device_id
            else:
                logger.error("Failed to obtain device ID after connection.")
                return None
        except Exception as e:
            logger.error(f"Exception while connecting new device: {e}")
            return None

    def check_devices(self) -> Tuple[List[DotDevice], List[DotDevice]]:
        """
        Detects USB-connected sensors to capture any new connections or disconnections.

        Returns:
            Tuple[List[DotDevice], List[DotDevice]]: Lists of last connected and last disconnected devices.
        """
        connected: List[DotDevice] = []
        for device in self.devices:
            try:
                if device.is_charging():
                    connected.append(device)
            except AttributeError as e:
                logger.error(f"DotDevice {device.deviceId} missing is_charging method: {e}")
            except Exception as e:
                logger.error(f"Error checking device {device.deviceId} charging status: {e}")

        last_connected = []
        last_disconnected = []
        if len(self.previousConnected) > len(connected):
            for device in self.previousConnected:
                if device not in connected:
                    try:
                        device.close_usb()
                        # device.close_bluetooth()
                        last_disconnected.append(device)
                        logger.info(f"Device {device.deviceId} disconnected.")
                    except AttributeError as e:
                        logger.error(f"DotDevice {device.deviceId} does not have expected close methods: {e}")
                    except Exception as e:
                        logger.error(f"Error disconnecting device {device.deviceId}: {e}")
        elif len(self.previousConnected) < len(connected):
            for device in connected:
                if device not in self.previousConnected:
                    try:
                        device.open_usb()
                        # device.open_bluetooth()
                        last_connected.append(device)
                        logger.info(f"Device {device.deviceId} connected.")
                    except AttributeError as e:
                        logger.error(f"DotDevice {device.deviceId} does not have expected open methods: {e}")
                    except Exception as e:
                        logger.error(f"Error connecting device {device.deviceId}: {e}")
        else:
            logger.info("No changes in device connections.")

        self.previousConnected = connected #.copy()
        return (last_connected, last_disconnected)

    def get_export_estimated_time(self) -> float:
        """
        Estimates the extraction time for all sensors simultaneously.

        Returns:
            float: The maximum estimated time among all devices.
        """
        estimated_times = [0]
        for device in self.devices:
            try:
                estimated_times.append(device.get_export_estimated_time())
            except Exception as e:
                logger.error(f"Error estimating export time for device {device.deviceId}: {e}")
                estimated_times.append(0)
        max_time = np.max(estimated_times)
        logger.info(f"Estimated export time: {max_time} seconds.")
        return max_time

    def get_devices(self) -> List[DotDevice]:
        """
        Retrieves the list of managed DotDevice instances.

        Returns:
            List[DotDevice]: The list of devices.
        """
        return self.devices #.copy()

