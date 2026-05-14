
#  Copyright (c) 2003-2023 Movella Technologies B.V. or subsidiaries worldwide.
#  All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification,
#  are permitted provided that the following conditions are met:
#  
#  1.	Redistributions of source code must retain the above copyright notice,
#  	this list of conditions and the following disclaimer.
#  
#  2.	Redistributions in binary form must reproduce the above copyright notice,
#  	this list of conditions and the following disclaimer in the documentation
#  	and/or other materials provided with the distribution.
#  
#  3.	Neither the names of the copyright holders nor the names of their contributors
#  	may be used to endorse or promote products derived from this software without
#  	specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
#  THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
#  OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR
#  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  

import logging
from typing import List
from user_settings import *
import time

from core.utils.movella_sdk_loader import sdk_bindings as movelladot_pc_sdk, XsDotConnectionManager, XsDotDevice, XsDotUsbDevice, XsPortInfo
from core.utils.device_support import is_valid_bluetooth_address

_logger = logging.getLogger(__name__)

waitForConnections = True

def on_press(key):
    global waitForConnections
    waitForConnections = False

class XdpcHandler(movelladot_pc_sdk.XsDotCallback):
    def __init__(self, whitelist=None):
        movelladot_pc_sdk.XsDotCallback.__init__(self)

        self.__manager : XsDotConnectionManager = 0
        self.__errorReceived = False
        self.__updateDone = False
        self.__detectedDots = list()
        self.__connectedDots = list()
        self.__connectedUsbDots = list()
        self.__whitelist = whitelist if whitelist is not None else globals().get("whitelist", [])

    def initialize(self):
        """
        Initialize the PC SDK

        - Prints the used PC SDK version to show we connected to XDPC
        - Constructs the connection manager used for discovering and connecting to DOTs
        - Connects this class as callback handler to the XDPC

        Returns:
            False if there was a problem creating a connection manager.
        """
        # Create connection manager
        self.__manager = movelladot_pc_sdk.XsDotConnectionManager()
        if self.__manager is None:
            _logger.error("Manager could not be constructed, exiting.")
            return False

        # Attach callback handler (self) to connection manager
        self.__manager.addXsDotCallbackHandler(self)
        return True

    def cleanup(self):
        """
        Close connections to any Movella DOT devices and destructs the connection manager created in initialize
        """
        self.__manager.close()

    def scanForDots(self, white_list=None):
        """
        Scan if any Movella DOT devices can be detected via Bluetooth

        Enables device detection in the connection manager and uses the
        onAdvertisementFound callback to detect active Movella DOT devices
        Disables device detection when done

        """
        # Start a scan and wait until we have found one or more DOT Devices
        _logger.info("Scanning for devices...")
        self.__manager.enableDeviceDetection()

        allowed_addresses = set(white_list or [])
        connectedDOTCount = 0
        startTime = movelladot_pc_sdk.XsTimeStamp_nowMs()
        while waitForConnections and not self.errorReceived() and movelladot_pc_sdk.XsTimeStamp_nowMs() - startTime <= 10000:
            time.sleep(0.1)

            nextCount = len(self.detectedDots())
            if nextCount != connectedDOTCount:
                _logger.info(f"Number of connected DOTs: {nextCount}.")
                connectedDOTCount = nextCount

            if allowed_addresses:
                found_addresses = {dot.bluetoothAddress() for dot in self.detectedDots() if hasattr(dot, "bluetoothAddress")}
                if allowed_addresses.issubset(found_addresses):
                    time.sleep(0.5)
                    break

        self.__manager.disableDeviceDetection()
        _logger.info("Stopped scanning for devices.")

    def connectDots(self):
        """
        Connects to Movella DOTs found via either USB or Bluetooth connection

        Uses the isBluetooth function of the XsPortInfo to determine if the device was detected
        via Bluetooth or via USB. Then connects to the device accordingly
        When using Bluetooth, a retry has been built in, since wireless connection sometimes just fails the 1st time
        Connected devices can be retrieved using either connectedDots() or connectedUsbDots()

        USB and Bluetooth devices should not be mixed in the same session!
        """
        for portInfo in self.detectedDots():
            if portInfo.isBluetooth():
                address = portInfo.bluetoothAddress()

                checkDevice = False

                retries = 5
                while not checkDevice and retries > 0:
                    if not self.__manager.openPort(portInfo):
                        _logger.warning(f"Connection to Device {address} failed")
                        checkDevice = False
                    else:
                        device : XsDotDevice = self.__manager.device(portInfo.deviceId())
                        if device is None:
                            checkDevice = False

                        devicesId = []
                        for x in self.__connectedDots:
                            devicesId.append(x.deviceId())

                        checkDevice = (device.deviceId() not in devicesId) and (device.deviceTagName() != '')
                    retries -= 1
                    if not checkDevice:
                        time.sleep(0.2)

                if checkDevice:
                    self.__connectedDots.append(device)
                    _logger.info(f"Found a device with Tag: {device.deviceTagName()} @ address: {address}")
                else:
                    _logger.error(f"Unable to connect to bluetooth device {address}")
            else:
                _logger.info(f"Opening DOT with ID: {portInfo.deviceId().toXsString()} @ port: {portInfo.portName()}, baudrate: {portInfo.baudrate()}")
                if not self.__manager.openPort(portInfo):
                    _logger.error(f"Could not open DOT. Reason: {self.__manager.lastResultText()}")
                    continue

                device = self.__manager.usbDevice(portInfo.deviceId())
                if device is None:
                    continue

                self.__connectedUsbDots.append(device)
                _logger.info(f"Device: {device.productCode()}, with ID: {device.deviceId().toXsString()} opened.")

    def detectUsbDevices(self):
        """
        Scans for USB connected Movella DOT devices for data export
        """
        self.__detectedDots = self.__manager.detectUsbDevices()

    def detectedDots(self) -> List[XsPortInfo]:
        """
        Returns:
             An XsPortInfoArray containing information on detected Movella DOT devices
        """
        return self.__detectedDots

    def connectedDots(self)-> List[XsDotDevice]:
        """
        Returns:
            A list containing an XsDotDevice pointer for each Movella DOT device connected via Bluetooth
        """
        return self.__connectedDots

    def connectedUsbDots(self) -> List[XsDotUsbDevice]:
        """
        Returns:
             A list containing an XsDotUsbDevice pointer for each Movella DOT device connected via USB */
        """
        return self.__connectedUsbDots

    def errorReceived(self):
        """
        Returns:
             True if an error was received through the onError callback
        """
        return self.__errorReceived

    def updateDone(self):
        """
        Returns:
             Whether update done was received through the onDeviceUpdateDone callback
        """
        return self.__updateDone

    def resetUpdateDone(self):
        """
        Resets the update done member variable to be ready for a next device update
        """
        self.__updateDone = False

    def onAdvertisementFound(self, port_info):
        """
        Called when an Movella DOT device advertisement was received. Updates m_detectedDots.
        Parameters:
            port_info: The XsPortInfo of the discovered information
        """
        address = port_info.bluetoothAddress()
        if not is_valid_bluetooth_address(address):
            _logger.debug(f"Ignoring invalid bluetooth address: {address}")
            return
        if not self.__whitelist or address in self.__whitelist:
            self.__detectedDots.append(port_info)
        else:
            _logger.debug(f"Ignoring {address}")

    def onError(self, result, errorString):
        """
        Called when an internal error has occurred. Prints to screen.
        Parameters:
            result: The XsResultValue related to this error
            errorString: The error string with information on the problem that occurred
        """
        _logger.error(f"{movelladot_pc_sdk.XsResultValueToString(result)}")
        _logger.error(f"Error received: {errorString}")
        self.__errorReceived = True

    def onDeviceUpdateDone(self, portInfo, result):
        """
        Called when the firmware update process has completed. Prints to screen.
        Parameters:
            portInfo: The XsPortInfo of the updated device
            result: The XsDotFirmwareUpdateResult of the firmware update
        """
        _logger.info(f"{portInfo.bluetoothAddress()} Firmware Update done. Result: {movelladot_pc_sdk.XsDotFirmwareUpdateResultToString(result)}")
        self.__updateDone = True
