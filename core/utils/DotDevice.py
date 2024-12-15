import logging
from datetime import datetime
import os
import sys
import time
from threading import Event
from PIL import Image, ImageDraw, ImageFont, ImageTk
import pandas as pd
import numpy as np
from movelladot_pc_sdk.movelladot_pc_sdk_py39_64 import (
    XsDotDevice,
    XsDotUsbDevice,
    XsDotConnectionManager,
    XsDotCallback,
    XsPortInfo,
    XsDataPacket,
)
from core.database.DatabaseManager import DatabaseManager, JumpData


class DotDevice(XsDotCallback):
    """
    Manages individual sensors (dots) connected via Bluetooth and USB.
    """

    def __init__(
        self,
        portInfoUsb: XsPortInfo,
        portInfoBt: XsPortInfo,
        db_manager: DatabaseManager,
    ):
        super().__init__()
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.is_recording = False  # Initialize early to prevent AttributeError
        self.portInfoUsb = portInfoUsb
        self.portInfoBt = portInfoBt
        self.db_manager = db_manager

        self.usbManager = XsDotConnectionManager()
        while self.usbManager is None:
            self.usbManager = XsDotConnectionManager()
        self.usbManager.addXsDotCallbackHandler(self)

        self.btManager = XsDotConnectionManager()
        while self.btManager is None:
            self.btManager = XsDotConnectionManager()
        self.btManager.addXsDotCallbackHandler(self)

        self.usbDevice: XsDotUsbDevice = None
        self.btDevice: XsDotDevice = None
        self.initialize_usb()
        self.initialize_bt()

        self.deviceId = str(self.usbDevice.deviceId()) if self.usbDevice else ""
        self.deviceTagName = str(self.btDevice.deviceTagName()) if self.btDevice else ""
        self.batteryLevel = self.btDevice.batteryLevel() if self.btDevice else 0
        self.recordingCount = (
            0 if self.is_recording else (self.usbDevice.recordingCount() if self.usbDevice else 0)
        )
        self.is_plugged = True
        self.timingRecord = datetime.now().timestamp()

        self.load_images()
        self.current_image = self.imageActive

        self.count = 0
        self.packetsReceived = []
        self.synchroTime = 0
        self.exportDone = False

        # Initialize charging status
        self.chargingStatus = False  # Default to not charging

        self.logger.info(f"DotDevice initialized: {self.deviceTagName} (ID: {self.deviceId})")

    def initialize_bt(self):
        self.btManager.closePort(self.portInfoBt)
        checkDevice = False
        device = None

        while not checkDevice:
            self.btManager.closePort(self.portInfoBt)
            if not self.btManager.openPort(self.portInfoBt):
                self.logger.warning(f"Connection to Bluetooth Device {self.portInfoBt.bluetoothAddress()} failed")
                checkDevice = False
            else:
                device: XsDotDevice = self.btManager.device(self.portInfoBt.deviceId())
                if device is None:
                    self.logger.warning("Bluetooth device not found after opening port.")
                    checkDevice = False
                else:
                    time.sleep(1)  # Wait for the device to initialize
                    checkDevice = (device.deviceTagName() != "") and (device.batteryLevel() != 0)
                    if checkDevice:
                        self.btDevice = device
                        self.deviceTagName = str(device.deviceTagName())
                        self.batteryLevel = device.batteryLevel()
                        self.logger.info(f"Bluetooth connection established with device: {self.deviceTagName}")

                        if self.is_recording:
                            self.logger.info("Bluetooth device is currently recording. Stopping recording...")
                            self.stop_record()
                    else:
                        self.logger.warning("Bluetooth device initialization incomplete.")

    def initialize_usb(self):
        self.usbManager.closePort(self.portInfoUsb)
        device = None
        while device is None:
            self.usbManager.openPort(self.portInfoUsb)
            device = self.usbManager.usbDevice(self.portInfoUsb.deviceId())
            if device is None:
                if hasattr(self.portInfoUsb, 'serial'):
                    serial_info = self.portInfoUsb.serial
                elif hasattr(self.portInfoUsb, 'serialNumber'):
                    serial_info = self.portInfoUsb.serialNumber
                else:
                    serial_info = "Unknown Serial"
                self.logger.warning(f"Connection to USB Device {serial_info} failed")
            else:
                self.usbDevice = device
                self.deviceId = str(device.deviceId())  # Assign deviceId before logging
                self.logger.info(f"USB connection established with device ID: {self.deviceId}")

                if self.is_recording:
                    self.logger.info("USB device is currently recording. Stopping recording...")
                    self.stop_record()
        self.is_plugged = True

    def load_images(self):
        try:
            fontTag = ImageFont.truetype(font="arialbd.ttf", size=60)
        except IOError:
            fontTag = ImageFont.load_default()
            self.logger.warning("Custom font 'arialbd.ttf' not found. Using default font.")

        # Load active image
        try:
            imgActive = Image.open(f"{sys._MEIPASS}/img/Dot_active.png")
        except Exception:
            imgActive = Image.open("img/Dot_active.png")
        d = ImageDraw.Draw(imgActive)
        text = self.deviceTagName
        x = 93 if len(text) == 1 else 75
        d.text((x, 65), text, font=fontTag, fill="black")
        imgActive = imgActive.resize((116, 139))
        self.imageActive = ImageTk.PhotoImage(imgActive)

        # Load inactive image
        try:
            imgInactive = Image.open(f"{sys._MEIPASS}/img/Dot_inactive.png")
        except Exception:
            imgInactive = Image.open("img/Dot_inactive.png")
        d = ImageDraw.Draw(imgInactive)
        d.text((x, 65), text, font=fontTag, fill="black")
        imgInactive = imgInactive.resize((116, 139))
        self.imageInactive = ImageTk.PhotoImage(imgInactive)

    def start_record(self) -> bool:
        self.is_recording = True
        if not self.btDevice.startRecording():
            self.logger.warning("Failed to start recording on Bluetooth device. Reinitializing connection.")
            self.initialize_bt()
            self.is_recording = self.btDevice.startRecording()
        if self.is_recording:
            self.timingRecord = datetime.now().timestamp()
            self.logger.info(f"Recording started at {self.timingRecord} seconds.")
        else:
            self.logger.error("Recording could not be started.")
        return self.is_recording

    def stop_record(self) -> bool:
        self.is_recording = False
        if not self.btDevice.stopRecording():
            self.logger.warning("Failed to stop recording on Bluetooth device. Reinitializing connection.")
            self.initialize_bt()
            self.is_recording = not self.btDevice.stopRecording()
        self.recordingCount = self.usbDevice.recordingCount() if self.usbDevice else 0
        if not self.is_recording:
            self.logger.info("Recording stopped successfully.")
        else:
            self.logger.error("Recording could not be stopped.")
        return not self.is_recording

    def export_data(self, save_file: bool, extract_event: Event):
        self.logger.info("Exporting data from sensor...")
        self.saveFile = save_file
        self.exportDone = False
        self.packetsReceived = []
        self.count = 0

        # Define the types of data to export
        exportData = movelladot_pc_sdk.XsIntArray()
        exportData.push_back(movelladot_pc_sdk.RecordingData_Timestamp)
        exportData.push_back(movelladot_pc_sdk.RecordingData_Euler)
        exportData.push_back(movelladot_pc_sdk.RecordingData_Acceleration)
        exportData.push_back(movelladot_pc_sdk.RecordingData_AngularVelocity)
        if self.saveFile:
            exportData.push_back(movelladot_pc_sdk.RecordingData_MagneticField)
            exportData.push_back(movelladot_pc_sdk.RecordingData_Quaternion)
            exportData.push_back(movelladot_pc_sdk.RecordingData_Status)

        # Select the data types for export
        if not self.usbDevice.selectExportData(exportData):
            self.logger.error(f"Could not select export data. Reason: {self.usbDevice.lastResultText()}")

        # Iterate through each recording and export data
        for recordingIndex in range(1, self.usbDevice.recordingCount() + 1):
            recInfo = self.usbDevice.getRecordingInfo(recordingIndex)
            if recInfo.empty():
                self.logger.error(f"Could not get recording info. Reason: {self.usbDevice.lastResultText()}")
                continue  # Skip to the next recording

            dateRecord = recInfo.startUTC()
            trainingId = self.db_manager.get_current_record(self.deviceId)
            if trainingId != "":
                self.db_manager.set_training_date(trainingId, dateRecord)
                if not self.usbDevice.startExportRecording(recordingIndex):
                    self.logger.error(f"Could not export recording. Reason: {self.usbDevice.lastResultText()}")
                else:
                    # Wait until export is done
                    while not self.exportDone:
                        time.sleep(0.1)
                    self.logger.info("File export finished!")

                    # Define columns based on whether all data is saved or not
                    if self.saveFile:
                        columnSelected = [
                            "PacketCounter",
                            "SampleTimeFine",
                            "Euler_X",
                            "Euler_Y",
                            "Euler_Z",
                            "Quat_W",
                            "Quat_X",
                            "Quat_Y",
                            "Quat_Z",
                            "Acc_X",
                            "Acc_Y",
                            "Acc_Z",
                            "Gyr_X",
                            "Gyr_Y",
                            "Gyr_Z",
                            "Mag_X",
                            "Mag_Y",
                            "Mag_Z",
                        ]
                    else:
                        columnSelected = [
                            "PacketCounter",
                            "SampleTimeFine",
                            "Euler_X",
                            "Euler_Y",
                            "Euler_Z",
                            "Acc_X",
                            "Acc_Y",
                            "Acc_Z",
                            "Gyr_X",
                            "Gyr_Y",
                            "Gyr_Z",
                        ]

                    # Create DataFrame from received packets
                    df = pd.DataFrame.from_records(
                        self.packetsReceived, columns=columnSelected
                    )
                    date = datetime.fromtimestamp(dateRecord).strftime("%Y_%m_%d")
                    startSampleTime = df["SampleTimeFine"].iloc[0]
                    newSampleTimeFine = []
                    for timeFine in df["SampleTimeFine"]:
                        newTime = timeFine - startSampleTime
                        if newTime < 0:
                            newSampleTimeFine.append(newTime + 2**32)
                        else:
                            newSampleTimeFine.append(newTime)
                    df["SampleTimeFine"] = newSampleTimeFine
                    self.synchroTime = max(0, self.synchroTime - startSampleTime)
                    os.makedirs(f"data/raw/{date}", exist_ok=True)
                    csv_path = f"data/raw/{date}/{self.synchroTime}_{trainingId}.csv"
                    df.to_csv(csv_path, index=False)
                    self.logger.info(f"Data exported to {csv_path}")

                    # Predict training data and update the database
                    self.predict_training(trainingId, df)
                    self.db_manager.remove_current_record(self.deviceId, trainingId)
                    self.recordingCount -= 1

        # Erase sensor's flash memory after exporting
        self.usbDevice.eraseFlash()
        self.logger.info("You can disconnect the dot.")
        self.recordingCount = 0
        extract_event.set()
        self.current_image = self.imageActive

    def predict_training(self, training_id: str, df: pd.DataFrame):
        from core.data_treatment.data_generation.exporter import export

        try:
            df = export(df)
            self.logger.info("End of data processing.")
            trainingJumps = []
            unknow_rotation = []

            for _, row in df.iterrows():
                jump_time_min, jump_time_sec = row["videoTimeStamp"].split(":")
                jump_time = "{:02d}:{:02d}".format(int(jump_time_min), int(jump_time_sec))
                val_rot = float(row["rotations"])

                if row["type"] < 5 and val_rot > 0.5:
                    if val_rot < 2:
                        val_rot = np.ceil(val_rot - 0.3)
                    else:
                        val_rot = np.ceil(val_rot - 0.15)
                    jump_data = JumpData(
                        0,
                        training_id,
                        jumpType(int(row["type"])).name,
                        val_rot,
                        bool(row["success"]),
                        jump_time,
                        float(row["rotation_speed"]),
                        float(row["length"]),
                    )
                    trainingJumps.append(jump_data.to_dict())
                elif row["type"] == 5 and val_rot > 0.8:
                    val_rot = np.ceil(val_rot - 0.7) + 0.5
                    jump_data = JumpData(
                        0,
                        training_id,
                        jumpType(int(row["type"])).name,
                        val_rot,
                        bool(row["success"]),
                        jump_time,
                        float(row["rotation_speed"]),
                        float(row["length"]),
                    )
                    trainingJumps.append(jump_data.to_dict())
                else:
                    jump_data = JumpData(
                        0,
                        training_id,
                        jumpType(int(row["type"])).name,
                        0,
                        bool(row["success"]),
                        jump_time,
                        float(row["rotation_speed"]),
                        float(row["length"]),
                    )
                    unknow_rotation.append(jump_data)

            if trainingJumps:
                self.db_manager.add_jumps_to_training(training_id, trainingJumps)
            else:
                for jump in unknow_rotation:
                    trainingJumps.append(jump.to_dict())
                self.db_manager.add_jumps_to_training(training_id, trainingJumps)

            self.logger.info(f"Training {training_id} updated with jump data.")

        except Exception as e:
            self.logger.error(f"Error during prediction training: {e}")

    def onRecordedDataAvailable(self, device, packet: XsDataPacket):
        self.count += 1
        euler = packet.orientationEuler()
        captor = packet.calibratedData()

        if self.saveFile:
            quaternion = packet.orientationQuaternion()
            data = np.concatenate(
                [
                    [int(self.count), packet.sampleTimeFine(), euler.x(), euler.y(), euler.z()],
                    quaternion,
                    captor.m_acc,
                    captor.m_gyr,
                    captor.m_mag,
                ]
            )
        else:
            data = np.concatenate(
                [
                    [int(self.count), packet.sampleTimeFine(), euler.x(), euler.y(), euler.z()],
                    captor.m_acc,
                    captor.m_gyr,
                ]
            )
        self.packetsReceived.append(data)

    def onRecordedDataDone(self, device):
        self.exportDone = True
        self.logger.info("Data export completed.")

    def __eq__(self, device) -> bool:
        return (self.usbDevice == device.usbDevice) and (self.btDevice == device.btDevice)

    def get_export_estimated_time(self) -> int:
        estimatedTime = 0
        for index in range(1, self.usbDevice.recordingCount() + 1):
            recInfo = self.usbDevice.getRecordingInfo(index)
            storage_size = recInfo.storageSize()
            estimatedTime += round(storage_size / (237568 * 8), 1)
        return estimatedTime + 1

    def onBatteryUpdated(self, device: XsDotDevice, batteryLevel: int, chargingStatus: int):
        self.batteryLevel = batteryLevel
        # Assuming chargingStatus: 1 = Charging, 0 = Not Charging
        self.chargingStatus = chargingStatus == 1
        self.logger.info(f"Battery level updated: {self.batteryLevel}%, Charging: {self.chargingStatus}")

    def onButtonClicked(self, device: XsDotDevice, timestamp: int):
        self.synchroTime = timestamp
        self.logger.info(f"Button clicked at timestamp: {self.synchroTime}")

    def is_charging(self) -> bool:
        """
        Check if the device is currently charging.

        Returns:
            bool: True if the device is charging, False otherwise.
        """
        self.logger.debug(f"Checking if device {self.deviceId} is charging: {self.chargingStatus}")
        return self.chargingStatus

    def close_usb(self):
        self.usbManager.closePort(self.portInfoUsb)
        self.is_plugged = False
        self.logger.info("USB connection closed.")

    def open_usb(self):
        device = None
        while device is None:
            self.usbManager.openPort(self.portInfoUsb)
            device = self.usbManager.usbDevice(self.portInfoUsb.deviceId())
            if device is None:
                serial_info = getattr(self.portInfoUsb, 'serial', 'Unknown Serial')
                self.logger.warning(f"Connection to USB Device {serial_info} failed")
            else:
                self.usbDevice = device
                self.deviceId = str(device.deviceId())
                self.logger.info(f"USB connection re-established with device ID: {self.deviceId}")

                if self.is_recording:
                    self.logger.info("USB device is currently recording. Stopping recording...")
                    self.stop_record()
        self.is_plugged = True
