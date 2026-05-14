from datetime import datetime
import logging
import os
import sys
from threading import Event
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageTk
from constants import *

from core.database.DatabaseManager import DatabaseManager, JumpData
from core.utils.movella_sdk_loader import (
    sdk_bindings as movelladot_pc_sdk,
    sdk_bindings,
    XsDataPacket,
    XsDotCallback,
    XsDotConnectionManager,
    XsDotDevice,
    XsDotUsbDevice,
    XsPortInfo,
)
from core.utils.device_support import normalize_sample_time_fine
from synergie.services.jump_predictions import build_training_jump_payload

_logger = logging.getLogger(__name__)

class DotDevice(XsDotCallback):
    """
    Class permettant de gérer individuellement les capteurs, elle remplace les classes fournis par movella_pc_sdk :
    XsDotDevice : capteur connecté en bluetooth
    XsDotUsbDevice : capteur connecté en USB
    Les deux classes sont fusionnéees dans celle-ci.
    La classe est une extension de XsDotCallback qui permet d'avoir des retours des capteurs
    """
    def __init__(self, portInfoUsb : XsPortInfo, portInfoBt : XsPortInfo, db_manager : DatabaseManager):
        XsDotCallback.__init__(self)
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

        self.usbDevice : XsDotUsbDevice = None
        self.btDevice : XsDotDevice = None
        self.initializeUsb()
        self.initializeBt()
        self.deviceId = str(self.usbDevice.deviceId())
        self.deviceTagName = str(self.btDevice.deviceTagName())
        self.batteryLevel = self.btDevice.batteryLevel()
        self.isRecording = self.usbDevice.recordingCount() == -1
        if self.isRecording:
            self.recordingCount = 0
        else:
            self.recordingCount = self.usbDevice.recordingCount()
        self.isPlugged = True
        self.timingRecord = datetime.now().timestamp()

        self.loadImages()
        self.currentImage = self.imageActive

        self.count = 0
        self.packetsReceived = []
        self.synchroTime = 0
        self.exportDone = False
        self.isBatteryCharging = False

    def _open_bluetooth_with_retries(self, retries: int = 5):
        self.btManager.closePort(self.portInfoBt)
        for _ in range(retries):
            self.btManager.closePort(self.portInfoBt)
            if not self.btManager.openPort(self.portInfoBt):
                _logger.warning(f"Connection to Device {self.portInfoBt.bluetoothAddress()} failed")
                time.sleep(0.2)
                continue
            device : XsDotDevice = self.btManager.device(self.portInfoBt.deviceId())
            if device is None:
                time.sleep(0.2)
                continue
            time.sleep(1)
            if device.deviceTagName() != '' and device.batteryLevel() != 0:
                return device
        return None

    def _open_usb_with_retries(self, retries: int = 5):
        self.usbManager.closePort(self.portInfoUsb)
        for _ in range(retries):
            self.usbManager.closePort(self.portInfoUsb)
            if not self.usbManager.openPort(self.portInfoUsb):
                _logger.warning(f"USB connection failed for {self.portInfoUsb.deviceId()}")
                time.sleep(0.2)
                continue
            device = self.usbManager.usbDevice(self.portInfoUsb.deviceId())
            if device is not None:
                return device
            time.sleep(0.2)
        return None

    def initializeBt(self):
        """
        Initialise la connexion bluetooth
        """
        device = self._open_bluetooth_with_retries()
        if device is None:
            raise ConnectionError(f"Unable to connect to bluetooth device {self.portInfoBt.bluetoothAddress()}")
        self.btDevice = device

    def initializeUsb(self):
        """
        Initialise la connexion USB
        """
        device = self._open_usb_with_retries()
        if device is None:
            raise ConnectionError(f"Unable to connect to USB device {self.portInfoUsb.deviceId()}")
        self.usbDevice = device
        self.isPlugged = True
    
    def loadImages(self):
        """
        Charge les images des capteurs pour l'interface graphique
        """
        try:
            fontTag = ImageFont.truetype(font="arialbd.ttf",size=60)
        except OSError:
            fontTag = ImageFont.load_default()
        try:
            imgActive = Image.open(f"{sys._MEIPASS}/img/Dot_active.png")
        except (AttributeError, FileNotFoundError, OSError):
            imgActive = Image.open(f"img/Dot_active.png")
        d = ImageDraw.Draw(imgActive)
        text = self.deviceTagName
        x = 0
        if len(text) == 1:
            x = 93
        else:
            x = 75
        d.text( (x,65), text,font=fontTag, fill="black")
        imgActive = imgActive.resize((116, 139))
        self.imageActive = ImageTk.PhotoImage(imgActive)

        try:
            imgInactive = Image.open(f"{sys._MEIPASS}/img/Dot_inactive.png")
        except (AttributeError, FileNotFoundError, OSError):
            imgInactive = Image.open(f"img/Dot_inactive.png")
        d = ImageDraw.Draw(imgInactive)
        d.text( (x,65), text,font=fontTag, fill="black")
        imgInactive = imgInactive.resize((116, 139))
        self.imageInactive = ImageTk.PhotoImage(imgInactive)
    
    def startRecord(self):
        """
        Commence un enregistrement sur le capteur
        """
        self.isRecording = True
        if not self.btDevice.startRecording():
            _logger.warning(f"Bluetooth startRecording failed for {self.deviceTagName}, retrying after reconnect")
            try:
                self.initializeBt()
                self.isRecording = self.btDevice.startRecording()
            except Exception as exc:
                _logger.error(f"Unable to start recording on {self.deviceTagName}: {exc}")
                self.isRecording = False
        if self.isRecording:
            self.timingRecord = datetime.now().timestamp()
            self.currentImage = self.imageActive
        return self.isRecording

    def stopRecord(self):
        """
        Arrête un enregistrement sur le capteur
        """
        self.isRecording = False
        if not self.btDevice.stopRecording():
            _logger.warning(f"Bluetooth stopRecording failed for {self.deviceTagName}, retrying after reconnect")
            try:
                self.initializeBt()
                self.isRecording = not self.btDevice.stopRecording()
            except Exception as exc:
                _logger.error(f"Unable to stop recording on {self.deviceTagName}: {exc}")
                self.isRecording = True
        if self.usbDevice is not None:
            self.recordingCount = self.usbDevice.recordingCount()
        self.currentImage = self.imageInactive
        return not self.isRecording

    def exportData(self, saveFile : bool, extractEvent : Event):
        """
        Export les données du capteurs
        saveFile : définis si on veut extraire toute les infos disponibles (et non seulement celle nécessaire aux modèles)
        extractEvent : event pour informer le thread principale que l'extraction est finie
        """
        self.saveFile = saveFile
        _logger.info(f"Exporting data for {self.deviceTagName}...")
        self.exportDone = False
        self.packetsReceived = []
        self.count = 0
        try:
            exportData = sdk_bindings.XsIntArray()
            exportData.push_back(movelladot_pc_sdk.RecordingData_Timestamp)
            exportData.push_back(movelladot_pc_sdk.RecordingData_Euler)
            exportData.push_back(movelladot_pc_sdk.RecordingData_Acceleration)
            exportData.push_back(movelladot_pc_sdk.RecordingData_AngularVelocity)
            if self.saveFile:
                exportData.push_back(movelladot_pc_sdk.RecordingData_MagneticField)
                exportData.push_back(movelladot_pc_sdk.RecordingData_Quaternion)
                exportData.push_back(movelladot_pc_sdk.RecordingData_Status)

            if not self.usbDevice.selectExportData(exportData):
                _logger.error(f"Could not select export data. Reason: {self.usbDevice.lastResultText()}")
                return

            for recordingIndex in range(1, self.usbDevice.recordingCount()+1):
                recInfo = self.usbDevice.getRecordingInfo(recordingIndex)
                if recInfo.empty():
                    _logger.warning(f"Could not get recording info. Reason: {self.usbDevice.lastResultText()}")
                    continue

                dateRecord = recInfo.startUTC()
                trainingId = self.db_manager.get_current_record(self.deviceId)
                if trainingId == "":
                    _logger.warning(f"No current training linked to sensor {self.deviceTagName}; skipping export {recordingIndex}")
                    continue

                self.db_manager.set_training_date(trainingId, dateRecord)
                if not self.usbDevice.startExportRecording(recordingIndex):
                    _logger.error(f"Could not export recording. Reason: {self.usbDevice.lastResultText()}")
                    continue

                self.exportDone = False
                deadline = time.time() + 180
                while not self.exportDone and time.time() < deadline:
                    time.sleep(0.1)
                if not self.exportDone:
                    _logger.error(f"Timed out while exporting recording {recordingIndex} for {self.deviceTagName}")
                    continue

                if self.saveFile:
                    columnSelected = ["PacketCounter","SampleTimeFine","Euler_X","Euler_Y","Euler_Z","Quat_W","Quat_X","Quat_Y","Quat_Z","Acc_X","Acc_Y","Acc_Z","Gyr_X","Gyr_Y","Gyr_Z","Mag_X","Mag_Y","Mag_Z"]
                else:
                    columnSelected = ["PacketCounter","SampleTimeFine","Euler_X","Euler_Y","Euler_Z","Acc_X","Acc_Y","Acc_Z","Gyr_X","Gyr_Y","Gyr_Z"]
                df = pd.DataFrame.from_records(self.packetsReceived, columns=columnSelected)
                if df.empty:
                    _logger.warning(f"No packet exported for recording {recordingIndex} on {self.deviceTagName}")
                    self.packetsReceived = []
                    continue
                date = datetime.fromtimestamp(dateRecord).strftime("%Y_%m_%d")
                sample_times = df["SampleTimeFine"].tolist()
                startSampleTime = int(sample_times[0]) if sample_times else 0
                df["SampleTimeFine"] = normalize_sample_time_fine(sample_times)
                self.synchroTime = max(0, self.synchroTime - startSampleTime)
                os.makedirs(f"data/raw/{date}", exist_ok = True)
                df.to_csv(f"data/raw/{date}/{self.synchroTime}_{trainingId}.csv", index=False)

                self.predict_training(trainingId, df)
                self.db_manager.remove_current_record(self.deviceId, trainingId)
                self.recordingCount = max(0, self.recordingCount - 1)
                self.packetsReceived = []

            for _ in range(5):
                if self.usbDevice.eraseFlash():
                    break
                time.sleep(1)
            else:
                _logger.error(f"Failed to erase flash for {self.deviceTagName}")
            self.recordingCount = 0
            self.currentImage = self.imageActive
            _logger.info(f"Export finished for {self.deviceTagName}. You can disconnect the dot.")
        finally:
            extractEvent.set()

    def predict_training(self, training_id : str, df : pd.DataFrame):
        from core.data_treatment.data_generation.exporter import export
        """
        Utilisation des modèles de prédiction pour avoir les infos de l'enregistrement
        """
        try:
            predictions = export(df)
            _logger.info(f"Prediction pipeline completed for training {training_id}")
            training_jumps = build_training_jump_payload(predictions, training_id)
            self.db_manager.add_jumps_to_training(training_id, training_jumps)
        except Exception as exc:
            _logger.error(f"Prediction failed for training {training_id}: {exc}")
        
    def onRecordedDataAvailable(self, device, packet : XsDataPacket):
        """
        Lorsque le capteur est en train d'exporter les données, cette fonction permet de capter les informations renvoyées
        """
        self.count += 1
        euler = packet.orientationEuler()
        captor = packet.calibratedData()
        if self.saveFile:
            quaternion = packet.orientationQuaternion()
            data = np.concatenate([[int(self.count), packet.sampleTimeFine(), euler.x(), euler.y(), euler.z()], quaternion, captor.m_acc, captor.m_gyr, captor.m_mag])
        else:
            data = np.concatenate([[int(self.count), packet.sampleTimeFine(), euler.x(), euler.y(), euler.z()], captor.m_acc, captor.m_gyr])
        self.packetsReceived.append(data)
    
    def onRecordedDataDone(self, device):
        """
        Fonction qui s'active lorsque le capteur a fini d'extraire les données
        """
        self.exportDone = True
    
    def __eq__(self, device) -> bool:
        return (self.usbDevice == device.usbDevice) and (self.btDevice == device.btDevice)
    
    def getExportEstimatedTime(self) -> int:
        """
        Calcul une estimation du temps d'extraction
        """
        estimatedTime = 0
        for index in range(1,self.usbDevice.recordingCount()+1):
            estimatedTime = estimatedTime + round(self.usbDevice.getRecordingInfo(index).storageSize()/(237568*8),1)
        return estimatedTime + 1

    def onBatteryUpdated(self, device: XsDotDevice, batteryLevel: int, chargingStatus: int):
        self.batteryLevel = batteryLevel
        self.isBatteryCharging = chargingStatus == 1
    
    def onButtonClicked(self, device: XsDotDevice, timestamp : int):
        """
        Appuyer sur le bouton pendant l'enregistrement stocke l'instant pour pouvoir synchroniser avec une vidéo lors des collectes de données
        """
        self.synchroTime = timestamp

    def closeUsb(self):
        """
        Ferme la connexion USB
        """
        self.usbManager.closePort(self.portInfoUsb)
        self.isPlugged = False
        self.isBatteryCharging = False
    
    def openUsb(self):
        """
        Ouvre la connexion USB
        """
        device = self._open_usb_with_retries()
        if device is None:
            _logger.error(f"Unable to reopen USB device {self.portInfoUsb.deviceId()}")
            return False
        self.usbDevice = device
        self.isPlugged = True
        self.isBatteryCharging = True
        return True
