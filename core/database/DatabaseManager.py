import os
import sys
import socket
import requests
from typing import List
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import firebase_admin.firestore

from dataclasses import dataclass
from datetime import datetime
from tkinter import messagebox

from core.utils.internet_check import check_internet_connection, has_internet_connection

@dataclass
class JumpData:
    """
    Represents a single jump's data.

    Attributes:
        jump_id (int): The unique identifier of the jump.
        jump_type (str): The type or category of the jump.
        jump_rotations (float): The number of rotations performed during the jump.
        jump_success (bool): Whether the jump was successfully landed or not.
        jump_time (int): The duration of the jump in some time unit (e.g., milliseconds).
        jump_max_speed (float): The maximum speed reached during the jump.
        jump_length (float): The horizontal distance covered during the jump.
    """
    jump_id: int
    jump_type: str
    jump_rotations: float
    jump_success: bool
    jump_time: int
    jump_max_speed: float
    jump_length: float

    def to_dict(self) -> dict:
        """
        Convert the jump data into a dictionary suitable for Firestore storage.

        Returns:
            dict: Dictionary representation of jump data.
        """
        return {
            "jump_type": self.jump_type,
            "jump_rotations": self.jump_rotations,
            "jump_success": self.jump_success,
            "jump_time": self.jump_time,
            "jump_max_speed": self.jump_max_speed,
            "jump_length": self.jump_length
        }


@dataclass
class TrainingData:
    """
    Represents a training session's data.

    Attributes:
        training_id (int): Unique identifier for the training session.
        skater_id (int): Unique identifier for the skater.
        training_date (datetime): The date and time of the training session.
        dot_id (str): The identifier of the dot/device used during training.
        training_jumps (List[str]): A list of jump document references or jump IDs associated with this training.
    """
    training_id: int
    skater_id: int
    training_date: datetime
    dot_id: str
    training_jumps: List[str]

    def to_dict(self) -> dict:
        """
        Convert the training data into a dictionary suitable for Firestore storage.

        Returns:
            dict: Dictionary representation of the training data.
        """
        return {
            "skater_id": self.skater_id,
            "training_date": self.training_date,
            "dot_id": self.dot_id,
            "training_jumps": self.training_jumps
        }


@dataclass
class SkaterData:
    """
    Represents a skater's basic information.

    Attributes:
        skater_id (int): Unique identifier for the skater.
        skater_name (str): The name of the skater.
    """
    skater_id: int
    skater_name: str

    def to_dict(self) -> dict:
        """
        Convert the skater data into a dictionary suitable for Firestore storage.

        Returns:
            dict: Dictionary representation of the skater data.
        """
        return {"skater_name": self.skater_name}


class DatabaseManager:
    """
    A manager class responsible for all Firestore database operations related to
    skaters, training sessions, jumps, and connected devices (dots).

    This class encapsulates CRUD (Create, Read, Update, Delete) operations
    interacting with various Firestore collections:
    - 'trainings'
    - 'jumps'
    - 'users' (for skaters and coaches)
    - 'dots'
    """

    def __init__(self):
        """
        Initialize the DatabaseManager by checking for internet connectivity,
        setting up the Firebase app, and creating a Firestore client if connected.

        If there is no internet connection, display an error message and do not
        initialize the database.
        """
        if not has_internet_connection():
            messagebox.showerror("Connexion Internet", "Aucune connexion Internet détectée. Veuillez vérifier votre réseau.")
            self.db = None
            return

        # Attempt to load the Firebase credentials from the PyInstaller bundle; if not present, load locally.
        try:
            json_path = os.path.join(sys._MEIPASS, 's2m-skating-firebase-adminsdk-3ofmb-cc861a6ad1.json')
        except:
            json_path = 's2m-skating-firebase-adminsdk-3ofmb-cc861a6ad1.json'

        cred = credentials.Certificate(json_path)

        # Initialize the Firebase app if it hasn't been initialized yet.
        try:
            firebase_admin.initialize_app(cred)
        except:
            pass

        # Create a Firestore client instance.
        self.db = firestore.client()


    def save_training_data(self, data: TrainingData) -> int:
        """
        Save a new training session document into the 'trainings' collection.

        Args:
            data (TrainingData): The training data to save.

        Returns:
            int: The ID of the newly created training document.
        """
        if self.db is None:
            return None
        add_time, new_ref = self.db.collection("trainings").add(data.to_dict())
        return new_ref.id

    def save_jump_data(self, data: JumpData) -> int:
        """
        Save a new jump document into the 'jumps' collection.

        Args:
            data (JumpData): The jump data to save.

        Returns:
            int: The ID of the newly created jump document.
        """
        if self.db is None:
            return None
        add_time, new_ref = self.db.collection("jumps").add(data.to_dict())
        return new_ref.id

    def get_skater_from_training(self, training_id: int) -> str:
        """
        Retrieve the skater_id associated with a given training session.

        Args:
            training_id (int): The ID of the training session document.

        Returns:
            str: The skater_id for that training session.
        """
        if self.db is None:
            return ""
        skater_id = self.db.collection("trainings").document(training_id).get().get("skater_id")
        return skater_id

    def get_all_skaters(self) -> List[SkaterData]:
        """
        Retrieve all skaters from the 'skaters' collection.

        Returns:
            List[SkaterData]: A list of SkaterData objects representing all skaters found.
        """
        if self.db is None:
            return []
        data_skaters = []
        for skater in self.db.collection("skaters").stream():
            data_skaters.append(SkaterData(skater.id, skater.get("skater_name")))
        return data_skaters

    def set_training_date(self, training_id: str, date: datetime) -> None:
        """
        Update the training_date field for a given training document.

        Args:
            training_id (str): The ID of the training document to update.
            date (datetime): The new training date to set.
        """
        if self.db is None:
            return
        self.db.collection("trainings").document(training_id).update({"training_date": date})

    def set_current_record(self, device_id: str, current_record: str) -> None:
        """
        Add a record ID to the 'current_record' array field of a device document.

        Args:
            device_id (str): The device (dot) ID.
            current_record (str): The record ID to add.
        """
        if self.db is None:
            return
        self.db.collection("dots").document(device_id).update({"current_record": firestore.ArrayUnion([current_record])})

    def get_current_record(self, deviceId: str) -> str:
        """
        Get the latest 'current_record' from a device's document.

        Args:
            deviceId (str): The ID of the device.

        Returns:
            str: The most recently added record ID, or an empty string if none found.
        """
        if self.db is None:
            return ""
        try:
            trainingId = self.db.collection("dots").document(deviceId).get().get("current_record")[-1]
            return trainingId
        except:
            return ""

    def remove_current_record(self, deviceId: str, trainingId: str) -> None:
        """
        Remove a specific training ID from the 'current_record' array of a given device.

        Args:
            deviceId (str): The ID of the device.
            trainingId (str): The training ID to remove.
        """
        if self.db is None:
            return
        self.db.collection("dots").document(deviceId).update({"current_record": firestore.ArrayRemove([trainingId])})

    def get_dot_from_bluetooth(self, bluetoothAddress: str):
        """
        Retrieve a dot document based on its associated Bluetooth address.

        Args:
            bluetoothAddress (str): The Bluetooth address to search for.

        Returns:
            DocumentSnapshot or None: The first matched dot document or None if not found.
        """
        if self.db is None:
            return None
        dots = self.db.collection("dots").where(
            filter=firestore.firestore.FieldFilter("bluetooth_address", "==", bluetoothAddress)
        ).get()
        if len(dots) > 0:
            return dots[0]
        else:
            return None

    def save_dot_data(self, deviceId: str, bluetoothAddress: str, tagName: str) -> None:
        """
        Save a new dot document with the given details.

        Args:
            deviceId (str): The device's unique ID.
            bluetoothAddress (str): The Bluetooth address of the device.
            tagName (str): A tag name (e.g., a human-readable identifier) for the device.
        """
        if self.db is None:
            return
        newDot = {
            "bluetooth_address": bluetoothAddress,
            "current_record": [],
            "tag_name": tagName
        }
        self.db.collection("dots").add(document_data=newDot, document_id=deviceId)

    def add_jumps_to_training(self, trainingId: str, trainingJumps: List[str]) -> None:
        """
        Update a training's jump list with new jumps.

        Args:
            trainingId (str): The ID of the training document to update.
            trainingJumps (List[str]): A list of jump IDs to associate with the training.
        """
        if self.db is None:
            return
        self.db.collection("trainings").document(trainingId).update({"training_jumps": trainingJumps})

    def findUserByEmail(self, email: str):
        """
        Find a user document based on their email.

        Args:
            email (str): The email address to search for.

        Returns:
            List[DocumentSnapshot]: A list of user documents matching the email.
        """
        if self.db is None:
            return []
        return self.db.collection("users").where(
            filter=firestore.firestore.FieldFilter("email", "==", email)
        ).get()

    def getAllSkaterFromCoach(self, coachId: str) -> List[SkaterData]:
        """
        Retrieve all skaters associated with a particular coach user.

        Assumes that the coach user's document contains an 'access' field listing skater IDs.

        Args:
            coachId (str): The user ID of the coach.

        Returns:
            List[SkaterData]: A list of SkaterData objects for each skater accessible by the coach.
        """
        if self.db is None:
            return []
        skatersData = []
        for skater in self.db.collection("users").document(coachId).get().get("access"):
            skatersData.append(
                SkaterData(
                    skater,
                    self.db.collection("users").document(skater).get().get("name")
                )
            )
        return skatersData

    def get_all_trainings_for_skater(self, skater_id: int) -> List[TrainingData]:
        """
        Retrieve all training documents associated with a given skater.

        Args:
            skater_id (int): The skater's unique ID.

        Returns:
            List[TrainingData]: A list of TrainingData objects representing each training found.
        """
        if self.db is None:
            return []
        trainings = self.db.collection("trainings").where("skater_id", "==", skater_id).stream()
        return [
            TrainingData(
                training.id,
                training.get("skater_id"),
                training.get("training_date"),
                training.get("dot_id"),
                training.get("training_jumps")
            ) for training in trainings
        ]

    def get_jump_by_id(self, jump_id: str) -> JumpData:
        """
        Retrieve a single jump's details by its document ID.

        Args:
            jump_id (str): The unique ID of the jump document.

        Returns:
            JumpData: The JumpData object representing the jump.

        Raises:
            ValueError: If no jump document with the given ID exists.
        """
        if self.db is None:
            raise ValueError("Aucune connexion à la base de données. Impossible de récupérer le jump.")
        jump_doc = self.db.collection("jumps").document(jump_id).get()

        if jump_doc.exists:
            jump_data = jump_doc.to_dict()
            return JumpData(
                jump_id=jump_id,
                jump_type=jump_data["jump_type"],
                jump_rotations=jump_data["jump_rotations"],
                jump_success=jump_data["jump_success"],
                jump_time=jump_data["jump_time"],
                jump_length=jump_data["jump_length"],
                jump_max_speed=jump_data["jump_max_speed"]
            )
        else:
            raise ValueError(f"Aucun jump trouvé avec l'identifiant {jump_id}")

    def get_skater_name_from_training_id(self, training_id: str) -> str:
        """
        Retrieve a skater's name given a training ID.

        This method first finds the skater_id from the training,
        then uses that skater_id to find the skater's name.

        Args:
            training_id (str): The ID of the training document.

        Returns:
            str: The skater's name.
        """
        if self.db is None:
            return ""
        skater_id = self.get_skater_from_training(training_id)
        skater_name = self.get_skater_name_from_id(skater_id)
        return skater_name

    def get_training_date_from_training_id(self, training_id: str) -> datetime:
        """
        Retrieve the training date for a given training session.

        Args:
            training_id (str): The ID of the training document.

        Returns:
            datetime: The date of the training session.
        """
        if self.db is None:
            return None
        training_date = self.db.collection("trainings").document(training_id).get().get("training_date")
        return training_date

    def get_skater_name_from_id(self, skater_id: str) -> str:
        """
        Retrieve a skater's name given their user/skater document ID.

        Args:
            skater_id (str): The skater's unique ID.

        Returns:
            str: The skater's name.
        """
        if self.db is None:
            return ""
        skater_name = self.db.collection("users").document(skater_id).get().get("name")
        return skater_name
