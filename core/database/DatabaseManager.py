import os
import sys
from typing import List
from pathlib import Path
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import firebase_admin.firestore

from dataclasses import dataclass
from datetime import datetime

@dataclass
class JumpData:
    jump_id : int
    training_id : str
    jump_type : str
    jump_rotations : float
    jump_success : bool
    jump_time : int
    jump_max_speed : float
    jump_length : float

    def to_dict(self):
        return {"training_id" : self.training_id,
         "jump_type" : self.jump_type,
         "jump_rotations" : self.jump_rotations,
         "jump_success" : self.jump_success,
         "jump_time" : self.jump_time,
         "jump_max_speed" : self.jump_max_speed,
         "jump_length" : self.jump_length}

@dataclass
class TrainingData:
    training_id : int
    skater_id : int
    training_date : datetime
    dot_id : str
    training_jumps : List[str]

    def to_dict(self):
        return {"skater_id" : self.skater_id,
         "training_date" : self.training_date,
         "dot_id" : self.dot_id,
         "training_jumps" : self.training_jumps}

@dataclass
class SkaterData:
    skater_id : int
    skater_name : str

    def to_dict(self):
        return {"skater_name" : self.skater_name}

class DatabaseManager:
    CREDENTIALS_FILENAME = "s2m-skating-firebase-adminsdk-3ofmb-8552d58146.json"
    CREDENTIALS_ENV_VARS = ("SYNERGIE_FIREBASE_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS")

    @classmethod
    def credential_search_paths(cls) -> list[Path]:
        candidate_paths: list[Path] = []

        for env_var in cls.CREDENTIALS_ENV_VARS:
            env_value = os.environ.get(env_var)
            if env_value:
                candidate_paths.append(Path(env_value).expanduser())

        try:
            candidate_paths.append(Path(sys._MEIPASS) / cls.CREDENTIALS_FILENAME)
        except AttributeError:
            pass

        repo_root = Path(__file__).resolve().parents[2]
        candidate_paths.extend(
            [
                Path.cwd() / cls.CREDENTIALS_FILENAME,
                repo_root / cls.CREDENTIALS_FILENAME,
                repo_root / "config" / cls.CREDENTIALS_FILENAME,
            ]
        )

        unique_candidates: list[Path] = []
        seen: set[str] = set()
        for candidate in candidate_paths:
            normalized = str(candidate.resolve(strict=False)).lower()
            if normalized not in seen:
                unique_candidates.append(candidate)
                seen.add(normalized)
        return unique_candidates

    @classmethod
    def resolve_credentials_path(cls) -> Path:
        candidates = cls.credential_search_paths()
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        searched = "\n".join(f"- {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "Firebase credentials file not found.\n"
            f"Expected filename: {cls.CREDENTIALS_FILENAME}\n"
            "Checked these locations:\n"
            f"{searched}\n"
            "You can also define SYNERGIE_FIREBASE_CREDENTIALS to point to the JSON file."
        )

    def __init__(self):
        json_path = self.resolve_credentials_path()
        cred = credentials.Certificate(str(json_path))
        try:
            firebase_admin.initialize_app(cred)
        except ValueError:
            pass
        self.db = firestore.client()
    
    def save_training_data(self, data : TrainingData) -> int:
        add_time, new_ref = self.db.collection("trainings").add(data.to_dict())
        return new_ref.id
    
    def save_jump_data(self, data : JumpData) -> int:
        add_time, new_ref = self.db.collection("jumps").add(data.to_dict())
        return new_ref.id
    
    def get_skater_from_training(self, training_id : int) -> str:
        skater_id = self.db.collection("trainings").document(training_id).get().get("skater_id")
        return skater_id
    
    def get_all_skaters(self) -> list[SkaterData]:
        data_skaters = []
        for skater in self.db.collection("skaters").stream():
            data_skaters.append(SkaterData(skater.id, skater.get("skater_name")))
        return data_skaters
    
    def set_training_date(self, training_id, date) -> None:
        self.db.collection("trainings").document(training_id).update({"training_date" : date})

    def set_current_record(self, device_id, current_record) -> None:
        self.db.collection("dots").document(device_id).update({"current_record" : firestore.ArrayUnion([current_record])})

    def get_current_record(self, deviceId) -> str:
        try:
            trainingId = self.db.collection("dots").document(deviceId).get().get("current_record")[-1]
            return trainingId
        except (TypeError, IndexError):
            return ""
        
    def remove_current_record(self, deviceId, trainingId):
        self.db.collection("dots").document(deviceId).update({"current_record" : firestore.ArrayRemove([trainingId])})
    
    def get_dot_from_bluetooth(self, bluetoothAddress):
        dots =  self.db.collection("dots").where(filter=firestore.firestore.FieldFilter("bluetooth_address", "==", bluetoothAddress)).get()
        if len(dots) > 0:
            return dots[0]
        else:
            return None
        
    def save_dot_data(self, deviceId : str, bluetoothAddress : str, tagName : str) -> None:
        newDot = {"bluetooth_address" : bluetoothAddress,
                  "current_record" : [],
                  "tag_name" : tagName}
        self.db.collection("dots").add(document_data=newDot, document_id=deviceId)
    
    def add_jumps_to_training(self, trainingId : str, trainingJumps : List[str]):
        self.db.collection("trainings").document(trainingId).update({"training_jumps" : trainingJumps})
    
    def findUserByEmail(self, email):
        return self.db.collection("users").where(filter=firestore.firestore.FieldFilter("email", "==", email)).get()
    
    def getAllSkaterFromCoach(self, coachId) -> List[SkaterData]:
        skatersData = []
        for skater in self.db.collection("users").document(coachId).get().get("access"):
            skatersData.append(SkaterData(skater, self.db.collection("users").document(skater).get().get("name")))
        return skatersData

    def get_all_trainings_for_skater(self, skater_id) -> List[TrainingData]:
        trainings = self.db.collection("trainings").where("skater_id", "==", skater_id).stream()
        return [TrainingData(training.id, training.get("skater_id"), training.get("training_date"), training.get("dot_id"), training.get("training_jumps")) for training in trainings]
    
    def get_jump_by_id(self, jump_id: str) -> JumpData:

        jump_doc = self.db.collection("jumps").document(jump_id).get()
        
        if jump_doc.exists:
            jump_data = jump_doc.to_dict()
            return JumpData(
                jump_id=jump_id,
                training_id=jump_data["training_id"],
                jump_type=jump_data["jump_type"],
                jump_rotations=jump_data["jump_rotations"],
                jump_success=jump_data["jump_success"],
                jump_time=jump_data["jump_time"],
                jump_length=jump_data["jump_length"],
                jump_max_speed=jump_data["jump_max_speed"]
            )
        else:
            raise ValueError(f"Aucun jump trouvé avec l'identifiant {jump_id}")
        
    def get_skater_name_from_training_id(self, training_id) -> str:
        skater_id = self.get_skater_from_training(training_id)
        skater_name = self.get_skater_name_from_id(skater_id)
        return skater_name
    
    def get_training_date_from_training_id(self, training_id) -> datetime:
        training_date = self.db.collection("trainings").document(training_id).get().get("training_date")
        return training_date
    
    def get_skater_name_from_id(self, skater_id : str) -> str:
        skater_name = self.db.collection("users").document(skater_id).get().get("name")
        return skater_name
