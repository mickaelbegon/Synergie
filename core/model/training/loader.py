import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass

import constants
from synergie.config import (
    ROTATION_MIRROR_COLUMNS,
    SUCCESS_WINDOW_FRAMES,
    SUCCESS_WINDOW_START,
    TYPE_WINDOW_FRAMES,
    TrainingConfig,
)
from synergie.services.training_cache_service import load_or_build_training_cache

@dataclass
class Dataset:
    temporal_features_train : np.array
    scalar_features_train : np.array
    labels_train : np.array
    temporal_features_test : np.array
    scalar_features_test : np.array
    labels_test : np.array
    val_dataset : tf.data.Dataset
    class_weight: dict[int, float] | None = None
    stratified_split_used: bool = False
    class_counts: dict[int, int] | None = None

class Loader:
    """
    This class is meant to load the data from the csv files,
    and make it ready to be used by the model for training
    """
    def __init__(
        self,
        folder_path: str,
        train_ratio: float = 0.8,
        augment_mirror: bool = True,
        use_scalar_features: bool = True,
        type_window_start: int = 0,
        type_window_frames: int = TYPE_WINDOW_FRAMES,
        success_window_start: int = SUCCESS_WINDOW_START,
        success_window_frames: int = SUCCESS_WINDOW_FRAMES,
    ):
        assert 0 <= train_ratio <= 1
        self.training_config = TrainingConfig(train_ratio=train_ratio, augment_mirror=augment_mirror)
        self.use_scalar_features = bool(use_scalar_features)
        self.type_window_start = int(type_window_start)
        self.type_window_frames = int(type_window_frames)
        self.success_window_start = int(success_window_start)
        self.success_window_frames = int(success_window_frames)

        self.folder_path = folder_path
        main_csv = os.path.join(folder_path, "jumplist.csv")
        mainFrame = pd.read_csv(main_csv)
        training_cache = load_or_build_training_cache(
            folder_path,
            type_window_start=self.type_window_start,
            type_window_frames=self.type_window_frames,
            success_window_start=self.success_window_start,
            success_window_frames=self.success_window_frames,
        )
        cached_type_windows = dict(zip(training_cache["paths"], training_cache["type_windows"]))
        cached_success_windows = dict(zip(training_cache["paths"], training_cache["success_windows"]))

        skaterData = pd.read_csv("data/annotated/total/skaterData.csv")

        jumps = []
        jumps_success = []
        labelstype = []
        labelssuccess = []
        fields_to_keep = constants.fields_to_keep
        self.path_jumps = []

        for index, row in mainFrame.iterrows():
            if row["success"] != 2 and row["type"] != 8:
                self.path_jumps.append(row["path"])
                skater_info = self._lookup_skater_info(skaterData, row["skater"], use_scalar_features=self.use_scalar_features)
                type_window = cached_type_windows[str(row["path"])]
                success_window = cached_success_windows[str(row["path"])]
                jumps.append((type_window, skater_info))
                jumps_success.append((success_window, skater_info))
                labelstype.append(row['type'])
                labelssuccess.append(row['success'])

                if self.training_config.augment_mirror:
                    jumps.append((self._mirror_temporal_features(type_window, fields_to_keep), skater_info))
                    jumps_success.append((self._mirror_temporal_features(success_window, fields_to_keep), skater_info))
                    labelstype.append(row['type'])
                    labelssuccess.append(row['success'])

        # label one hot encoding
        type_label_encoder = LabelEncoder()
        encoded_type_labels = type_label_encoder.fit_transform(labelstype)
        labelstype = np.eye(len(type_label_encoder.classes_))[encoded_type_labels]

        success_label_encoder = LabelEncoder()
        encoded_success_labels = success_label_encoder.fit_transform(labelssuccess)
        labelssuccess = np.eye(len(success_label_encoder.classes_))[encoded_success_labels]

        # make a training and validation dataset

        (
            features_train,
            features_val,
            self.labels_train,
            self.labels_val,
            self.type_stratified_split_used,
        ) = self._split_dataset(
            jumps,
            labelstype,
            encoded_type_labels,
            train_ratio=self.training_config.train_ratio,
        )

        self.temporal_features_test = []
        self.scalar_features_test = []
        for x in features_val:
            self.temporal_features_test.append(x[0])
            self.scalar_features_test.append(x[1])

        self.temporal_features_train = []
        self.scalar_features_train = []
        for x in features_train:
            self.temporal_features_train.append(x[0])
            self.scalar_features_train.append(x[1])

        self.val_dataset = tf.data.Dataset.from_tensor_slices(({"temporal_input" : self.temporal_features_test, "scalar_input" : self.scalar_features_test}, self.labels_val)).batch(16)

        (
            features_train_success,
            features_val_success,
            self.labels_train_success,
            self.labels_val_success,
            self.success_stratified_split_used,
        ) = self._split_dataset(
            jumps_success,
            labelssuccess,
            encoded_success_labels,
            train_ratio=self.training_config.train_ratio,
        )

        self.type_class_counts = self._class_counts(encoded_type_labels)
        self.success_class_counts = self._class_counts(encoded_success_labels)
        self.type_class_weight = self._compute_class_weight(encoded_type_labels)
        self.success_class_weight = self._compute_class_weight(encoded_success_labels)

        self.temporal_features_test_success = []
        self.scalar_features_test_success = []
        for x in features_val_success:
            self.temporal_features_test_success.append(x[0])
            self.scalar_features_test_success.append(x[1])

        self.temporal_features_train_success = []
        self.scalar_features_train_success = []
        for x in features_train_success:
            self.temporal_features_train_success.append(x[0])
            self.scalar_features_train_success.append(x[1])

        self.val_dataset_success = tf.data.Dataset.from_tensor_slices(({"temporal_input" : self.temporal_features_test_success, "scalar_input" : self.scalar_features_test_success}, self.labels_val_success)).batch(16)

    def _mirror_temporal_features(self, features: np.ndarray, field_names: list[str]) -> np.ndarray:
        mirrored = features.copy()
        for column_name in ROTATION_MIRROR_COLUMNS:
            if column_name in field_names:
                mirrored[:, field_names.index(column_name)] *= -1
        return mirrored

    @staticmethod
    def _lookup_skater_info(
        skater_data: pd.DataFrame,
        skater_id,
        *,
        use_scalar_features: bool = True,
    ) -> np.ndarray:
        """Return athlete scalars, or neutral zeros when scalar features are disabled."""
        if not use_scalar_features:
            return np.zeros(2, dtype=np.float32)
        matches = skater_data[skater_data["skater"].astype(str) == str(skater_id)][["weight", "height"]].to_numpy()
        if len(matches) == 0:
            raise ValueError(
                f"Unknown skater '{skater_id}' in jumplist.csv. "
                "Add this athlete to skaterData.csv or normalize the legacy skater id before training."
            )
        return matches[0]

    @staticmethod
    def _class_counts(encoded_labels) -> dict[int, int]:
        counts: dict[int, int] = {}
        for label in encoded_labels:
            normalized = int(label)
            counts[normalized] = counts.get(normalized, 0) + 1
        return counts

    @classmethod
    def _compute_class_weight(cls, encoded_labels) -> dict[int, float]:
        counts = cls._class_counts(encoded_labels)
        total = sum(counts.values())
        class_count = len(counts)
        if total == 0 or class_count == 0:
            return {}
        return {
            label: float(total / (class_count * count))
            for label, count in counts.items()
            if count > 0
        }

    @staticmethod
    def _can_use_stratify(encoded_labels, train_ratio: float) -> bool:
        if len(encoded_labels) < 2:
            return False
        counts: dict[int, int] = {}
        for label in encoded_labels:
            normalized = int(label)
            counts[normalized] = counts.get(normalized, 0) + 1
        if len(counts) < 2 or min(counts.values()) < 2:
            return False
        test_count = len(encoded_labels) - int(len(encoded_labels) * train_ratio)
        if test_count <= 0 and len(encoded_labels) > 1:
            test_count = 1
        return test_count >= len(counts)

    @classmethod
    def _split_dataset(cls, features, one_hot_labels, encoded_labels, train_ratio: float):
        use_stratify = cls._can_use_stratify(encoded_labels, train_ratio)
        features_train, features_val, labels_train, labels_val = train_test_split(
            features,
            one_hot_labels,
            train_size=train_ratio,
            shuffle=True,
            stratify=encoded_labels if use_stratify else None,
        )
        return features_train, features_val, labels_train, labels_val, use_stratify

    def get_type_data(self):
        data = Dataset(
            np.array(self.temporal_features_train),
            np.array(self.scalar_features_train),
            self.labels_train,
            np.array(self.temporal_features_test),
            np.array(self.scalar_features_test),
            self.labels_val,
            self.val_dataset,
            class_weight=self.type_class_weight,
            stratified_split_used=self.type_stratified_split_used,
            class_counts=self.type_class_counts,
        )
        return data
    
    def get_success_data(self):
        data = Dataset(
            np.array(self.temporal_features_train_success),
            np.array(self.scalar_features_train_success),
            self.labels_train_success,
            np.array(self.temporal_features_test_success),
            np.array(self.scalar_features_test_success),
            self.labels_val_success,
            self.val_dataset_success,
            class_weight=self.success_class_weight,
            stratified_split_used=self.success_stratified_split_used,
            class_counts=self.success_class_counts,
        )
        return data
