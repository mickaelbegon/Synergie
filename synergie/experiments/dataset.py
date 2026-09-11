from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import constants
from synergie.config import SUCCESS_WINDOW_FRAMES, SUCCESS_WINDOW_START, TYPE_WINDOW_FRAMES, TYPE_WINDOW_START
from synergie.services.hdf5_archive_service import load_segment_dataframe


@dataclass
class ClassificationDataset:
    X_series: np.ndarray
    X_scalar: np.ndarray
    y: np.ndarray
    labels: list[int]
    n_samples: int
    sequence_length: int
    n_channels: int


def _resolve_jump_path(dataset_path: Path, jump_path: str) -> Path:
    candidate = Path(jump_path)
    if candidate.exists():
        return candidate

    repo_root = dataset_path.parents[1]
    candidate = repo_root / jump_path
    if candidate.exists():
        return candidate

    return Path(jump_path)


def load_classification_dataset(task: str, dataset_path: str) -> ClassificationDataset:
    dataset_root = Path(dataset_path)
    main_frame = pd.read_csv(dataset_root / "jumplist.csv")
    skater_data = pd.read_csv(dataset_root / "skaterData.csv")
    fields_to_keep = constants.fields_to_keep

    X_series = []
    X_scalar = []
    y = []
    labels = []

    for _, row in main_frame.iterrows():
        if row["success"] == 2 or row["type"] == 8:
            continue

        jump_frame = load_segment_dataframe(_resolve_jump_path(dataset_root, row["path"]))
        jump_frame = jump_frame[fields_to_keep]

        if task == "type":
            temporal = jump_frame[TYPE_WINDOW_START : TYPE_WINDOW_START + TYPE_WINDOW_FRAMES].copy().to_numpy()
            label = int(row["type"])
        elif task == "success":
            temporal = jump_frame[SUCCESS_WINDOW_START : SUCCESS_WINDOW_START + SUCCESS_WINDOW_FRAMES].copy().reset_index(drop=True).to_numpy()
            label = int(row["success"])
        else:
            raise ValueError(f"Unsupported task: {task}")

        skater_info = skater_data[skater_data["skater"] == row["skater"]][["weight", "height"]].to_numpy()[0]
        temporal = np.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0)

        X_series.append(temporal.T)
        X_scalar.append(skater_info)
        y.append(label)
        labels.append(label)

    if not X_series:
        raise ValueError(f"No usable samples found for task '{task}' in {dataset_path}")

    X_series_np = np.stack(X_series).astype(np.float32)
    X_scalar_np = np.asarray(X_scalar, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.int64)

    return ClassificationDataset(
        X_series=X_series_np,
        X_scalar=X_scalar_np,
        y=y_np,
        labels=sorted(set(labels)),
        n_samples=int(X_series_np.shape[0]),
        sequence_length=int(X_series_np.shape[2]),
        n_channels=int(X_series_np.shape[1]),
    )
