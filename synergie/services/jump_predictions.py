from __future__ import annotations

import numpy as np
import pandas as pd

import constants
from core.database.DatabaseManager import JumpData


def _rounded_rotation_value(row: pd.Series) -> float:
    val_rot = float(row["rotations"])
    if row["type"] < 5 and val_rot > 0.5:
        if val_rot < 2:
            return float(np.ceil(val_rot - 0.3))
        return float(np.ceil(val_rot - 0.15))
    if row["type"] == 5 and val_rot > 0.8:
        return float(np.ceil(val_rot - 0.7) + 0.5)
    return 0.0


def build_training_jump_payload(predictions: pd.DataFrame, training_id: str) -> list[dict]:
    training_jumps = []
    unknown_rotations = []

    for _, row in predictions.iterrows():
        jump_time_min, jump_time_sec = row["videoTimeStamp"].split(":")
        jump_time = "{:02d}:{:02d}".format(int(jump_time_min), int(jump_time_sec))
        rounded_rotation = _rounded_rotation_value(row)
        jump_data = JumpData(
            0,
            training_id,
            constants.jumpType(int(row["type"])).name,
            rounded_rotation,
            bool(row["success"]),
            jump_time,
            float(row["rotation_speed"]),
            float(row["length"]),
        )

        if rounded_rotation > 0:
            training_jumps.append(jump_data.to_dict())
        else:
            unknown_rotations.append(jump_data)

    if training_jumps:
        return training_jumps

    return [jump.to_dict() for jump in unknown_rotations]
