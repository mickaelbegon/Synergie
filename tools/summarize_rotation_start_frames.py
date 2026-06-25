from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.utils.jump import rotation_start_index_for_turns
from synergie.services.annotation_service import JUMP_TYPE_LABELS
from synergie.services.rotation_audit_service import _load_segment_frame, _rotation_interval_indices, _safe_float


def main() -> None:
    frame = pd.read_csv("data/annotated/total/jumplist.csv")
    rows = []
    for _, row in frame.iterrows():
        jump_type_value = _safe_float(row.get("type"), default=None)
        annotated_turns = _safe_float(row.get("rotations"), default=None)
        if jump_type_value is None or int(jump_type_value) == 8 or annotated_turns is None or annotated_turns <= 0:
            continue
        segment = _load_segment_frame(row.get("path"))
        if segment is None:
            continue
        pair = _rotation_interval_indices(segment)
        if pair is None:
            continue
        takeoff_index, landing_index = pair
        rotation_start_index = rotation_start_index_for_turns(takeoff_index, segment, landing_index)
        jump_type = int(jump_type_value)
        rows.append(
            {
                "type": jump_type,
                "label": JUMP_TYPE_LABELS.get(jump_type, str(jump_type)),
                "frames_before_takeoff": int(takeoff_index - rotation_start_index),
            }
        )

    summary = (
        pd.DataFrame(rows)
        .groupby(["type", "label"])["frames_before_takeoff"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    print(f"N={len(rows)}")
    print(summary.to_csv(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
