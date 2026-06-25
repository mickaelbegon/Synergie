from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synergie.operations import annotation_turn_value_for_storage, suggest_turns_from_rotation
from core.utils.jump import rotation_start_index_for_turns
from synergie.services.rotation_audit_service import (
    _load_segment_frame,
    _rotation_interval_indices,
    _safe_float,
)


def detected_interval(frame) -> tuple[int, int] | None:
    crossing = frame["X_gyr_second_derivative_crossing"].astype(int).to_numpy()
    begins = np.where(np.diff(crossing) == 1)[0]
    ends = np.where(np.diff(crossing) == -1)[0]
    pairs = [(int(begin), int(end)) for begin in begins for end in ends if end > begin]
    if not pairs:
        return None
    return min(pairs, key=lambda pair: pair[1] - pair[0])


def integrate_rotation(frame, start_index: int, end_index: int) -> float | None:
    interval = frame.iloc[int(start_index) : int(end_index)]
    if len(interval) < 2:
        return None
    timestamps = interval["SampleTimeFine"].to_numpy(dtype="float64")
    speeds = interval["Gyr_X"].to_numpy(dtype="float64")
    valid = np.isfinite(speeds[:-1]) & np.isfinite(np.diff(timestamps))
    if not valid.any():
        return None
    return abs(float(np.nansum(speeds[:-1][valid] * (np.diff(timestamps)[valid] / 1e6)) / 360.0))


def estimate_turns(measured_rotation: float, jump_type, offset: float) -> float:
    ui_turns = suggest_turns_from_rotation(measured_rotation, contact_offset_turns=offset, jump_type=jump_type)
    return float(annotation_turn_value_for_storage(jump_type, ui_turns))


def accuracy_for_offset(records: list[dict], offset: float) -> tuple[float, float]:
    predicted = np.array([estimate_turns(row["rotation"], row["type"], offset) for row in records])
    annotated = np.array([row["annotated"] for row in records])
    return float((predicted == annotated).mean()), float(np.abs(predicted - annotated).mean())


def best_offsets(records: list[dict]) -> list[dict]:
    results = []
    for offset in np.arange(-0.10, 0.81, 0.01):
        accuracy, mae = accuracy_for_offset(records, float(offset))
        results.append({"offset": round(float(offset), 2), "accuracy": accuracy, "mae": mae})
    return sorted(results, key=lambda row: (row["accuracy"], -row["mae"]), reverse=True)


def load_labelled_rows(jumplist_path: Path):
    frame = pd.read_csv(jumplist_path)
    for row_index, row in frame.iterrows():
        jump_type_value = _safe_float(row.get("type"), default=None)
        jump_type = 8 if jump_type_value is None else int(jump_type_value)
        annotated_turns = _safe_float(row.get("rotations"), default=None)
        if jump_type == 8 or annotated_turns is None or annotated_turns <= 0:
            continue
        yield row_index, row, jump_type, annotated_turns


def main() -> None:
    jumplist_path = Path("data/annotated/total/jumplist.csv")
    strategies = {
        "detected_start_refined_landing": lambda frame, begin, landing: begin,
        "dynamic_rotation_start": lambda frame, begin, landing: rotation_start_index_for_turns(begin, frame, landing),
        "dynamic_rotation_start_strict": lambda frame, begin, landing: rotation_start_index_for_turns(
            begin,
            frame,
            landing,
            min_speed_dps=180.0,
            peak_fraction=0.20,
            sustain_frames=4,
        ),
        "dynamic_rotation_start_very_strict": lambda frame, begin, landing: rotation_start_index_for_turns(
            begin,
            frame,
            landing,
            min_speed_dps=240.0,
            peak_fraction=0.25,
            sustain_frames=4,
        ),
        "dynamic_strict_max_30f": lambda frame, begin, landing: rotation_start_index_for_turns(
            begin,
            frame,
            landing,
            max_search_frames_before_takeoff=30,
            min_speed_dps=240.0,
            peak_fraction=0.25,
            sustain_frames=4,
        ),
        "dynamic_strict_max_40f": lambda frame, begin, landing: rotation_start_index_for_turns(
            begin,
            frame,
            landing,
            max_search_frames_before_takeoff=40,
            min_speed_dps=240.0,
            peak_fraction=0.25,
            sustain_frames=4,
        ),
        "dynamic_strict_max_50f": lambda frame, begin, landing: rotation_start_index_for_turns(
            begin,
            frame,
            landing,
            max_search_frames_before_takeoff=50,
            min_speed_dps=240.0,
            peak_fraction=0.25,
            sustain_frames=4,
        ),
        "dynamic_strict_max_60f": lambda frame, begin, landing: rotation_start_index_for_turns(
            begin,
            frame,
            landing,
            max_search_frames_before_takeoff=60,
            min_speed_dps=240.0,
            peak_fraction=0.25,
            sustain_frames=4,
        ),
        "fixed_pre_start_05f": lambda frame, begin, landing: max(0, begin - 5),
        "fixed_pre_start_10f": lambda frame, begin, landing: max(0, begin - 10),
        "fixed_pre_start_15f": lambda frame, begin, landing: max(0, begin - 15),
        "fixed_pre_start_20f": lambda frame, begin, landing: max(0, begin - 20),
        "fixed_pre_start_25f": lambda frame, begin, landing: max(0, begin - 25),
        "fixed_pre_start_30f": lambda frame, begin, landing: max(0, begin - 30),
    }
    records_by_strategy = {name: [] for name in strategies}
    shifts_by_strategy = {name: [] for name in strategies}

    for row_index, row, jump_type, annotated_turns in load_labelled_rows(jumplist_path):
        segment = _load_segment_frame(row.get("path"))
        if segment is None:
            continue
        detected = detected_interval(segment)
        refined = _rotation_interval_indices(segment)
        if detected is None or refined is None:
            continue
        begin, _detected_end = detected
        _refined_begin, landing = refined
        for name, resolve_start in strategies.items():
            start = int(resolve_start(segment, begin, landing))
            rotation = integrate_rotation(segment, start, landing)
            if rotation is None:
                continue
            records_by_strategy[name].append(
                {
                    "row_index": int(row_index),
                    "type": jump_type,
                    "annotated": float(annotated_turns),
                    "rotation": float(rotation),
                }
            )
            shifts_by_strategy[name].append(int(begin) - start)

    for name, records in records_by_strategy.items():
        top = best_offsets(records)[:5]
        offset_045 = accuracy_for_offset(records, 0.45)
        shifts = np.array(shifts_by_strategy[name], dtype="float64")
        print(name)
        print(f"  n={len(records)} offset_0.45 acc={offset_045[0]:.4f} mae={offset_045[1]:.4f}")
        print(f"  best={top}")
        print(
            "  shift_frames "
            f"mean={float(np.mean(shifts)):.2f} p50={float(np.quantile(shifts, 0.50)):.1f} "
            f"p90={float(np.quantile(shifts, 0.90)):.1f} moved={float(np.mean(shifts > 0)):.3f}"
        )


if __name__ == "__main__":
    main()
