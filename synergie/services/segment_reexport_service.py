from __future__ import annotations

import re
from pathlib import Path

from synergie.config import SEGMENT_FRAMES_AFTER_TAKEOFF, SEGMENT_FRAMES_BEFORE_TAKEOFF


def ensure_pre_takeoff_context(dataset_path: str | Path, required_before_frames: int) -> dict:
    """Extend legacy annotated segments from raw IMU files when earlier context is required."""
    import pandas as pd

    if required_before_frames <= SEGMENT_FRAMES_BEFORE_TAKEOFF:
        return {"reexported": 0, "skipped": 0}
    frame = pd.read_csv(Path(dataset_path) / "jumplist.csv")
    reexported = 0
    skipped = 0
    for path_value in frame["path"].dropna().astype(str):
        segment_path = Path(path_value)
        try:
            changed = reexport_segment_with_context(segment_path, required_before_frames)
        except FileNotFoundError:
            skipped += 1
            continue
        reexported += int(changed)
    if reexported == 0:
        raise ValueError(
            "Unable to re-export earlier context automatically: no matching raw IMU files were found for the annotated segments."
        )
    return {"reexported": reexported, "skipped": skipped}


def reexport_segment_with_context(segment_path: str | Path, frames_before_takeoff: int) -> bool:
    """Rebuild one legacy segment from its matching raw IMU file when possible."""
    import pandas as pd
    from core.data_treatment.data_generation.trainingSession import trainingSession

    path = Path(segment_path)
    segment = pd.read_csv(path)
    existing_before = len(segment) - SEGMENT_FRAMES_AFTER_TAKEOFF
    if existing_before >= frames_before_takeoff:
        return False
    raw_path = _matching_raw_path(path)
    if raw_path is None:
        raise FileNotFoundError(f"No matching raw IMU file found for {path}")
    raw_session = trainingSession(pd.read_csv(raw_path))
    takeoff_sample = int(segment.iloc[existing_before]["SampleTimeFine"])
    matches = raw_session.df.index[raw_session.df["SampleTimeFine"].astype("int64") == takeoff_sample].tolist()
    if not matches:
        raise FileNotFoundError(f"Unable to locate takeoff frame in raw IMU file for {path}")
    takeoff_index = int(matches[0])
    start = takeoff_index - frames_before_takeoff
    end = takeoff_index + SEGMENT_FRAMES_AFTER_TAKEOFF
    if start < 0 or end > len(raw_session.df):
        raise FileNotFoundError(f"Not enough raw context to rebuild {path}")
    rebuilt = raw_session.df.iloc[start:end].copy()
    rebuilt["Combination"] = int(segment["Combination"].iloc[0]) if "Combination" in segment else 0
    rebuilt.to_csv(path, index=False)
    return True


def _matching_raw_path(segment_path: Path) -> Path | None:
    match = re.fullmatch(r"(\d{8})_(\d{4})_(\d+)_\d+\.csv", segment_path.name)
    if not match:
        return None
    date, time, sensor = match.groups()
    raw_dir = Path("data/raw") / f"{date[6:8]}{date[4:6]}" / time
    candidates = sorted(raw_dir.glob(f"{sensor}_*.csv"))
    return candidates[0] if candidates else None
