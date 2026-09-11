from __future__ import annotations

import re
from pathlib import Path

from synergie.config import SEGMENT_FRAMES_AFTER_TAKEOFF, SEGMENT_FRAMES_BEFORE_TAKEOFF
from synergie.services.hdf5_archive_service import load_segment_dataframe


def ensure_pre_takeoff_context(dataset_path: str | Path, required_before_frames: int, *, progress_callback=None) -> dict:
    """Extend legacy annotated segments from raw IMU files when earlier context is required."""
    import pandas as pd

    if required_before_frames <= SEGMENT_FRAMES_BEFORE_TAKEOFF:
        return {"reexported": 0, "skipped": 0}
    frame = pd.read_csv(Path(dataset_path) / "jumplist.csv")
    reexported = 0
    skipped = 0
    already_sufficient = 0
    raw_session_cache: dict[Path, object] = {}
    paths = frame["path"].dropna().astype(str).tolist()
    for index, path_value in enumerate(paths, start=1):
        segment_path = Path(path_value)
        try:
            changed = reexport_segment_with_context(segment_path, required_before_frames, raw_session_cache=raw_session_cache)
        except FileNotFoundError:
            skipped += 1
            continue
        reexported += int(changed)
        already_sufficient += int(not changed)
        if progress_callback is not None:
            progress_callback({"current": index, "total": len(paths), "reexported": reexported})
    if skipped > 0:
        raise ValueError(
            f"Unable to re-export all segments automatically: {skipped} segment(s) have no matching raw IMU file. "
            "The offset benchmark needs a uniform pre-takeoff context for every training segment."
        )
    if reexported == 0 and already_sufficient == 0:
        raise ValueError(
            "Unable to re-export earlier context automatically: no matching raw IMU files were found for the annotated segments."
        )
    return {"reexported": reexported, "already_sufficient": already_sufficient, "skipped": skipped}


def ensure_post_takeoff_context(dataset_path: str | Path, required_after_frames: int, *, progress_callback=None) -> dict:
    """Extend annotated segments after takeoff when later success windows are required."""
    import pandas as pd

    if required_after_frames <= SEGMENT_FRAMES_AFTER_TAKEOFF:
        return {"reexported": 0, "skipped": 0}
    frame = pd.read_csv(Path(dataset_path) / "jumplist.csv")
    reexported = 0
    skipped = 0
    already_sufficient = 0
    raw_session_cache: dict[Path, object] = {}
    paths = frame["path"].dropna().astype(str).tolist()
    for index, path_value in enumerate(paths, start=1):
        segment_path = Path(path_value)
        try:
            changed = reexport_segment_with_post_context(
                segment_path,
                required_after_frames,
                raw_session_cache=raw_session_cache,
            )
        except FileNotFoundError:
            skipped += 1
            continue
        reexported += int(changed)
        already_sufficient += int(not changed)
        if progress_callback is not None:
            progress_callback({"current": index, "total": len(paths), "reexported": reexported})
    if reexported == 0 and already_sufficient == 0:
        raise ValueError(
            "Unable to re-export later context automatically: no matching raw IMU files were found for the annotated segments."
        )
    return {"reexported": reexported, "already_sufficient": already_sufficient, "skipped": skipped}


def reexport_segment_with_context(
    segment_path: str | Path,
    frames_before_takeoff: int,
    *,
    raw_session_cache: dict[Path, object] | None = None,
) -> bool:
    """Rebuild one legacy segment from its matching raw IMU file when possible."""
    import pandas as pd
    from core.data_treatment.data_generation.trainingSession import trainingSession

    path = Path(segment_path)
    segment = load_segment_dataframe(path)
    existing_before = len(segment) - SEGMENT_FRAMES_AFTER_TAKEOFF
    if existing_before >= frames_before_takeoff:
        return False
    raw_path = _matching_raw_path(path)
    if raw_path is None:
        raise FileNotFoundError(f"No matching raw IMU file found for {path}")
    cache = raw_session_cache if raw_session_cache is not None else {}
    raw_session = cache.get(raw_path)
    if raw_session is None:
        raw_session = trainingSession(pd.read_csv(raw_path, low_memory=False))
        cache[raw_path] = raw_session
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


def reexport_segment_with_post_context(
    segment_path: str | Path,
    frames_after_takeoff: int,
    *,
    raw_session_cache: dict[Path, object] | None = None,
) -> bool:
    """Rebuild one segment with more post-takeoff context when possible."""
    import pandas as pd
    from core.data_treatment.data_generation.trainingSession import trainingSession

    path = Path(segment_path)
    segment = load_segment_dataframe(path)
    # Historical segments were expanded on the left while keeping 180 frames after takeoff.
    existing_before = len(segment) - SEGMENT_FRAMES_AFTER_TAKEOFF
    existing_after = len(segment) - existing_before
    if existing_after >= frames_after_takeoff:
        return False
    raw_path = _matching_raw_path(path)
    if raw_path is None:
        raise FileNotFoundError(f"No matching raw IMU file found for {path}")
    cache = raw_session_cache if raw_session_cache is not None else {}
    raw_session = cache.get(raw_path)
    if raw_session is None:
        raw_session = trainingSession(pd.read_csv(raw_path, low_memory=False))
        cache[raw_path] = raw_session
    takeoff_sample = int(segment.iloc[existing_before]["SampleTimeFine"])
    matches = raw_session.df.index[raw_session.df["SampleTimeFine"].astype("int64") == takeoff_sample].tolist()
    if not matches:
        raise FileNotFoundError(f"Unable to locate takeoff frame in raw IMU file for {path}")
    takeoff_index = int(matches[0])
    start = takeoff_index - existing_before
    end = takeoff_index + frames_after_takeoff
    if start < 0 or end > len(raw_session.df):
        raise FileNotFoundError(f"Not enough raw context to rebuild {path}")
    rebuilt = raw_session.df.iloc[start:end].copy()
    rebuilt["Combination"] = int(segment["Combination"].iloc[0]) if "Combination" in segment else 0
    rebuilt.to_csv(path, index=False)
    return True


def _matching_raw_path(segment_path: Path) -> Path | None:
    match = re.fullmatch(r"(\d{8})_(\d{4})_(\d+)_\d+\.csv", segment_path.name)
    if match:
        date, time, sensor = match.groups()
        raw_dir = Path("data/raw") / f"{date[6:8]}{date[4:6]}" / time
    else:
        legacy_match = re.fullmatch(r"(\d{4})_(\d+)_\d+\.csv", segment_path.name)
        if legacy_match is None:
            return None
        time, sensor = legacy_match.groups()
        parts = segment_path.as_posix().split("/")
        if len(parts) < 4:
            return None
        raw_dir = Path("data/raw") / parts[-3] / time
    candidates = sorted(raw_dir.glob(f"{sensor}_*.csv"))
    return candidates[0] if candidates else None
