from __future__ import annotations

import hashlib
import json
from pathlib import Path

import constants
from synergie.config import ACCELERATION_ABERRANT_LIMIT_G
from synergie.services.hdf5_archive_service import load_segment_dataframe
from synergie.services.signal_cleaning_service import clean_imu_outliers


def training_cache_path(dataset_path: str | Path) -> Path:
    """Return the on-disk cache path for one training dataset."""
    return Path(dataset_path) / ".training_cache.npz"


def load_or_build_training_cache(
    dataset_path: str | Path,
    *,
    type_window_start: int,
    type_window_frames: int,
    success_window_start: int,
    success_window_frames: int,
    skip_incomplete_windows: bool = False,
) -> dict:
    """Load a valid numeric-window cache or rebuild it from segment CSV files."""
    import numpy as np
    import pandas as pd

    dataset_root = Path(dataset_path)
    main_frame = pd.read_csv(dataset_root / "jumplist.csv")
    trainable = main_frame[(main_frame["success"] != 2) & (main_frame["type"] != 8)].copy()
    paths = trainable["path"].astype(str).tolist()
    fingerprint = _cache_fingerprint(
        dataset_root,
        paths,
        type_window_start=type_window_start,
        type_window_frames=type_window_frames,
        success_window_start=success_window_start,
        success_window_frames=success_window_frames,
        skip_incomplete_windows=skip_incomplete_windows,
    )
    cache_path = training_cache_path(dataset_root)
    cached = _load_valid_cache(cache_path, fingerprint)
    if cached is not None:
        return cached

    type_windows = []
    success_windows = []
    retained_paths = []
    skipped_paths = []
    for path_value in paths:
        frame = load_segment_dataframe(path_value)[constants.fields_to_keep]
        frame, _cleaning_report = clean_imu_outliers(frame, acceleration_limit_g=ACCELERATION_ABERRANT_LIMIT_G)
        type_window = frame[type_window_start : type_window_start + type_window_frames].to_numpy(dtype="float32")
        success_window = frame[success_window_start : success_window_start + success_window_frames].to_numpy(dtype="float32")
        if len(type_window) != type_window_frames:
            if skip_incomplete_windows:
                skipped_paths.append(path_value)
                continue
            raise ValueError(
                f"Type window requires {type_window_frames} frames starting at {type_window_start}, "
                f"but segment '{path_value}' only provides {len(type_window)} frames."
            )
        if len(success_window) != success_window_frames:
            if skip_incomplete_windows:
                skipped_paths.append(path_value)
                continue
            raise ValueError(
                f"Success window requires {success_window_frames} frames starting at {success_window_start}, "
                f"but segment '{path_value}' only provides {len(success_window)} frames."
            )
        type_windows.append(np.nan_to_num(type_window, nan=0.0, posinf=0.0, neginf=0.0))
        success_windows.append(np.nan_to_num(success_window, nan=0.0, posinf=0.0, neginf=0.0))
        retained_paths.append(path_value)

    metadata = {
        "fingerprint": fingerprint,
        "count": len(retained_paths),
        "skipped_paths": skipped_paths,
        "type_window_start": int(type_window_start),
        "type_window_frames": int(type_window_frames),
        "success_window_start": int(success_window_start),
        "success_window_frames": int(success_window_frames),
    }
    payload = {
        "metadata": metadata,
        "paths": retained_paths,
        "type_windows": np.asarray(type_windows, dtype="float32"),
        "success_windows": np.asarray(success_windows, dtype="float32"),
        "from_cache": False,
    }
    np.savez_compressed(
        cache_path,
        metadata=json.dumps(metadata),
        paths=np.asarray(retained_paths),
        type_windows=payload["type_windows"],
        success_windows=payload["success_windows"],
    )
    return payload


def _load_valid_cache(cache_path: Path, fingerprint: str) -> dict | None:
    """Return cache contents only when the recorded fingerprint still matches."""
    import numpy as np

    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata"]))
        if metadata.get("fingerprint") != fingerprint:
            return None
        return {
            "metadata": metadata,
            "paths": cached["paths"].astype(str).tolist(),
            "type_windows": cached["type_windows"],
            "success_windows": cached["success_windows"],
            "from_cache": True,
        }


def _cache_fingerprint(
    dataset_root: Path,
    paths: list[str],
    *,
    type_window_start: int,
    type_window_frames: int,
    success_window_start: int,
    success_window_frames: int,
    skip_incomplete_windows: bool,
) -> str:
    """Hash all inputs that change the numeric training windows."""
    digest = hashlib.sha256()
    jumplist_path = dataset_root / "jumplist.csv"
    digest.update(jumplist_path.read_bytes())
    digest.update(",".join(constants.fields_to_keep).encode("utf-8"))
    digest.update(
        f"{type_window_start}:{type_window_frames}:{success_window_start}:{success_window_frames}:{skip_incomplete_windows}".encode("utf-8")
    )
    for path_value in paths:
        path = Path(path_value)
        if path.exists():
            stat = path.stat()
            digest.update(f"{path.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        else:
            archive_path = Path("data/synergie_archive.h5")
            if archive_path.exists():
                stat = archive_path.stat()
                digest.update(f"{path.as_posix()}:hdf5:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
            else:
                path.stat()
    return digest.hexdigest()
