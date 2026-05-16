from __future__ import annotations

from datetime import datetime
from pathlib import Path


def parse_new_imu_filename(path: str | Path) -> dict:
    """Parse one incoming IMU filename into structured metadata."""
    file_path = Path(path)
    stem_parts = file_path.stem.split("_")
    if len(stem_parts) != 4:
        raise ValueError(f"Unexpected IMU filename format: {file_path.name}")
    sensor_id, device_id, date_token, time_token = stem_parts
    recorded_at = datetime.strptime(f"{date_token}_{time_token}", "%Y%m%d_%H%M%S")
    return {
        "path": file_path,
        "name": file_path.name,
        "sensor_id": sensor_id,
        "device_id": device_id,
        "date_token": date_token,
        "time_token": time_token,
        "recorded_at": recorded_at,
    }


def new_imu_session_key(metadata: dict) -> str:
    """Return the shared timestamp key used to group one session."""
    return f"{metadata['date_token']}_{metadata['time_token']}"


def list_new_data_directories(root: str | Path = "data/new") -> list[dict]:
    """List usable incoming-data directories, excluding archival folders."""
    root_path = Path(root)
    directories: list[dict] = []
    if not root_path.exists():
        return directories
    for directory in sorted(root_path.rglob("*")):
        if not directory.is_dir():
            continue
        relative_parts = directory.relative_to(root_path).parts
        if not relative_parts:
            continue
        if any(part.lower() in {"done", "non"} for part in relative_parts):
            continue
        directories.append(
            {
                "path": directory,
                "relative_path": str(directory.relative_to(root_path)).replace("\\", "/"),
                "name": directory.name,
            }
        )
    return directories


def list_new_imu_files(root: str | Path = "data/new", directory: str | Path | None = None) -> list[dict]:
    """List incoming IMU CSV files and attach parsed metadata."""
    root_path = Path(root)
    search_root = root_path / directory if directory else root_path
    files: list[dict] = []
    if not search_root.exists():
        return files
    for file_path in sorted(search_root.rglob("*.csv")):
        if file_path.name.startswith("._"):
            continue
        if any(part.lower() in {"done", "non"} for part in file_path.parts):
            continue
        metadata = parse_new_imu_filename(file_path)
        metadata["relative_directory"] = str(file_path.parent.relative_to(root_path)).replace("\\", "/")
        files.append(metadata)
    return files


def list_new_imu_sessions(root: str | Path = "data/new", directory: str | Path | None = None) -> list[dict]:
    """Group incoming IMU files by timestamped recording session."""
    sessions: dict[str, dict] = {}
    for metadata in list_new_imu_files(root=root, directory=directory):
        session_key = new_imu_session_key(metadata)
        entry = sessions.setdefault(
            session_key,
            {
                "session_key": session_key,
                "date_token": metadata["date_token"],
                "time_token": metadata["time_token"],
                "recorded_at": metadata["recorded_at"],
                "files": [],
            },
        )
        entry["files"].append(metadata)
    session_list = list(sessions.values())
    session_list.sort(key=lambda item: item["recorded_at"])
    for session in session_list:
        session["files"].sort(key=lambda item: int(item["sensor_id"]))
    return session_list


def files_for_new_imu_session(selected_file: str | Path, root: str | Path = "data/new") -> list[dict]:
    """Return all files belonging to the selected incoming recording session."""
    selected_metadata = parse_new_imu_filename(selected_file)
    session_key = new_imu_session_key(selected_metadata)
    for session in list_new_imu_sessions(root=root):
        if session["session_key"] == session_key:
            return session["files"]
    return [selected_metadata]


def suggest_for_annotation_output_path(raw_csv_path: str | Path, pending_root: str | Path = "data/pending") -> Path:
    """Return a collision-free annotation filename for an incoming IMU CSV."""
    metadata = parse_new_imu_filename(raw_csv_path)
    pending_root_path = Path(pending_root)
    pending_root_path.mkdir(parents=True, exist_ok=True)
    base_name = f"{metadata['date_token']}_{metadata['time_token']}_for_annotation.csv"
    candidate = pending_root_path / base_name
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = pending_root_path / f"{metadata['date_token']}_{metadata['time_token']}_sensor{metadata['sensor_id']}_for_annotation_{index}.csv"
        if not candidate.exists():
            return candidate
        index += 1
