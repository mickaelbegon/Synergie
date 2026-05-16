from __future__ import annotations

from pathlib import Path

import constants
from synergie import session_store


def list_sessions() -> list[str]:
    """Return configured session identifiers in sorted order."""
    constants.sessions = session_store.load_sessions()
    return sorted(constants.sessions)


def session_metadata(session_name: str) -> dict:
    """Return metadata for one configured session."""
    constants.sessions = session_store.load_sessions()
    return constants.get_session(session_name)


def add_session(session_name: str, path: str, sample_time_fine_synchro: int) -> dict:
    """Persist one session and refresh in-memory constants."""
    metadata = session_store.add_session(session_name, path, sample_time_fine_synchro)
    constants.sessions = session_store.load_sessions()
    return metadata


def session_synchro(session_name: str) -> int:
    """Return the configured sync offset for one session."""
    return int(session_metadata(session_name)["sample_time_fine_synchro"])


def list_session_csv_files(session_name: str, raw_root: str = "data/raw") -> list[Path]:
    """Return processable raw CSV files inside one configured session folder.

    Session folders can also contain derived exports such as
    ``jumplist_partie1.csv``. Those are processing outputs, not IMU inputs, so
    they must stay out of the GUI lists and batch-processing workflows.
    """
    metadata = session_metadata(session_name)
    session_dir = Path(raw_root) / metadata["path"]
    if not session_dir.exists():
        return []
    return sorted(path for path in session_dir.glob("*.csv") if _is_processable_session_csv(path))


def list_all_session_csv_files(raw_root: str = "data/raw") -> list[dict]:
    """Return all session CSV files with processing metadata."""
    records: list[dict] = []
    for session_name in list_sessions():
        metadata = session_metadata(session_name)
        for path in list_session_csv_files(session_name, raw_root=raw_root):
            records.append(
                {
                    "path": path,
                    "session_name": session_name,
                    "sample_time_fine_synchro": int(metadata["sample_time_fine_synchro"]),
                }
            )
    return records


def list_directory_files(directory: str | Path) -> list[Path]:
    """Return regular files directly inside one directory."""
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return []
    return sorted(path for path in directory_path.iterdir() if path.is_file())


def session_directory(session_name: str, raw_root: str = "data/raw") -> Path:
    """Return the raw-data directory for one session."""
    return Path(raw_root) / session_metadata(session_name)["path"]


def describe_file(path: str | Path) -> dict:
    """Return lightweight filesystem metadata for one file."""
    file_path = Path(path)
    stat = file_path.stat()
    return {
        "path": file_path,
        "name": file_path.name,
        "suffix": file_path.suffix.lower(),
        "size_bytes": stat.st_size,
    }


def next_jumplist_output_path(directory: str | Path) -> Path:
    """Return the next collision-free `jumplist_partieX.csv` path."""
    directory_path = Path(directory)
    index = 1
    while True:
        candidate = directory_path / f"jumplist_partie{index}.csv"
        if not candidate.exists():
            return candidate
        index += 1


def _is_processable_session_csv(path: Path) -> bool:
    """Return whether a session CSV is a raw IMU candidate for processing."""
    return "jumplist" not in path.stem.lower()
