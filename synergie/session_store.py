from __future__ import annotations

import json
from pathlib import Path


SESSIONS_FILE = Path("config") / "sessions.json"


def load_sessions() -> dict[str, dict]:
    with SESSIONS_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): value for key, value in data.items()}


def save_sessions(sessions: dict[str, dict]) -> None:
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        str(session_id): {
            "path": str(metadata["path"]).replace("\\", "/"),
            "sample_time_fine_synchro": int(metadata["sample_time_fine_synchro"]),
        }
        for session_id, metadata in sorted(sessions.items())
    }
    with SESSIONS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
        handle.write("\n")


def add_session(session_id: str, path: str, sample_time_fine_synchro: int) -> dict:
    sessions = load_sessions()
    if session_id in sessions:
        raise ValueError(f"Session '{session_id}' already exists.")

    sessions[session_id] = {
        "path": path.replace("\\", "/"),
        "sample_time_fine_synchro": int(sample_time_fine_synchro),
    }
    save_sessions(sessions)
    return sessions[session_id]
