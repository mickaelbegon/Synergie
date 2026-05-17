from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def workflow_state_path(root: str | Path = "data/pending") -> Path:
    """Return the JSON file used to track session workflow state."""
    return Path(root) / "workflow_state.json"


def load_workflow_state(root: str | Path = "data/pending") -> dict:
    """Load workflow state with a stable empty structure."""
    path = workflow_state_path(root)
    if not path.exists():
        return {"sessions": {}}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        return {"sessions": {}}
    loaded.setdefault("sessions", {})
    return loaded


def save_workflow_state(state: dict, root: str | Path = "data/pending") -> Path:
    """Persist normalized workflow state."""
    path = workflow_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
    return path


def record_pending_session(
    *,
    session_key: str,
    source_files: list[str | Path],
    annotation_csv: str | Path,
    raw_destination: str | Path,
    prediction_status: str,
    root: str | Path = "data/pending",
) -> dict:
    """Record that one incoming session has been processed into pending data."""
    state = load_workflow_state(root)
    entry = {
        "status": "pending_annotation",
        "source_files": [str(Path(path)).replace("\\", "/") for path in source_files],
        "annotation_csv": str(Path(annotation_csv)).replace("\\", "/"),
        "raw_destination": str(Path(raw_destination)).replace("\\", "/"),
        "prediction_status": str(prediction_status),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    state["sessions"][session_key] = entry
    save_workflow_state(state, root)
    return entry


def session_workflow_entry(session_key: str, root: str | Path = "data/pending") -> dict | None:
    """Return tracked workflow metadata for one session if available."""
    return load_workflow_state(root)["sessions"].get(session_key)


def mark_finalized_session(session_key: str, root: str | Path = "data/pending") -> dict | None:
    """Mark one tracked session as merged into the annotated dataset."""
    state = load_workflow_state(root)
    entry = state["sessions"].get(session_key)
    if entry is None:
        return None
    entry["status"] = "finalized"
    entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_workflow_state(state, root)
    return entry
