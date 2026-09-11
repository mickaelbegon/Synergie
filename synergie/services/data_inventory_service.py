from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from synergie.services.new_data_service import list_new_imu_sessions
from synergie.services.workflow_state_service import load_workflow_state


def build_data_inventory(
    *,
    new_root: str | Path = "data/new",
    pending_root: str | Path = "data/pending",
    annotated_root: str | Path = "data/annotated",
    training_dataset_root: str | Path = "data/annotated/total",
    hdf5_archive_path: str | Path | None = "data/synergie_archive.h5",
) -> list[dict]:
    """Summarize where each discovered data unit currently lives."""
    rows: dict[str, dict] = defaultdict(_empty_row)
    local_segment_paths: set[str] = set()

    for session in list_new_imu_sessions(root=new_root):
        row = rows[session["session_key"]]
        row["session"] = session["session_key"]
        row["new_files"] = len(session["files"])

    pending_root_path = Path(pending_root)
    for path in pending_root_path.glob("*_for_annotation*.csv"):
        session_key = path.name.split("_for_annotation", 1)[0]
        row = rows[session_key]
        row["session"] = session_key
        row["pending_files"] += 1

    for session_key, entry in load_workflow_state(pending_root).get("sessions", {}).items():
        row = rows[session_key]
        row["session"] = session_key
        row["workflow_status"] = entry.get("status", "")
        row["prediction_status"] = entry.get("prediction_status", "")

    annotated_root_path = Path(annotated_root)
    if annotated_root_path.exists():
        for first_level in annotated_root_path.iterdir():
            if not first_level.is_dir() or first_level.name == "total":
                continue
            for second_level in first_level.iterdir():
                if not second_level.is_dir():
                    continue
                key = f"{first_level.name}/{second_level.name}"
                row = rows[key]
                row["session"] = key
                segment_paths = [
                    path for path in second_level.rglob("*.csv") if path.is_file() and not path.name.lower().startswith("jumplist")
                ]
                local_segment_paths.update(path.as_posix() for path in segment_paths)
                row["stored_segments"] = len(segment_paths)
                row["trainable_labels"] = _count_trainable_local_labels(second_level)

    jumplist_path = Path(training_dataset_root) / "jumplist.csv"
    if jumplist_path.exists():
        import pandas as pd

        frame = pd.read_csv(jumplist_path)
        normalized_keys = frame["path"].fillna("").astype(str).map(_annotated_parent_key)
        for path_value, count in normalized_keys.value_counts().items():
            if not path_value:
                continue
            row = rows[path_value]
            row["session"] = path_value
            row["total_rows"] = int(count)
        if {"type", "success"}.issubset(frame.columns):
            trainable_mask = (
                frame["type"].apply(_safe_numeric).isin([0, 1, 2, 3, 4, 5])
                & frame["success"].apply(_safe_numeric).isin([0, 1])
            )
            for path_value, count in normalized_keys[trainable_mask].value_counts().items():
                if not path_value:
                    continue
                row = rows[path_value]
                row["session"] = path_value
                row["trainable_total_rows"] = int(count)

    if hdf5_archive_path is not None:
        try:
            from synergie.services.hdf5_archive_service import hdf5_segment_paths

            for path_value in hdf5_segment_paths(hdf5_archive_path):
                normalized = path_value.replace("\\", "/")
                if normalized in local_segment_paths or Path(normalized).exists():
                    continue
                key = _annotated_parent_key(normalized)
                if not key:
                    continue
                row = rows[key]
                row["session"] = key
                row["stored_segments"] += 1
        except (OSError, ValueError, KeyError):
            pass

    result = list(rows.values())
    result.sort(key=lambda item: item["session"])
    return result


def _empty_row() -> dict:
    return {
        "session": "",
        "new_files": 0,
        "pending_files": 0,
        "stored_segments": 0,
        "trainable_labels": 0,
        "total_rows": 0,
        "trainable_total_rows": 0,
        "workflow_status": "",
        "prediction_status": "",
    }


def _annotated_parent_key(path_value: str) -> str:
    normalized = path_value.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) < 4 or parts[0:2] != ["data", "annotated"]:
        return ""
    return f"{parts[2]}/{parts[3]}"


def _count_trainable_local_labels(session_dir: Path) -> int:
    """Count local legacy labels that are trainable, if a jumplist exists."""
    import pandas as pd

    candidates = sorted(path for path in session_dir.glob("jumplist*.csv") if path.is_file())
    if not candidates:
        return 0
    try:
        frame = pd.read_csv(candidates[0])
    except Exception:
        return 0
    success_column = "success" if "success" in frame else "sucess" if "sucess" in frame else None
    if "type" not in frame or success_column is None:
        return 0
    jump_types = pd.to_numeric(frame["type"], errors="coerce")
    success = pd.to_numeric(frame[success_column], errors="coerce")
    return int((jump_types.isin([0, 1, 2, 3, 4, 5]) & success.isin([0, 1])).sum())


def _safe_numeric(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
