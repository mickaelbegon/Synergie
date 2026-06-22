from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from synergie.services.annotation_service import load_annotation_metadata
from synergie.services.data_inventory_service import build_data_inventory
from synergie.services.session_service import list_sessions, session_metadata


TRIAL_COLUMNS = [
    "path",
    "videoTimeStamp",
    "type",
    "turns",
    "success",
    "skater",
    "athlete_id",
    "sensor_id",
    "device_id",
    "recorded_at",
    "annotation_status",
    "video_status",
    "detection_status",
    "impact_offset_ms",
    "start_ms",
    "end_ms",
    "synced_start_ms",
    "rotations",
    "source_file",
    "session_key",
    "added_by",
    "manual_takeoff_video_ms",
    "manual_landing_video_ms",
]

TEXT_COLUMNS = {"sensor_id", "device_id", "skater", "athlete_id", "session_key", "source_file"}


def build_backup_manifest(
    *,
    new_root: str | Path = "data/new",
    pending_root: str | Path = "data/pending",
    annotated_root: str | Path = "data/annotated",
    training_dataset_root: str | Path = "data/annotated/total",
    generated_at: datetime | None = None,
) -> dict:
    """Build a human-readable project manifest for backup and audit workflows."""
    generated = generated_at or datetime.now(timezone.utc)
    pending_files = _pending_annotation_files(Path(pending_root))
    training_dataset = _training_dataset(Path(training_dataset_root))
    configured_sessions = _configured_sessions()
    inventory = build_data_inventory(
        new_root=new_root,
        pending_root=pending_root,
        annotated_root=annotated_root,
        training_dataset_root=training_dataset_root,
    )
    return {
        "format": "synergie-backup-manifest-v1",
        "generated_at": generated.isoformat(),
        "roots": {
            "new": _posix(new_root),
            "pending": _posix(pending_root),
            "annotated": _posix(annotated_root),
            "training_dataset": _posix(training_dataset_root),
        },
        "summary": {
            "configured_sessions": len(configured_sessions),
            "pending_annotation_files": len(pending_files),
            "pending_trials": sum(item["trial_count"] for item in pending_files),
            "training_trials": training_dataset["trial_count"],
            "trainable_training_trials": training_dataset["trainable_trial_count"],
        },
        "configured_sessions": configured_sessions,
        "inventory": inventory,
        "pending_annotation_files": pending_files,
        "training_dataset": training_dataset,
    }


def write_backup_manifest(manifest_path: str | Path = "data/backup_manifest.json", **kwargs) -> Path:
    """Write the backup manifest as indented JSON."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_backup_manifest(**kwargs)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return path


def _configured_sessions() -> list[dict]:
    sessions = []
    for session_id in list_sessions():
        metadata = session_metadata(session_id)
        sessions.append(
            {
                "session_id": session_id,
                "path": str(metadata.get("path", "")),
                "sample_time_fine_synchro": int(metadata.get("sample_time_fine_synchro", 0) or 0),
            }
        )
    return sessions


def _pending_annotation_files(pending_root: Path) -> list[dict]:
    import pandas as pd

    if not pending_root.exists():
        return []
    records = []
    for csv_path in sorted(pending_root.glob("*for_annotation*.csv")):
        frame = pd.read_csv(csv_path)
        metadata = load_annotation_metadata(csv_path)
        records.append(
            {
                "path": _posix(csv_path),
                "trial_count": int(len(frame)),
                "metadata": metadata,
                "trials": [_trial_from_row(row) for _, row in frame.iterrows()],
            }
        )
    return records


def _training_dataset(dataset_root: Path) -> dict:
    import pandas as pd

    jumplist_path = dataset_root / "jumplist.csv"
    if not jumplist_path.exists():
        return {"path": _posix(dataset_root), "jumplist": _posix(jumplist_path), "trial_count": 0, "trainable_trial_count": 0, "trials": []}
    frame = pd.read_csv(jumplist_path)
    trainable_count = 0
    if {"type", "success"}.issubset(frame.columns):
        types = pd.to_numeric(frame["type"], errors="coerce")
        successes = pd.to_numeric(frame["success"], errors="coerce")
        trainable_count = int((types.isin([0, 1, 2, 3, 4, 5]) & successes.isin([0, 1])).sum())
    return {
        "path": _posix(dataset_root),
        "jumplist": _posix(jumplist_path),
        "trial_count": int(len(frame)),
        "trainable_trial_count": trainable_count,
        "trials": [_trial_from_row(row) for _, row in frame.iterrows()],
    }


def _trial_from_row(row) -> dict:
    trial = {}
    for column in TRIAL_COLUMNS:
        if column in row.index:
            value = _json_value(row[column])
            trial[column] = None if value is None else str(value) if column in TEXT_COLUMNS else value
    return trial


def _json_value(value):
    if value is None or value != value:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Path):
        return _posix(value)
    return value


def _posix(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")
