from __future__ import annotations

import json
from pathlib import Path


ANNOTATION_JUMP_TYPE_OPTIONS = [
    ("toe_loop", "Toe loop", 0),
    ("flip", "Flip", 1),
    ("lutz", "Lutz", 2),
    ("salchow", "Salchow", 3),
    ("loop", "Loop", 4),
    ("axel", "Axel", 5),
]

ANNOTATION_TOE_JUMP_OPTIONS = [
    ("toe_loop", "Toe loop", 0),
    ("flip", "Flip", 1),
    ("lutz", "Lutz", 2),
]

ANNOTATION_EDGE_JUMP_OPTIONS = [
    ("salchow", "Salchow", 3),
    ("loop", "Loop", 4),
    ("axel", "Axel", 5),
]

ANNOTATION_REVIEW_STATUS_OPTIONS = [
    ("normal", "Seen jump"),
    ("not_seen_on_video", "Unseen on video"),
    ("weird_signal", "Weird signal / bad bounds"),
    ("not_a_jump", "Not a jump"),
    ("manual_missing_jump", "Missed jump added manually"),
]

ANNOTATION_METADATA_DEFAULTS = {
    "video_path": "",
    "video_directory": "",
    "sensor_sync_offsets_ms": {},
}

JUMP_TYPE_LABELS = {
    0: "Toe loop",
    1: "Flip",
    2: "Lutz",
    3: "Salchow",
    4: "Loop",
    5: "Axel",
    8: "Exclude",
}


def annotation_turn_options(jump_type: str | int | None) -> list[str]:
    """Return the UI turn choices for one jump type."""
    return ["1", "2", "3", "4"]


def annotation_turn_value_for_storage(jump_type: str | int | None, ui_turn_value: str | int | float | None) -> str:
    """Convert the UI turn value into the persisted label representation."""
    if ui_turn_value is None:
        return ""
    normalized_turn = str(ui_turn_value).strip()
    if not normalized_turn:
        return ""
    normalized_type = "" if jump_type is None else str(jump_type).strip().lower()
    if normalized_type in {"5", "axel"}:
        return f"{int(float(normalized_turn))}.5"
    return str(int(float(normalized_turn))) if "." in normalized_turn else normalized_turn


def annotation_turn_value_for_ui(jump_type: str | int | None, stored_turn_value) -> str:
    """Convert the persisted turn label back into the radio-button value."""
    if stored_turn_value is None or stored_turn_value != stored_turn_value:
        return ""
    normalized_turn = str(stored_turn_value).strip()
    if not normalized_turn:
        return ""
    numeric_value = _safe_float(normalized_turn, default=float("nan"))
    if numeric_value != numeric_value:
        return normalized_turn
    return str(int(numeric_value))


def annotation_review_status_from_row(row) -> str:
    """Map backend status columns to the review status shown by the GUI."""
    detection_status = str(row.get("detection_status", "detected_jump"))
    video_status = str(row.get("video_status", "visible"))
    if detection_status == "manual_missing_jump":
        return "manual_missing_jump"
    if detection_status == "weird_signal":
        return "weird_signal"
    if detection_status == "not_a_jump":
        return "not_a_jump"
    if video_status == "not_seen_on_video":
        return "not_seen_on_video"
    return "normal"


def annotation_review_status_to_backend(review_status: str) -> dict:
    """Map one GUI review choice to the persisted backend status columns."""
    normalized = str(review_status or "normal")
    if normalized == "manual_missing_jump":
        return {"video_status": "visible", "detection_status": "manual_missing_jump"}
    if normalized == "weird_signal":
        return {"video_status": "visible", "detection_status": "weird_signal"}
    if normalized == "not_a_jump":
        return {"video_status": "visible", "detection_status": "not_a_jump"}
    if normalized == "not_seen_on_video":
        return {"video_status": "not_seen_on_video", "detection_status": "detected_jump"}
    return {"video_status": "visible", "detection_status": "detected_jump"}


def annotation_metadata_path(annotation_csv_path: str | Path) -> Path:
    """Return the sidecar metadata path for one annotation CSV."""
    path = Path(annotation_csv_path)
    return path.with_suffix(".annotation_meta.json")


def load_annotation_metadata(annotation_csv_path: str | Path) -> dict:
    """Load annotation sidecar metadata with stable defaults."""
    metadata_path = annotation_metadata_path(annotation_csv_path)
    if not metadata_path.exists():
        return dict(ANNOTATION_METADATA_DEFAULTS)

    with metadata_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    metadata = dict(ANNOTATION_METADATA_DEFAULTS)
    metadata.update(loaded if isinstance(loaded, dict) else {})
    metadata["sensor_sync_offsets_ms"] = {
        str(sensor_id): float(offset_ms)
        for sensor_id, offset_ms in metadata.get("sensor_sync_offsets_ms", {}).items()
    }
    return metadata


def save_annotation_metadata(annotation_csv_path: str | Path, metadata: dict) -> Path:
    """Persist normalized annotation sidecar metadata."""
    metadata_path = annotation_metadata_path(annotation_csv_path)
    payload = dict(ANNOTATION_METADATA_DEFAULTS)
    payload.update(metadata)
    payload["sensor_sync_offsets_ms"] = {
        str(sensor_id): float(offset_ms)
        for sensor_id, offset_ms in payload.get("sensor_sync_offsets_ms", {}).items()
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return metadata_path


def set_annotation_video_path(annotation_csv_path: str | Path, video_path: str | Path) -> dict:
    """Store the chosen video path for one annotation file."""
    metadata = load_annotation_metadata(annotation_csv_path)
    metadata["video_path"] = str(Path(video_path))
    save_annotation_metadata(annotation_csv_path, metadata)
    return metadata


def set_annotation_sensor_sync_offset(annotation_csv_path: str | Path, sensor_id: str | int, offset_ms: float) -> dict:
    """Store one IMU-to-video sync offset."""
    metadata = load_annotation_metadata(annotation_csv_path)
    offsets = dict(metadata.get("sensor_sync_offsets_ms", {}))
    offsets[str(sensor_id)] = round(float(offset_ms), 3)
    metadata["sensor_sync_offsets_ms"] = offsets
    save_annotation_metadata(annotation_csv_path, metadata)
    return metadata


def set_annotation_video_directory(annotation_csv_path: str | Path, video_directory: str | Path) -> dict:
    """Store the preferred video directory for one annotation file."""
    metadata = load_annotation_metadata(annotation_csv_path)
    metadata["video_directory"] = str(Path(video_directory))
    save_annotation_metadata(annotation_csv_path, metadata)
    return metadata


def get_annotation_sensor_sync_offset(metadata: dict | str | Path, sensor_id: str | int) -> float:
    """Return one stored sensor sync offset in milliseconds."""
    loaded = load_annotation_metadata(metadata) if isinstance(metadata, (str, Path)) else metadata
    offsets = loaded.get("sensor_sync_offsets_ms", {})
    return float(offsets.get(str(sensor_id), 0.0))


def compute_annotation_jump_video_time_ms(row, sensor_sync_offset_ms: float = 0.0) -> float:
    """Return the synced video timestamp for one annotation row."""
    base_ms = _safe_float(row.get("synced_start_ms", row.get("start_ms", 0.0)), default=0.0)
    return max(base_ms + float(sensor_sync_offset_ms), 0.0)


def summarize_annotation_progress(annotation_rows) -> dict:
    """Summarize pending versus completed annotation rows."""
    total = int(len(annotation_rows))
    if total == 0:
        return {"total": 0, "pending": 0, "completed": 0}

    if "annotation_status" in annotation_rows:
        statuses = annotation_rows["annotation_status"].fillna("pending").astype(str).str.lower()
        pending = int((statuses == "pending").sum())
    else:
        pending = 0
        for _, row in annotation_rows.iterrows():
            jump_type = int(_safe_float(row.get("type", 8), default=8))
            success = int(_safe_float(row.get("success", 2), default=2))
            if jump_type == 8 and success == 2:
                pending += 1
    return {"total": total, "pending": pending, "completed": total - pending}


def annotate_combination_flags(annotation_rows, max_gap_ms: float = 1500.0):
    """Mark jumps occurring close together on the same sensor as combinations."""
    frame = annotation_rows.copy()
    if frame.empty:
        frame["combination"] = []
        return frame

    if "combination" not in frame:
        frame["combination"] = False
    else:
        frame["combination"] = frame["combination"].fillna(False).astype(bool)
    for _sensor_id, sensor_rows in frame.groupby("sensor_id", sort=False):
        ordered = sensor_rows.sort_values("synced_start_ms")
        previous_index = None
        previous_start = None
        for index, row in ordered.iterrows():
            current_start = _safe_float(row.get("synced_start_ms", row.get("start_ms", 0.0)))
            if previous_index is not None and current_start - previous_start < max_gap_ms:
                frame.at[previous_index, "combination"] = True
                frame.at[index, "combination"] = True
            previous_index = index
            previous_start = current_start
    return frame


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
