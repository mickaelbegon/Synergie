from __future__ import annotations

import csv
import itertools
import json
import os
import random
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import constants
from synergie import pretrained_models
from synergie import session_store


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


def list_sessions() -> list[str]:
    constants.sessions = session_store.load_sessions()
    return sorted(constants.sessions)


def session_metadata(session_name: str) -> dict:
    constants.sessions = session_store.load_sessions()
    return constants.get_session(session_name)


def add_session(session_name: str, path: str, sample_time_fine_synchro: int) -> dict:
    metadata = session_store.add_session(session_name, path, sample_time_fine_synchro)
    constants.sessions = session_store.load_sessions()
    return metadata


def session_synchro(session_name: str) -> int:
    return int(session_metadata(session_name)["sample_time_fine_synchro"])


def list_session_csv_files(session_name: str, raw_root: str = "data/raw") -> list[Path]:
    metadata = session_metadata(session_name)
    session_dir = Path(raw_root) / metadata["path"]
    if not session_dir.exists():
        return []
    return sorted(session_dir.glob("*.csv"))


def list_directory_files(directory: str | Path) -> list[Path]:
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return []
    return sorted(path for path in directory_path.iterdir() if path.is_file())


def session_directory(session_name: str, raw_root: str = "data/raw") -> Path:
    return Path(raw_root) / session_metadata(session_name)["path"]


def describe_file(path: str | Path) -> dict:
    file_path = Path(path)
    stat = file_path.stat()
    return {
        "path": file_path,
        "name": file_path.name,
        "suffix": file_path.suffix.lower(),
        "size_bytes": stat.st_size,
    }


def next_jumplist_output_path(directory: str | Path) -> Path:
    directory_path = Path(directory)
    index = 1
    while True:
        candidate = directory_path / f"jumplist_partie{index}.csv"
        if not candidate.exists():
            return candidate
        index += 1


def annotation_turn_options(jump_type: str | int | None) -> list[str]:
    if jump_type is None:
        return ["1", "2", "3", "4"]
    return ["1", "2", "3", "4"]


def annotation_turn_value_for_storage(jump_type: str | int | None, ui_turn_value: str | int | float | None) -> str:
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
    if stored_turn_value is None or stored_turn_value != stored_turn_value:
        return ""
    normalized_type = "" if jump_type is None else str(jump_type).strip().lower()
    normalized_turn = str(stored_turn_value).strip()
    if not normalized_turn:
        return ""
    numeric_value = _safe_float(normalized_turn, default=float("nan"))
    if numeric_value != numeric_value:
        return normalized_turn
    if normalized_type in {"5", "axel"}:
        return str(int(numeric_value))
    return str(int(numeric_value))


def annotation_review_status_from_row(row) -> str:
    detection_status = str(row.get("detection_status", "detected_jump"))
    video_status = str(row.get("video_status", "visible"))
    if detection_status == "manual_missing_jump":
        return "manual_missing_jump"
    if detection_status == "not_a_jump":
        return "not_a_jump"
    if video_status == "not_seen_on_video":
        return "not_seen_on_video"
    return "normal"


def annotation_review_status_to_backend(review_status: str) -> dict:
    normalized = str(review_status or "normal")
    if normalized == "manual_missing_jump":
        return {"video_status": "visible", "detection_status": "manual_missing_jump"}
    if normalized == "not_a_jump":
        return {"video_status": "visible", "detection_status": "not_a_jump"}
    if normalized == "not_seen_on_video":
        return {"video_status": "not_seen_on_video", "detection_status": "detected_jump"}
    return {"video_status": "visible", "detection_status": "detected_jump"}


def annotation_metadata_path(annotation_csv_path: str | Path) -> Path:
    path = Path(annotation_csv_path)
    return path.with_suffix(".annotation_meta.json")


def load_annotation_metadata(annotation_csv_path: str | Path) -> dict:
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
    metadata = load_annotation_metadata(annotation_csv_path)
    metadata["video_path"] = str(Path(video_path))
    save_annotation_metadata(annotation_csv_path, metadata)
    return metadata


def set_annotation_sensor_sync_offset(annotation_csv_path: str | Path, sensor_id: str | int, offset_ms: float) -> dict:
    metadata = load_annotation_metadata(annotation_csv_path)
    offsets = dict(metadata.get("sensor_sync_offsets_ms", {}))
    offsets[str(sensor_id)] = round(float(offset_ms), 3)
    metadata["sensor_sync_offsets_ms"] = offsets
    save_annotation_metadata(annotation_csv_path, metadata)
    return metadata


def set_annotation_video_directory(annotation_csv_path: str | Path, video_directory: str | Path) -> dict:
    metadata = load_annotation_metadata(annotation_csv_path)
    metadata["video_directory"] = str(Path(video_directory))
    save_annotation_metadata(annotation_csv_path, metadata)
    return metadata


def get_annotation_sensor_sync_offset(metadata: dict | str | Path, sensor_id: str | int) -> float:
    loaded = load_annotation_metadata(metadata) if isinstance(metadata, (str, Path)) else metadata
    offsets = loaded.get("sensor_sync_offsets_ms", {})
    return float(offsets.get(str(sensor_id), 0.0))


def compute_annotation_jump_video_time_ms(row, sensor_sync_offset_ms: float = 0.0) -> float:
    base_ms = _safe_float(row.get("synced_start_ms", row.get("start_ms", 0.0)), default=0.0)
    return max(base_ms + float(sensor_sync_offset_ms), 0.0)


def list_video_files(directory: str | Path, recursive: bool = False) -> list[Path]:
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return []
    suffixes = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    iterator = directory_path.rglob("*") if recursive else directory_path.iterdir()
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in suffixes
    )


def annotation_reference_datetime(annotation_csv_path: str | Path, annotation_rows=None) -> datetime | None:
    if annotation_rows is not None:
        candidate_values: list[datetime] = []
        for value in annotation_rows.get("recorded_at", []):
            parsed = _parse_datetime_value(value)
            if parsed is not None:
                candidate_values.append(parsed)
        if candidate_values:
            return min(candidate_values)

    for value in Path(annotation_csv_path).stem.split("_"):
        parsed = _parse_datetime_from_text(value)
        if parsed is not None:
            return parsed

    parsed = _parse_datetime_from_text(Path(annotation_csv_path).stem)
    if parsed is not None:
        return parsed
    return None


def read_video_metadata(video_path: str | Path) -> dict:
    path = Path(video_path)
    stat = path.stat()
    candidates: list[tuple[str, datetime]] = []

    ffprobe_datetime = _read_video_creation_time_with_ffprobe(path)
    if ffprobe_datetime is not None:
        candidates.append(("ffprobe.creation_time", ffprobe_datetime))

    filename_datetime = _parse_datetime_from_text(path.stem)
    if filename_datetime is not None:
        candidates.append(("filename", filename_datetime))

    created_datetime = datetime.fromtimestamp(stat.st_ctime)
    modified_datetime = datetime.fromtimestamp(stat.st_mtime)
    candidates.append(("filesystem.modified", modified_datetime))
    candidates.append(("filesystem.created", created_datetime))

    best_source, best_datetime = _select_best_video_datetime(candidates)
    return {
        "path": path,
        "name": path.name,
        "recorded_at": best_datetime,
        "recorded_at_source": best_source,
        "candidates": [{"source": source, "recorded_at": value} for source, value in candidates],
    }


def find_matching_videos(
    annotation_csv_path: str | Path,
    video_directory: str | Path,
    *,
    annotation_rows=None,
    recursive: bool = True,
    limit: int = 5,
) -> dict:
    reference_datetime = annotation_reference_datetime(annotation_csv_path, annotation_rows=annotation_rows)
    videos = list_video_files(video_directory, recursive=recursive)
    matches: list[dict] = []
    for video_path in videos:
        metadata = read_video_metadata(video_path)
        delta_seconds = None
        if reference_datetime is not None:
            delta_seconds = abs((metadata["recorded_at"] - reference_datetime).total_seconds())
        matches.append(
            {
                "path": metadata["path"],
                "name": metadata["name"],
                "recorded_at": metadata["recorded_at"],
                "recorded_at_source": metadata["recorded_at_source"],
                "delta_seconds": delta_seconds,
            }
        )

    matches.sort(
        key=lambda item: (
            float("inf") if item["delta_seconds"] is None else item["delta_seconds"],
            item["recorded_at"],
            item["name"].lower(),
        )
    )
    return {
        "reference_datetime": reference_datetime,
        "video_directory": Path(video_directory),
        "matches": matches[:limit],
        "match_count": len(matches),
    }


def parse_new_imu_filename(path: str | Path) -> dict:
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
    return f"{metadata['date_token']}_{metadata['time_token']}"


def list_new_data_directories(root: str | Path = "data/new") -> list[dict]:
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
    selected_metadata = parse_new_imu_filename(selected_file)
    session_key = new_imu_session_key(selected_metadata)
    for session in list_new_imu_sessions(root=root):
        if session["session_key"] == session_key:
            return session["files"]
    return [selected_metadata]


def suggest_for_annotation_output_path(raw_csv_path: str | Path, pending_root: str | Path = "data/pending") -> Path:
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


def parse_training_id_from_csv_path(csv_path: Path) -> str | None:
    training_parts = csv_path.stem.split("_")
    if len(training_parts) < 2:
        return None
    return training_parts[1]


def describe_training_dataset(task: str, dataset_path: str, augment_mirror: bool = True) -> dict:
    dataset_root = Path(dataset_path)
    filtered: list[dict] = []
    with (dataset_root / "jumplist.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            jump_type = int(float(row["type"]))
            success = int(float(row["success"]))
            if success == 2 or jump_type == 8:
                continue
            filtered.append({"type": jump_type, "success": success, "skater": row["skater"]})

    if task == "type":
        labels = [row["type"] for row in filtered]
    elif task == "success":
        labels = [row["success"] for row in filtered]
    else:
        raise ValueError(f"Unsupported training task: {task}")

    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    base_samples = int(len(filtered))
    recommended_class_weight = {
        str(label): round(base_samples / (len(counts) * count), 6)
        for label, count in sorted(counts.items())
        if count > 0
    }

    validation_samples = base_samples - int(base_samples * 0.8)
    stratified_split_possible = (
        bool(counts)
        and len(counts) >= 2
        and min(counts.values()) >= 2
        and validation_samples >= len(counts)
    )

    return {
        "task": task,
        "base_samples": base_samples,
        "effective_samples": int(base_samples * (2 if augment_mirror else 1)),
        "unique_skaters": len({row["skater"] for row in filtered}),
        "class_counts": {str(label): counts[label] for label in sorted(counts)},
        "recommended_class_weight": recommended_class_weight,
        "stratified_split_possible": stratified_split_possible,
        "augment_mirror": bool(augment_mirror),
    }


def analyze_jump_quality(
    dataset_path: str | Path,
    *,
    outlier_threshold: float = 3.5,
    saturation_threshold: float = 1900.0,
) -> dict:
    import numpy as np
    import pandas as pd

    jumplist_path = Path(dataset_path) / "jumplist.csv"
    if not jumplist_path.exists():
        raise FileNotFoundError(f"Unable to find jumplist.csv in {dataset_path}")

    jumplist = pd.read_csv(jumplist_path)
    records: list[dict] = []
    skipped = 0

    for row_index, row in jumplist.iterrows():
        jump_type = int(_safe_float(row.get("type", 8), default=8))
        success = int(_safe_float(row.get("success", 2), default=2))
        if jump_type == 8 or success == 2:
            skipped += 1
            continue

        segment_path = Path(str(row.get("path", "")))
        if not segment_path.exists():
            records.append(
                {
                    "row_index": int(row_index),
                    "path": str(segment_path).replace("\\", "/"),
                    "type": jump_type,
                    "type_label": JUMP_TYPE_LABELS.get(jump_type, str(jump_type)),
                    "skater": str(row.get("skater", "")),
                    "success": success,
                    "rotations": _safe_float(row.get("rotations", 0.0), default=0.0),
                    "duration_ms": 0.0,
                    "max_abs_gyr_x": 0.0,
                    "max_abs_acc_x": 0.0,
                    "max_abs_second_derivative": 0.0,
                    "missing_file": True,
                    "suspicious": True,
                    "reasons": ["missing_segment_file"],
                }
            )
            continue

        segment = pd.read_csv(segment_path)
        duration_ms = float(segment["ms"].iloc[-1] - segment["ms"].iloc[0]) if len(segment) > 1 else 0.0
        max_abs_gyr_x = float(segment["Gyr_X"].abs().max()) if "Gyr_X" in segment else 0.0
        max_abs_acc_x = float(segment["Acc_X"].abs().max()) if "Acc_X" in segment else 0.0
        max_abs_second_derivative = float(segment["X_gyr_second_derivative"].abs().max()) if "X_gyr_second_derivative" in segment else 0.0
        nan_ratio = float(segment.isna().mean().mean()) if not segment.empty else 0.0
        flat_signal = bool(segment["Gyr_X"].std() < 1e-6) if "Gyr_X" in segment and len(segment) > 1 else True
        reasons: list[str] = []
        if nan_ratio > 0.02:
            reasons.append("nan_ratio")
        if flat_signal:
            reasons.append("flat_signal")
        if max_abs_gyr_x >= saturation_threshold:
            reasons.append("gyro_saturation")

        records.append(
            {
                "row_index": int(row_index),
                "path": str(segment_path).replace("\\", "/"),
                "type": jump_type,
                "type_label": JUMP_TYPE_LABELS.get(jump_type, str(jump_type)),
                "skater": str(row.get("skater", "")),
                "success": success,
                "rotations": _safe_float(row.get("rotations", 0.0), default=0.0),
                "duration_ms": duration_ms,
                "max_abs_gyr_x": max_abs_gyr_x,
                "max_abs_acc_x": max_abs_acc_x,
                "max_abs_second_derivative": max_abs_second_derivative,
                "nan_ratio": nan_ratio,
                "missing_file": False,
                "suspicious": False,
                "reasons": reasons,
            }
        )

    if not records:
        return {
            "dataset_path": str(dataset_path),
            "total_labelled_jumps": 0,
            "skipped_rows": skipped,
            "suspicious_count": 0,
            "records": [],
            "suspicious_records": [],
            "type_summary": [],
        }

    features = ["duration_ms", "max_abs_gyr_x", "max_abs_acc_x", "max_abs_second_derivative", "rotations"]
    by_type: dict[int, list[dict]] = {}
    for record in records:
        by_type.setdefault(int(record["type"]), []).append(record)

    for jump_type, typed_records in by_type.items():
        if len(typed_records) < 4:
            continue
        for feature in features:
            values = np.array([float(record[feature]) for record in typed_records], dtype=float)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            scale = 1.4826 * mad if mad > 0 else 0.0
            for record, value in zip(typed_records, values):
                z_key = f"{feature}_robust_z"
                if scale == 0.0:
                    record[z_key] = 0.0
                    continue
                robust_z = abs((float(value) - median) / scale)
                record[z_key] = robust_z
                if robust_z >= outlier_threshold:
                    record["reasons"].append(f"{feature}_outlier")

    for record in records:
        deduped = []
        for reason in record["reasons"]:
            if reason not in deduped:
                deduped.append(reason)
        record["reasons"] = deduped
        record["suspicious"] = bool(deduped)

    suspicious_records = [record for record in records if record["suspicious"]]
    type_summary = []
    for jump_type in sorted(by_type):
        typed_records = by_type[jump_type]
        type_summary.append(
            {
                "type": jump_type,
                "label": JUMP_TYPE_LABELS.get(jump_type, str(jump_type)),
                "count": len(typed_records),
                "suspicious_count": sum(1 for record in typed_records if record["suspicious"]),
                "median_duration_ms": float(np.median([record["duration_ms"] for record in typed_records])),
                "median_rotations": float(np.median([record["rotations"] for record in typed_records])),
                "median_max_abs_gyr_x": float(np.median([record["max_abs_gyr_x"] for record in typed_records])),
            }
        )

    return {
        "dataset_path": str(dataset_path),
        "total_labelled_jumps": len(records),
        "skipped_rows": skipped,
        "suspicious_count": len(suspicious_records),
        "records": records,
        "suspicious_records": suspicious_records,
        "type_summary": type_summary,
    }


def list_pretrained_training_models(task: str | None = None, compatible_only: bool = False) -> list[dict]:
    return pretrained_models.list_pretrained_models(task=task, compatible_only=compatible_only)


def format_pretrained_model_label(model_entry: dict) -> str:
    performance = model_entry.get("performance") or {}
    accuracy = performance.get("test_accuracy")
    if accuracy is None:
        accuracy_text = "acc n/a"
    else:
        accuracy_text = f"acc {accuracy:.3f}"
    status = "compatible" if model_entry.get("compatible") and model_entry.get("path_exists") else "not compatible"
    return (
        f"{model_entry['label']} | {model_entry['architecture']} | "
        f"{accuracy_text} | {status}"
    )


def latest_model_path_for_task(task: str) -> str:
    """Return the active model alias used by inference for one task."""
    if task == "type":
        return constants.modeltype_filepath
    if task == "success":
        return constants.modelsuccess_filepath
    raise ValueError(f"Unsupported training task: {task}")


def summarize_annotation_progress(annotation_rows) -> dict:
    """
    Summarize pending versus completed annotation rows.

    New annotation exports carry an explicit ``annotation_status`` column. Older
    files may not, so the fallback treats rows with the default placeholder
    labels as pending and every other row as completed.
    """
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


def summarize_pending_annotation_files(root: str | Path = "data/pending") -> dict:
    """Return per-file and global pending annotation counts."""
    import pandas as pd

    file_summaries: list[dict] = []
    global_summary = {"total": 0, "pending": 0, "completed": 0}
    for path in list_pending_annotation_files(root):
        frame = pd.read_csv(path)
        summary = summarize_annotation_progress(frame)
        file_summaries.append({"path": path, **summary})
        for key in global_summary:
            global_summary[key] += summary[key]
    return {"files": file_summaries, **global_summary}


def annotate_combination_flags(annotation_rows, max_gap_ms: float = 1500.0):
    """
    Mark jumps occurring close together on the same sensor as combinations.

    The first jump in a pair remains marked too, so a reviewer can see the whole
    combination rather than only the second element.
    """
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


def list_pretrained_models_by_performance(task: str) -> list[dict]:
    """Return compatible models ordered by reported test accuracy."""
    models = list_pretrained_training_models(task=task, compatible_only=True)
    return sorted(
        models,
        key=lambda item: (
            item.get("performance", {}).get("test_accuracy") is not None,
            item.get("performance", {}).get("test_accuracy") or -1.0,
        ),
        reverse=True,
    )


def suggest_turns_from_rotation(rotation_value) -> str:
    """Convert a measured absolute rotation into a conservative 1-4 turn guess."""
    turns = int(round(_safe_float(rotation_value, default=1.0)))
    return str(min(4, max(1, turns)))


def prefill_annotation_predictions(
    annotation_rows,
    *,
    type_model_path: str,
    success_model_path: str,
) -> dict:
    """
    Predict type/success labels for annotation rows and prefill turn guesses.

    Annotation candidates are not yet linked to a validated athlete profile, so
    this V1 uses neutral scalar inputs. These predictions are intended as review
    aids only; the human annotation remains authoritative.
    """
    import numpy as np
    import pandas as pd
    from core.model import model
    from synergie.config import SUCCESS_WINDOW_START, TYPE_WINDOW_FRAMES

    frame = annotation_rows.copy()
    if frame.empty:
        return {"rows": frame, "updated": 0, "skipped": 0}

    type_predictor = model.load_model(type_model_path, for_training=False)
    success_predictor = model.load_model(success_model_path, for_training=False)
    type_temporal = []
    success_temporal = []
    valid_indices = []
    for index, row in frame.iterrows():
        path = Path(str(row.get("path", "")))
        if not path.exists():
            continue
        segment = pd.read_csv(path)
        if len(segment) < TYPE_WINDOW_FRAMES:
            continue
        temporal = np.nan_to_num(
            segment[constants.fields_to_keep].to_numpy(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        type_temporal.append(temporal[:TYPE_WINDOW_FRAMES])
        success_temporal.append(temporal[SUCCESS_WINDOW_START:])
        valid_indices.append(index)

    if not valid_indices:
        return {"rows": frame, "updated": 0, "skipped": int(len(frame))}

    scalar_features = np.zeros((len(valid_indices), 2), dtype=float)
    type_predictions = type_predictor.predict(
        {"temporal_input": np.asarray(type_temporal), "scalar_input": scalar_features},
        verbose=0,
    )
    success_predictions = success_predictor.predict(
        {"temporal_input": np.asarray(success_temporal), "scalar_input": scalar_features},
        verbose=0,
    )
    for row_position, index in enumerate(valid_indices):
        frame.at[index, "type"] = int(np.argmax(type_predictions[row_position]))
        frame.at[index, "success"] = int(np.argmax(success_predictions[row_position]))
        frame.at[index, "turns"] = suggest_turns_from_rotation(frame.at[index, "rotations"])
        frame.at[index, "prediction_source"] = "batch_model_prefill"

    return {
        "rows": frame,
        "updated": len(valid_indices),
        "skipped": int(len(frame) - len(valid_indices)),
    }


def analyze_detection_review_labels(root: str | Path = "data/pending") -> dict:
    """Collect reviewed false positives and false negatives across annotation files."""
    import pandas as pd

    false_positives: list[dict] = []
    false_negatives: list[dict] = []
    reviewed_detected = 0
    for path in list_pending_annotation_files(root):
        frame = pd.read_csv(path)
        for index, row in frame.iterrows():
            status = annotation_review_status_from_row(row)
            record = {
                "annotation_file": str(path),
                "row_index": int(index),
                "sensor_id": str(row.get("sensor_id", "")),
                "athlete_id": str(row.get("athlete_id", row.get("skater", ""))),
                "path": str(row.get("path", "")),
                "start_ms": _safe_float(row.get("start_ms", 0.0)),
                "synced_start_ms": _safe_float(row.get("synced_start_ms", row.get("start_ms", 0.0))),
            }
            if status == "not_a_jump":
                false_positives.append(record)
            elif status == "manual_missing_jump":
                false_negatives.append(record)
            elif status == "normal":
                reviewed_detected += 1
    total_reviewed = reviewed_detected + len(false_positives) + len(false_negatives)
    return {
        "reviewed_detected": reviewed_detected,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "total_reviewed": total_reviewed,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def audit_saved_models() -> list[dict]:
    """
    Inspect active and registered models under the current Python environment.

    The audit distinguishes Keras 3 training-compatible files from legacy
    TensorFlow SavedModel directories. Legacy directories can still be used for
    inference through ``LegacySavedModelPredictor`` but cannot be resumed for
    training with Keras 3.
    """
    from core.model import model

    candidates: list[dict] = [
        {"label": "Active type model", "task": "type", "path": latest_model_path_for_task("type")},
        {"label": "Active success model", "task": "success", "path": latest_model_path_for_task("success")},
        {"label": "Legacy type model alias", "task": "type", "path": str(Path(latest_model_path_for_task("type")).with_suffix(""))},
        {"label": "Legacy success model alias", "task": "success", "path": str(Path(latest_model_path_for_task("success")).with_suffix(""))},
    ]
    candidates.extend(
        {
            "label": entry["label"],
            "task": entry["task"],
            "architecture": entry["architecture"],
            "path": entry["path"],
            "registered": True,
        }
        for entry in list_pretrained_training_models()
    )

    audited: list[dict] = []
    seen_paths: set[str] = set()
    for candidate in candidates:
        path = str(candidate["path"]).replace("\\", "/")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        model_path = Path(path)
        item = dict(candidate)
        item["path"] = path
        item["exists"] = model_path.exists()
        item["format"] = _model_format(model_path)
        item["training_compatible"] = pretrained_models.supports_training_reload(model_path)
        item["can_infer"] = False
        item["can_resume_training"] = False
        item["input_shapes"] = []
        item["error"] = ""

        if not item["exists"]:
            item["error"] = "missing path"
            audited.append(item)
            continue

        try:
            loaded = model.load_model(path, for_training=False)
            item["can_infer"] = True
            item["input_shapes"] = [
                tuple(dimension if dimension is None else int(dimension) for dimension in model_input.shape)
                for model_input in getattr(loaded, "inputs", [])
            ]
        except Exception as exc:
            item["error"] = f"inference load failed: {exc}"

        if item["training_compatible"]:
            try:
                model.load_model(path, for_training=True)
                item["can_resume_training"] = True
            except Exception as exc:
                if item["error"]:
                    item["error"] += f" | training reload failed: {exc}"
                else:
                    item["error"] = f"training reload failed: {exc}"
        audited.append(item)
    return audited


def _model_format(path: Path) -> str:
    if path.is_file() and path.suffix.lower() == ".keras":
        return "keras_v3"
    if path.is_file() and path.suffix.lower() == ".h5":
        return "keras_h5"
    if path.is_dir() and (path / "saved_model.pb").exists():
        return "legacy_saved_model"
    return "unknown"


def compute_signal_importance(
    task: str,
    dataset_path: str,
    *,
    model_path: str | None = None,
    repeats: int = 3,
    temporal_windows: int = 6,
    random_seed: int = 42,
) -> dict:
    """
    Estimate which inputs matter using grouped permutation tests.

    Channel importance shuffles one full signal trace between validation
    samples, preserving realistic within-signal shapes while breaking the link
    to the label. Temporal importance shuffles all channels inside a time window
    to show when the model uses the sequence. The metric is balanced accuracy so
    the success task is not dominated by the majority class.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from core.model import model
    from core.model.training.loader import Loader

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if temporal_windows < 1:
        raise ValueError("temporal_windows must be at least 1")

    loader = Loader(dataset_path, augment_mirror=False)
    dataset = loader.get_type_data() if task == "type" else loader.get_success_data()
    temporal = np.asarray(dataset.temporal_features_test)
    scalar = np.asarray(dataset.scalar_features_test)
    labels = np.asarray(dataset.labels_test)
    if len(temporal) < 2:
        raise ValueError("At least two validation samples are required for permutation importance.")

    predictor = model.load_model(model_path or latest_model_path_for_task(task), for_training=False)
    true_labels = np.argmax(labels, axis=1)
    baseline_predictions = _predict_class_labels(predictor, temporal, scalar)
    baseline_accuracy = float(accuracy_score(true_labels, baseline_predictions))
    baseline_balanced_accuracy = float(balanced_accuracy_score(true_labels, baseline_predictions))
    rng = np.random.default_rng(random_seed)

    channel_importance = []
    for channel_index, channel_name in enumerate(constants.fields_to_keep):
        drops = []
        for _ in range(repeats):
            permuted = temporal.copy()
            permuted[:, :, channel_index] = permuted[rng.permutation(len(permuted)), :, channel_index]
            score = balanced_accuracy_score(true_labels, _predict_class_labels(predictor, permuted, scalar))
            drops.append(float(baseline_balanced_accuracy - score))
        channel_importance.append(_summarize_importance(channel_name, drops))

    scalar_importance = []
    for scalar_index, scalar_name in enumerate(("weight", "height")):
        drops = []
        for _ in range(repeats):
            permuted_scalar = scalar.copy()
            permuted_scalar[:, scalar_index] = permuted_scalar[rng.permutation(len(permuted_scalar)), scalar_index]
            score = balanced_accuracy_score(true_labels, _predict_class_labels(predictor, temporal, permuted_scalar))
            drops.append(float(baseline_balanced_accuracy - score))
        scalar_importance.append(_summarize_importance(scalar_name, drops))

    temporal_importance = []
    for window_index, indices in enumerate(np.array_split(np.arange(temporal.shape[1]), temporal_windows), start=1):
        if len(indices) == 0:
            continue
        drops = []
        for _ in range(repeats):
            permuted = temporal.copy()
            permuted[:, indices, :] = permuted[rng.permutation(len(permuted))][:, indices, :]
            score = balanced_accuracy_score(true_labels, _predict_class_labels(predictor, permuted, scalar))
            drops.append(float(baseline_balanced_accuracy - score))
        temporal_importance.append(
            {
                **_summarize_importance(f"window_{window_index}", drops),
                "start_frame": int(indices[0]),
                "end_frame": int(indices[-1]),
            }
        )

    channel_importance.sort(key=lambda item: item["mean_drop"], reverse=True)
    scalar_importance.sort(key=lambda item: item["mean_drop"], reverse=True)
    return {
        "task": task,
        "dataset_path": dataset_path,
        "model_path": model_path or latest_model_path_for_task(task),
        "validation_samples": int(len(temporal)),
        "baseline_accuracy": baseline_accuracy,
        "baseline_balanced_accuracy": baseline_balanced_accuracy,
        "channel_importance": channel_importance,
        "scalar_importance": scalar_importance,
        "temporal_importance": temporal_importance,
        "method": "grouped_permutation_balanced_accuracy",
    }


def _predict_class_labels(predictor, temporal, scalar):
    import numpy as np

    predictions = predictor.predict(
        {"temporal_input": temporal, "scalar_input": scalar},
        verbose=0,
    )
    return np.argmax(predictions, axis=1)


def _summarize_importance(label: str, drops: list[float]) -> dict:
    import numpy as np

    return {
        "label": label,
        "mean_drop": float(np.mean(drops)),
        "std_drop": float(np.std(drops)),
    }


def hyperparameter_search_space(task: str, architecture: str) -> list[dict]:
    """Return the bounded exploratory search space used by the GUI tuner."""
    if task == "success" and architecture == "tcn":
        return [
            {"filters": filters, "dropout": dropout, "learning_rate": learning_rate, "batch_size": batch_size}
            for filters, dropout, learning_rate, batch_size in itertools.product(
                [32, 64],
                [0.1, 0.2, 0.3],
                [0.00001, 0.00003, 0.0001],
                [16, 32],
            )
        ]
    if task == "success" and architecture == "lstm":
        return [
            {
                "first_units": first_units,
                "second_units": second_units,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
            for first_units, second_units, dropout, learning_rate, batch_size in itertools.product(
                [64, 128],
                [32, 64],
                [0.3, 0.4],
                [0.00001, 0.00003],
                [16, 32],
            )
        ]
    if task == "type" and architecture == "inceptiontime":
        return [
            {
                "filters": filters,
                "bottleneck_filters": bottleneck_filters,
                "modules": modules,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
            for filters, bottleneck_filters, modules, dropout, learning_rate, batch_size in itertools.product(
                [16, 32],
                [16, 32],
                [2, 3],
                [0.1, 0.2],
                [0.00001, 0.00003],
                [16, 32],
            )
        ]
    if task == "type" and architecture == "transformer":
        return [
            {
                "head_size": head_size,
                "num_heads": num_heads,
                "ff_dim": ff_dim,
                "num_transformer_blocks": blocks,
                "dropout": dropout,
                "mlp_dropout": 0.1,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
            for head_size, num_heads, ff_dim, blocks, dropout, learning_rate, batch_size in itertools.product(
                [64, 128],
                [2, 4],
                [32, 64],
                [2, 4],
                [0.2, 0.3],
                [0.00001, 0.00005],
                [16, 32],
            )
        ]
    raise ValueError(f"Unsupported search combination: task={task}, architecture={architecture}")


def sample_hyperparameter_trials(
    task: str,
    architecture: str,
    max_trials: int,
    *,
    random_seed: int = 42,
) -> list[dict]:
    """Choose a reproducible subset of the bounded search space."""
    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")
    candidates = hyperparameter_search_space(task, architecture)
    if max_trials >= len(candidates):
        return candidates
    return random.Random(random_seed).sample(candidates, k=max_trials)


def run_hyperparameter_search(
    task: str,
    dataset_path: str,
    architecture: str,
    *,
    max_trials: int = 6,
    epochs: int = 8,
    random_seed: int = 42,
) -> dict:
    """
    Run a compact exploratory hyperparameter search on the validation split.

    This is intentionally a practical first pass rather than a publication
    protocol: it ranks bounded candidates on the current validation split, uses
    early stopping, and does not claim an unbiased final test estimate.
    """
    import keras
    import numpy as np
    from core.model import model
    from core.model.training.loader import Loader

    loader = Loader(dataset_path, augment_mirror=True)
    dataset = loader.get_type_data() if task == "type" else loader.get_success_data()
    trials = sample_hyperparameter_trials(task, architecture, max_trials, random_seed=random_seed)
    results: list[dict] = []
    validation_data = (
        {"temporal_input": dataset.temporal_features_test, "scalar_input": dataset.scalar_features_test},
        dataset.labels_test,
    )

    for trial_index, parameters in enumerate(trials, start=1):
        keras.backend.clear_session()
        build_parameters = dict(parameters)
        batch_size = int(build_parameters.pop("batch_size"))
        model_instance = model.build_model(task, architecture, **build_parameters)
        history = model_instance.fit(
            {"temporal_input": dataset.temporal_features_train, "scalar_input": dataset.scalar_features_train},
            dataset.labels_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            class_weight=(dataset.class_weight or None) if task == "success" else None,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    mode="max",
                    patience=3,
                    restore_best_weights=True,
                )
            ],
        )
        history_data = history.history
        results.append(
            {
                "trial": trial_index,
                "parameters": parameters,
                "best_val_accuracy": float(max(history_data.get("val_accuracy", [0.0]))),
                "final_val_accuracy": float(history_data.get("val_accuracy", [0.0])[-1]),
                "epochs_ran": int(len(history_data.get("loss", []))),
            }
        )

    results.sort(key=lambda item: item["best_val_accuracy"], reverse=True)
    return {
        "task": task,
        "architecture": architecture,
        "dataset_path": dataset_path,
        "max_trials": int(max_trials),
        "epochs": int(epochs),
        "objective": "best_val_accuracy",
        "results": results,
        "best_trial": results[0] if results else None,
        "note": "Exploratory validation search only; use grouped cross-validation before publication.",
    }


def train_model(
    task: str,
    dataset_path: str,
    epochs: int | None = None,
    architecture: str | None = None,
    pretrained_model_id: str | None = None,
    model_overrides: dict | None = None,
    batch_size: int | None = None,
) -> dict:
    from core.model import model
    from core.model.training.loader import Loader
    from core.model.training.training import Trainer

    dataset = Loader(dataset_path, augment_mirror=True)
    pretrained_entry = None
    if pretrained_model_id:
        pretrained_entry = pretrained_models.get_pretrained_model(pretrained_model_id)
        if pretrained_entry["task"] != task:
            raise ValueError(f"Pretrained model '{pretrained_model_id}' is not compatible with task '{task}'.")
        if not pretrained_entry.get("compatible") or not pretrained_entry.get("path_exists"):
            raise ValueError(f"Pretrained model '{pretrained_model_id}' is not currently compatible.")

    selected_architecture = architecture or (
        pretrained_entry["architecture"] if pretrained_entry else model.default_architecture(task)
    )
    run_path = pretrained_models.build_trained_model_path(task, selected_architecture)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    if task == "type":
        model_instance = (
            model.load_model(pretrained_entry["path"], for_training=True)
            if pretrained_entry
            else model.build_model(task, selected_architecture, **(model_overrides or {}))
        )
        trainer = Trainer(
            dataset.get_type_data(),
            model_instance,
            str(run_path),
        )
        summary = trainer.train(epochs=epochs or 10, batch_size=batch_size)
        latest_path = Path(constants.modeltype_filepath)
        _promote_trained_model(run_path, latest_path)
        registered = pretrained_models.register_trained_model(
            model_id=run_path.stem,
            label=f"Type {selected_architecture} {run_path.stem}",
            task=task,
            architecture=selected_architecture,
            path=str(run_path).replace("\\", "/"),
            dataset=dataset_path,
            performance=summary,
            notes="Automatically registered after GUI/CLI training.",
        )
        summary["saved_model"] = registered
        summary["latest_model_path"] = str(latest_path).replace("\\", "/")
        summary["task"] = task
        return summary

    if task == "success":
        model_instance = (
            model.load_model(pretrained_entry["path"], for_training=True)
            if pretrained_entry
            else model.build_model(task, selected_architecture, **(model_overrides or {}))
        )
        trainer = Trainer(
            dataset.get_success_data(),
            model_instance,
            str(run_path),
        )
        summary = trainer.train_success(epochs=epochs or 20, batch_size=batch_size)
        latest_path = Path(constants.modelsuccess_filepath)
        _promote_trained_model(run_path, latest_path)
        registered = pretrained_models.register_trained_model(
            model_id=run_path.stem,
            label=f"Success {selected_architecture} {run_path.stem}",
            task=task,
            architecture=selected_architecture,
            path=str(run_path).replace("\\", "/"),
            dataset=dataset_path,
            performance=summary,
            notes="Automatically registered after GUI/CLI training.",
        )
        summary["saved_model"] = registered
        summary["latest_model_path"] = str(latest_path).replace("\\", "/")
        summary["task"] = task
        return summary

    raise ValueError(f"Unsupported training task: {task}")


def _promote_trained_model(source_path: Path, latest_path: Path) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    if latest_path.is_dir():
        shutil.rmtree(latest_path, onerror=_handle_remove_readonly)
    elif latest_path.exists():
        latest_path.unlink()
    shutil.copy2(source_path, latest_path)


def _handle_remove_readonly(function, path, _excinfo) -> None:
    os.chmod(path, 0o700)
    function(path)


def process_csv_file(csv_path: str, synchro: int = 0, output_path: str | None = None) -> Path:
    from core.data_treatment.data_generation.exporter import export
    import pandas as pd

    input_path = Path(csv_path)
    dataframe = pd.read_csv(input_path)
    result = export(dataframe, sampleTimeFineSynchro=synchro)
    destination = Path(output_path) if output_path else input_path.with_name(f"{input_path.stem}_jumps.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return destination


def process_new_imu_file_for_annotation(
    raw_csv_path: str | Path,
    *,
    output_path: str | Path | None = None,
    pending_root: str | Path = "data/pending",
    sample_time_fine_synchro: int = 0,
) -> dict:
    return process_new_imu_session_for_annotation(
        raw_csv_path,
        output_path=output_path,
        pending_root=pending_root,
        sample_time_fine_synchro=sample_time_fine_synchro,
    )


def process_new_imu_session_for_annotation(
    raw_csv_path: str | Path,
    *,
    output_path: str | Path | None = None,
    pending_root: str | Path = "data/pending",
    sample_time_fine_synchro: int = 0,
) -> dict:
    import pandas as pd
    from core.data_treatment.data_generation.trainingSession import trainingSession

    raw_path = Path(raw_csv_path)
    metadata = parse_new_imu_filename(raw_path)
    pending_root_path = Path(pending_root)
    output_csv_path = Path(output_path) if output_path else suggest_for_annotation_output_path(raw_path, pending_root=pending_root_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    session_files = files_for_new_imu_session(raw_path)
    segment_root = pending_root_path / "segments" / new_imu_session_key(metadata)
    segment_root.mkdir(parents=True, exist_ok=True)

    for file_metadata in session_files:
        sensor_segment_root = segment_root / f"sensor{file_metadata['sensor_id']}"
        sensor_segment_root.mkdir(parents=True, exist_ok=True)
        dataframe = pd.read_csv(file_metadata["path"])
        session = trainingSession(dataframe, sampleTimefineSynchro=sample_time_fine_synchro)
        impact_offset_ms = estimate_sensor_impact_offset_ms(session.df)

        for jump_index, jump in enumerate(session.jumps, start=1):
            segment_name = (
                f"{file_metadata['date_token']}_{file_metadata['time_token']}_sensor{file_metadata['sensor_id']}_"
                f"jump{jump_index:03d}.csv"
            )
            segment_path = sensor_segment_root / segment_name
            jump.df.to_csv(segment_path, index=False)
            synced_start_ms = round(jump.startTimestamp - impact_offset_ms, 3)
            records.append(
                {
                    "path": str(segment_path).replace("\\", "/"),
                    "videoTimeStamp": _ms_to_timestamp(max(synced_start_ms, 0.0)),
                    "type": 8,
                    "turns": "",
                    "skater": f"sensor_{file_metadata['sensor_id']}",
                    "athlete_id": f"sensor_{file_metadata['sensor_id']}",
                    "success": 2,
                    "rotations": round(jump.rotation, 1),
                    "source_file": file_metadata["path"].name,
                    "sensor_id": file_metadata["sensor_id"],
                    "device_id": file_metadata["device_id"],
                    "recorded_at": file_metadata["recorded_at"].isoformat(),
                    "annotation_status": "pending",
                    "video_status": "visible",
                    "detection_status": "detected_jump",
                    "impact_offset_ms": round(impact_offset_ms, 3),
                    "start_ms": round(jump.startTimestamp, 3),
                    "end_ms": round(jump.endTimestamp, 3),
                    "synced_start_ms": synced_start_ms,
                    "session_key": new_imu_session_key(file_metadata),
                }
            )

    annotation_frame = pd.DataFrame(records)
    if not annotation_frame.empty:
        annotation_frame = annotation_frame.sort_values(by=["synced_start_ms", "sensor_id", "start_ms"]).reset_index(drop=True)
        annotation_frame = annotate_combination_flags(annotation_frame)
    annotation_frame.to_csv(output_csv_path, index=False)
    return {
        "annotation_csv": output_csv_path,
        "segment_directory": segment_root,
        "jump_count": len(records),
        "source_file": raw_path,
        "session_key": new_imu_session_key(metadata),
        "sensor_count": len(session_files),
    }


_TEXT_DATETIME_PATTERNS = (
    re.compile(r"(?P<date>\d{8})(?P<time>\d{6})"),
    re.compile(r"(?P<date>\d{8})(?P<time>\d{4})(?!\d)"),
    re.compile(r"(?P<date>\d{8})[_-]?(?P<time>\d{6})"),
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})[_T -]?(?P<time>\d{2}[-:]\d{2}[-:]\d{2})"),
)


def _parse_datetime_value(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return _parse_datetime_from_text(text)


def _parse_datetime_from_text(text: str) -> datetime | None:
    for pattern in _TEXT_DATETIME_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        date_token = match.group("date").replace("-", "")
        time_token = match.group("time").replace("-", "").replace(":", "")
        if len(time_token) == 4:
            time_token = f"{time_token}00"
        try:
            return datetime.strptime(f"{date_token}_{time_token}", "%Y%m%d_%H%M%S")
        except ValueError:
            continue
    return None


def _read_video_creation_time_with_ffprobe(video_path: Path) -> datetime | None:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_entries",
                "format_tags=creation_time",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    creation_time = (((payload.get("format") or {}).get("tags") or {}).get("creation_time"))
    return _parse_datetime_value(creation_time)


def _select_best_video_datetime(candidates: list[tuple[str, datetime]]) -> tuple[str, datetime]:
    if not candidates:
        raise ValueError("No datetime candidates available for video metadata.")
    priority = {
        "ffprobe.creation_time": 0,
        "filename": 1,
        "filesystem.modified": 2,
        "filesystem.created": 3,
    }
    return min(
        candidates,
        key=lambda item: (
            priority.get(item[0], 99),
            item[1],
        ),
    )


def _ms_to_timestamp(ms: float) -> str:
    total_seconds = round(ms / 1000)
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def estimate_sensor_impact_offset_ms(session_df) -> float:
    import numpy as np

    if session_df is None or session_df.empty:
        return 0.0
    acc_norm = np.sqrt(
        np.square(session_df["Acc_X"].to_numpy())
        + np.square(session_df["Acc_Y"].to_numpy())
        + np.square(session_df["Acc_Z"].to_numpy())
    )
    impact_signal = np.abs(np.diff(acc_norm, prepend=acc_norm[0]))
    search_window = min(len(impact_signal), 600)
    if search_window == 0:
        return 0.0
    impact_index = int(np.argmax(impact_signal[:search_window]))
    return float(session_df.iloc[impact_index]["ms"])


def list_pending_annotation_files(root: str | Path = "data/pending") -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(
        path for path in root_path.glob("*for_annotation*.csv")
        if path.is_file()
    )


def repredict_raw_trainings(raw_root: str = "data/raw") -> int:
    from core.data_treatment.data_generation.exporter import export
    from core.database.DatabaseManager import DatabaseManager
    import pandas as pd
    from synergie.services.jump_predictions import build_training_jump_payload

    raw_root_path = Path(raw_root)
    database = DatabaseManager()
    processed = 0

    for csv_path in sorted(raw_root_path.rglob("*.csv")):
        training_id = parse_training_id_from_csv_path(csv_path)
        if training_id is None:
            continue
        dataframe = pd.read_csv(csv_path)
        predictions = export(dataframe)
        training_jumps = build_training_jump_payload(predictions, training_id)
        database.add_jumps_to_training(training_id, training_jumps)
        processed += 1

    return processed
