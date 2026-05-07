from __future__ import annotations

import csv
import json
import os
import shutil
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


def get_annotation_sensor_sync_offset(metadata: dict | str | Path, sensor_id: str | int) -> float:
    loaded = load_annotation_metadata(metadata) if isinstance(metadata, (str, Path)) else metadata
    offsets = loaded.get("sensor_sync_offsets_ms", {})
    return float(offsets.get(str(sensor_id), 0.0))


def compute_annotation_jump_video_time_ms(row, sensor_sync_offset_ms: float = 0.0) -> float:
    base_ms = _safe_float(row.get("synced_start_ms", row.get("start_ms", 0.0)), default=0.0)
    return max(base_ms + float(sensor_sync_offset_ms), 0.0)


def list_video_files(directory: str | Path) -> list[Path]:
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return []
    suffixes = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    return sorted(
        path for path in directory_path.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


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
    return {
        "task": task,
        "base_samples": base_samples,
        "effective_samples": int(base_samples * (2 if augment_mirror else 1)),
        "unique_skaters": len({row["skater"] for row in filtered}),
        "class_counts": {str(label): counts[label] for label in sorted(counts)},
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


def train_model(
    task: str,
    dataset_path: str,
    epochs: int | None = None,
    architecture: str | None = None,
    pretrained_model_id: str | None = None,
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
            else model.build_model(task, selected_architecture)
        )
        trainer = Trainer(
            dataset.get_type_data(),
            model_instance,
            str(run_path),
        )
        summary = trainer.train(epochs=epochs or 10)
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
        return summary

    if task == "success":
        model_instance = (
            model.load_model(pretrained_entry["path"], for_training=True)
            if pretrained_entry
            else model.build_model(task, selected_architecture)
        )
        trainer = Trainer(
            dataset.get_success_data(),
            model_instance,
            str(run_path),
        )
        summary = trainer.train_success(epochs=epochs or 20)
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
                    "synced_start_ms": synced_start_ms,
                    "session_key": new_imu_session_key(file_metadata),
                }
            )

    annotation_frame = pd.DataFrame(records)
    if not annotation_frame.empty:
        annotation_frame = annotation_frame.sort_values(by=["synced_start_ms", "sensor_id", "start_ms"]).reset_index(drop=True)
    annotation_frame.to_csv(output_csv_path, index=False)
    return {
        "annotation_csv": output_csv_path,
        "segment_directory": segment_root,
        "jump_count": len(records),
        "source_file": raw_path,
        "session_key": new_imu_session_key(metadata),
        "sensor_count": len(session_files),
    }


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
