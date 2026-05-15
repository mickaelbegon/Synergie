from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import constants
from synergie import pretrained_models
from synergie import session_store
from synergie.services.annotation_service import (
    ANNOTATION_EDGE_JUMP_OPTIONS,
    ANNOTATION_JUMP_TYPE_OPTIONS,
    ANNOTATION_METADATA_DEFAULTS,
    ANNOTATION_REVIEW_STATUS_OPTIONS,
    ANNOTATION_TOE_JUMP_OPTIONS,
    JUMP_TYPE_LABELS,
    annotate_combination_flags,
    annotation_metadata_path,
    annotation_review_status_from_row,
    annotation_review_status_to_backend,
    annotation_turn_options,
    annotation_turn_value_for_storage,
    annotation_turn_value_for_ui,
    compute_annotation_jump_video_time_ms,
    get_annotation_sensor_sync_offset,
    load_annotation_metadata,
    save_annotation_metadata,
    set_annotation_sensor_sync_offset,
    set_annotation_video_directory,
    set_annotation_video_path,
    summarize_annotation_progress,
)
from synergie.services.hyperparameter_search_service import (
    hyperparameter_search_space,
    sample_hyperparameter_trials,
)
from synergie.services.detection_tuning_service import (
    analyze_detection_review_labels,
    optimize_detection_parameters,
)
from synergie.services.quality_service import analyze_jump_quality


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
    duplicate_report = find_training_dataset_duplicates(dataset_path)
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
        "duplicate_rows": duplicate_report["duplicate_rows"],
        "duplicate_paths": duplicate_report["duplicate_paths"],
        "has_duplicates": duplicate_report["has_duplicates"],
    }


def find_training_dataset_duplicates(dataset_path: str | Path) -> dict:
    """Detect duplicate training rows, primarily by repeated segment path."""
    import pandas as pd

    jumplist_path = Path(dataset_path) / "jumplist.csv"
    if not jumplist_path.exists():
        raise FileNotFoundError(f"Unable to find jumplist.csv in {dataset_path}")

    frame = pd.read_csv(jumplist_path)
    if "path" not in frame:
        return {
            "has_duplicates": False,
            "duplicate_paths": [],
            "duplicate_rows": 0,
            "rows": [],
        }
    normalized_paths = frame["path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    duplicate_mask = normalized_paths.duplicated(keep=False) & normalized_paths.ne("")
    duplicate_rows = frame.loc[duplicate_mask].copy()
    duplicate_rows["_normalized_path"] = normalized_paths[duplicate_mask]
    duplicate_paths = sorted(duplicate_rows["_normalized_path"].unique().tolist()) if not duplicate_rows.empty else []
    return {
        "has_duplicates": bool(duplicate_paths),
        "duplicate_paths": duplicate_paths,
        "duplicate_rows": int(duplicate_mask.sum()),
        "rows": duplicate_rows.to_dict(orient="records"),
    }


def training_dataset_state_path(dataset_path: str | Path = "data/annotated/total") -> Path:
    """Return the sidecar file used to record the last training-set update."""
    return Path(dataset_path) / "dataset_state.json"


def load_training_dataset_state(dataset_path: str | Path = "data/annotated/total") -> dict:
    path = training_dataset_state_path(dataset_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def finalize_annotation_file(
    annotation_csv_path: str | Path,
    *,
    dataset_path: str | Path = "data/annotated/total",
    annotated_root: str | Path = "data/annotated",
) -> dict:
    """
    Merge a fully reviewed annotation file into the training dataset.

    The operation validates the annotation status, archives the previous global
    jumplist, moves trainable segment files into the annotated tree, refuses
    duplicate paths, writes the merged jumplist, and records a dataset-change
    marker consumed by the GUI.
    """
    import pandas as pd

    annotation_path = Path(annotation_csv_path)
    dataset_root = Path(dataset_path)
    annotated_root_path = Path(annotated_root)
    jumplist_path = dataset_root / "jumplist.csv"
    if not jumplist_path.exists():
        raise FileNotFoundError(f"Unable to find jumplist.csv in {dataset_root}")

    annotation_rows = pd.read_csv(annotation_path)
    progress = summarize_annotation_progress(annotation_rows)
    if progress["pending"] > 0:
        raise ValueError(f"Cannot finalize: {progress['pending']} annotations are still pending.")

    trainable = annotation_rows[
        (annotation_rows["type"].astype(float).astype(int) != 8)
        & (annotation_rows["success"].astype(float).astype(int) != 2)
    ].copy()
    if trainable.empty:
        raise ValueError("Cannot finalize: no trainable labelled jumps are available.")

    existing = pd.read_csv(jumplist_path)
    session_key = str(trainable.iloc[0].get("session_key", annotation_path.stem.replace("_for_annotation", "")))
    destination_rows: list[dict] = []
    planned_moves: list[tuple[Path, Path]] = []
    for _, row in trainable.iterrows():
        source = Path(str(row["path"]))
        if not source.exists():
            raise FileNotFoundError(f"Missing annotated segment: {source}")
        sensor_id = str(row.get("sensor_id", "unknown"))
        destination = annotated_root_path / session_key / f"sensor{sensor_id}" / source.name
        destination_path = str(destination).replace("\\", "/")
        copied = row.to_dict()
        copied["path"] = destination_path
        copied["skater"] = copied.get("athlete_id", copied.get("skater", ""))
        destination_rows.append(copied)
        planned_moves.append((source, destination))

    destination_paths = [row["path"] for row in destination_rows]
    if len(destination_paths) != len(set(destination_paths)):
        raise ValueError("Cannot finalize: duplicate destination paths exist inside the annotation file.")
    existing_paths = set(existing.get("path", pd.Series(dtype=str)).fillna("").astype(str).str.replace("\\", "/", regex=False))
    duplicate_paths = sorted(path for path in destination_paths if path in existing_paths)
    if duplicate_paths:
        raise ValueError(f"Cannot finalize: paths already present in training jumplist: {', '.join(duplicate_paths[:3])}")
    existing_destinations = [destination for _source, destination in planned_moves if destination.exists()]
    if existing_destinations:
        raise ValueError(f"Cannot finalize: destination segment already exists: {existing_destinations[0]}")

    archive_dir = dataset_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_path = archive_dir / f"jumplist_{timestamp}.csv"
    shutil.copy2(jumplist_path, archive_path)

    for source, destination in planned_moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    merged = pd.concat([existing, pd.DataFrame(destination_rows)], ignore_index=True, sort=False)
    merged.to_csv(jumplist_path, index=False)
    duplicate_report = find_training_dataset_duplicates(dataset_root)
    if duplicate_report["has_duplicates"]:
        raise ValueError("Finalization created duplicate training rows; inspect the archived jumplist before continuing.")

    completed_dir = annotated_root_path / session_key
    completed_annotation_path = completed_dir / annotation_path.name.replace("_for_annotation", "_annotated")
    completed_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(annotation_path), str(completed_annotation_path))
    metadata_path = annotation_metadata_path(annotation_path)
    if metadata_path.exists():
        shutil.move(str(metadata_path), str(completed_annotation_path.with_suffix(".annotation_meta.json")))

    state = {
        "changed_at": datetime.now().isoformat(timespec="seconds"),
        "source_annotation_file": str(completed_annotation_path).replace("\\", "/"),
        "rows_added": int(len(destination_rows)),
        "jumplist_archive": str(archive_path).replace("\\", "/"),
        "retraining_recommended": True,
    }
    state_path = training_dataset_state_path(dataset_root)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return {
        "rows_added": int(len(destination_rows)),
        "archive_path": archive_path,
        "completed_annotation_path": completed_annotation_path,
        "state": state,
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
    from synergie.services.prediction_service import PredictionService

    frame = annotation_rows.copy()
    if frame.empty:
        return {"rows": frame, "updated": 0, "skipped": 0}

    service = PredictionService.from_paths(type_model_path, success_model_path)
    result = service.prefill_annotation_rows(frame)
    frame = result["rows"]
    for index in frame.index:
        if frame.at[index, "type"] == 8:
            continue
        frame.at[index, "turns"] = suggest_turns_from_rotation(frame.at[index, "rotations"])
        frame.at[index, "prediction_source"] = "batch_model_prefill"
    result["rows"] = frame
    return result


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

    duplicate_report = find_training_dataset_duplicates(dataset_path)
    if duplicate_report["has_duplicates"]:
        preview = ", ".join(duplicate_report["duplicate_paths"][:3])
        suffix = "..." if len(duplicate_report["duplicate_paths"]) > 3 else ""
        raise ValueError(
            f"Training dataset contains duplicate jump paths ({duplicate_report['duplicate_rows']} rows): "
            f"{preview}{suffix}"
        )
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


def process_csv_file(
    csv_path: str,
    synchro: int = 0,
    output_path: str | None = None,
    *,
    type_model_path: str | None = None,
    success_model_path: str | None = None,
) -> dict:
    from core.data_treatment.data_generation.exporter import export
    import pandas as pd

    input_path = Path(csv_path)
    dataframe = pd.read_csv(input_path)
    result = export(
        dataframe,
        sampleTimeFineSynchro=synchro,
        type_model_path=type_model_path,
        success_model_path=success_model_path,
    )
    destination = Path(output_path) if output_path else input_path.with_name(f"{input_path.stem}_jumps.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return {
        "path": destination,
        "prediction_status": result.attrs.get("prediction_status", "unknown"),
    }


def process_new_imu_file_for_annotation(
    raw_csv_path: str | Path,
    *,
    output_path: str | Path | None = None,
    pending_root: str | Path = "data/pending",
    sample_time_fine_synchro: int = 0,
    type_model_path: str | None = None,
    success_model_path: str | None = None,
) -> dict:
    return process_new_imu_session_for_annotation(
        raw_csv_path,
        output_path=output_path,
        pending_root=pending_root,
        sample_time_fine_synchro=sample_time_fine_synchro,
        type_model_path=type_model_path,
        success_model_path=success_model_path,
    )


def process_new_imu_session_for_annotation(
    raw_csv_path: str | Path,
    *,
    output_path: str | Path | None = None,
    pending_root: str | Path = "data/pending",
    sample_time_fine_synchro: int = 0,
    type_model_path: str | None = None,
    success_model_path: str | None = None,
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
        if type_model_path and success_model_path:
            try:
                annotation_frame = prefill_annotation_predictions(
                    annotation_frame,
                    type_model_path=type_model_path,
                    success_model_path=success_model_path,
                )["rows"]
            except ModuleNotFoundError as exc:
                if exc.name not in {"tensorflow", "keras"}:
                    raise
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
