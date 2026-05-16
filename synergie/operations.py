from __future__ import annotations

import re
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
from synergie.services.annotation_finalization_service import finalize_annotation_file
from synergie.services.annotation_generation_service import (
    estimate_sensor_impact_offset_ms,
    process_new_imu_file_for_annotation,
    process_new_imu_session_for_annotation,
)
from synergie.services.csv_processing_service import process_csv_file
from synergie.services.hyperparameter_search_service import (
    hyperparameter_search_space,
    sample_hyperparameter_trials,
)
from synergie.services.detection_tuning_service import (
    analyze_detection_review_labels,
    optimize_detection_parameters,
)
from synergie.services.quality_service import analyze_jump_quality
from synergie.services.signal_importance_service import compute_signal_importance
from synergie.services.new_data_service import (
    files_for_new_imu_session,
    list_new_data_directories,
    list_new_imu_files,
    list_new_imu_sessions,
    new_imu_session_key,
    parse_new_imu_filename,
    suggest_for_annotation_output_path,
)
from synergie.services.model_registry_service import (
    audit_saved_models,
    format_pretrained_model_label,
    latest_model_path_for_task,
    list_pretrained_models_by_performance,
    list_pretrained_training_models,
    model_format as _model_format,
)
from synergie.services.training_dataset_service import (
    describe_training_dataset,
    find_training_dataset_duplicates,
    load_training_dataset_state,
    training_dataset_state_path,
)
from synergie.services.training_service import (
    promote_trained_model as _promote_trained_model,
    run_hyperparameter_search,
    train_model,
)
from synergie.services.video_service import (
    annotation_reference_datetime,
    list_video_files,
    parse_datetime_from_text as _parse_datetime_from_text,
    parse_datetime_value as _parse_datetime_value,
    read_video_creation_time_with_ffprobe as _read_video_creation_time_with_ffprobe,
    select_best_video_datetime as _select_best_video_datetime,
)


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


def list_all_session_csv_files(raw_root: str = "data/raw") -> list[dict]:
    """Return all session CSV files with the metadata needed for processing."""
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


def read_video_metadata(video_path: str | Path) -> dict:
    """Read video timestamps while preserving patchable helper aliases."""
    path = Path(video_path)
    stat = path.stat()
    candidates: list[tuple[str, datetime]] = []
    ffprobe_datetime = _read_video_creation_time_with_ffprobe(path)
    if ffprobe_datetime is not None:
        candidates.append(("ffprobe.creation_time", ffprobe_datetime))
    filename_datetime = _parse_datetime_from_text(path.stem)
    if filename_datetime is not None:
        candidates.append(("filename", filename_datetime))
    candidates.append(("filesystem.modified", datetime.fromtimestamp(stat.st_mtime)))
    candidates.append(("filesystem.created", datetime.fromtimestamp(stat.st_ctime)))
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
    """Find nearest videos while preserving the historical operations API."""
    reference_datetime = annotation_reference_datetime(annotation_csv_path, annotation_rows=annotation_rows)
    matches: list[dict] = []
    for video_path in list_video_files(video_directory, recursive=recursive):
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


def parse_training_id_from_csv_path(csv_path: Path) -> str | None:
    training_parts = csv_path.stem.split("_")
    if len(training_parts) < 2:
        return None
    return training_parts[1]


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


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
