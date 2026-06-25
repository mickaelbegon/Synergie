from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

from synergie import pretrained_models
from synergie.services.add_jump_service import (
    annotation_index_for_path,
    append_manual_annotation_row,
    build_manual_annotation_row,
    build_manual_jump_segment,
    compute_rotation_between_ms,
    detect_add_jump_candidates,
    format_add_jump_candidate_labels,
    manual_segment_path,
    template_for_sensor,
)
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
    clear_annotation_block_sync,
    clear_annotation_sensor_sync,
    compute_annotation_jump_video_time_ms,
    get_annotation_block_sync_offset,
    get_annotation_block_sync_source,
    get_annotation_sensor_sync_offset,
    get_annotation_sensor_sync_source,
    annotation_ui_values_from_row,
    load_annotation_metadata,
    save_annotation_file_values,
    save_annotation_values,
    save_annotation_metadata,
    set_annotation_block_sync_offset,
    set_annotation_sensor_sync_offset,
    set_annotation_video_directory,
    set_annotation_video_path,
    summarize_annotation_progress,
)
from synergie.services.annotation_sync_service import (
    annotation_imu_to_video_ms,
    annotation_review_video_target_ms,
    annotation_row_video_time_ms,
    annotation_sync_context_text,
    annotation_sync_source_summary,
    annotation_sync_summary,
    annotation_video_to_imu_ms,
    block_offset_from_impact,
    block_offset_from_jump,
    sync_impact_diagnostic_text,
    sync_impact_selection_text,
)
from synergie.services.annotation_timeline_service import build_annotation_timeline_items
from synergie.services.annotation_video_player import AnnotationVideoPlayer
from synergie.services.annotation_finalization_service import finalize_annotation_file
from synergie.services.annotation_generation_service import (
    estimate_sensor_impact_offset_from_file,
    estimate_sensor_impact_offset_ms,
    process_new_imu_file_for_annotation,
    process_new_imu_session_for_annotation,
)
from synergie.services.annotation_impact_repair_service import redetect_annotation_sync_impacts
from synergie.services.annotation_plot_service import (
    annotation_gyro_column,
    annotation_segment_available,
    annotation_segment_path,
    annotation_type_window_start_imu_ms,
    prepare_annotation_segment_signal,
)
from synergie.services.annotation_shortcut_service import (
    ANNOTATION_JUMP_TYPE_SHORTCUTS,
    ANNOTATION_SHORTCUTS_HELP_TEXT,
    ANNOTATION_SUCCESS_SHORTCUTS,
    resolve_annotation_shortcut,
)
from synergie.services.annotation_source_service import find_annotation_raw_source_path
from synergie.services.csv_processing_service import process_csv_file
from synergie.services.hyperparameter_search_service import (
    hyperparameter_search_space,
    load_optimized_model_parameters,
    sample_hyperparameter_trials,
    save_optimized_model_parameters,
)
from synergie.services.legacy_import_service import import_legacy_jumplist, migrate_legacy_annotation_workbook
from synergie.services.detection_tuning_service import (
    analyze_detection_review_labels,
    load_optimized_detection_parameters,
    optimize_detection_parameters,
    save_optimized_detection_parameters,
)
from synergie.services.data_inventory_service import build_data_inventory
from synergie.services.data_validation_service import (
    data_validation_action_summary,
    data_validation_status_text,
    format_data_validation_report,
    validate_data_files,
)
from synergie.services.detection_explanation_service import (
    angular_velocity_peak_context,
    detection_threshold_context,
    nearest_interval,
    threshold_intervals,
)
from synergie.services.formatting_service import (
    format_annotation_video_info,
    format_file_size,
    format_video_cache_cleared_status,
    format_video_cache_button,
    format_video_ms,
    format_video_preparation_message,
    parse_video_ms,
)
from synergie.services.backup_manifest_service import build_backup_manifest, write_backup_manifest
from synergie.services.hdf5_archive_service import archive_segment_csvs, export_hdf5_archive, plan_segment_csv_cleanup
from synergie.services.quality_service import analyze_jump_quality
from synergie.services.signal_importance_service import compute_signal_importance
from synergie.services.rotation_audit_service import audit_turn_estimation, load_turn_audit_signal
from synergie.services.inspect_signal_service import (
    inspect_axis_labels,
    inspect_detection_threshold_spec,
    inspect_drag_zoom_selection,
    inspect_jump_center_markers,
    inspect_jump_center_ms,
    inspect_jump_has_gyro_saturation,
    inspect_jump_legend_specs,
    inspect_jump_list_labels,
    inspect_jump_window_bounds_ms,
    inspect_jump_zoom_view,
    inspect_ready_status,
    inspect_selected_jump_title,
    inspect_selected_jump_status,
    inspect_signal_series_specs,
    inspect_text,
    inspect_zoom_range_title,
    inspect_zoom_range_view,
)
from synergie.services.sync_impact_service import (
    detect_sync_impacts,
    prepare_sync_impact_signal,
    sync_impact_list_labels,
    sync_impact_plot_title,
    sync_impact_review_warning,
    sync_impact_selected_status,
    sync_impact_signal_window,
)
from synergie.services.session_service import (
    add_session,
    describe_file,
    list_all_session_csv_files,
    list_directory_files,
    list_session_csv_files,
    list_sessions,
    next_jumplist_output_path,
    session_directory,
    session_metadata,
    session_synchro,
    suggest_session_from_imu_file,
)
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
    delete_pretrained_training_model,
    evaluate_registered_model,
    format_pretrained_model_label,
    latest_model_path_for_task,
    list_pretrained_models_by_performance,
    list_pretrained_training_models,
    model_format as _model_format,
)
from synergie.services.training_dataset_service import (
    describe_training_dataset,
    exclude_training_dataset_row,
    find_training_dataset_duplicates,
    load_training_dataset_rows,
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
    cached_video_path,
    clear_video_cache,
    list_video_files,
    optimized_playback_video_path,
    parse_datetime_from_text as _parse_datetime_from_text,
    parse_datetime_value as _parse_datetime_value,
    read_video_creation_time_with_ffprobe as _read_video_creation_time_with_ffprobe,
    select_best_video_datetime as _select_best_video_datetime,
    video_cache_status,
    video_proxy_status,
    video_cache_size_bytes,
)
from synergie.services.window_benchmark_service import benchmark_temporal_offsets, benchmark_temporal_windows
from synergie.services.segment_reexport_service import ensure_pre_takeoff_context, reexport_segment_with_context


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


def suggest_turns_from_rotation(
    rotation_value,
    *,
    contact_offset_turns: float = 0.45,
    jump_type: str | int | None = None,
) -> str:
    """Convert measured airborne rotation into a 1-4 UI turn guess with on-ice offset."""
    total_turns = _safe_float(rotation_value, default=1.0) + float(contact_offset_turns)
    normalized_type = "" if jump_type is None else str(jump_type).strip().lower()
    if normalized_type in {"5", "axel"}:
        turns = int(round(total_turns - 0.5))
    else:
        turns = int(round(total_turns))
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
        frame.at[index, "turns"] = suggest_turns_from_rotation(
            frame.at[index, "rotations"],
            jump_type=frame.at[index, "type"],
        )
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
