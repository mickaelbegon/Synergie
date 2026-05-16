from __future__ import annotations

from pathlib import Path

from synergie.services.annotation_service import annotate_combination_flags
from synergie.services.new_data_service import (
    files_for_new_imu_session,
    new_imu_session_key,
    parse_new_imu_filename,
    suggest_for_annotation_output_path,
)


def process_new_imu_file_for_annotation(
    raw_csv_path: str | Path,
    *,
    output_path: str | Path | None = None,
    pending_root: str | Path = "data/pending",
    sample_time_fine_synchro: int = 0,
    type_model_path: str | None = None,
    success_model_path: str | None = None,
) -> dict:
    """Process the selected file's full session into one annotation CSV."""
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
    """Detect jumps for every sensor in one session and export annotation rows."""
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
        records.extend(_records_for_sensor(file_metadata, session, sensor_segment_root, impact_offset_ms))

    annotation_frame = pd.DataFrame(records)
    if not annotation_frame.empty:
        annotation_frame = annotation_frame.sort_values(by=["synced_start_ms", "sensor_id", "start_ms"]).reset_index(drop=True)
        annotation_frame = annotate_combination_flags(annotation_frame)
        if type_model_path and success_model_path:
            annotation_frame = _prefill_predictions(annotation_frame, type_model_path, success_model_path)
    annotation_frame.to_csv(output_csv_path, index=False)
    return {
        "annotation_csv": output_csv_path,
        "segment_directory": segment_root,
        "jump_count": len(records),
        "source_file": raw_path,
        "session_key": new_imu_session_key(metadata),
        "sensor_count": len(session_files),
    }


def estimate_sensor_impact_offset_ms(session_df) -> float:
    """Estimate the first strong impact timestamp for one sensor session."""
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


def _records_for_sensor(file_metadata: dict, session, sensor_segment_root: Path, impact_offset_ms: float) -> list[dict]:
    records = []
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
    return records


def _prefill_predictions(annotation_frame, type_model_path: str, success_model_path: str):
    from synergie.operations import prefill_annotation_predictions

    try:
        return prefill_annotation_predictions(
            annotation_frame,
            type_model_path=type_model_path,
            success_model_path=success_model_path,
        )["rows"]
    except ModuleNotFoundError as exc:
        if exc.name not in {"tensorflow", "keras"}:
            raise
        return annotation_frame


def _ms_to_timestamp(ms: float) -> str:
    total_seconds = round(ms / 1000)
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"
