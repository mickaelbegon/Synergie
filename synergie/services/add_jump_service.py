from __future__ import annotations

from pathlib import Path


def detect_add_jump_candidates(
    raw_path: str | Path,
    *,
    detection_threshold: float,
    smoothing_sigma: float,
    combination_gap_frames: int,
    training_session_factory=None,
) -> list[dict]:
    """Detect candidate jumps for the manual ADD JUMP workflow."""
    import pandas as pd

    if training_session_factory is None:
        from core.data_treatment.data_generation.trainingSession import trainingSession as training_session_factory

    raw_path = Path(raw_path)
    session = training_session_factory(
        pd.read_csv(raw_path, low_memory=False),
        detection_threshold=float(detection_threshold),
        smoothing_sigma=float(smoothing_sigma),
        combination_gap_frames=int(combination_gap_frames),
    )
    return [{"index": index, "jump": jump, "raw_path": raw_path, "session": session} for index, jump in enumerate(session.jumps, start=1)]


def format_add_jump_candidate_labels(candidates: list[dict]) -> list[str]:
    """Return listbox labels for ADD JUMP candidates."""
    return [
        f"{candidate['index']:03d} | start {candidate['jump'].startTimestamp:.0f} ms | "
        f"end {candidate['jump'].endTimestamp:.0f} ms | rotation {candidate['jump'].rotation:.1f}"
        for candidate in candidates
    ]


def build_manual_jump_segment(
    candidate: dict,
    start_ms: float,
    *,
    frames_before_takeoff: int,
    window_frames: int,
):
    """Return a fixed-size segment around a manually selected jump takeoff."""
    session_df = candidate["session"].df
    start_index = int((session_df["ms"] - float(start_ms)).abs().idxmin())
    segment_start = max(0, start_index - int(frames_before_takeoff))
    segment_end = min(len(session_df), segment_start + int(window_frames))
    if segment_end - segment_start < int(window_frames):
        segment_start = max(0, segment_end - int(window_frames))
    segment = session_df.iloc[segment_start:segment_end].copy()
    segment["Combination"] = int(candidate["jump"].combinate)
    return segment, start_index


def compute_rotation_between_ms(frame, start_ms: float, end_ms: float) -> float:
    """Estimate absolute rotations between two IMU timestamps from Gyr_X."""
    if end_ms <= start_ms or not {"ms", "SampleTimeFine", "Gyr_X"}.issubset(frame.columns):
        return 0.0
    interval = frame[(frame["ms"] >= float(start_ms)) & (frame["ms"] <= float(end_ms))].copy()
    interval = interval[interval["Gyr_X"].abs() <= 1e6]
    if len(interval) < 2:
        return 0.0
    timestamps = interval["SampleTimeFine"].to_numpy(dtype="float64")
    speeds = interval["Gyr_X"].to_numpy(dtype="float64")[:-1]
    dt = (timestamps[1:] - timestamps[:-1]) / 1e6
    return abs(float((speeds * dt).sum()) / 360.0)


def template_for_sensor(annotation_frame, sensor_id: str) -> dict:
    """Return the first existing annotation row for a sensor as a reusable template."""
    sensor_rows = annotation_frame[annotation_frame["sensor_id"].astype(str) == str(sensor_id)]
    return sensor_rows.iloc[0].to_dict() if not sensor_rows.empty else {}


def manual_segment_path(template: dict, annotation_file_path: str | Path, sensor_id: str, row_count: int) -> Path:
    """Return the output CSV path for one manually added jump segment."""
    segment_dir = Path(str(template.get("path", Path(annotation_file_path).parent))).parent
    return segment_dir / f"manual_sensor{sensor_id}_jump{int(row_count) + 1:03d}.csv"


def build_manual_annotation_row(
    template: dict,
    *,
    segment_path: str | Path,
    source_file_name: str,
    sensor_id: str,
    start_ms: float,
    end_ms: float,
    impact_offset_ms: float,
    rotations: float,
    takeoff_video_text: str,
    landing_video_text: str,
    detection_threshold: float,
    smoothing_sigma: float,
    combination_gap_frames: int,
    format_video_ms,
) -> dict:
    """Build the annotation CSV row for a manually added jump."""
    synced_start_ms = round(float(start_ms) - float(impact_offset_ms), 3)
    normalized_segment_path = str(segment_path).replace("\\", "/")
    row = dict(template)
    row.update(
        {
            "path": normalized_segment_path,
            "videoTimeStamp": format_video_ms(max(synced_start_ms, 0.0)),
            "type": 8,
            "turns": "",
            "skater": template.get("skater", f"sensor_{sensor_id}"),
            "athlete_id": template.get("athlete_id", f"sensor_{sensor_id}"),
            "success": 2,
            "rotations": round(float(rotations), 1),
            "source_file": source_file_name,
            "sensor_id": sensor_id,
            "annotation_status": "pending",
            "video_status": "visible",
            "detection_status": "manual_missing_jump",
            "start_ms": round(float(start_ms), 3),
            "end_ms": round(float(end_ms), 3),
            "synced_start_ms": synced_start_ms,
            "added_by": "add_jump_popup",
            "manual_takeoff_video_ms": takeoff_video_text,
            "manual_landing_video_ms": landing_video_text,
            "detection_threshold": float(detection_threshold),
            "smoothing_sigma": float(smoothing_sigma),
            "combination_gap_frames": int(combination_gap_frames),
        }
    )
    return row


def append_manual_annotation_row(annotation_frame, row: dict):
    """Append one manual annotation row and keep the annotation table ordered."""
    updated = annotation_frame.copy()
    updated.loc[len(updated)] = row
    return updated.sort_values(by=["synced_start_ms", "sensor_id", "start_ms"]).reset_index(drop=True)


def annotation_index_for_path(annotation_frame, path: str) -> int:
    """Return the first annotation row index matching a segment path."""
    matches = annotation_frame.index[annotation_frame["path"].astype(str) == str(path)]
    if len(matches) == 0:
        raise ValueError(f"Annotation path not found: {path}")
    return int(matches[0])
