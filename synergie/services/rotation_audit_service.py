from __future__ import annotations

from pathlib import Path

from synergie.services.annotation_service import JUMP_TYPE_LABELS


def audit_turn_estimation(
    annotation_root: str | Path = "data/annotated/total",
    *,
    annotation_pattern: str = "*for_annotation*.csv",
) -> dict:
    """Audit turn estimation from either historical jumplist data or pending annotations."""
    root = Path(annotation_root)
    jumplist_path = root / "jumplist.csv"
    if jumplist_path.exists():
        return _audit_training_jumplist(jumplist_path)
    return _audit_annotation_files(root, annotation_pattern)


def _audit_training_jumplist(jumplist_path: Path) -> dict:
    """Recompute measured rotations from labelled historical training segments."""
    import pandas as pd

    frame = pd.read_csv(jumplist_path)
    records = []
    for row_index, row in frame.iterrows():
        jump_type_value = _safe_float(row.get("type"), default=None)
        jump_type = 8 if jump_type_value is None else int(jump_type_value)
        annotated_turns = _safe_float(row.get("rotations"), default=None)
        if jump_type == 8 or annotated_turns is None or annotated_turns <= 0:
            continue
        measured_rotation = _measured_rotation_from_segment(row.get("path"))
        if measured_rotation is None:
            continue
        records.append(_build_record(jumplist_path, row_index, row, annotated_turns, measured_rotation))
    return _summarize_records(jumplist_path.parent, scanned_files=1, records=records)


def _audit_annotation_files(root: Path, annotation_pattern: str) -> dict:
    import pandas as pd

    records: list[dict] = []
    scanned_files = 0
    for path in sorted(root.glob(annotation_pattern)):
        frame = pd.read_csv(path)
        if not {"rotations", "turns"}.issubset(frame.columns):
            continue
        scanned_files += 1
        records.extend(_records_from_frame(frame, path))
    return _summarize_records(root, scanned_files, records)


def _records_from_frame(frame, source_path: Path) -> list[dict]:
    records: list[dict] = []
    for row_index, row in frame.iterrows():
        annotation_status = str(row.get("annotation_status", "")).strip().lower()
        detection_status = str(row.get("detection_status", "detected_jump")).strip().lower()
        if annotation_status != "completed" or detection_status == "not_a_jump":
            continue
        annotated_turns = _safe_float(row.get("turns"), default=None)
        measured_rotation = _safe_float(row.get("rotations"), default=None)
        if annotated_turns is None or measured_rotation is None:
            continue
        records.append(_build_record(source_path, row_index, row, annotated_turns, measured_rotation))
    return records


def _build_record(source_file: Path, row_index: int, row, annotated_turns: float, measured_rotation: float) -> dict:
    from synergie.operations import annotation_turn_value_for_storage, suggest_turns_from_rotation

    estimated_ui_turns = suggest_turns_from_rotation(measured_rotation)
    estimated_turns = float(annotation_turn_value_for_storage(row.get("type"), estimated_ui_turns))
    signed_error = measured_rotation - annotated_turns
    absolute_error = abs(signed_error)
    jump_type_value = _safe_float(row.get("type"), default=None)
    jump_type = 8 if jump_type_value is None else int(jump_type_value)
    reasons = []
    if absolute_error >= 0.35:
        reasons.append("large_rotation_error")
    if annotated_turns != estimated_turns:
        reasons.append("rounded_turn_mismatch")
    return {
        "source_file": str(source_file).replace("\\", "/"),
        "row_index": int(row_index),
        "path": str(row.get("path", "")),
        "type": jump_type,
        "type_label": JUMP_TYPE_LABELS.get(jump_type, str(jump_type)),
        "annotated_turns": annotated_turns,
        "measured_rotation": measured_rotation,
        "estimated_turns": estimated_turns,
        "signed_error": signed_error,
        "absolute_error": absolute_error,
        "suspicious": bool(reasons),
        "reasons": reasons,
    }


def _measured_rotation_from_segment(segment_path) -> float | None:
    import numpy as np
    import pandas as pd

    path = Path(str(segment_path))
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if not {"SampleTimeFine", "Gyr_X"}.issubset(frame.columns):
        return None
    if "X_gyr_second_derivative_crossing" not in frame:
        from core.data_treatment.data_generation.trainingSession import trainingSession

        frame = trainingSession(frame).df
    crossing = frame["X_gyr_second_derivative_crossing"].astype(int).to_numpy()
    begins = np.where(np.diff(crossing) == 1)[0]
    ends = np.where(np.diff(crossing) == -1)[0]
    pairs = [(int(begin), int(end)) for begin in begins for end in ends if end > begin]
    if not pairs:
        return None
    begin, end = min(pairs, key=lambda pair: pair[1] - pair[0])
    interval = frame.iloc[begin:end]
    if len(interval) < 2:
        return None
    timestamps = interval["SampleTimeFine"].to_numpy(dtype="float64")
    speeds = interval["Gyr_X"].to_numpy(dtype="float64")
    valid = np.isfinite(speeds[:-1]) & np.isfinite(np.diff(timestamps))
    if not valid.any():
        return None
    rotation = float(np.nansum(speeds[:-1][valid] * (np.diff(timestamps)[valid] / 1e6)) / 360.0)
    if not np.isfinite(rotation):
        return None
    return abs(rotation)


def _summarize_records(root: Path, scanned_files: int, records: list[dict]) -> dict:
    import pandas as pd

    if not records:
        return {
            "annotation_root": str(root),
            "scanned_files": scanned_files,
            "labelled_jumps": 0,
            "exact_accuracy": None,
            "mean_absolute_error": None,
            "records": [],
            "suspicious_records": [],
            "type_summary": [],
            "confusion_matrix": [],
            "labels": [],
        }
    labelled = pd.DataFrame(records)
    suspicious = labelled[labelled["suspicious"]].to_dict("records")
    labels = sorted(set(labelled["annotated_turns"]).union(labelled["estimated_turns"]))
    matrix = pd.crosstab(labelled["annotated_turns"], labelled["estimated_turns"]).reindex(index=labels, columns=labels, fill_value=0)
    return {
        "annotation_root": str(root),
        "scanned_files": scanned_files,
        "labelled_jumps": int(len(labelled)),
        "exact_accuracy": float((labelled["annotated_turns"] == labelled["estimated_turns"]).mean()),
        "mean_absolute_error": float(labelled["absolute_error"].mean()),
        "records": labelled.to_dict("records"),
        "suspicious_records": suspicious,
        "type_summary": _type_summary(labelled),
        "confusion_matrix": matrix.to_numpy(dtype=int).tolist(),
        "labels": labels,
    }


def _type_summary(labelled) -> list[dict]:
    summary = []
    for jump_type, rows in labelled.groupby("type"):
        summary.append(
            {
                "type": int(jump_type),
                "label": JUMP_TYPE_LABELS.get(int(jump_type), str(jump_type)),
                "count": int(len(rows)),
                "exact_accuracy": float((rows["annotated_turns"] == rows["estimated_turns"]).mean()),
                "mean_absolute_error": float(rows["absolute_error"].mean()),
                "suspicious_count": int(rows["suspicious"].sum()),
            }
        )
    return summary


def _safe_float(value, default=None):
    try:
        if value != value:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
