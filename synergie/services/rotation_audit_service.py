from __future__ import annotations

from pathlib import Path

from synergie.services.annotation_service import JUMP_TYPE_LABELS


def audit_turn_estimation(
    annotation_root: str | Path = "data/pending",
    *,
    annotation_pattern: str = "*for_annotation*.csv",
) -> dict:
    """Compare measured IMU rotations with human turn labels in annotation files."""
    import pandas as pd

    root = Path(annotation_root)
    records: list[dict] = []
    scanned_files = 0
    for path in sorted(root.glob(annotation_pattern)):
        frame = pd.read_csv(path)
        if not {"rotations", "turns"}.issubset(frame.columns):
            continue
        scanned_files += 1
        records.extend(_records_from_frame(frame, path))

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
    matrix = pd.crosstab(
        labelled["annotated_turns"],
        labelled["estimated_turns"],
    ).reindex(index=labels, columns=labels, fill_value=0)
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


def _records_from_frame(frame, source_path: Path) -> list[dict]:
    from synergie.operations import annotation_turn_value_for_storage, suggest_turns_from_rotation

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
        estimated_ui_turns = suggest_turns_from_rotation(measured_rotation)
        estimated_turns = float(annotation_turn_value_for_storage(row.get("type"), estimated_ui_turns))
        signed_error = measured_rotation - annotated_turns
        absolute_error = abs(signed_error)
        jump_type = int(_safe_float(row.get("type"), default=8) or 8)
        reasons = []
        if absolute_error >= 0.35:
            reasons.append("large_rotation_error")
        if annotated_turns != estimated_turns:
            reasons.append("rounded_turn_mismatch")
        records.append(
            {
                "source_file": str(source_path).replace("\\", "/"),
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
        )
    return records


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
