from __future__ import annotations

from pathlib import Path

from synergie.services.annotation_service import JUMP_TYPE_LABELS
from synergie.config import DEFAULT_DETECTION_THRESHOLD, DEFAULT_SMOOTHING_SIGMA
from synergie.services.hdf5_archive_service import load_segment_dataframe


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


def load_turn_audit_signal(segment_path: str | Path) -> dict | None:
    """Load one audited segment with the takeoff/landing bounds used for rotation."""
    frame = _load_segment_frame(segment_path)
    if frame is None:
        return None
    pair = _rotation_interval_indices(frame)
    if pair is None:
        return None
    begin, end = pair
    return {
        "path": str(segment_path),
        "frame": frame,
        "takeoff_index": begin,
        "landing_index": end,
    }


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
        if annotation_status != "completed" or detection_status in {"not_a_jump", "weird_signal"}:
            continue
        annotated_turns = _safe_float(row.get("turns"), default=None)
        measured_rotation = _safe_float(row.get("rotations"), default=None)
        if annotated_turns is None or measured_rotation is None:
            continue
        records.append(_build_record(source_path, row_index, row, annotated_turns, measured_rotation))
    return records


def _build_record(source_file: Path, row_index: int, row, annotated_turns: float, measured_rotation: float) -> dict:
    from synergie.operations import annotation_turn_value_for_storage, suggest_turns_from_rotation

    estimated_ui_turns = suggest_turns_from_rotation(measured_rotation, jump_type=row.get("type"))
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
        "skater": _safe_float(row.get("skater"), default=None),
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

    frame = _load_segment_frame(segment_path)
    if frame is None:
        return None
    pair = _rotation_interval_indices(frame)
    if pair is None:
        return None
    begin, end = pair
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


def _load_segment_frame(segment_path):
    path = Path(str(segment_path))
    if not str(segment_path or "").strip():
        return None
    try:
        frame = load_segment_dataframe(path)
    except FileNotFoundError:
        return None
    if not {"SampleTimeFine", "Gyr_X"}.issubset(frame.columns):
        return None
    if "X_gyr_second_derivative_crossing" not in frame:
        frame = _add_rotation_detection_columns(frame)
    return frame


def _add_rotation_detection_columns(frame):
    import scipy as sp

    prepared = frame.copy()
    prepared["Gyr_X_smoothed"] = sp.ndimage.gaussian_filter1d(prepared["Gyr_X"], sigma=DEFAULT_SMOOTHING_SIGMA)
    prepared["X_gyr_derivative"] = prepared["Gyr_X_smoothed"].diff()
    prepared["X_gyr_second_derivative"] = prepared["X_gyr_derivative"].diff()
    prepared["X_gyr_second_derivative_crossing"] = [
        False if value > DEFAULT_DETECTION_THRESHOLD else True
        for value in prepared["X_gyr_second_derivative"]
    ]
    return prepared


def _rotation_interval_indices(frame) -> tuple[int, int] | None:
    import numpy as np

    crossing = frame["X_gyr_second_derivative_crossing"].astype(int).to_numpy()
    begins = np.where(np.diff(crossing) == 1)[0]
    ends = np.where(np.diff(crossing) == -1)[0]
    pairs = [(int(begin), int(end)) for begin in begins for end in ends if end > begin]
    if not pairs:
        return None
    return min(pairs, key=lambda pair: pair[1] - pair[0])


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
            "rounding_rule_summary": [],
            "strategy_summary": [],
            "contact_offset_summary": [],
            "strategy_by_skater": [],
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
        "rounding_rule_summary": _rounding_rule_summary(labelled),
        "strategy_summary": _strategy_summary(labelled),
        "contact_offset_summary": _contact_offset_summary(labelled),
        "strategy_by_skater": _strategy_by_skater(labelled),
        "confusion_matrix": matrix.to_numpy(dtype=int).tolist(),
        "labels": labels,
    }


def _type_summary(labelled) -> list[dict]:
    summary = []
    for jump_type, rows in labelled.groupby("type"):
        best_rule = _rounding_rule_summary(rows)[0]
        summary.append(
            {
                "type": int(jump_type),
                "label": JUMP_TYPE_LABELS.get(int(jump_type), str(jump_type)),
                "count": int(len(rows)),
                "exact_accuracy": float((rows["annotated_turns"] == rows["estimated_turns"]).mean()),
                "mean_absolute_error": float(rows["absolute_error"].mean()),
                "suspicious_count": int(rows["suspicious"].sum()),
                "best_rule": best_rule["label"],
                "best_rule_accuracy": best_rule["exact_accuracy"],
            }
        )
    return summary


def _rounding_rule_summary(labelled) -> list[dict]:
    """Compare simple post-processing rules before considering a learned model."""
    rules = {
        "round": lambda value: round(value),
        "ceil_minus_0.15": lambda value: _ceil_shift(value, 0.15),
        "ceil_minus_0.30": lambda value: _ceil_shift(value, 0.30),
        "floor_plus_0.50": lambda value: int(value + 0.50),
    }
    summaries = []
    for label, rule in rules.items():
        predicted = labelled.apply(lambda row: _estimate_turns_for_rule(row, rule), axis=1)
        summaries.append(
            {
                "label": label,
                "exact_accuracy": float((predicted == labelled["annotated_turns"]).mean()),
                "mean_absolute_error": float((predicted - labelled["annotated_turns"]).abs().mean()),
            }
        )
    return sorted(summaries, key=lambda item: (item["exact_accuracy"], -item["mean_absolute_error"]), reverse=True)


def _strategy_summary(labelled) -> list[dict]:
    """Compare deployable and optimistic post-processing strategies."""
    strategies = {
        "current_round": lambda row: _estimate_turns_for_rule(row, round),
        "fixed_contact_offset_0.45": lambda row: _estimate_turns_with_contact_offset(row, 0.45),
        "hybrid_non_axel_shift": _estimate_turns_for_hybrid_rule,
        "best_rule_per_type_observed": _estimate_turns_for_best_observed_type_rule,
    }
    summaries = []
    for label, strategy in strategies.items():
        predicted = labelled.apply(strategy, axis=1)
        summaries.append(
            {
                "label": label,
                "exact_accuracy": float((predicted == labelled["annotated_turns"]).mean()),
                "mean_absolute_error": float((predicted - labelled["annotated_turns"]).abs().mean()),
            }
        )
    return summaries


def _contact_offset_summary(labelled) -> list[dict]:
    summaries = []
    for offset in _float_range(0.30, 0.60, 0.05):
        predicted = labelled.apply(lambda row: _estimate_turns_with_contact_offset(row, offset), axis=1)
        summaries.append(
            {
                "offset_turns": round(offset, 2),
                "label": f"contact_plus_{offset:.2f}",
                "exact_accuracy": float((predicted == labelled["annotated_turns"]).mean()),
                "mean_absolute_error": float((predicted - labelled["annotated_turns"]).abs().mean()),
            }
        )
    return sorted(summaries, key=lambda item: (item["exact_accuracy"], -item["mean_absolute_error"]), reverse=True)


def _strategy_by_skater(labelled) -> list[dict]:
    """Check whether strategy gains hold across skaters instead of one pooled total."""
    if "skater" not in labelled or labelled["skater"].isna().all():
        return []
    summaries = []
    for skater, rows in labelled.dropna(subset=["skater"]).groupby("skater"):
        strategies = {item["label"]: item for item in _strategy_summary(rows)}
        current = strategies["current_round"]
        hybrid = strategies["hybrid_non_axel_shift"]
        summaries.append(
            {
                "skater": int(skater) if float(skater).is_integer() else float(skater),
                "count": int(len(rows)),
                "current_accuracy": current["exact_accuracy"],
                "hybrid_accuracy": hybrid["exact_accuracy"],
                "accuracy_gain": hybrid["exact_accuracy"] - current["exact_accuracy"],
            }
        )
    return summaries


def _estimate_turns_for_hybrid_rule(row) -> float:
    """Use the empirically better non-Axel rule while keeping Axel on round()."""
    rule = round if int(row["type"]) == 5 else lambda value: _ceil_shift(value, 0.15)
    return _estimate_turns_for_rule(row, rule)


def _estimate_turns_for_best_observed_type_rule(row) -> float:
    """Upper-bound strategy based on the best historical rule seen for each jump type."""
    rules_by_type = {
        0: lambda value: _ceil_shift(value, 0.15),
        1: lambda value: _ceil_shift(value, 0.15),
        2: lambda value: _ceil_shift(value, 0.15),
        3: lambda value: _ceil_shift(value, 0.15),
        4: lambda value: _ceil_shift(value, 0.15),
        5: round,
    }
    return _estimate_turns_for_rule(row, rules_by_type.get(int(row["type"]), round))


def _estimate_turns_with_contact_offset(row, contact_offset_turns: float) -> float:
    from synergie.operations import annotation_turn_value_for_storage, suggest_turns_from_rotation

    estimate = int(
        suggest_turns_from_rotation(
            row["measured_rotation"],
            contact_offset_turns=contact_offset_turns,
            jump_type=row["type"],
        )
    )
    return float(annotation_turn_value_for_storage(row["type"], estimate))


def _estimate_turns_for_rule(row, rule) -> float:
    from synergie.operations import annotation_turn_value_for_storage

    estimate = min(4, max(1, int(rule(float(row["measured_rotation"])))))
    return float(annotation_turn_value_for_storage(row["type"], estimate))


def _ceil_shift(value: float, shift: float) -> int:
    import math

    return int(math.ceil(value - shift))


def _float_range(start: float, stop: float, step: float):
    current = start
    while current <= stop + 1e-9:
        yield current
        current += step


def _safe_float(value, default=None):
    try:
        if value != value:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
