from __future__ import annotations

from pathlib import Path

from synergie.services.annotation_service import JUMP_TYPE_LABELS
from synergie.services.hdf5_archive_service import load_segment_dataframe


def analyze_jump_quality(
    dataset_path: str | Path,
    *,
    outlier_threshold: float = 3.5,
    saturation_threshold: float = 1900.0,
) -> dict:
    """Analyze labelled jump segments for missing files, saturation, and outliers."""
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
        if not segment_path.exists() and not _segment_exists_in_hdf5(segment_path):
            records.append(_missing_segment_record(row_index, row, segment_path, jump_type, success))
            continue

        records.append(_segment_quality_record(row_index, row, segment_path, jump_type, success, saturation_threshold))

    if not records:
        return _empty_quality_result(dataset_path, skipped)

    _append_outlier_reasons(records, outlier_threshold)
    suspicious_records = [record for record in records if record["suspicious"]]
    return {
        "dataset_path": str(dataset_path),
        "total_labelled_jumps": len(records),
        "skipped_rows": skipped,
        "suspicious_count": len(suspicious_records),
        "records": records,
        "suspicious_records": suspicious_records,
        "type_summary": _summarize_types(records),
    }


def _missing_segment_record(row_index, row, segment_path: Path, jump_type: int, success: int) -> dict:
    return {
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


def _segment_quality_record(row_index, row, segment_path: Path, jump_type: int, success: int, saturation_threshold: float) -> dict:
    segment = load_segment_dataframe(segment_path)
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

    return {
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


def _append_outlier_reasons(records: list[dict], outlier_threshold: float) -> None:
    import numpy as np

    features = ["duration_ms", "max_abs_gyr_x", "max_abs_acc_x", "max_abs_second_derivative", "rotations"]
    by_type: dict[int, list[dict]] = {}
    for record in records:
        by_type.setdefault(int(record["type"]), []).append(record)

    for typed_records in by_type.values():
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
        record["reasons"] = list(dict.fromkeys(record["reasons"]))
        record["suspicious"] = bool(record["reasons"])


def _summarize_types(records: list[dict]) -> list[dict]:
    import numpy as np

    by_type: dict[int, list[dict]] = {}
    for record in records:
        by_type.setdefault(int(record["type"]), []).append(record)

    return [
        {
            "type": jump_type,
            "label": JUMP_TYPE_LABELS.get(jump_type, str(jump_type)),
            "count": len(typed_records),
            "suspicious_count": sum(1 for record in typed_records if record["suspicious"]),
            "median_duration_ms": float(np.median([record["duration_ms"] for record in typed_records])),
            "median_rotations": float(np.median([record["rotations"] for record in typed_records])),
            "median_max_abs_gyr_x": float(np.median([record["max_abs_gyr_x"] for record in typed_records])),
        }
        for jump_type, typed_records in sorted(by_type.items())
    ]


def _empty_quality_result(dataset_path: str | Path, skipped: int) -> dict:
    return {
        "dataset_path": str(dataset_path),
        "total_labelled_jumps": 0,
        "skipped_rows": skipped,
        "suspicious_count": 0,
        "records": [],
        "suspicious_records": [],
        "type_summary": [],
    }


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _segment_exists_in_hdf5(segment_path: Path) -> bool:
    from synergie.services.hdf5_archive_service import hdf5_segment_paths

    return str(segment_path).replace("\\", "/") in hdf5_segment_paths()
