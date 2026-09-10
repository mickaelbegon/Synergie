from __future__ import annotations

from pathlib import Path
from collections import Counter
from hashlib import sha256
import json

from synergie.services.annotation_service import annotation_review_status_from_row
from synergie.services.hdf5_archive_service import hdf5_segment_paths, load_segment_dataframe
from synergie.services.numeric_utils import safe_float


OPTIMIZED_DETECTION_PARAMETERS_FILE = Path("config") / "optimized_detection_parameters.json"


# These values are deliberately an audit vocabulary, not detector settings.  A
# reviewer can add one of the optional category columns below to a *copy* of an
# annotation CSV; this module never writes it back.
DETECTION_REGRESSION_CATEGORIES = (
    "immobile",
    "slow_movement",
    "twizzle",
    "flying_spin",
    "walley",
    "duplicate_detection",
    "left_rotation",
    "missed_jump",
    "unclassified_false_positive",
    "unclassified_false_negative",
)
_AUDIT_CATEGORY_COLUMNS = (
    "detection_issue_category",
    "audit_category",
    "issue_category",
    "error_category",
)
_ROTATION_DIRECTION_COLUMNS = ("rotation_direction", "turn_direction", "direction")
_CATEGORY_ALIASES = {
    "immobile": "immobile",
    "still": "immobile",
    "no_movement": "immobile",
    "mouvement_lent": "slow_movement",
    "slow_movement": "slow_movement",
    "walkthrough": "slow_movement",
    "twizzle": "twizzle",
    "flying_spin": "flying_spin",
    "flying spin": "flying_spin",
    "walley": "walley",
    "duplicate": "duplicate_detection",
    "duplicate_detection": "duplicate_detection",
    "doublon": "duplicate_detection",
    "left_rotation": "left_rotation",
    "rotation_gauche": "left_rotation",
    "rotation gauche": "left_rotation",
    "missed_jump": "missed_jump",
    "saut_manque": "missed_jump",
    "saut manqué": "missed_jump",
}


def audit_detection_regressions(root: str | Path = "data/pending") -> dict:
    """Read reviewed labels into a non-mutating, issue-oriented regression report.

    The report groups false positives and manual missing jumps by an optional
    reviewer category.  It is safe to run on Maksim's files: CSVs and metadata
    are only read, and input SHA-256 values are included for traceability.
    """
    import pandas as pd

    records: list[dict] = []
    source_files: list[dict] = []
    reviewed_detected = 0
    for path in _list_pending_annotation_files(root):
        frame = pd.read_csv(path)
        source_files.append({
            "path": str(path),
            "sha256": _file_sha256(path),
            "rows": int(len(frame)),
        })
        for row_index, row in frame.iterrows():
            if not _row_is_reviewed(row):
                continue
            status = annotation_review_status_from_row(row)
            outcome = _detection_outcome(status)
            if outcome == "reviewed_detected":
                reviewed_detected += 1
                continue
            if outcome is None:
                continue
            supplied_category, category_column = _row_audit_category(row)
            category = supplied_category or _default_audit_category(outcome)
            records.append(
                {
                    "annotation_file": str(path),
                    "annotation_file_sha256": source_files[-1]["sha256"],
                    "row_index": int(row_index),
                    "outcome": outcome,
                    "review_status": status,
                    "category": category,
                    "category_source": category_column or "inferred_from_review_status",
                    "rotation_direction": _row_rotation_direction(row),
                    "sensor_id": str(row.get("sensor_id", "")),
                    "athlete_id": str(row.get("athlete_id", row.get("skater", ""))),
                    "path": str(row.get("path", "")),
                    "start_ms": _safe_float(row.get("start_ms", 0.0)),
                    "synced_start_ms": _safe_float(row.get("synced_start_ms", row.get("start_ms", 0.0))),
                }
            )
    false_positives = [record for record in records if record["outcome"] == "false_positive"]
    false_negatives = [record for record in records if record["outcome"] == "false_negative"]
    return {
        "scope": "read_only_annotation_regression_audit_v1",
        "annotation_root": str(Path(root)),
        "source_files": source_files,
        "reviewed_detected": reviewed_detected,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "total_reviewed": reviewed_detected + len(records),
        "category_summary": _audit_category_summary(records),
        "false_positive_categories": _audit_category_summary(false_positives),
        "false_negative_categories": _audit_category_summary(false_negatives),
        "records": records,
    }


def write_detection_regression_audit(root: str | Path, output_path: str | Path) -> Path:
    """Write an explicit JSON copy of :func:`audit_detection_regressions`.

    No annotation input is changed.  The caller controls the output location so
    reports can be kept alongside a backup manifest or outside labelled data.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(audit_detection_regressions(root), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output


def analyze_detection_review_labels(root: str | Path = "data/pending") -> dict:
    """Collect reviewed false positives and false negatives across annotation files."""
    import pandas as pd

    false_positives: list[dict] = []
    false_negatives: list[dict] = []
    reviewed_detected = 0
    for path in _list_pending_annotation_files(root):
        frame = pd.read_csv(path)
        for index, row in frame.iterrows():
            status = annotation_review_status_from_row(row)
            record = {
                "annotation_file": str(path),
                "row_index": int(index),
                "sensor_id": str(row.get("sensor_id", "")),
                "athlete_id": str(row.get("athlete_id", row.get("skater", ""))),
                "path": str(row.get("path", "")),
                "start_ms": _safe_float(row.get("start_ms", 0.0)),
                "synced_start_ms": _safe_float(row.get("synced_start_ms", row.get("start_ms", 0.0))),
            }
            if status in {"not_a_jump", "not_a_jump_or_drill", "jump_drill", "weird_signal"}:
                false_positives.append(record)
            elif status == "manual_missing_jump":
                false_negatives.append(record)
            elif status == "normal":
                reviewed_detected += 1
    total_reviewed = reviewed_detected + len(false_positives) + len(false_negatives)
    return {
        "reviewed_detected": reviewed_detected,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "total_reviewed": total_reviewed,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def optimize_detection_parameters(
    root: str | Path = "data/pending",
    *,
    thresholds: list[float] | None = None,
    smoothing_sigmas: list[float] | None = None,
    derivative_polarities: list[int] | None = None,
) -> dict:
    """Score detector parameters on unique, explicitly reviewed segments."""
    import numpy as np
    import pandas as pd
    import scipy as sp

    from synergie.services.signal_cleaning_service import clean_imu_outliers

    thresholds = thresholds or [-0.4, -0.3, -0.2, -0.1, -0.05]
    smoothing_sigmas = smoothing_sigmas or [10, 20, 30, 40]
    derivative_polarities = derivative_polarities or [-1]
    if any(polarity not in {-1, 1} for polarity in derivative_polarities):
        raise ValueError("derivative_polarities must contain only -1 or 1")
    labeled_segments = _load_reviewed_segments(root)
    prepared_segments: list[tuple[dict, np.ndarray]] = []
    for segment in labeled_segments:
        data = load_segment_dataframe(segment["path"])
        cleaned, _report = clean_imu_outliers(data)
        gyro = pd.to_numeric(cleaned["Gyr_X"], errors="coerce").interpolate(limit_direction="both").fillna(0.0)
        prepared_segments.append((segment, gyro.to_numpy()))

    results: list[dict] = []
    for sigma in smoothing_sigmas:
        derivatives: list[tuple[dict, np.ndarray]] = []
        for segment, gyro in prepared_segments:
            smoothed = sp.ndimage.gaussian_filter1d(gyro, sigma=float(sigma))
            derivatives.append((segment, np.diff(np.diff(smoothed, prepend=smoothed[0]), prepend=0.0)))
        for threshold in thresholds:
            for polarity in derivative_polarities:
                true_positive = true_negative = false_positive = false_negative = 0
                for segment, second_derivative in derivatives:
                    if polarity == -1:
                        predicted = bool(np.any(second_derivative <= float(threshold)))
                    else:
                        predicted = bool(np.any(second_derivative >= abs(float(threshold))))
                    expected = segment["should_detect"]
                    if predicted and expected:
                        true_positive += 1
                    elif predicted and not expected:
                        false_positive += 1
                    elif not predicted and expected:
                        false_negative += 1
                    else:
                        true_negative += 1
                positive_count = true_positive + false_negative
                negative_count = true_negative + false_positive
                false_negative_rate = false_negative / positive_count if positive_count else 0.0
                false_positive_rate = false_positive / negative_count if negative_count else 0.0
                results.append(
                    {
                        "threshold": float(threshold),
                        "smoothing_sigma": float(sigma),
                        "derivative_polarity": int(polarity),
                        "true_positive": true_positive,
                        "true_negative": true_negative,
                        "false_positive": false_positive,
                        "false_negative": false_negative,
                        "false_positive_rate": false_positive_rate,
                        "false_negative_rate": false_negative_rate,
                        "balanced_error": (false_negative_rate + false_positive_rate) / 2.0,
                    }
                )
    results.sort(key=lambda item: (item["balanced_error"], item["false_negative_rate"], item["false_positive_rate"]))
    return {
        "reviewed_segments": len(labeled_segments),
        "results": results,
        "best": results[0] if results else None,
        "scope": "reviewed_candidate_windows_v1",
    }


def save_optimized_detection_parameters(best: dict, path: str | Path = OPTIMIZED_DETECTION_PARAMETERS_FILE) -> Path:
    """Persist the best reviewed detection parameters for later reuse."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": float(best["threshold"]),
        "smoothing_sigma": float(best["smoothing_sigma"]),
        "derivative_polarity": int(best.get("derivative_polarity", -1)),
        "balanced_error": float(best["balanced_error"]),
        "false_positive": int(best["false_positive"]),
        "false_negative": int(best["false_negative"]),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return output_path


def load_optimized_detection_parameters(path: str | Path = OPTIMIZED_DETECTION_PARAMETERS_FILE) -> dict | None:
    """Load persisted optimized detection parameters when available."""
    input_path = Path(path)
    if not input_path.exists():
        return None
    with input_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return {
        "threshold": float(loaded["threshold"]),
        "smoothing_sigma": float(loaded["smoothing_sigma"]),
        "derivative_polarity": int(loaded.get("derivative_polarity", -1)),
        "balanced_error": float(loaded.get("balanced_error", 0.0)),
        "false_positive": int(loaded.get("false_positive", 0)),
        "false_negative": int(loaded.get("false_negative", 0)),
    }


def _load_reviewed_segments(root: str | Path) -> list[dict]:
    import pandas as pd

    labeled_segments_by_path: dict[str, dict] = {}
    for path in _list_pending_annotation_files(root):
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            if not _row_is_reviewed(row):
                continue
            status = annotation_review_status_from_row(row)
            if status not in {"normal", "weird_signal", "jump_drill", "not_a_jump", "not_a_jump_or_drill", "manual_missing_jump"}:
                continue
            segment_path = Path(str(row.get("path", "")))
            if not segment_path.exists() and str(segment_path).replace("\\", "/") not in hdf5_segment_paths():
                continue
            normalized_path = str(segment_path).replace("\\", "/")
            record = {
                "path": segment_path,
                "should_detect": status in {"normal", "manual_missing_jump"},
            }
            previous = labeled_segments_by_path.get(normalized_path)
            if previous is not None and previous["should_detect"] != record["should_detect"]:
                raise ValueError(f"Conflicting reviewed detection labels for segment: {normalized_path}")
            labeled_segments_by_path[normalized_path] = record
    return list(labeled_segments_by_path.values())


def _list_pending_annotation_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(root_path.glob("*_for_annotation*.csv"))


def _detection_outcome(status: str) -> str | None:
    if status == "manual_missing_jump":
        return "false_negative"
    if status in {"not_a_jump", "not_a_jump_or_drill", "jump_drill", "weird_signal"}:
        return "false_positive"
    if status == "normal":
        return "reviewed_detected"
    return None


def _row_is_reviewed(row) -> bool:
    """Require an explicit completed status when the annotation column exists.

    Legacy annotation files without ``annotation_status`` predate that field and
    remain auditable. New pending files must never be mistaken for validated
    normal detections merely because their default detector status is normal.
    """
    if "annotation_status" not in row.index:
        return True
    return str(row.get("annotation_status", "")).strip().lower() in {"annotated", "completed", "finalized"}


def _row_audit_category(row) -> tuple[str | None, str | None]:
    for column in _AUDIT_CATEGORY_COLUMNS:
        value = str(row.get(column, "")).strip()
        if not value or value.lower() in {"nan", "none"}:
            continue
        normalized = value.lower().replace("-", "_")
        return _CATEGORY_ALIASES.get(normalized, normalized), column
    return None, None


def _row_rotation_direction(row) -> str:
    for column in _ROTATION_DIRECTION_COLUMNS:
        value = str(row.get(column, "")).strip().lower()
        if value in {"left", "gauche", "l"}:
            return "left"
        if value in {"right", "droite", "r"}:
            return "right"
    return "unknown"


def _default_audit_category(outcome: str) -> str:
    return "unclassified_false_negative" if outcome == "false_negative" else "unclassified_false_positive"


def _audit_category_summary(records: list[dict]) -> list[dict]:
    counts = Counter((record["outcome"], record["category"], record["rotation_direction"]) for record in records)
    return [
        {
            "outcome": outcome,
            "category": category,
            "rotation_direction": direction,
            "count": count,
        }
        for (outcome, category, direction), count in sorted(counts.items())
    ]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_float(value, default: float = 0.0) -> float:
    return safe_float(value, default)
