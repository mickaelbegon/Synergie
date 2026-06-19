from __future__ import annotations

from pathlib import Path
import json

from synergie.services.annotation_service import annotation_review_status_from_row


OPTIMIZED_DETECTION_PARAMETERS_FILE = Path("config") / "optimized_detection_parameters.json"


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
) -> dict:
    """Score detector parameters on reviewed annotation segments."""
    import numpy as np
    import pandas as pd
    import scipy as sp

    thresholds = thresholds or [-0.4, -0.3, -0.2, -0.1, -0.05]
    smoothing_sigmas = smoothing_sigmas or [10, 20, 30, 40]
    labeled_segments = _load_reviewed_segments(root)

    results: list[dict] = []
    for sigma in smoothing_sigmas:
        for threshold in thresholds:
            true_positive = true_negative = false_positive = false_negative = 0
            for segment in labeled_segments:
                data = pd.read_csv(segment["path"])
                smoothed = sp.ndimage.gaussian_filter1d(data["Gyr_X"].to_numpy(), sigma=float(sigma))
                second_derivative = np.diff(np.diff(smoothed, prepend=smoothed[0]), prepend=0.0)
                predicted = bool(np.any(second_derivative <= float(threshold)))
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
        "balanced_error": float(loaded.get("balanced_error", 0.0)),
        "false_positive": int(loaded.get("false_positive", 0)),
        "false_negative": int(loaded.get("false_negative", 0)),
    }


def _load_reviewed_segments(root: str | Path) -> list[dict]:
    import pandas as pd

    labeled_segments: list[dict] = []
    for path in _list_pending_annotation_files(root):
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            status = annotation_review_status_from_row(row)
            if status not in {"normal", "weird_signal", "jump_drill", "not_a_jump", "not_a_jump_or_drill", "manual_missing_jump"}:
                continue
            segment_path = Path(str(row.get("path", "")))
            if not segment_path.exists():
                continue
            labeled_segments.append(
                {
                    "path": segment_path,
                    "should_detect": status in {"normal", "manual_missing_jump"},
                }
            )
    return labeled_segments


def _list_pending_annotation_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(root_path.glob("*_for_annotation*.csv"))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
