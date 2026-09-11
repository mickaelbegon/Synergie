from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def redetect_annotation_sync_impacts(
    annotation_frame,
    *,
    raw_path_for_sensor: Callable[[str], Path | None],
    estimate_impact_ms: Callable[[Path], float | None],
):
    """Return an annotation frame with refreshed per-sensor sync impacts."""
    if "sensor_id" not in annotation_frame.columns:
        raise ValueError("Annotation file has no sensor_id column.")

    repaired = annotation_frame.copy()
    updated: list[dict] = []
    skipped: list[str] = []
    sensor_ids = sorted({str(value) for value in repaired["sensor_id"].dropna().astype(str)})
    for sensor_id in sensor_ids:
        raw_path = raw_path_for_sensor(sensor_id)
        if raw_path is None:
            skipped.append(f"sensor_{sensor_id}: raw CSV not found")
            continue
        try:
            impact_ms = estimate_impact_ms(raw_path)
        except Exception as exc:
            skipped.append(f"sensor_{sensor_id}: {exc}")
            continue
        if impact_ms is None:
            skipped.append(f"sensor_{sensor_id}: no reliable impact found")
            continue

        impact_ms = round(float(impact_ms), 3)
        mask = repaired["sensor_id"].astype(str) == sensor_id
        repaired.loc[mask, "impact_offset_ms"] = impact_ms
        if "start_ms" in repaired.columns:
            starts = repaired.loc[mask, "start_ms"].astype(float)
            repaired.loc[mask, "synced_start_ms"] = (starts - impact_ms).round(3)
        if "end_ms" in repaired.columns:
            ends = repaired.loc[mask, "end_ms"].astype(float)
            repaired.loc[mask, "synced_end_ms"] = (ends - impact_ms).round(3)
        updated.append({"sensor_id": sensor_id, "impact_ms": impact_ms, "raw_path": str(raw_path)})

    return {"frame": repaired, "updated": updated, "skipped": skipped}
