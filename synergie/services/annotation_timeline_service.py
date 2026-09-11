from __future__ import annotations

from collections.abc import Callable

from synergie.services.annotation_service import (
    JUMP_TYPE_LABELS,
    get_annotation_block_sync_offset,
    get_annotation_block_sync_source,
    get_annotation_sensor_sync_source,
)


def build_annotation_timeline_items(
    annotation_frame,
    metadata: dict,
    *,
    row_video_time_ms: Callable[[object, str], float],
    imu_to_video_ms: Callable[[float, str], float],
    format_ms: Callable[[float], str],
) -> list[dict]:
    """Build sorted listbox items for the Annotate session timeline."""
    items: list[dict] = []
    if annotation_frame is None:
        return items

    if "sensor_id" in annotation_frame and "impact_offset_ms" in annotation_frame:
        for sensor_id, rows in annotation_frame.groupby("sensor_id", sort=False):
            sensor_key = str(sensor_id)
            impact_ms = float(rows["impact_offset_ms"].dropna().iloc[0]) if not rows["impact_offset_ms"].dropna().empty else 0.0
            impact_video_ms = imu_to_video_ms(impact_ms, sensor_key)
            raw_impact_video_ms = _raw_impact_video_ms(metadata, impact_ms, impact_video_ms)
            items.append(
                {
                    "kind": "sync_impact",
                    "sensor_id": sensor_key,
                    "impact_ms": impact_ms,
                    "video_ms": impact_video_ms,
                    "raw_video_ms": raw_impact_video_ms,
                    "sort_ms": impact_video_ms,
                }
            )

    for index, row in annotation_frame.iterrows():
        sensor_id = str(row.get("sensor_id", ""))
        video_ms = row_video_time_ms(row, sensor_id)
        label = _jump_label(index, row, video_ms, format_ms)
        items.append({"kind": "jump", "dataframe_index": index, "label": label, "sort_ms": video_ms})

    sorted_items = sorted(items, key=lambda item: (float(item.get("sort_ms", 0.0)), 0 if item["kind"] == "sync_impact" else 1))
    for item in sorted_items:
        item["label"] = _sync_label(item, metadata, format_ms) if item["kind"] == "sync_impact" else item["label"]
    return sorted_items


def _jump_label(index: int, row, video_ms: float, format_ms: Callable[[float], str]) -> str:
    label = (
        f"{index + 1:03d} | {format_ms(video_ms)} | "
        f"{row.get('athlete_id', row.get('skater', 'unknown'))} | "
        f"{row.get('detection_status', 'detected_jump')}"
    )
    if str(row.get("prediction_source", "") or ""):
        jump_type = int(float(row.get("type", 8)))
        success = int(float(row.get("success", 2)))
        label += f" | model: {JUMP_TYPE_LABELS.get(jump_type, jump_type)} / success {success}"
    return label


def _sync_label(item: dict, metadata: dict, format_ms: Callable[[float], str]) -> str:
    block_source = get_annotation_block_sync_source(metadata)
    source = block_source or get_annotation_sensor_sync_source(metadata, item["sensor_id"])
    source_label = source.get("method", "not synced") if source else "not synced"
    timing_note = _sync_timing_note(item)
    return (
        f"SYNC | {format_ms(item['video_ms'])} | sensor_{item['sensor_id']} | "
        f"impact IMU {item['impact_ms']:.0f} ms ({format_ms(item['impact_ms'])}) | {source_label}{timing_note}"
    )


def _raw_impact_video_ms(metadata: dict, impact_ms: float, displayed_video_ms: float) -> float:
    block_offset_ms = get_annotation_block_sync_offset(metadata)
    if block_offset_ms is not None:
        return float(impact_ms) + block_offset_ms
    return float(displayed_video_ms)


def _sync_timing_note(item: dict) -> str:
    raw_video_ms = item.get("raw_video_ms", item.get("video_ms", 0.0))
    try:
        raw_value = float(raw_video_ms)
    except (TypeError, ValueError):
        return ""
    if raw_value < 0:
        return f" | maps before video start ({raw_value:.0f} ms)"
    return ""
