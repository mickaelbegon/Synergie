from __future__ import annotations

from collections.abc import Callable, Iterable

from synergie.services.annotation_service import (
    compute_annotation_jump_video_time_ms,
    get_annotation_block_sync_offset,
    get_annotation_block_sync_source,
    get_annotation_sensor_sync_offset,
    get_annotation_sensor_sync_source,
)


def annotation_row_video_time_ms(metadata: dict, row, sensor_id: str | None = None) -> float:
    """Return the video timestamp for one annotation row using block sync when available."""
    block_offset_ms = get_annotation_block_sync_offset(metadata)
    if block_offset_ms is not None:
        return compute_annotation_jump_video_time_ms(row, block_sync_offset_ms=block_offset_ms)
    resolved_sensor_id = sensor_id if sensor_id is not None else str(row.get("sensor_id", ""))
    offset_ms = get_annotation_sensor_sync_offset(metadata, resolved_sensor_id)
    return compute_annotation_jump_video_time_ms(row, offset_ms)


def annotation_imu_to_video_ms(metadata: dict, imu_ms: float, sensor_id: str | None = None) -> float:
    """Return the video timestamp for one IMU timestamp using the active sync mode."""
    block_offset_ms = get_annotation_block_sync_offset(metadata)
    if block_offset_ms is not None:
        return max(float(imu_ms) + block_offset_ms, 0.0)
    offset_ms = get_annotation_sensor_sync_offset(metadata, sensor_id or "")
    return max(float(imu_ms) + offset_ms, 0.0)


def annotation_video_to_imu_ms(
    metadata: dict,
    video_ms: float,
    sensor_id: str | None = None,
    *,
    impact_offset_ms: float = 0.0,
) -> float:
    """Return the IMU timestamp corresponding to one video timestamp."""
    block_offset_ms = get_annotation_block_sync_offset(metadata)
    if block_offset_ms is not None:
        return float(video_ms) - block_offset_ms
    offset_ms = get_annotation_sensor_sync_offset(metadata, sensor_id or "")
    return float(video_ms) - offset_ms + float(impact_offset_ms)


def block_offset_from_jump(video_ms: float, row) -> float:
    """Return the block sync offset implied by aligning one selected jump to video time."""
    start_ms = float(row.get("start_ms", row.get("synced_start_ms", 0.0)) or 0.0)
    return float(video_ms) - start_ms


def block_offset_from_impact(video_ms: float, impact_ms: float) -> float:
    """Return the block sync offset implied by aligning one IMU impact to video time."""
    return float(video_ms) - float(impact_ms)


def annotation_sync_summary(metadata: dict, sensor_ids: Iterable[str], current_sensor_id: str = "") -> str:
    """Return a compact human-readable status for annotation video sync."""
    block_offset = get_annotation_block_sync_offset(metadata)
    if block_offset is not None:
        return "Block sync active for all sensors"
    offsets = metadata.get("sensor_sync_offsets_ms", {})
    synced = sorted(str(sensor_id) for sensor_id in offsets)
    sensors = sorted(str(sensor_id) for sensor_id in sensor_ids)
    missing = [sensor_id for sensor_id in sensors if sensor_id not in offsets]
    current_status = "synced" if current_sensor_id in offsets else "not synced"
    return (
        f"Current sensor {current_status}; "
        f"synced: {', '.join(synced) if synced else 'none'}; "
        f"missing: {', '.join(missing) if missing else 'none'}"
    )


def annotation_sync_source_summary(
    metadata: dict,
    sensor_id: str,
    *,
    format_ms: Callable[[float], str],
) -> str:
    """Return a compact description of how the active sync offset was produced."""
    block_source = get_annotation_block_sync_source(metadata)
    if block_source:
        method = str(block_source.get("method", "block")).strip().lower()
        sensor = block_source.get("sensor_id")
        video_ms = block_source.get("video_ms")
        imu_ms = block_source.get("imu_impact_ms")
        if method == "block_impact" and imu_ms is not None and video_ms is not None:
            sensor_text = f" sensor {sensor}" if sensor is not None else ""
            return f"block sync by impact{sensor_text}: IMU {format_ms(float(imu_ms))} -> video {format_ms(float(video_ms))}"
        if method == "block_jump" and video_ms is not None:
            return f"block sync by selected jump at video {format_ms(float(video_ms))}"
        return f"block sync by {method}"

    source = get_annotation_sensor_sync_source(metadata, sensor_id)
    method = str(source.get("method", "")).strip().lower()
    if method == "impact":
        imu_ms = source.get("imu_impact_ms")
        video_ms = source.get("video_ms")
        if imu_ms is not None and video_ms is not None:
            return f"synced by impact: IMU {format_ms(float(imu_ms))} -> video {format_ms(float(video_ms))}"
        return "synced by impact"
    if method == "jump":
        video_ms = source.get("video_ms")
        if video_ms is not None:
            return f"synced by selected jump at video {format_ms(float(video_ms))}"
        return "synced by selected jump"
    if method:
        return f"synced by {method}"
    return "sync source unknown"
