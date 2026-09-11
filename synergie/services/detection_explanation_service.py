from __future__ import annotations

from synergie.services.numeric_utils import safe_float


def threshold_intervals(ms, active_mask) -> list[tuple[float, float]]:
    """Return contiguous time intervals where a boolean mask is active."""
    intervals: list[tuple[float, float]] = []
    start = None
    for index, is_active in enumerate(active_mask):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            end = max(start, index - 1)
            intervals.append((float(ms[start]), float(ms[end])))
            start = None
    if start is not None:
        intervals.append((float(ms[start]), float(ms[len(active_mask) - 1])))
    return intervals


def nearest_interval(intervals: list[tuple[float, float]], reference_ms: float) -> tuple[float, float] | None:
    """Return the interval containing or closest to a reference timestamp."""
    if not intervals:
        return None
    containing = [interval for interval in intervals if interval[0] <= reference_ms <= interval[1]]
    if containing:
        return min(containing, key=lambda interval: interval[1] - interval[0])
    return min(intervals, key=lambda interval: min(abs(reference_ms - interval[0]), abs(reference_ms - interval[1])))


def angular_velocity_peak_context(dataframe, reference_ms: float) -> tuple[str, str]:
    """Describe the closest biomechanical context between threshold timing and |Gyr_X| peak."""
    if "ms" not in dataframe:
        return "", ""
    gyro_column = "Gyr_X_smoothed" if "Gyr_X_smoothed" in dataframe else "Gyr_X" if "Gyr_X" in dataframe else None
    if gyro_column is None:
        return "", ""

    import numpy as np

    ms = dataframe["ms"].to_numpy(dtype="float64")
    gyro = dataframe[gyro_column].to_numpy(dtype="float64")
    valid = np.isfinite(ms) & np.isfinite(gyro)
    if not valid.any():
        return "", ""
    valid_indices = np.where(valid)[0]
    peak_index = int(valid_indices[np.nanargmax(np.abs(gyro[valid]))])
    peak_ms = float(ms[peak_index])
    peak_value = float(gyro[peak_index])
    lag_ms = float(reference_ms - peak_ms)
    direction = "positive" if peak_value >= 0 else "negative"
    relation = "after" if lag_ms > 0 else "before"
    text = (
        "Angular-speed peak uses |Gyr_X| because rotation sign can flip: "
        f"peak at {peak_ms:.0f} ms ({peak_value:.0f} deg/s, {direction}); "
        f"threshold reference is {abs(lag_ms):.0f} ms {relation} that peak."
    )
    warning = ""
    if abs(lag_ms) > 250:
        warning = (
            f"Biomech check: threshold timing is {abs(lag_ms):.0f} ms from the |Gyr_X| peak. "
            "This may be a false positive or poor takeoff/landing bound."
        )
    return text, warning


def detection_threshold_context(
    dataframe,
    row,
    *,
    threshold: float,
    plot_scale: float,
) -> dict:
    """Return diagnostics explaining why the detector flagged one annotation row."""
    if "X_gyr_second_derivative" not in dataframe or "ms" not in dataframe:
        return {"diagnostic": "No derivative signal available to explain this detection.", "warning": "", "intervals": []}

    import numpy as np

    ms = dataframe["ms"].to_numpy(dtype="float64")
    derivative = dataframe["X_gyr_second_derivative"].to_numpy(dtype="float64")
    below_threshold = np.isfinite(derivative) & (derivative <= threshold)
    intervals = threshold_intervals(ms, below_threshold)

    if len(ms) == 0 or not np.isfinite(derivative).any():
        return {"diagnostic": "Derivative signal is empty or invalid.", "warning": "", "intervals": intervals}

    min_index = int(np.nanargmin(derivative))
    min_ms = float(ms[min_index])
    min_value = float(derivative[min_index])
    jump_start = safe_float(row.get("start_ms"), default=min_ms)
    selected_interval = nearest_interval(intervals, jump_start)

    if selected_interval is None:
        peak_text, _peak_warning = angular_velocity_peak_context(dataframe, min_ms)
        return {
            "diagnostic": (
                f"Detection threshold: Gyr_X_ddot <= {threshold:.3f}. "
                f"No under-threshold interval remains after outlier cleaning; minimum is {min_value:.3f} at {min_ms:.0f} ms. "
                f"{peak_text}"
            ),
            "warning": (
                "This looks like an old/noisy false positive: mark it 'No jump or drill' or 'Weird signal / bad bounds', "
                "or regenerate annotations after cleaning."
            ),
            "intervals": intervals,
            "min_ms": min_ms,
            "min_value": min_value,
        }

    start_ms, end_ms = selected_interval
    interval_mask = (ms >= start_ms) & (ms <= end_ms)
    interval_min = float(np.nanmin(derivative[interval_mask])) if interval_mask.any() else min_value
    margin = threshold - interval_min
    duration_ms = max(0.0, end_ms - start_ms)
    interval_center_ms = (start_ms + end_ms) / 2.0
    peak_text, peak_warning = angular_velocity_peak_context(dataframe, interval_center_ms)
    return {
        "diagnostic": (
            f"Why detected: Gyr_X_ddot crossed below threshold {threshold:.3f}. "
            f"Selected under-threshold zone: {start_ms:.0f}-{end_ms:.0f} ms ({duration_ms:.0f} ms). "
            f"Minimum {interval_min:.3f}, margin below threshold {margin:.3f}. "
            f"{peak_text} "
            f"Red shaded zones show all threshold crossings; red curve is displayed x{plot_scale:g} on the right axis."
        ),
        "warning": peak_warning,
        "intervals": intervals,
        "selected_interval": selected_interval,
        "min_ms": min_ms,
        "min_value": min_value,
    }
