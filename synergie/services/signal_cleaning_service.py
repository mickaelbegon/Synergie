from __future__ import annotations


ACCELERATION_COLUMNS = ("Acc_X", "Acc_Y", "Acc_Z")
GYROSCOPE_COLUMNS = ("Gyr_X", "Gyr_Y", "Gyr_Z", "Gyr_X_unfiltered", "Gyr_X_smoothed")


def _replace_outliers(cleaned, columns, limit: float, report_key: str):
    report = {"limit": float(limit), "replaced": 0, "columns": {}}
    for column in columns:
        if column not in cleaned:
            continue
        values = cleaned[column]
        mask = values.abs() > float(limit)
        replaced = int(mask.sum())
        report["columns"][column] = replaced
        report["replaced"] += replaced
        if replaced:
            cleaned.loc[mask, column] = float("nan")
            cleaned[column] = cleaned[column].interpolate(limit_direction="both").fillna(0.0)
    return {report_key: report}


def clean_acceleration_outliers(frame, limit_g: float = 32.0):
    """Replace impossible acceleration spikes with interpolated values."""
    cleaned = frame.copy()
    report = _replace_outliers(cleaned, ACCELERATION_COLUMNS, limit_g, "acceleration")["acceleration"]
    report["limit_g"] = report.pop("limit")
    return cleaned, report


def clean_imu_outliers(frame, acceleration_limit_g: float = 32.0, gyroscope_limit_dps: float = 5000.0):
    """Replace physically impossible IMU spikes with interpolated values."""
    cleaned = frame.copy()
    report = {}
    report.update(_replace_outliers(cleaned, ACCELERATION_COLUMNS, acceleration_limit_g, "acceleration"))
    report.update(_replace_outliers(cleaned, GYROSCOPE_COLUMNS, gyroscope_limit_dps, "gyroscope"))
    report["replaced"] = report["acceleration"]["replaced"] + report["gyroscope"]["replaced"]
    return cleaned, report


def recompute_gyro_x_derivatives(frame, *, smoothing_sigma: float = 30.0, threshold: float | None = None):
    """Rebuild smoothed gyro and derivative columns from the cleaned Gyr_X signal."""
    import scipy as sp

    prepared = frame.copy()
    if "Gyr_X" not in prepared:
        return prepared
    prepared["Gyr_X_smoothed"] = sp.ndimage.gaussian_filter1d(prepared["Gyr_X"], sigma=float(smoothing_sigma))
    prepared["X_gyr_derivative"] = prepared["Gyr_X_smoothed"].diff().fillna(0.0)
    prepared["X_gyr_second_derivative"] = prepared["X_gyr_derivative"].diff().fillna(0.0)
    if threshold is not None:
        prepared["X_gyr_second_derivative_crossing"] = [
            False if value > float(threshold) else True
            for value in prepared["X_gyr_second_derivative"]
        ]
    return prepared
