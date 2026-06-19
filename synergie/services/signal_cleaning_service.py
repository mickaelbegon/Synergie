from __future__ import annotations


ACCELERATION_COLUMNS = ("Acc_X", "Acc_Y", "Acc_Z")


def clean_acceleration_outliers(frame, limit_g: float = 32.0):
    """Replace impossible acceleration spikes with interpolated values."""
    cleaned = frame.copy()
    report = {"limit_g": float(limit_g), "replaced": 0, "columns": {}}
    for column in ACCELERATION_COLUMNS:
        if column not in cleaned:
            continue
        values = cleaned[column]
        mask = values.abs() > float(limit_g)
        replaced = int(mask.sum())
        report["columns"][column] = replaced
        report["replaced"] += replaced
        if replaced:
            cleaned.loc[mask, column] = float("nan")
            cleaned[column] = cleaned[column].interpolate(limit_direction="both").fillna(0.0)
    return cleaned, report
