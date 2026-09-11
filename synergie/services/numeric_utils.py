from __future__ import annotations


def safe_float(value, default: float = 0.0):
    """Return value as float, falling back when conversion fails or value is NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result
