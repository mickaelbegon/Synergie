from __future__ import annotations


def format_video_ms(milliseconds: float) -> str:
    """Format milliseconds as H:MM:SS.mmm or MM:SS.mmm for UI display."""
    value = max(float(milliseconds), 0.0)
    total_ms = int(round(value))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}.{ms:03d}"
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"


def parse_video_ms(text: str) -> float | None:
    """Parse H:MM:SS.mmm or MM:SS.mmm video timestamps into milliseconds."""
    value = str(text or "").strip()
    if not value:
        return None
    time_parts = value.split(":")
    if len(time_parts) == 2:
        hours = 0
        minutes_text, seconds_part = time_parts
    elif len(time_parts) == 3:
        hours_text, minutes_text, seconds_part = time_parts
        hours = int(hours_text)
    else:
        return None
    seconds_text, dot, millis_text = seconds_part.partition(".")
    if not dot:
        millis_text = "0"
    return float(
        (
            int(hours) * 3600
            + int(minutes_text) * 60
            + int(seconds_text)
        )
        * 1000
        + int(millis_text[:3].ljust(3, "0"))
    )


def format_file_size(size_bytes: int) -> str:
    """Format a byte count using compact binary units."""
    value = float(max(int(size_bytes), 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"
