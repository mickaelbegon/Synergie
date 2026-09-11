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


def format_video_cache_button(size_bytes: int) -> str:
    """Return the annotation video cache clear button label."""
    return f"Clear cache ({format_file_size(size_bytes)})"


def format_annotation_video_info(
    *,
    video_name: str,
    frame_count: int,
    duration_ms: float | None,
    cache_note: str = "",
) -> str:
    """Return the compact annotation video info line shown below the chooser."""
    duration_text = format_video_ms(duration_ms) if duration_ms else "unknown"
    return f"{video_name} | {int(frame_count)} frames | {duration_text}{cache_note}"


def format_video_preparation_message(cache_status: dict, proxy_status: dict) -> str:
    """Return the modal progress message used while preparing annotation video playback."""
    cache_needed = bool(cache_status.get("copy_needed"))
    proxy_needed = bool(proxy_status.get("proxy_needed"))
    if cache_needed and proxy_needed:
        return (
            "Preparing video for smooth playback.\n"
            f"1/2 Caching local copy ({format_file_size(cache_status['size_bytes'])}).\n"
            "2/2 Optimizing playback proxy with ffmpeg."
        )
    if proxy_needed:
        return "Optimizing video for smoother playback with ffmpeg.\nThis happens only once per video."
    return f"Caching video locally ({format_file_size(cache_status['size_bytes'])}).\nPlease wait..."


def format_video_cache_cleared_status(result: dict) -> str:
    """Return the status-bar text shown after clearing cached annotation videos."""
    return f"Video cache cleared: {int(result['removed_files'])} file(s), {format_file_size(result['freed_bytes'])} freed."
