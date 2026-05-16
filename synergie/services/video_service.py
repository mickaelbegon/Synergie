from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


_TEXT_DATETIME_PATTERNS = (
    re.compile(r"(?P<date>\d{8})(?P<time>\d{6})"),
    re.compile(r"(?P<date>\d{8})(?P<time>\d{4})(?!\d)"),
    re.compile(r"(?P<date>\d{8})[_-]?(?P<time>\d{6})"),
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})[_T -]?(?P<time>\d{2}[-:]\d{2}[-:]\d{2})"),
)


def list_video_files(directory: str | Path, recursive: bool = False) -> list[Path]:
    """Return supported video files from one directory."""
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return []
    suffixes = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    iterator = directory_path.rglob("*") if recursive else directory_path.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in suffixes)


def annotation_reference_datetime(annotation_csv_path: str | Path, annotation_rows=None) -> datetime | None:
    """Infer the recording datetime that should be matched to session videos."""
    if annotation_rows is not None:
        candidate_values: list[datetime] = []
        for value in annotation_rows.get("recorded_at", []):
            parsed = parse_datetime_value(value)
            if parsed is not None:
                candidate_values.append(parsed)
        if candidate_values:
            return min(candidate_values)
    for value in Path(annotation_csv_path).stem.split("_"):
        parsed = parse_datetime_from_text(value)
        if parsed is not None:
            return parsed
    return parse_datetime_from_text(Path(annotation_csv_path).stem)


def read_video_metadata(video_path: str | Path) -> dict:
    """Read candidate timestamps for one video and choose the best source."""
    path = Path(video_path)
    stat = path.stat()
    candidates: list[tuple[str, datetime]] = []
    ffprobe_datetime = read_video_creation_time_with_ffprobe(path)
    if ffprobe_datetime is not None:
        candidates.append(("ffprobe.creation_time", ffprobe_datetime))
    filename_datetime = parse_datetime_from_text(path.stem)
    if filename_datetime is not None:
        candidates.append(("filename", filename_datetime))
    candidates.append(("filesystem.modified", datetime.fromtimestamp(stat.st_mtime)))
    candidates.append(("filesystem.created", datetime.fromtimestamp(stat.st_ctime)))
    best_source, best_datetime = select_best_video_datetime(candidates)
    return {
        "path": path,
        "name": path.name,
        "recorded_at": best_datetime,
        "recorded_at_source": best_source,
        "candidates": [{"source": source, "recorded_at": value} for source, value in candidates],
    }


def find_matching_videos(
    annotation_csv_path: str | Path,
    video_directory: str | Path,
    *,
    annotation_rows=None,
    recursive: bool = True,
    limit: int = 5,
) -> dict:
    """Find videos whose timestamps are closest to the annotation session."""
    reference_datetime = annotation_reference_datetime(annotation_csv_path, annotation_rows=annotation_rows)
    matches: list[dict] = []
    for video_path in list_video_files(video_directory, recursive=recursive):
        metadata = read_video_metadata(video_path)
        delta_seconds = None
        if reference_datetime is not None:
            delta_seconds = abs((metadata["recorded_at"] - reference_datetime).total_seconds())
        matches.append(
            {
                "path": metadata["path"],
                "name": metadata["name"],
                "recorded_at": metadata["recorded_at"],
                "recorded_at_source": metadata["recorded_at_source"],
                "delta_seconds": delta_seconds,
            }
        )
    matches.sort(
        key=lambda item: (
            float("inf") if item["delta_seconds"] is None else item["delta_seconds"],
            item["recorded_at"],
            item["name"].lower(),
        )
    )
    return {
        "reference_datetime": reference_datetime,
        "video_directory": Path(video_directory),
        "matches": matches[:limit],
        "match_count": len(matches),
    }


def parse_datetime_value(value) -> datetime | None:
    """Parse datetime-like values coming from CSVs or metadata."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return parse_datetime_from_text(text)


def parse_datetime_from_text(text: str) -> datetime | None:
    """Extract supported datetime patterns from free text."""
    for pattern in _TEXT_DATETIME_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        date_token = match.group("date").replace("-", "")
        time_token = match.group("time").replace("-", "").replace(":", "")
        if len(time_token) == 4:
            time_token = f"{time_token}00"
        try:
            return datetime.strptime(f"{date_token}_{time_token}", "%Y%m%d_%H%M%S")
        except ValueError:
            continue
    return None


def read_video_creation_time_with_ffprobe(video_path: Path) -> datetime | None:
    """Read container creation time with ffprobe when available."""
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_entries",
                "format_tags=creation_time",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    creation_time = (((payload.get("format") or {}).get("tags") or {}).get("creation_time"))
    return parse_datetime_value(creation_time)


def select_best_video_datetime(candidates: list[tuple[str, datetime]]) -> tuple[str, datetime]:
    """Choose the best timestamp source for one video."""
    if not candidates:
        raise ValueError("No datetime candidates available for video metadata.")
    priority = {
        "ffprobe.creation_time": 0,
        "filename": 1,
        "filesystem.modified": 2,
        "filesystem.created": 3,
    }
    return min(candidates, key=lambda item: (priority.get(item[0], 99), item[1]))
