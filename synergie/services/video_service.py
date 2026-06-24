from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import hashlib


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


def cached_video_path(video_path: str | Path, cache_root: str | Path = ".tmp/video_cache") -> dict:
    """Return a local cached copy for videos opened from another drive or a network path."""
    source = Path(video_path).resolve()
    status = video_cache_status(source, cache_root=cache_root)
    if not status["should_cache"]:
        return {"path": source, "source_path": source, "from_cache": False, "copied": False}

    stat = status["source_stat"]
    cached = status["cache_path"]
    cached.parent.mkdir(parents=True, exist_ok=True)
    copied = False
    if not cached.exists() or cached.stat().st_size != stat.st_size:
        try:
            shutil.copy2(source, cached)
            copied = True
        except OSError as exc:
            if cached.exists() and cached.stat().st_size == stat.st_size:
                return {"path": cached.resolve(), "source_path": source, "from_cache": True, "copied": False, "copy_error": str(exc)}
            return {"path": source, "source_path": source, "from_cache": False, "copied": False, "cache_failed": True, "copy_error": str(exc)}
    return {"path": cached.resolve(), "source_path": source, "from_cache": True, "copied": copied}


def optimized_playback_video_path(video_path: str | Path, cache_root: str | Path = ".tmp/video_cache") -> dict:
    """Return the fastest local playback path available for annotation review."""
    cache_result = cached_video_path(video_path, cache_root=cache_root)
    proxy_status = video_proxy_status(video_path, cache_root=cache_root)
    if not proxy_status["can_create_proxy"]:
        return {**cache_result, "from_proxy": False, "proxy_created": False, "proxy_failed": False}
    if proxy_status["proxy_needed"]:
        input_path = Path(cache_result["path"])
        try:
            _create_video_proxy(input_path, proxy_status["proxy_path"])
        except (OSError, subprocess.SubprocessError):
            return {**cache_result, "from_proxy": False, "proxy_created": False, "proxy_failed": True}
        proxy_created = True
    else:
        proxy_created = False
    return {
        **cache_result,
        "path": proxy_status["proxy_path"].resolve(),
        "from_proxy": True,
        "proxy_created": proxy_created,
        "proxy_failed": False,
    }


def video_cache_status(video_path: str | Path, cache_root: str | Path = ".tmp/video_cache") -> dict:
    """Return cache target and whether copying will be needed before playback."""
    source = Path(video_path).resolve()
    should_cache = should_cache_video(source, Path.cwd().resolve())
    stat = source.stat()
    fingerprint = hashlib.sha1(f"{source}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8")).hexdigest()[:16]
    cache_path = Path(cache_root) / f"{source.stem}-{fingerprint}{source.suffix.lower()}"
    copy_needed = should_cache and (not cache_path.exists() or cache_path.stat().st_size != stat.st_size)
    return {
        "source_path": source,
        "source_stat": stat,
        "should_cache": should_cache,
        "cache_path": cache_path,
        "copy_needed": copy_needed,
        "size_bytes": stat.st_size,
    }


def video_proxy_status(video_path: str | Path, cache_root: str | Path = ".tmp/video_cache") -> dict:
    """Return the optional low-resolution proxy path used for faster playback."""
    source = Path(video_path).resolve()
    stat = source.stat()
    fingerprint = hashlib.sha1(f"{source}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8")).hexdigest()[:16]
    proxy_path = Path(cache_root) / f"{source.stem}-{fingerprint}-proxy.mp4"
    ffmpeg_path = shutil.which("ffmpeg")
    proxy_exists = proxy_path.exists() and proxy_path.stat().st_size > 0
    return {
        "source_path": source,
        "proxy_path": proxy_path,
        "can_create_proxy": ffmpeg_path is not None,
        "ffmpeg_path": ffmpeg_path,
        "proxy_needed": ffmpeg_path is not None and not proxy_exists,
        "proxy_exists": proxy_exists,
    }


def _create_video_proxy(input_path: Path, proxy_path: Path) -> None:
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = proxy_path.with_suffix(".tmp.mp4")
    if temporary_path.exists():
        temporary_path.unlink()
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise OSError("ffmpeg is not available")
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vf",
        "scale='min(960,iw)':-2,fps=30",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    subprocess.run(command, check=True)
    temporary_path.replace(proxy_path)


def should_cache_video(source_path, workspace_path) -> bool:
    """Return whether a video should be copied locally before playback."""
    source_text = str(source_path)
    if source_text.startswith("\\\\"):
        return True

    source_drive = source_path.drive.lower()
    workspace_drive = workspace_path.drive.lower()
    if source_drive or workspace_drive:
        return source_drive != workspace_drive

    source_mount = _external_mount_key(source_path)
    workspace_mount = _external_mount_key(workspace_path)
    if source_mount is not None:
        return source_mount != workspace_mount
    return False


def _external_mount_key(path) -> tuple[str, ...] | None:
    parts = tuple(str(part).lower() for part in path.parts)
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "volumes":
        return ("volumes", parts[2])
    if len(parts) >= 4 and parts[0] == "/" and parts[1] == "media":
        return ("media", parts[2], parts[3])
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt":
        return ("mnt", parts[2])
    return None


def video_cache_size_bytes(cache_root: str | Path = ".tmp/video_cache") -> int:
    """Return the total size of cached local video copies."""
    cache_dir = Path(cache_root)
    if not cache_dir.exists() or not cache_dir.is_dir():
        return 0
    return sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())


def clear_video_cache(cache_root: str | Path = ".tmp/video_cache") -> dict:
    """Delete cached video files and return how much space was freed."""
    cache_dir = Path(cache_root)
    before_bytes = video_cache_size_bytes(cache_dir)
    removed_files = 0
    if cache_dir.exists() and cache_dir.is_dir():
        for path in sorted(cache_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
                removed_files += 1
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    return {"removed_files": removed_files, "freed_bytes": before_bytes}


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
