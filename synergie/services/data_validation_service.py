from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from synergie.services.annotation_service import annotation_metadata_path, load_annotation_metadata
from synergie.services.hdf5_archive_service import hdf5_segment_paths
from synergie.services.session_service import list_sessions, session_metadata


def validate_data_files(
    *,
    raw_root: str | Path = "data/raw",
    pending_root: str | Path = "data/pending",
    annotated_root: str | Path = "data/annotated",
    hdf5_archive_path: str | Path | None = "data/synergie_archive.h5",
) -> dict:
    """Validate cross-folder data references used by the GUI workflow."""
    raw_root = Path(raw_root)
    pending_root = Path(pending_root)
    annotated_root = Path(annotated_root)
    issues: list[dict] = []
    referenced_pending_segments: set[str] = set()
    archive_segments = _safe_hdf5_segment_paths(hdf5_archive_path)
    raw_files_by_name = _raw_files_by_name(raw_root)

    _validate_configured_raw_sessions(raw_root, pending_root, archive_segments, issues)
    _validate_unregistered_raw_csvs(raw_root, issues)
    _validate_pending_annotations(
        pending_root,
        issues,
        referenced_pending_segments,
        archive_segments,
        raw_files_by_name,
    )
    _validate_orphan_pending_segments(pending_root, issues, referenced_pending_segments, archive_segments)
    _validate_training_dataset(annotated_root, issues, archive_segments)

    counts = defaultdict(int)
    for issue in issues:
        counts[issue["severity"]] += 1
    return {
        "issues": issues,
        "summary": {
            "errors": counts["error"],
            "warnings": counts["warning"],
            "info": counts["info"],
            "total": len(issues),
        },
    }


def format_data_validation_report(result: dict) -> str:
    """Return a compact text report for the GUI."""
    summary = result.get("summary", {})
    issues = result.get("issues", [])
    header = (
        f"Data validation: {summary.get('errors', 0)} error(s), "
        f"{summary.get('warnings', 0)} warning(s), {summary.get('info', 0)} info."
    )
    if not issues:
        return f"{header}\n\nNo obvious data consistency issue found."
    lines = [header, ""]
    for issue in issues:
        severity = str(issue.get("severity", "")).upper()
        area = issue.get("area", "data")
        path = issue.get("path", "")
        message = issue.get("message", "")
        suggestion = issue.get("suggestion", "")
        lines.append(f"[{severity}] {area}: {message}")
        if path:
            lines.append(f"  Path: {path}")
        if suggestion:
            lines.append(f"  Suggestion: {suggestion}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _validate_configured_raw_sessions(
    raw_root: Path,
    pending_root: Path,
    archive_segments: set[str],
    issues: list[dict],
) -> None:
    for session_name in list_sessions():
        metadata = session_metadata(session_name)
        session_dir = raw_root / metadata["path"]
        if not session_dir.exists():
            if _session_has_derived_data(session_name, pending_root, archive_segments):
                _add_issue(
                    issues,
                    "info",
                    "raw",
                    session_dir,
                    f"Configured session {session_name} has no local raw folder, but derived annotation/segment data is available.",
                    "Raw files are only needed if you want to regenerate detections or inspect the full raw session.",
                )
                continue
            _add_issue(
                issues,
                "error",
                "raw",
                session_dir,
                f"Configured session {session_name} points to a missing raw folder.",
                "Copy the raw files for this session or update the session path in Data | Sessions.",
            )
            continue
        if not _processable_raw_csvs(session_dir):
            _add_issue(
                issues,
                "warning",
                "raw",
                session_dir,
                f"Configured session {session_name} has no processable raw CSV.",
                "Check whether the folder contains only derived files or whether the session path is wrong.",
            )


def _validate_unregistered_raw_csvs(raw_root: Path, issues: list[dict]) -> None:
    if not raw_root.exists():
        return
    registered_dirs = {(raw_root / session_metadata(session)["path"]).resolve() for session in list_sessions()}
    seen_dirs: set[Path] = set()
    for csv_path in raw_root.rglob("*.csv"):
        if "jumplist" in csv_path.stem.lower():
            continue
        try:
            parent = csv_path.parent.resolve()
        except OSError:
            parent = csv_path.parent
        if parent in registered_dirs or parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        _add_issue(
            issues,
            "warning",
            "raw",
            parent,
            "Raw CSV folder is not referenced by any configured session.",
            "Add the session in Data | Sessions if these files should be processed.",
        )


def _validate_pending_annotations(
    pending_root: Path,
    issues: list[dict],
    referenced_pending_segments: set[str],
    archive_segments: set[str],
    raw_files_by_name: dict[str, list[Path]],
) -> None:
    for annotation_csv in sorted(pending_root.glob("*_for_annotation*.csv")):
        try:
            import pandas as pd

            frame = pd.read_csv(annotation_csv)
        except Exception as exc:
            _add_issue(
                issues,
                "error",
                "pending",
                annotation_csv,
                f"Pending annotation CSV cannot be read: {exc}",
                "Open or regenerate this annotation file.",
            )
            continue
        if "path" not in frame.columns:
            _add_issue(
                issues,
                "error",
                "pending",
                annotation_csv,
                "Pending annotation CSV has no 'path' column.",
                "Regenerate this annotation file.",
            )
            continue
        for path_value in frame["path"].fillna("").astype(str):
            if not path_value:
                continue
            normalized = _normalize_path(path_value)
            referenced_pending_segments.add(normalized)
            if _segment_exists(path_value, archive_segments):
                continue
            _add_issue(
                issues,
                "error",
                "pending",
                annotation_csv,
                f"Annotation references a missing segment: {path_value}",
                "Restore the segment CSV, rebuild the HDF5 archive, or regenerate annotations.",
            )
        if "source_file" in frame.columns:
            missing_sources = []
            for source_name in sorted(set(frame["source_file"].dropna().astype(str))):
                if source_name and source_name not in raw_files_by_name:
                    missing_sources.append(source_name)
            if missing_sources:
                preview = ", ".join(missing_sources[:5])
                suffix = "" if len(missing_sources) <= 5 else f", +{len(missing_sources) - 5} more"
                segment_paths = [path_value for path_value in frame["path"].fillna("").astype(str) if path_value]
                segments_available = bool(segment_paths) and all(
                    _segment_exists(path_value, archive_segments) for path_value in segment_paths
                )
                severity = "info" if segments_available else "warning"
                suggestion = (
                    "Raw CSVs are optional for normal annotation/training because the segments are available; "
                    "copy raw files back only if you need full-session inspection or regeneration."
                    if segments_available
                    else "Copy raw CSVs back locally if sync-impact plots or manual jump review need them."
                )
                _add_issue(
                    issues,
                    severity,
                    "pending",
                    annotation_csv,
                    f"Annotation references {len(missing_sources)} source_file(s) not found under data/raw: {preview}{suffix}",
                    suggestion,
                )
        _validate_annotation_metadata(annotation_csv, frame, issues)


def _validate_annotation_metadata(annotation_csv: Path, frame, issues: list[dict]) -> None:
    metadata_path = annotation_metadata_path(annotation_csv)
    if not metadata_path.exists():
        return
    try:
        metadata = load_annotation_metadata(annotation_csv)
    except Exception as exc:
        _add_issue(
            issues,
            "error",
            "pending",
            metadata_path,
            f"Annotation metadata JSON cannot be read: {exc}",
            "Delete or repair this sidecar JSON, then reload the annotation file.",
        )
        return
    video_path = str(metadata.get("video_path", "") or "")
    if video_path and not Path(video_path).exists():
        _add_issue(
            issues,
            "info",
            "pending",
            metadata_path,
            f"Saved video path does not exist: {video_path}",
            "Choose the video again from Data | Annotate on this computer.",
        )
    if "sensor_id" in frame.columns:
        sensors = {str(sensor_id) for sensor_id in frame["sensor_id"].dropna().astype(str)}
        offsets = {str(sensor_id) for sensor_id in metadata.get("sensor_sync_offsets_ms", {})}
        unknown = sorted(offsets - sensors)
        if unknown:
            _add_issue(
                issues,
                "warning",
                "pending",
                metadata_path,
                f"Sync offsets exist for sensor(s) not present in the annotation file: {', '.join(unknown)}",
                "Clear obsolete sync metadata if it came from an older annotation file.",
            )


def _validate_orphan_pending_segments(
    pending_root: Path,
    issues: list[dict],
    referenced_pending_segments: set[str],
    archive_segments: set[str],
) -> None:
    segment_root = pending_root / "segments"
    if not segment_root.exists():
        return
    for segment_path in sorted(segment_root.rglob("*.csv")):
        normalized = _normalize_path(segment_path)
        if normalized in referenced_pending_segments or normalized in archive_segments:
            continue
        _add_issue(
            issues,
            "warning",
            "pending",
            segment_path,
            "Pending segment CSV is not referenced by any pending annotation file.",
            "It may be leftover from a regenerated annotation run; archive or delete it after confirming.",
        )


def _validate_training_dataset(annotated_root: Path, issues: list[dict], archive_segments: set[str]) -> None:
    jumplist_path = annotated_root / "total" / "jumplist.csv"
    if not jumplist_path.exists():
        return
    try:
        import pandas as pd

        frame = pd.read_csv(jumplist_path)
    except Exception as exc:
        _add_issue(
            issues,
            "error",
            "annotated",
            jumplist_path,
            f"Training jumplist cannot be read: {exc}",
            "Repair or restore data/annotated/total/jumplist.csv.",
        )
        return
    if "path" not in frame.columns:
        _add_issue(
            issues,
            "error",
            "annotated",
            jumplist_path,
            "Training jumplist has no 'path' column.",
            "Repair the training jumplist before retraining models.",
        )
        return
    for path_value in frame["path"].fillna("").astype(str):
        if path_value and not _segment_exists(path_value, archive_segments):
            _add_issue(
                issues,
                "error",
                "annotated",
                jumplist_path,
                f"Training jumplist references a missing segment: {path_value}",
                "Restore the segment CSV or ensure it is present in data/synergie_archive.h5.",
            )


def _safe_hdf5_segment_paths(hdf5_archive_path: str | Path | None) -> set[str]:
    if hdf5_archive_path is None:
        return set()
    try:
        return {_normalize_path(path) for path in hdf5_segment_paths(hdf5_archive_path)}
    except (OSError, ValueError, KeyError):
        return set()


def _session_has_derived_data(session_name: str, pending_root: Path, archive_segments: set[str]) -> bool:
    if pending_root.exists() and any(pending_root.glob(f"{session_name}*_for_annotation*.csv")):
        return True
    segment_prefix = _normalize_path(pending_root / "segments" / session_name)
    return any(path.startswith(f"{segment_prefix}/") for path in archive_segments)


def _raw_files_by_name(raw_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    if not raw_root.exists():
        return result
    for path in raw_root.rglob("*.csv"):
        result[path.name].append(path)
    return result


def _processable_raw_csvs(session_dir: Path) -> list[Path]:
    return sorted(path for path in session_dir.glob("*.csv") if "jumplist" not in path.stem.lower())


def _segment_exists(path_value: str | Path, archive_segments: set[str]) -> bool:
    normalized = _normalize_path(path_value)
    return Path(str(path_value)).exists() or normalized in archive_segments


def _normalize_path(path_value: str | Path) -> str:
    return str(path_value).replace("\\", "/")


def _add_issue(
    issues: list[dict],
    severity: str,
    area: str,
    path: str | Path,
    message: str,
    suggestion: str = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "area": area,
            "path": str(path),
            "message": message,
            "suggestion": suggestion,
        }
    )
