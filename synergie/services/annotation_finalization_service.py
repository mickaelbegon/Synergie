from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from synergie.services.annotation_service import annotation_metadata_path, summarize_annotation_progress
from synergie.services.training_dataset_service import (
    find_training_dataset_duplicates,
    training_dataset_state_path,
)
from synergie.services.workflow_state_service import mark_finalized_session, session_workflow_entry


def finalize_annotation_file(
    annotation_csv_path: str | Path,
    *,
    dataset_path: str | Path = "data/annotated/total",
    annotated_root: str | Path = "data/annotated",
    segment_archive_root: str | Path = "data/segment_csv_archive",
    hdf5_archive_path: str | Path | None = "data/synergie_archive.h5",
    dry_run: bool = False,
) -> dict:
    """
    Merge one fully reviewed annotation file into the training dataset.

    The workflow validates labels, archives the old jumplist, transfers
    trainable segments, refuses duplicate paths, writes the merged jumplist,
    and records a retraining marker consumed by the GUI.

    Segment CSV files previously moved under ``segment_archive_root`` are
    resolved through their exact archived path and copied to the training
    destination.  Archived sources are never moved or overwritten.  A dry run
    performs the complete read-only preflight and returns the planned actions.
    """
    import pandas as pd

    annotation_path = Path(annotation_csv_path)
    dataset_root = Path(dataset_path)
    annotated_root_path = Path(annotated_root)
    jumplist_path = dataset_root / "jumplist.csv"
    if not jumplist_path.exists():
        raise FileNotFoundError(f"Unable to find jumplist.csv in {dataset_root}")

    annotation_rows = pd.read_csv(annotation_path)
    progress = summarize_annotation_progress(annotation_rows)
    if progress["pending"] > 0:
        raise ValueError(f"Cannot finalize: {progress['pending']} annotations are still pending.")

    trainable = annotation_rows[
        (annotation_rows["type"].astype(float).astype(int) != 8)
        & (annotation_rows["success"].astype(float).astype(int) != 2)
    ].copy()
    if not trainable.empty:
        _validate_trainable_source_paths(trainable)
        _validate_annotation_file_duplicates(annotation_path, trainable)
    if trainable.empty:
        raise ValueError("Cannot finalize: no trainable labelled jumps are available.")

    existing = pd.read_csv(jumplist_path)
    session_key = str(trainable.iloc[0].get("session_key", annotation_path.stem.replace("_for_annotation", "")))
    hdf5_paths = _load_hdf5_segment_paths(hdf5_archive_path)
    destination_rows, planned_transfers = _plan_segment_moves(
        trainable,
        annotated_root_path,
        session_key,
        segment_archive_root=Path(segment_archive_root),
        hdf5_paths=hdf5_paths,
    )
    _validate_destinations(existing, destination_rows, planned_transfers, dataset_root)
    planned_raw_moves = _plan_source_raw_moves(session_key)
    completed_annotation_path = _completed_annotation_path(annotation_path, annotated_root_path, session_key)
    _validate_completed_annotation_destination(annotation_path, completed_annotation_path)

    archive_segment_count = sum(item[2] == "archive_csv" for item in planned_transfers)
    live_segment_count = len(planned_transfers) - archive_segment_count
    if dry_run:
        return {
            "dry_run": True,
            "session_key": session_key,
            "rows_added": 0,
            "rows_to_add": int(len(destination_rows)),
            "archive_segments_to_copy": int(archive_segment_count),
            "live_segments_to_move": int(live_segment_count),
            "segment_transfers": [
                {
                    "source": source,
                    "destination": destination,
                    "source_kind": source_kind,
                    "operation": "copy" if source_kind == "archive_csv" else "move",
                }
                for source, destination, source_kind in planned_transfers
            ],
            "raw_files_moved": 0,
            "raw_files_to_move": len(planned_raw_moves),
            "completed_annotation_path": completed_annotation_path,
            "annotation_source_preserved_until_commit": True,
            "state": None,
        }

    archive_path = _archive_jumplist(jumplist_path, dataset_root)
    _transfer_segments(planned_transfers)
    _move_files(planned_raw_moves)
    _merge_rows(existing, destination_rows, jumplist_path, dataset_root)
    completed_annotation_path = _move_completed_annotation(annotation_path, annotated_root_path, session_key)
    mark_finalized_session(session_key)
    state = _write_dataset_state(dataset_root, completed_annotation_path, destination_rows, archive_path)
    return {
        "rows_added": int(len(destination_rows)),
        "archive_path": archive_path,
        "completed_annotation_path": completed_annotation_path,
        "raw_files_moved": len(planned_raw_moves),
        "archive_segments_copied": int(archive_segment_count),
        "live_segments_moved": int(live_segment_count),
        "dry_run": False,
        "state": state,
    }


def _plan_segment_moves(
    trainable,
    annotated_root_path: Path,
    session_key: str,
    *,
    segment_archive_root: Path,
    hdf5_paths: set[str],
) -> tuple[list[dict], list[tuple[Path, Path, str]]]:
    destination_rows: list[dict] = []
    planned_transfers: list[tuple[Path, Path, str]] = []
    for _, row in trainable.iterrows():
        requested_source = Path(str(row["path"]))
        source, source_kind = _resolve_segment_source(
            requested_source,
            segment_archive_root=segment_archive_root,
            hdf5_paths=hdf5_paths,
        )
        sensor_id = str(row.get("sensor_id", "unknown"))
        destination = annotated_root_path / session_key / f"sensor{sensor_id}" / requested_source.name
        copied = row.to_dict()
        copied["path"] = str(destination).replace("\\", "/")
        copied["skater"] = copied.get("athlete_id", copied.get("skater", ""))
        destination_rows.append(copied)
        planned_transfers.append((source, destination, source_kind))
    return destination_rows, planned_transfers


def _validate_destinations(
    existing,
    destination_rows: list[dict],
    planned_transfers: list[tuple[Path, Path, str]],
    dataset_root: Path,
) -> None:
    import pandas as pd

    existing_duplicates = find_training_dataset_duplicates(dataset_root)
    if existing_duplicates["has_duplicates"]:
        examples = ", ".join(existing_duplicates["duplicate_paths"][:3])
        raise ValueError(
            "Cannot finalize: the existing training jumplist already contains duplicate paths"
            f" ({examples}). Resolve those duplicates first; no files were changed."
        )
    destination_paths = [row["path"] for row in destination_rows]
    if len(destination_paths) != len(set(destination_paths)):
        raise ValueError("Cannot finalize: duplicate destination paths exist inside the annotation file.")
    existing_paths = set(existing.get("path", pd.Series(dtype=str)).fillna("").astype(str).str.replace("\\", "/", regex=False))
    duplicate_paths = sorted(path for path in destination_paths if path in existing_paths)
    if duplicate_paths:
        raise ValueError(f"Cannot finalize: paths already present in training jumplist: {', '.join(duplicate_paths[:3])}")
    existing_destinations = [destination for _source, destination, _kind in planned_transfers if destination.exists()]
    if existing_destinations:
        raise ValueError(f"Cannot finalize: destination segment already exists: {existing_destinations[0]}")


def _archive_jumplist(jumplist_path: Path, dataset_root: Path) -> Path:
    archive_dir = dataset_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"jumplist_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.csv"
    _copy_file_exclusive(jumplist_path, archive_path)
    return archive_path


def _transfer_segments(planned_transfers: list[tuple[Path, Path, str]]) -> None:
    for source, destination, source_kind in planned_transfers:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_kind == "archive_csv":
            _copy_file_exclusive(source, destination)
        else:
            shutil.move(str(source), str(destination))


def _move_files(planned_moves: list[tuple[Path, Path]]) -> None:
    for source, destination in planned_moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def _merge_rows(existing, destination_rows: list[dict], jumplist_path: Path, dataset_root: Path) -> None:
    import pandas as pd

    merged = pd.concat([existing, pd.DataFrame(destination_rows)], ignore_index=True, sort=False)
    normalized_paths = merged.get("path", pd.Series(dtype=str)).fillna("").astype(str).str.replace("\\", "/", regex=False)
    duplicate_mask = normalized_paths.duplicated(keep=False) & normalized_paths.ne("")
    if duplicate_mask.any():
        raise ValueError("Finalization would create duplicate training rows; no jumplist changes were written.")

    temporary = jumplist_path.with_name(f".{jumplist_path.name}.{uuid4().hex}.tmp")
    try:
        merged.to_csv(temporary, index=False)
        os.replace(temporary, jumplist_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _move_completed_annotation(annotation_path: Path, annotated_root_path: Path, session_key: str) -> Path:
    completed_annotation_path = _completed_annotation_path(annotation_path, annotated_root_path, session_key)
    completed_dir = completed_annotation_path.parent
    completed_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(annotation_path), str(completed_annotation_path))
    metadata_path = annotation_metadata_path(annotation_path)
    if metadata_path.exists():
        shutil.move(str(metadata_path), str(completed_annotation_path.with_suffix(".annotation_meta.json")))
    return completed_annotation_path


def _resolve_segment_source(
    requested_source: Path,
    *,
    segment_archive_root: Path,
    hdf5_paths: set[str],
) -> tuple[Path, str]:
    if requested_source.is_file():
        return requested_source, "live_csv"

    archived_source = _archived_segment_path(segment_archive_root, requested_source)
    if archived_source.is_file():
        return archived_source, "archive_csv"

    normalized_requested = _normalize_path(requested_source)
    if normalized_requested in hdf5_paths:
        raise FileNotFoundError(
            f"Cannot finalize: segment {requested_source} is available only in the HDF5 archive. "
            "Restore or re-export its original CSV under data/segment_csv_archive before finalizing; "
            "HDF5-only reconstruction is intentionally refused to preserve the labelled source exactly."
        )
    raise FileNotFoundError(
        f"Missing annotated segment: {requested_source}. Expected the live CSV or archived copy at "
        f"{archived_source}. No files were changed."
    )


def _archived_segment_path(segment_archive_root: Path, source: Path) -> Path:
    if source.is_absolute():
        drive = source.drive.replace(":", "") or "root"
        relative_parts = source.parts[1:] if source.anchor else source.parts
        return segment_archive_root / drive / Path(*relative_parts)

    candidate = segment_archive_root / source
    try:
        candidate.resolve().relative_to(segment_archive_root.resolve())
    except ValueError as error:
        raise ValueError(f"Cannot finalize: archived segment path escapes its archive root: {source}") from error
    return candidate


def _load_hdf5_segment_paths(hdf5_archive_path: str | Path | None) -> set[str]:
    if hdf5_archive_path is None or not Path(hdf5_archive_path).is_file():
        return set()
    from synergie.services.hdf5_archive_service import hdf5_segment_paths

    return {_normalize_path(path) for path in hdf5_segment_paths(hdf5_archive_path)}


def _normalize_path(path: str | Path) -> str:
    return str(Path(str(path))).replace("\\", "/")


def _validate_annotation_file_duplicates(annotation_path: Path, trainable) -> None:
    """Refuse ambiguous labelled sibling files before changing any data."""
    import pandas as pd

    target_paths = {_normalize_path(path) for path in trainable["path"].fillna("").astype(str) if str(path).strip()}
    overlaps: list[tuple[str, int]] = []
    for sibling in sorted(annotation_path.parent.glob("*for_annotation*.csv")):
        if sibling.resolve() == annotation_path.resolve():
            continue
        try:
            candidate = pd.read_csv(sibling)
            if not {"path", "type", "success"}.issubset(candidate.columns):
                continue
            candidate_types = pd.to_numeric(candidate["type"], errors="coerce")
            candidate_success = pd.to_numeric(candidate["success"], errors="coerce")
            candidate_trainable_mask = (
                candidate_types.notna()
                & candidate_success.notna()
                & (candidate_types != 8)
                & (candidate_success != 2)
            )
            if "annotation_status" in candidate:
                reviewed = (
                    candidate["annotation_status"]
                    .fillna("pending")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin({"annotated", "completed", "finalized"})
                )
                candidate_trainable_mask &= reviewed
            candidate_trainable = candidate[candidate_trainable_mask]
            candidate_paths = {
                _normalize_path(path)
                for path in candidate_trainable["path"].fillna("").astype(str)
                if str(path).strip()
            }
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        overlap_count = len(target_paths & candidate_paths)
        if overlap_count:
            overlaps.append((sibling.name, overlap_count))
    if overlaps:
        examples = ", ".join(f"{name} ({count} shared segments)" for name, count in overlaps[:3])
        raise ValueError(
            "Cannot finalize: another labelled annotation file overlaps this one: "
            f"{examples}. Choose the canonical annotation file first; no files were changed."
        )


def _validate_trainable_source_paths(trainable) -> None:
    normalized_paths = trainable["path"].fillna("").astype(str).map(_normalize_path)
    duplicate_mask = normalized_paths.duplicated(keep=False) & normalized_paths.ne("")
    if duplicate_mask.any():
        examples = ", ".join(sorted(normalized_paths[duplicate_mask].unique())[:3])
        raise ValueError(
            "Cannot finalize: duplicate source segment paths exist inside the annotation file: "
            f"{examples}. Choose one labelled row per source segment; no files were changed."
        )


def _completed_annotation_path(annotation_path: Path, annotated_root_path: Path, session_key: str) -> Path:
    return annotated_root_path / session_key / annotation_path.name.replace("_for_annotation", "_annotated")


def _validate_completed_annotation_destination(annotation_path: Path, completed_annotation_path: Path) -> None:
    if completed_annotation_path.exists():
        raise ValueError(f"Cannot finalize: completed annotation already exists: {completed_annotation_path}")
    metadata_path = annotation_metadata_path(annotation_path)
    completed_metadata_path = completed_annotation_path.with_suffix(".annotation_meta.json")
    if metadata_path.exists() and completed_metadata_path.exists():
        raise ValueError(f"Cannot finalize: completed annotation metadata already exists: {completed_metadata_path}")


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    """Copy one file without ever overwriting an existing destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_created = False
    try:
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            destination_created = True
            shutil.copyfileobj(source_handle, destination_handle)
        shutil.copystat(source, destination)
    except Exception:
        if destination_created and destination.exists():
            destination.unlink()
        raise


def _plan_source_raw_moves(session_key: str) -> list[tuple[Path, Path]]:
    """Plan and validate moves from incoming IMU files into raw storage."""
    entry = session_workflow_entry(session_key)
    if not entry:
        return []
    destination_root = Path(entry["raw_destination"])
    planned_moves = [(Path(source), destination_root / Path(source).name) for source in entry.get("source_files", [])]
    collisions = [destination for _source, destination in planned_moves if destination.exists()]
    if collisions:
        raise ValueError(f"Cannot finalize: raw destination already exists: {collisions[0]}")
    missing = [source for source, _destination in planned_moves if not source.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize: missing source raw IMU file: {missing[0]}")
    return planned_moves


def _write_dataset_state(dataset_root: Path, completed_annotation_path: Path, destination_rows: list[dict], archive_path: Path) -> dict:
    state = {
        "changed_at": datetime.now().isoformat(timespec="seconds"),
        "source_annotation_file": str(completed_annotation_path).replace("\\", "/"),
        "rows_added": int(len(destination_rows)),
        "jumplist_archive": str(archive_path).replace("\\", "/"),
        "retraining_recommended": True,
    }
    state_path = training_dataset_state_path(dataset_root)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return state
