from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from synergie.services.annotation_service import annotation_metadata_path, summarize_annotation_progress
from synergie.services.training_dataset_service import (
    find_training_dataset_duplicates,
    training_dataset_state_path,
)


def finalize_annotation_file(
    annotation_csv_path: str | Path,
    *,
    dataset_path: str | Path = "data/annotated/total",
    annotated_root: str | Path = "data/annotated",
) -> dict:
    """
    Merge one fully reviewed annotation file into the training dataset.

    The workflow validates labels, archives the old jumplist, moves trainable
    segments, refuses duplicate paths, writes the merged jumplist, and records
    a retraining marker consumed by the GUI.
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
    if trainable.empty:
        raise ValueError("Cannot finalize: no trainable labelled jumps are available.")

    existing = pd.read_csv(jumplist_path)
    session_key = str(trainable.iloc[0].get("session_key", annotation_path.stem.replace("_for_annotation", "")))
    destination_rows, planned_moves = _plan_segment_moves(trainable, annotated_root_path, session_key)
    _validate_destinations(existing, destination_rows, planned_moves)

    archive_path = _archive_jumplist(jumplist_path, dataset_root)
    _move_segments(planned_moves)
    _merge_rows(existing, destination_rows, jumplist_path, dataset_root)
    completed_annotation_path = _move_completed_annotation(annotation_path, annotated_root_path, session_key)
    state = _write_dataset_state(dataset_root, completed_annotation_path, destination_rows, archive_path)
    return {
        "rows_added": int(len(destination_rows)),
        "archive_path": archive_path,
        "completed_annotation_path": completed_annotation_path,
        "state": state,
    }


def _plan_segment_moves(trainable, annotated_root_path: Path, session_key: str) -> tuple[list[dict], list[tuple[Path, Path]]]:
    destination_rows: list[dict] = []
    planned_moves: list[tuple[Path, Path]] = []
    for _, row in trainable.iterrows():
        source = Path(str(row["path"]))
        if not source.exists():
            raise FileNotFoundError(f"Missing annotated segment: {source}")
        sensor_id = str(row.get("sensor_id", "unknown"))
        destination = annotated_root_path / session_key / f"sensor{sensor_id}" / source.name
        copied = row.to_dict()
        copied["path"] = str(destination).replace("\\", "/")
        copied["skater"] = copied.get("athlete_id", copied.get("skater", ""))
        destination_rows.append(copied)
        planned_moves.append((source, destination))
    return destination_rows, planned_moves


def _validate_destinations(existing, destination_rows: list[dict], planned_moves: list[tuple[Path, Path]]) -> None:
    import pandas as pd

    destination_paths = [row["path"] for row in destination_rows]
    if len(destination_paths) != len(set(destination_paths)):
        raise ValueError("Cannot finalize: duplicate destination paths exist inside the annotation file.")
    existing_paths = set(existing.get("path", pd.Series(dtype=str)).fillna("").astype(str).str.replace("\\", "/", regex=False))
    duplicate_paths = sorted(path for path in destination_paths if path in existing_paths)
    if duplicate_paths:
        raise ValueError(f"Cannot finalize: paths already present in training jumplist: {', '.join(duplicate_paths[:3])}")
    existing_destinations = [destination for _source, destination in planned_moves if destination.exists()]
    if existing_destinations:
        raise ValueError(f"Cannot finalize: destination segment already exists: {existing_destinations[0]}")


def _archive_jumplist(jumplist_path: Path, dataset_root: Path) -> Path:
    archive_dir = dataset_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"jumplist_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    shutil.copy2(jumplist_path, archive_path)
    return archive_path


def _move_segments(planned_moves: list[tuple[Path, Path]]) -> None:
    for source, destination in planned_moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def _merge_rows(existing, destination_rows: list[dict], jumplist_path: Path, dataset_root: Path) -> None:
    import pandas as pd

    merged = pd.concat([existing, pd.DataFrame(destination_rows)], ignore_index=True, sort=False)
    merged.to_csv(jumplist_path, index=False)
    duplicate_report = find_training_dataset_duplicates(dataset_root)
    if duplicate_report["has_duplicates"]:
        raise ValueError("Finalization created duplicate training rows; inspect the archived jumplist before continuing.")


def _move_completed_annotation(annotation_path: Path, annotated_root_path: Path, session_key: str) -> Path:
    completed_dir = annotated_root_path / session_key
    completed_annotation_path = completed_dir / annotation_path.name.replace("_for_annotation", "_annotated")
    completed_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(annotation_path), str(completed_annotation_path))
    metadata_path = annotation_metadata_path(annotation_path)
    if metadata_path.exists():
        shutil.move(str(metadata_path), str(completed_annotation_path.with_suffix(".annotation_meta.json")))
    return completed_annotation_path


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
