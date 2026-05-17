from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
import re

from synergie.services.training_dataset_service import find_training_dataset_duplicates


def import_legacy_jumplist(
    jumplist_path: str | Path,
    *,
    dataset_path: str | Path = "data/annotated/total",
) -> dict:
    """Merge one legacy local jumplist into the total training jumplist safely."""
    import pandas as pd

    source_path = Path(jumplist_path)
    dataset_root = Path(dataset_path)
    total_path = dataset_root / "jumplist.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Legacy jumplist not found: {source_path}")
    if not total_path.exists():
        raise FileNotFoundError(f"Training jumplist not found: {total_path}")

    legacy = pd.read_csv(source_path)
    normalized = _normalize_legacy_frame(legacy, source_path.parent)
    trainable = normalized[
        (pd.to_numeric(normalized["type"], errors="coerce") != 8)
        & (pd.to_numeric(normalized["success"], errors="coerce") != 2)
    ].copy()
    if trainable.empty:
        raise ValueError("Legacy jumplist has no trainable rows (type != 8 and success != 2).")
    missing_paths = [path for path in trainable["path"].astype(str) if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Legacy jumplist references missing segment: {missing_paths[0]}")

    existing = pd.read_csv(total_path)
    existing_paths = set(existing["path"].fillna("").astype(str).str.replace("\\", "/", regex=False))
    duplicate_paths = sorted(path for path in trainable["path"].astype(str) if path in existing_paths)
    if duplicate_paths:
        raise ValueError(f"Legacy jumplist already imported: {duplicate_paths[0]}")

    archive_dir = dataset_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"jumplist_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    shutil.copy2(total_path, archive_path)
    merged = pd.concat([existing, trainable], ignore_index=True, sort=False)
    merged.to_csv(total_path, index=False)
    duplicate_report = find_training_dataset_duplicates(dataset_root)
    if duplicate_report["has_duplicates"]:
        raise ValueError("Legacy import created duplicate paths; restore the archived jumplist before continuing.")
    return {
        "rows_added": int(len(trainable)),
        "archive_path": archive_path,
        "source_path": source_path,
    }


def migrate_legacy_annotation_workbook(
    workbook_path: str | Path,
    *,
    annotated_root: str | Path = "data/annotated",
    dataset_path: str | Path = "data/annotated/total",
) -> dict:
    """Normalize a human-annotated workbook into session jumplists and import trainable rows."""
    import pandas as pd

    workbook = Path(workbook_path)
    frame = pd.read_excel(workbook)
    normalized = _normalize_legacy_frame(frame, Path("."))
    normalized = _normalize_training_labels(normalized)
    normalized["session_key"] = normalized["path"].map(_session_key_from_annotated_path)
    if normalized["session_key"].eq("").any():
        raise ValueError("Workbook contains paths outside data/annotated/<date>/<time>.")

    session_outputs: list[Path] = []
    for session_key, session_rows in normalized.groupby("session_key", sort=True):
        output_path = Path(annotated_root) / session_key / "jumplist.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        session_rows.drop(columns=["session_key"]).to_csv(output_path, index=False)
        session_outputs.append(output_path)

    total_before = pd.read_csv(Path(dataset_path) / "jumplist.csv")
    trainable = normalized[
        normalized["type"].isin([0, 1, 2, 3, 4, 5]) & normalized["success"].isin([0, 1])
    ].drop(columns=["session_key"])
    imported = _merge_trainable_rows(trainable, dataset_path)
    return {
        "workbook_path": workbook,
        "rows_total": int(len(normalized)),
        "rows_trainable": int(len(trainable)),
        "session_outputs": session_outputs,
        "rows_added": imported["rows_added"],
        "archive_path": imported["archive_path"],
        "rows_total_before": int(len(total_before)),
    }


def _normalize_legacy_frame(frame, session_dir: Path):
    """Normalize common legacy columns and relative segment paths."""
    normalized = frame.copy()
    if "success" not in normalized and "sucess" in normalized:
        normalized["success"] = normalized["sucess"]
    required = {"path", "type", "success"}
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"Legacy jumplist missing required columns: {', '.join(sorted(missing))}")
    normalized["path"] = normalized["path"].fillna("").astype(str).map(
        lambda value: _resolve_legacy_segment_path(value, session_dir)
    )
    if "skater" in normalized:
        normalized["skater"] = normalized["skater"].map(_normalize_legacy_skater_id)
    return normalized


def _normalize_training_labels(frame):
    """Map unsupported legacy rows to current exclude conventions while preserving source labels."""
    import pandas as pd

    normalized = frame.copy()
    normalized["legacy_type"] = pd.to_numeric(normalized["type"], errors="coerce")
    normalized["legacy_success"] = pd.to_numeric(normalized["success"], errors="coerce")
    valid_type = normalized["legacy_type"].isin([0, 1, 2, 3, 4, 5])
    valid_success = normalized["legacy_success"].isin([0, 1])
    trainable = valid_type & valid_success
    normalized["type"] = normalized["legacy_type"].where(trainable, 8).astype(int)
    normalized["success"] = normalized["legacy_success"].where(trainable, 2).astype(int)
    return normalized


def _resolve_legacy_segment_path(path_value: str, session_dir: Path) -> str:
    path = Path(path_value)
    resolved = path if path.is_absolute() else session_dir / path
    return str(resolved).replace("\\", "/")


def _session_key_from_annotated_path(path_value: str) -> str:
    normalized = str(path_value).replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) < 4 or parts[:2] != ["data", "annotated"]:
        return ""
    return f"{parts[2]}/{parts[3]}"


def _normalize_legacy_skater_id(value):
    """Collapse legacy session-prefixed skater ids such as 20250901_0910_10 to 10."""
    text = str(value)
    match = re.fullmatch(r"\d{8}_\d{4}_(\d+)", text)
    return match.group(1) if match else value


def _merge_trainable_rows(trainable, dataset_path: str | Path) -> dict:
    """Merge normalized trainable rows into the total dataset with backup and duplicate checks."""
    import pandas as pd

    dataset_root = Path(dataset_path)
    total_path = dataset_root / "jumplist.csv"
    existing = pd.read_csv(total_path)
    existing_paths = set(existing["path"].fillna("").astype(str).str.replace("\\", "/", regex=False))
    duplicate_paths = sorted(path for path in trainable["path"].astype(str) if path in existing_paths)
    if duplicate_paths:
        raise ValueError(f"Legacy workbook already imported: {duplicate_paths[0]}")
    archive_dir = dataset_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"jumplist_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    shutil.copy2(total_path, archive_path)
    merged = pd.concat([existing, trainable], ignore_index=True, sort=False)
    merged.to_csv(total_path, index=False)
    duplicate_report = find_training_dataset_duplicates(dataset_root)
    if duplicate_report["has_duplicates"]:
        raise ValueError("Legacy workbook import created duplicate paths; inspect the archived jumplist.")
    return {"rows_added": int(len(trainable)), "archive_path": archive_path}
