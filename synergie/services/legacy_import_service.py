from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

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
    return normalized


def _resolve_legacy_segment_path(path_value: str, session_dir: Path) -> str:
    path = Path(path_value)
    resolved = path if path.is_absolute() else session_dir / path
    return str(resolved).replace("\\", "/")
