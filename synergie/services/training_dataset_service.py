from __future__ import annotations

import csv
import json
from pathlib import Path


def describe_training_dataset(task: str, dataset_path: str, augment_mirror: bool = True) -> dict:
    """Summarize labelled rows, balance, and duplicate state for one dataset."""
    dataset_root = Path(dataset_path)
    filtered: list[dict] = []
    duplicate_report = find_training_dataset_duplicates(dataset_path)
    with (dataset_root / "jumplist.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            jump_type = int(float(row["type"]))
            success = int(float(row["success"]))
            if success == 2 or jump_type == 8:
                continue
            filtered.append({"type": jump_type, "success": success, "skater": row["skater"]})

    if task == "type":
        labels = [row["type"] for row in filtered]
    elif task == "success":
        labels = [row["success"] for row in filtered]
    else:
        raise ValueError(f"Unsupported training task: {task}")

    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    base_samples = int(len(filtered))
    recommended_class_weight = {
        str(label): round(base_samples / (len(counts) * count), 6)
        for label, count in sorted(counts.items())
        if count > 0
    }
    validation_samples = base_samples - int(base_samples * 0.8)
    stratified_split_possible = (
        bool(counts)
        and len(counts) >= 2
        and min(counts.values()) >= 2
        and validation_samples >= len(counts)
    )
    return {
        "task": task,
        "base_samples": base_samples,
        "effective_samples": int(base_samples * (2 if augment_mirror else 1)),
        "unique_skaters": len({row["skater"] for row in filtered}),
        "class_counts": {str(label): counts[label] for label in sorted(counts)},
        "recommended_class_weight": recommended_class_weight,
        "stratified_split_possible": stratified_split_possible,
        "augment_mirror": bool(augment_mirror),
        "duplicate_rows": duplicate_report["duplicate_rows"],
        "duplicate_paths": duplicate_report["duplicate_paths"],
        "has_duplicates": duplicate_report["has_duplicates"],
    }


def load_training_dataset_rows(dataset_path: str | Path, *, include_excluded: bool = False) -> list[dict]:
    """Load training dataset rows with stable row indices for GUI review workflows."""
    import pandas as pd

    jumplist_path = Path(dataset_path) / "jumplist.csv"
    if not jumplist_path.exists():
        raise FileNotFoundError(f"Unable to find jumplist.csv in {dataset_path}")

    frame = pd.read_csv(jumplist_path)
    records: list[dict] = []
    for row_index, row in frame.iterrows():
        jump_type = int(float(row.get("type", 8)))
        success = int(float(row.get("success", 2)))
        if not include_excluded and (jump_type == 8 or success == 2):
            continue
        record = row.to_dict()
        record["row_index"] = int(row_index)
        record["type"] = jump_type
        record["success"] = success
        records.append(record)
    return records


def exclude_training_dataset_row(
    dataset_path: str | Path,
    row_index: int,
    *,
    excluded_reason: str = "weird_signal",
) -> dict:
    """Exclude one already-finalized training row from future training runs."""
    import pandas as pd

    dataset_root = Path(dataset_path)
    jumplist_path = dataset_root / "jumplist.csv"
    if not jumplist_path.exists():
        raise FileNotFoundError(f"Unable to find jumplist.csv in {dataset_path}")

    frame = pd.read_csv(jumplist_path)
    index = int(row_index)
    if index < 0 or index >= len(frame):
        raise IndexError(f"Training dataset row index out of range: {row_index}")

    frame.at[index, "type"] = 8
    frame.at[index, "success"] = 2
    frame.at[index, "excluded_reason"] = str(excluded_reason)
    frame.at[index, "review_status"] = str(excluded_reason)
    frame.to_csv(jumplist_path, index=False)

    record = frame.iloc[index].to_dict()
    record["row_index"] = index
    record["type"] = int(float(record.get("type", 8)))
    record["success"] = int(float(record.get("success", 2)))
    return record


def find_training_dataset_duplicates(dataset_path: str | Path) -> dict:
    """Detect duplicate training rows, primarily by repeated segment path."""
    import pandas as pd

    jumplist_path = Path(dataset_path) / "jumplist.csv"
    if not jumplist_path.exists():
        raise FileNotFoundError(f"Unable to find jumplist.csv in {dataset_path}")

    frame = pd.read_csv(jumplist_path)
    if "path" not in frame:
        return {"has_duplicates": False, "duplicate_paths": [], "duplicate_rows": 0, "rows": []}
    normalized_paths = frame["path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    duplicate_mask = normalized_paths.duplicated(keep=False) & normalized_paths.ne("")
    duplicate_rows = frame.loc[duplicate_mask].copy()
    duplicate_rows["_normalized_path"] = normalized_paths[duplicate_mask]
    duplicate_paths = sorted(duplicate_rows["_normalized_path"].unique().tolist()) if not duplicate_rows.empty else []
    return {
        "has_duplicates": bool(duplicate_paths),
        "duplicate_paths": duplicate_paths,
        "duplicate_rows": int(duplicate_mask.sum()),
        "rows": duplicate_rows.to_dict(orient="records"),
    }


def training_dataset_state_path(dataset_path: str | Path = "data/annotated/total") -> Path:
    """Return the sidecar file used to record the last training-set update."""
    return Path(dataset_path) / "dataset_state.json"


def load_training_dataset_state(dataset_path: str | Path = "data/annotated/total") -> dict:
    """Load the persisted training dataset state when available."""
    path = training_dataset_state_path(dataset_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}
