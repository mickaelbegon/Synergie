from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from synergie.services.backup_manifest_service import build_backup_manifest


NUMERIC_COLUMNS = [
    "SampleTimeFine",
    "ms",
    "Euler_X",
    "Euler_Y",
    "Euler_Z",
    "Gyr_X",
    "Gyr_Y",
    "Gyr_Z",
    "Gyr_X_unfiltered",
    "Gyr_X_smoothed",
    "Acc_X",
    "Acc_Y",
    "Acc_Z",
    "X_gyr_second_derivative",
    "Combination",
]


def export_hdf5_archive(
    archive_path: str | Path = "data/synergie_archive.h5",
    *,
    manifest: dict | None = None,
    compression: str = "gzip",
    compression_opts: int = 4,
    include_segments: bool = True,
    **manifest_kwargs,
) -> dict:
    """Export a compact HDF5 archive that complements the human-readable manifest."""
    import h5py

    path = Path(archive_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = manifest if manifest is not None else build_backup_manifest(**manifest_kwargs)
    summary = {"archive_path": path, "segments_written": 0, "segments_missing": 0, "trials_indexed": 0}
    segment_index: dict[str, str] = {}

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "synergie-hdf5-archive-v1"
        handle.attrs["manifest_json"] = json.dumps(manifest_payload, ensure_ascii=True)
        _write_json_dataset(handle, "manifest", manifest_payload)
        trials_group = handle.create_group("trials")
        summary["trials_indexed"] += _write_trial_collection(
            trials_group,
            "pending",
            _pending_trials(manifest_payload),
            include_segments=include_segments,
            compression=compression,
            compression_opts=compression_opts,
            summary=summary,
            segment_index=segment_index,
        )
        summary["trials_indexed"] += _write_trial_collection(
            trials_group,
            "training",
            manifest_payload.get("training_dataset", {}).get("trials", []),
            include_segments=include_segments,
            compression=compression,
            compression_opts=compression_opts,
            summary=summary,
            segment_index=segment_index,
        )
        handle.attrs["segment_path_index_json"] = json.dumps(segment_index, ensure_ascii=True)
    return summary


def load_segment_dataframe(segment_path: str | Path, archive_path: str | Path = "data/synergie_archive.h5"):
    """Load one segment from HDF5 when available, otherwise from its CSV path."""
    import pandas as pd

    csv_path = Path(str(segment_path))
    archive = Path(archive_path)
    if archive.exists():
        hdf_frame = _load_segment_dataframe_from_archive(csv_path, archive)
        if hdf_frame is not None:
            return hdf_frame
    return pd.read_csv(csv_path)


def hdf5_segment_paths(archive_path: str | Path = "data/synergie_archive.h5") -> set[str]:
    """Return segment CSV paths represented in one HDF5 archive."""
    import h5py

    archive = Path(archive_path)
    if not archive.exists():
        return set()
    with h5py.File(archive, "r") as handle:
        raw = handle.attrs.get("segment_path_index_json", "{}")
        return set(json.loads(str(raw)).keys())


def plan_segment_csv_cleanup(
    archive_path: str | Path = "data/synergie_archive.h5",
    *,
    roots: list[str | Path] | None = None,
) -> dict:
    """Report CSV segment files that are safely represented in HDF5."""
    represented = hdf5_segment_paths(archive_path)
    candidates = []
    total_bytes = 0
    for path_text in sorted(represented):
        path = Path(path_text)
        if roots is not None and not _is_under_any_root(path, roots):
            continue
        if path.exists() and path.suffix.lower() == ".csv":
            size = path.stat().st_size
            candidates.append({"path": path.as_posix(), "size_bytes": size})
            total_bytes += size
    return {
        "archive_path": str(Path(archive_path)).replace("\\", "/"),
        "candidate_count": len(candidates),
        "candidate_bytes": total_bytes,
        "candidates": candidates,
    }


def archive_segment_csvs(
    archive_path: str | Path = "data/synergie_archive.h5",
    *,
    destination_root: str | Path = "data/segment_csv_archive",
    roots: list[str | Path] | None = None,
    apply: bool = False,
) -> dict:
    """Move HDF5-covered segment CSV files into an archive folder when explicitly applied."""
    plan = plan_segment_csv_cleanup(archive_path, roots=roots)
    destination = Path(destination_root)
    moved = []
    skipped = []
    if apply:
        destination.mkdir(parents=True, exist_ok=True)
        for candidate in plan["candidates"]:
            source = Path(candidate["path"])
            target = _archive_target_path(destination, source)
            if not source.exists():
                skipped.append({"path": source.as_posix(), "reason": "missing_source"})
                continue
            if target.exists():
                skipped.append({"path": source.as_posix(), "reason": "target_exists"})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append({"source": source.as_posix(), "target": target.as_posix(), "size_bytes": candidate["size_bytes"]})
    return {
        **plan,
        "destination_root": str(destination).replace("\\", "/"),
        "applied": bool(apply),
        "moved_count": len(moved),
        "moved_bytes": sum(item["size_bytes"] for item in moved),
        "moved": moved,
        "skipped": skipped,
    }


def _archive_target_path(destination: Path, source: Path) -> Path:
    if not source.is_absolute():
        return destination / source
    drive = source.drive.replace(":", "") or "root"
    relative_parts = source.parts[1:] if source.anchor else source.parts
    return destination / drive / Path(*relative_parts)


def _write_json_dataset(handle, name: str, payload: dict) -> None:
    import h5py

    encoded = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
    handle.create_dataset(name, data=encoded, dtype=h5py.string_dtype(encoding="utf-8"))


def _write_trial_collection(
    parent,
    name: str,
    trials: list[dict],
    *,
    include_segments: bool,
    compression: str,
    compression_opts: int,
    summary: dict,
    segment_index: dict[str, str],
) -> int:
    collection = parent.create_group(name)
    collection.attrs["count"] = len(trials)
    for index, trial in enumerate(trials):
        trial_group = collection.create_group(f"trial_{index:06d}")
        trial_group.attrs["metadata_json"] = json.dumps(trial, ensure_ascii=True)
        if include_segments:
            _write_segment_dataset(
                trial_group,
                trial,
                compression=compression,
                compression_opts=compression_opts,
                summary=summary,
                segment_index=segment_index,
            )
    return len(trials)


def _write_segment_dataset(group, trial: dict, *, compression: str, compression_opts: int, summary: dict, segment_index: dict[str, str]) -> None:
    segment_path = Path(str(trial.get("path") or ""))
    if not segment_path.exists():
        summary["segments_missing"] += 1
        group.attrs["segment_missing"] = True
        return
    if segment_path.suffix.lower() != ".csv":
        summary["segments_missing"] += 1
        group.attrs["segment_missing"] = True
        group.attrs["segment_skip_reason"] = "unsupported_segment_format"
        return

    import pandas as pd

    frame = pd.read_csv(segment_path)
    columns = [column for column in NUMERIC_COLUMNS if column in frame.columns]
    if not columns:
        summary["segments_missing"] += 1
        group.attrs["segment_missing"] = True
        group.attrs["segment_skip_reason"] = "no_numeric_columns"
        return

    data = frame[columns].to_numpy(dtype="float32")
    dataset = group.create_dataset("segment", data=data, compression=compression, compression_opts=compression_opts)
    dataset.attrs["columns_json"] = json.dumps(columns, ensure_ascii=True)
    dataset.attrs["source_path"] = str(segment_path).replace("\\", "/")
    segment_index[str(segment_path).replace("\\", "/")] = dataset.name
    summary["segments_written"] += 1


def _load_segment_dataframe_from_archive(segment_path: Path, archive_path: Path):
    import h5py
    import pandas as pd

    normalized = str(segment_path).replace("\\", "/")
    with h5py.File(archive_path, "r") as handle:
        index = json.loads(str(handle.attrs.get("segment_path_index_json", "{}")))
        dataset_name = index.get(normalized)
        if not dataset_name or dataset_name not in handle:
            return None
        dataset = handle[dataset_name]
        columns = json.loads(str(dataset.attrs["columns_json"]))
        return pd.DataFrame(dataset[()], columns=columns)


def _is_under_any_root(path: Path, roots: list[str | Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(Path(root).resolve())
            return True
        except ValueError:
            continue
    return False


def _pending_trials(manifest: dict) -> list[dict]:
    trials: list[dict] = []
    for file_record in manifest.get("pending_annotation_files", []):
        for trial in file_record.get("trials", []):
            enriched = dict(trial)
            enriched["annotation_file"] = file_record.get("path")
            trials.append(enriched)
    return trials


def safe_hdf5_name(value: str) -> str:
    """Return a stable group name fragment for arbitrary labels."""
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unnamed"
