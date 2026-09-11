from __future__ import annotations

from pathlib import Path


def find_annotation_raw_source_path(annotation_frame, sensor_id: str, *, search_roots=None) -> Path | None:
    """Find the raw CSV source referenced by one annotation sensor."""
    if annotation_frame is None or "sensor_id" not in annotation_frame.columns:
        return None
    roots = [Path(root) for root in (search_roots or [Path("data/new"), Path("data/raw"), Path("data/pending")])]
    rows = annotation_frame[annotation_frame["sensor_id"].astype(str) == str(sensor_id)]
    for _, row in rows.iterrows():
        source_name = str(row.get("source_file", "")).strip()
        if not source_name:
            continue
        direct = Path(source_name)
        if direct.exists():
            return direct
        for root in roots:
            if root.exists():
                matches = sorted(root.rglob(source_name))
                if matches:
                    return matches[0]
    return None
