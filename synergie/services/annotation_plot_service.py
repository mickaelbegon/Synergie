from __future__ import annotations

from pathlib import Path

from synergie.services.hdf5_archive_service import hdf5_segment_paths, load_segment_dataframe
from synergie.services.signal_cleaning_service import clean_imu_outliers, recompute_gyro_x_derivatives


def annotation_segment_available(path: str | Path, *, archive_paths_factory=hdf5_segment_paths) -> bool:
    """Return whether an annotation segment can be loaded from CSV or the HDF5 archive."""
    segment_path = Path(str(path))
    return segment_path.exists() or str(segment_path).replace("\\", "/") in archive_paths_factory()


def prepare_annotation_segment_signal(
    row,
    *,
    acceleration_limit_g: float,
    smoothing_sigma: float,
    threshold: float,
    load_dataframe=load_segment_dataframe,
    archive_paths_factory=hdf5_segment_paths,
    clean_frame=clean_imu_outliers,
    recompute_derivatives=recompute_gyro_x_derivatives,
):
    """Load and prepare one annotation segment for the Jump Signals plot."""
    path = annotation_segment_path(row)
    if path is None:
        return None
    if not annotation_segment_available(path, archive_paths_factory=archive_paths_factory):
        return None
    dataframe = load_dataframe(path)
    dataframe, cleaning_report = clean_frame(dataframe, acceleration_limit_g=acceleration_limit_g)
    dataframe = recompute_derivatives(dataframe, smoothing_sigma=smoothing_sigma, threshold=threshold)
    return {
        "dataframe": dataframe,
        "cleaning_report": cleaning_report,
        "gyro_column": annotation_gyro_column(dataframe),
    }


def annotation_type_window_start_imu_ms(
    row,
    *,
    type_window_start: int,
    load_dataframe=load_segment_dataframe,
    archive_paths_factory=hdf5_segment_paths,
) -> float | None:
    """Return the IMU timestamp for the first frame used by the jump-type model."""
    path = annotation_segment_path(row)
    if path is None:
        return None
    if not annotation_segment_available(path, archive_paths_factory=archive_paths_factory):
        return None
    dataframe = load_dataframe(path)
    if dataframe is None or dataframe.empty or "ms" not in dataframe:
        return None
    index = int(type_window_start)
    if index < 0 or index >= len(dataframe):
        return None
    return float(dataframe.iloc[index]["ms"])


def annotation_gyro_column(dataframe) -> str:
    """Return the preferred gyroscope column for plotting annotation segments."""
    return "Gyr_X_smoothed" if "Gyr_X_smoothed" in dataframe else "Gyr_X"


def annotation_segment_path(row) -> Path | None:
    """Return the segment path from an annotation row, or None when unavailable."""
    try:
        value = row.get("path", None)
    except AttributeError:
        return None
    if value is None or value != value:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)
