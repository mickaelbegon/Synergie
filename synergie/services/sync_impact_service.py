from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SyncImpact:
    index: int
    ms: float
    strength: float
    confidence: str


def detect_sync_impacts(
    session_df,
    *,
    max_candidates: int = 5,
    search_ms: float | None = 8000.0,
    min_impact_ms: float = 1000.0,
    stability_window_ms: float = 500.0,
    impact_exclusion_ms: float = 100.0,
    max_stability_ratio: float = 0.25,
) -> list[SyncImpact]:
    """Find likely clap/tap impacts used for IMU-video synchronization."""
    if session_df is None or session_df.empty or "ms" not in session_df:
        return []
    required = {"Acc_X", "Acc_Y", "Acc_Z"}
    if not required.issubset(session_df.columns):
        return []

    frame = session_df
    if search_ms is not None:
        frame = frame[frame["ms"] <= float(search_ms)]
    if len(frame) < 3:
        return []

    acc_norm = np.sqrt(
        np.square(frame["Acc_X"].to_numpy(dtype="float64"))
        + np.square(frame["Acc_Y"].to_numpy(dtype="float64"))
        + np.square(frame["Acc_Z"].to_numpy(dtype="float64"))
    )
    impact_signal = np.abs(np.diff(acc_norm, prepend=acc_norm[0]))
    median = float(np.nanmedian(impact_signal))
    mad = float(np.nanmedian(np.abs(impact_signal - median)))
    robust_scale = max(mad * 1.4826, 1e-6)

    ordered = np.argsort(impact_signal)[::-1]
    impacts: list[SyncImpact] = []
    min_gap_ms = 250.0
    for local_index in ordered:
        strength = float(impact_signal[local_index])
        score = (strength - median) / robust_scale
        if score < 6.0:
            break
        ms = float(frame.iloc[int(local_index)]["ms"])
        if ms < float(min_impact_ms):
            continue
        if not _has_stable_context(
            frame,
            impact_signal,
            local_index=int(local_index),
            impact_ms=ms,
            strength=strength,
            stability_window_ms=stability_window_ms,
            impact_exclusion_ms=impact_exclusion_ms,
            max_stability_ratio=max_stability_ratio,
        ):
            continue
        if any(abs(ms - impact.ms) < min_gap_ms for impact in impacts):
            continue
        confidence = "high" if score >= 12.0 else "medium" if score >= 6.0 else "low"
        impacts.append(SyncImpact(index=int(frame.index[int(local_index)]), ms=ms, strength=strength, confidence=confidence))
        if len(impacts) >= max_candidates:
            break
    return sorted(impacts, key=lambda impact: impact.ms)


def sync_impact_signal_window(
    dataframe,
    impact_ms: float,
    *,
    before_ms: float = 2000.0,
    after_ms: float = 3000.0,
    fallback_rows: int = 300,
):
    """Return a display window and 3D acceleration norm around a sync impact."""
    if dataframe is None or dataframe.empty or "ms" not in dataframe:
        return dataframe, np.array([], dtype="float64")
    required = {"Acc_X", "Acc_Y", "Acc_Z"}
    if not required.issubset(dataframe.columns):
        return dataframe.iloc[:0].copy(), np.array([], dtype="float64")

    window_start_ms = max(0.0, float(impact_ms) - float(before_ms))
    window_end_ms = float(impact_ms) + float(after_ms)
    view_df = dataframe[(dataframe["ms"] >= window_start_ms) & (dataframe["ms"] <= window_end_ms)]
    if view_df.empty:
        view_df = dataframe.iloc[: min(len(dataframe), int(fallback_rows))]

    acc_norm = np.sqrt(
        np.square(view_df["Acc_X"].to_numpy(dtype="float64"))
        + np.square(view_df["Acc_Y"].to_numpy(dtype="float64"))
        + np.square(view_df["Acc_Z"].to_numpy(dtype="float64"))
    )
    return view_df, acc_norm


def sync_impact_list_labels(impacts: list[SyncImpact]) -> list[str]:
    """Return stable labels for the Inspect IMU sync impact list."""
    return [
        f"{index + 1:02d} | {impact.ms:.0f} ms | strength {impact.strength:.2f} | {impact.confidence}"
        for index, impact in enumerate(impacts)
    ]


def sync_impact_review_warning(impact_count: int) -> str:
    """Explain whether sync impact candidates need manual review."""
    if impact_count > 1:
        return (
            f"{impact_count} sync impact candidates remain. Review the acceleration signals and select the one "
            "that matches the visible tap on video."
        )
    if impact_count == 1:
        return "One sync impact candidate found. Select it to review the acceleration signal."
    return "No reliable sync impact candidate found. Review the recording or choose sync manually."


def sync_impact_selected_status(impact: SyncImpact, impact_number: int) -> str:
    """Return a short status bar message for a selected sync impact."""
    return (
        f"Selected sync impact {impact_number}: {impact.ms:.0f} ms. "
        "Detection uses the sudden change in 3D acceleration norm; use this selected candidate only if it matches the video tap."
    )


def sync_impact_plot_title(impact: SyncImpact, impact_number: int) -> str:
    """Return the Inspect IMU zoom title for a selected sync impact."""
    return (
        f"Sync impact #{impact_number}: {impact.ms:.0f} ms | "
        f"3D acceleration-norm change strength {impact.strength:.2f} | {impact.confidence}"
    )


def prepare_sync_impact_signal(
    raw_path: str | Path,
    impact_ms: float,
    *,
    acceleration_limit_g: float,
    smoothing_sigma: float,
    threshold: float,
    read_csv=None,
    training_session_factory=None,
    clean_frame=None,
    recompute_derivatives=None,
) -> dict:
    """Load and prepare raw IMU acceleration signals around a sync impact."""
    if read_csv is None:
        import pandas as pd

        read_csv = pd.read_csv
    if training_session_factory is None:
        from core.data_treatment.data_generation.trainingSession import trainingSession as training_session_factory
    if clean_frame is None:
        from synergie.services.signal_cleaning_service import clean_imu_outliers as clean_frame
    if recompute_derivatives is None:
        from synergie.services.signal_cleaning_service import recompute_gyro_x_derivatives as recompute_derivatives

    session = training_session_factory(read_csv(raw_path, low_memory=False))
    dataframe = session.df.copy()
    if dataframe.empty or "ms" not in dataframe:
        return {"status": "missing_timeline", "dataframe": dataframe}

    dataframe, cleaning_report = clean_frame(dataframe, acceleration_limit_g=acceleration_limit_g)
    dataframe = recompute_derivatives(dataframe, smoothing_sigma=smoothing_sigma, threshold=threshold)
    view_df, acc_norm = sync_impact_signal_window(dataframe, impact_ms)
    if view_df.empty:
        return {
            "status": "missing_acceleration",
            "dataframe": dataframe,
            "cleaning_report": cleaning_report,
            "view_df": view_df,
            "acc_norm": acc_norm,
        }
    return {
        "status": "ok",
        "dataframe": dataframe,
        "cleaning_report": cleaning_report,
        "view_df": view_df,
        "acc_norm": acc_norm,
    }


def _has_stable_context(
    frame,
    impact_signal,
    *,
    local_index: int,
    impact_ms: float,
    strength: float,
    stability_window_ms: float,
    impact_exclusion_ms: float,
    max_stability_ratio: float,
) -> bool:
    """Return whether acceleration changes are quiet before and after one impact."""
    ms_values = frame["ms"].to_numpy(dtype="float64")
    before_mask = (ms_values >= impact_ms - stability_window_ms) & (ms_values <= impact_ms - impact_exclusion_ms)
    after_mask = (ms_values >= impact_ms + impact_exclusion_ms) & (ms_values <= impact_ms + stability_window_ms)
    before = impact_signal[before_mask]
    after = impact_signal[after_mask]
    if len(before) < 2 or len(after) < 2:
        return False

    # Remove the candidate sample if coarse sampling puts it inside one context mask.
    before = np.delete(before, np.where(np.flatnonzero(before_mask) == local_index)[0])
    after = np.delete(after, np.where(np.flatnonzero(after_mask) == local_index)[0])
    if len(before) < 2 or len(after) < 2:
        return False

    stability_limit = max(float(strength) * float(max_stability_ratio), 1e-6)
    before_level = float(np.nanpercentile(np.abs(before), 90))
    after_level = float(np.nanpercentile(np.abs(after), 90))
    return before_level <= stability_limit and after_level <= stability_limit
