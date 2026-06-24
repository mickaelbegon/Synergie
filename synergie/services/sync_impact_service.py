from __future__ import annotations

from dataclasses import dataclass

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
