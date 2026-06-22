from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyncImpact:
    index: int
    ms: float
    strength: float
    confidence: str


def detect_sync_impacts(session_df, *, max_candidates: int = 5, search_ms: float | None = 8000.0) -> list[SyncImpact]:
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
        if any(abs(ms - impact.ms) < min_gap_ms for impact in impacts):
            continue
        confidence = "high" if score >= 12.0 else "medium" if score >= 6.0 else "low"
        impacts.append(SyncImpact(index=int(frame.index[int(local_index)]), ms=ms, strength=strength, confidence=confidence))
        if len(impacts) >= max_candidates:
            break
    return sorted(impacts, key=lambda impact: impact.ms)
