import numpy as np
import pandas as pd

import constants
from constants import jumpType,jumpSuccess
import math
from synergie.config import (
    JUMP_WINDOW_FRAMES,
    SEGMENT_FRAMES_BEFORE_TAKEOFF,
    SUCCESS_WINDOW_FRAMES,
    SUCCESS_WINDOW_START,
    TYPE_WINDOW_START,
    TYPE_WINDOW_FRAMES,
)


ACCELERATION_COLUMNS = ("Acc_X", "Acc_Y", "Acc_Z")
LANDING_ACCELERATION_SEARCH_FRAMES_BEFORE = 20
LANDING_ACCELERATION_SEARCH_FRAMES_AFTER = 25
ROTATION_START_MAX_SEARCH_FRAMES_BEFORE_TAKEOFF = 30


def refine_landing_index_by_acceleration(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    *,
    max_landing_index: int | None = None,
    search_frames_before: int = LANDING_ACCELERATION_SEARCH_FRAMES_BEFORE,
    search_frames_after: int = LANDING_ACCELERATION_SEARCH_FRAMES_AFTER,
    min_prominence_g: float = 1.5,
    prominence_mad_multiplier: float = 6.0,
) -> int:
    """Snap the landing bound to a nearby acceleration-impact peak when clear."""
    available_columns = [column for column in ACCELERATION_COLUMNS if column in df]
    if not available_columns or len(df) == 0:
        return int(end_index)

    start_index = int(start_index)
    end_index = int(end_index)
    lower = max(start_index + 1, end_index - int(search_frames_before), 0)
    upper = min(len(df) - 1, end_index + int(search_frames_after))
    if max_landing_index is not None:
        upper = min(upper, int(max_landing_index))
    if upper <= lower:
        return end_index

    acceleration = df[available_columns].apply(pd.to_numeric, errors="coerce")
    if len(available_columns) == 1:
        acceleration_norm = acceleration[available_columns[0]].abs().to_numpy(dtype="float64")
    else:
        acceleration_norm = np.sqrt(np.nansum(np.square(acceleration.to_numpy(dtype="float64")), axis=1))

    search_values = acceleration_norm[lower : upper + 1]
    finite_mask = np.isfinite(search_values)
    if not finite_mask.any():
        return end_index

    finite_values = search_values[finite_mask]
    baseline = float(np.nanmedian(finite_values))
    mad = float(np.nanmedian(np.abs(finite_values - baseline)))
    threshold = baseline + max(float(min_prominence_g), float(prominence_mad_multiplier) * mad)

    local_peak_offset = int(np.nanargmax(search_values))
    peak_value = float(search_values[local_peak_offset])
    if not np.isfinite(peak_value) or peak_value < threshold:
        return end_index

    return lower + local_peak_offset


def rotation_start_index_for_turns(
    start_index: int,
    frame: pd.DataFrame | None = None,
    landing_index: int | None = None,
    *,
    max_search_frames_before_takeoff: int = ROTATION_START_MAX_SEARCH_FRAMES_BEFORE_TAKEOFF,
    min_speed_dps: float = 240.0,
    peak_fraction: float = 0.25,
    sustain_frames: int = 4,
) -> int:
    """Find where useful rotation starts, allowing normal pre-rotation on ice."""
    start_index = int(start_index)
    lower = max(0, start_index - int(max_search_frames_before_takeoff))
    if frame is None or "Gyr_X" not in frame:
        return lower

    gyro_column = "Gyr_X_smoothed" if "Gyr_X_smoothed" in frame else "Gyr_X"
    gyro = pd.to_numeric(frame[gyro_column], errors="coerce").to_numpy(dtype="float64")
    if len(gyro) == 0:
        return lower

    start_index = min(max(0, start_index), len(gyro) - 1)
    landing_index = len(gyro) - 1 if landing_index is None else min(max(start_index + 1, int(landing_index)), len(gyro))
    rotation_window = gyro[start_index:landing_index]
    finite_rotation_window = rotation_window[np.isfinite(rotation_window)]
    if len(finite_rotation_window) == 0:
        return start_index

    direction = 1.0 if np.nansum(finite_rotation_window) >= 0 else -1.0
    projected = direction * gyro
    peak = float(np.nanmax(projected[start_index:landing_index])) if landing_index > start_index else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return start_index

    threshold = max(float(min_speed_dps), float(peak_fraction) * peak)
    upper = min(len(projected), start_index + int(sustain_frames))
    active = np.isfinite(projected[lower:upper]) & (projected[lower:upper] >= threshold)
    if not active.any():
        return start_index

    for offset in range(0, start_index - lower + 1):
        if active[offset : offset + int(sustain_frames)].sum() == int(sustain_frames):
            return lower + offset
    return start_index


class Jump:
    def __init__(
        self,
        start: int,
        end: int,
        df: pd.DataFrame,
        combinate : bool,
        jump_type: jumpType = jumpType.NONE,
        jump_success: jumpSuccess = jumpSuccess.NONE,
        max_landing_index: int | None = None,
    ):
        """
        :param start: the frame index where the jump starts
        :param end: the frame index where the jump ends
        :param df: the dataframe containing the session where the jump is
        :param jump_type: the type of the jump (has to be set to NONE before annotation)
        """
        self.start = start
        self.detected_end = end
        self.end = refine_landing_index_by_acceleration(df, start, end, max_landing_index=max_landing_index)
        self.rotation_start = rotation_start_index_for_turns(start, df, self.end)
        self.landing_refined_by_acceleration = self.end != self.detected_end
        self.type = jump_type
        self.success = jump_success
        self.combinate = combinate

        self.startTimestamp = (df['SampleTimeFine'][start] - df['SampleTimeFine'][0]) / 1000
        self.rotationStartTimestamp = (df['SampleTimeFine'][self.rotation_start] - df['SampleTimeFine'][0]) / 1000
        self.endTimestamp = (df['SampleTimeFine'][self.end] - df['SampleTimeFine'][0]) / 1000

        # timestamps are in microseconds, I want to have the lenghs in seconds
        self.length = round(np.longlong(df['ms'][self.end] - df['ms'][start]) / 1000,3)

        self.signed_rotation = self.calculate_rotation(df[self.rotation_start:self.end].copy().reset_index())
        self.rotation = abs(self.signed_rotation)
        self.rotation_direction = self._rotation_direction_label(self.signed_rotation)

        self.df = self.dynamic_resize(df) # The dataframe containing the jump
        self.df["Combination"] = [int(self.combinate)]*len(self.df)
        self.df_success = self.df[SUCCESS_WINDOW_START : SUCCESS_WINDOW_START + SUCCESS_WINDOW_FRAMES]
        self.df_type = self.df[TYPE_WINDOW_START : TYPE_WINDOW_START + TYPE_WINDOW_FRAMES]
        gyr_source = df["Gyr_X_unfiltered"].replace([np.inf, -np.inf], np.nan).dropna()
        window = gyr_source.iloc[self.rotation_start:self.end]
        self.max_rotation_speed = round(window.abs().max() / 360, 1) if not window.empty else 0.0

    def calculate_rotation(self, df):
        """
        calculates the rotation in degrees around the vertical axis, the initial frame is a frame where the skater is
        standing still
        :param df: the dataframe containing the jump
        :return: the absolute value of the rotation in degrees
        """
        # initial frame is the reference frame, I want to compute rotations around the "Euler_X" axis
        df_rots = df[["SampleTimeFine", "Gyr_X"]]
        def check(s):
            return math.isinf(s["Gyr_X"]) or np.abs(s["Gyr_X"]) > 1e6

        df_rots = df_rots.drop(df_rots[df_rots.apply(check,axis=1)].index)
        n = len(df_rots)

        tps = df_rots['SampleTimeFine'].to_numpy().reshape(1,n)[0]
        tps = tps - tps[0]
        difftps = np.diff(tps)/1e6
        vit = df_rots['Gyr_X'].to_numpy().reshape(1,n)[0][:-1]
        pos = np.nansum(np.array(vit) * np.array(difftps))
        return pos / 360

    def _rotation_direction_label(self, signed_rotation: float) -> str:
        if abs(signed_rotation) < 0.05:
            return "unknown"
        if signed_rotation > 0:
            return "positive"
        return "negative"

    def dynamic_resize(self, df: pd.DataFrame = None):
        """
        normalize the jump to the configured frame window around takeoff.
        :param df: the dataframe containing the session where the jump is
        :return: the new dataframe
        """
        frames_before_takeoff = SEGMENT_FRAMES_BEFORE_TAKEOFF
        frames_after_takeoff = JUMP_WINDOW_FRAMES - frames_before_takeoff
        resampled_df = df[self.start - frames_before_takeoff:self.start + frames_after_takeoff].copy(deep=True)

        return resampled_df

    def generate_csv(self, path: str):
        """
        exports the jump to a csv file
        :param path:
        :return:
        """
        self.df.to_csv(path, index=False)
