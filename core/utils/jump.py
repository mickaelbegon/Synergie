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

class Jump:
    def __init__(self, start: int, end: int, df: pd.DataFrame, combinate : bool, jump_type: jumpType = jumpType.NONE, jump_success: jumpSuccess = jumpSuccess.NONE):
        """
        :param start: the frame index where the jump starts
        :param end: the frame index where the jump ends
        :param df: the dataframe containing the session where the jump is
        :param jump_type: the type of the jump (has to be set to NONE before annotation)
        """
        self.start = start
        self.end = end
        self.type = jump_type
        self.success = jump_success
        self.combinate = combinate

        self.startTimestamp = (df['SampleTimeFine'][start] - df['SampleTimeFine'][0]) / 1000
        self.endTimestamp = (df['SampleTimeFine'][end] - df['SampleTimeFine'][0]) / 1000

        # timestamps are in microseconds, I want to have the lenghs in seconds
        self.length = round(np.longlong(df['ms'][end] - df['ms'][start]) / 1000,3)

        self.signed_rotation = self.calculate_rotation(df[self.start:self.end].copy().reset_index())
        self.rotation = abs(self.signed_rotation)
        self.rotation_direction = self._rotation_direction_label(self.signed_rotation)

        self.df = self.dynamic_resize(df) # The dataframe containing the jump
        self.df["Combination"] = [int(self.combinate)]*len(self.df)
        self.df_success = self.df[SUCCESS_WINDOW_START : SUCCESS_WINDOW_START + SUCCESS_WINDOW_FRAMES]
        self.df_type = self.df[TYPE_WINDOW_START : TYPE_WINDOW_START + TYPE_WINDOW_FRAMES]
        gyr_source = df["Gyr_X_unfiltered"].replace([np.inf, -np.inf], np.nan).dropna()
        window = gyr_source.iloc[start:end]
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
