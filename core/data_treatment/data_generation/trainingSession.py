import numpy as np
import pandas as pd
import scipy as sp

import constants
from core.utils import plot
from core.utils.jump import Jump
from synergie.config import (
    ACCELERATION_ABERRANT_LIMIT_G,
    DEFAULT_COMBINATION_GAP_FRAMES,
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_SMOOTHING_SIGMA,
)
from synergie.services.signal_cleaning_service import clean_imu_outliers


def gather_jumps(df: pd.DataFrame, combination_gap_frames: int = DEFAULT_COMBINATION_GAP_FRAMES) -> list[Jump]:
    """
    detects and gathers all the jumps in a dataframe
    :param df: the dataframe containing the session data
    :return: list of jumps done
    """
    jumps = []
    # Find indices where 'X_gyr_second_derivative_crossing' transitions from False to True
    begin = np.where(np.diff(df['X_gyr_second_derivative_crossing'].astype(int)) == 1)[0]
    # Find indices where 'X_gyr_second_derivative_crossing' transitions from True to False
    end = np.where(np.diff(df['X_gyr_second_derivative_crossing'].astype(int)) == -1)[0]
    
    end_cursor = 0
    previous_begin = None
    begin_list = [int(value) for value in begin]
    for begin_position, begin_index in enumerate(begin_list):
        while end_cursor < len(end) and end[end_cursor] <= begin_index:
            end_cursor += 1
        if end_cursor >= len(end):
            break

        end_index = int(end[end_cursor])
        end_cursor += 1
        max_landing_index = begin_list[begin_position + 1] - 1 if begin_position + 1 < len(begin_list) else None
        combinate = previous_begin is not None and (begin_index - previous_begin) < combination_gap_frames
        jumps.append(Jump(begin_index, end_index, df, combinate, max_landing_index=max_landing_index))
        previous_begin = begin_index

    return jumps

class trainingSession:
    """
    This class is meant to describe a training session in a sport context. Not to be confused with a training session in a machine learning context (class training)
    contains the preprocessed dataframe and the jumps
    """
    def __init__(
        self,
        df: pd.DataFrame,
        sampleTimefineSynchro: int = 0,
        detection_threshold: float | None = None,
        smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA,
        combination_gap_frames: int = DEFAULT_COMBINATION_GAP_FRAMES,
    ):
        """
        :param path: path of the CSV
        :param synchroFrame: the frame where the synchro tap is
        """
        self.detection_threshold = DEFAULT_DETECTION_THRESHOLD if detection_threshold is None else detection_threshold
        self.smoothing_sigma = smoothing_sigma
        self.combination_gap_frames = combination_gap_frames
        df = self.__load_and_preprocess_data(df, sampleTimefineSynchro)
        self.initFromDataFrame(df)

    def __load_and_preprocess_data(self, df: pd.DataFrame, sampleTimefineSynchro: int = 0) -> pd.DataFrame:
        """
        loads a dataframe from a csv, and preprocess data
        :param self: path to the csv file
        :return: the dataframe with preprocessed fields
        """
        df = df.astype({'PacketCounter': 'int64', 'SampleTimeFine': 'ulonglong', 'Euler_X': 'float64', 'Euler_Y': 'float64','Euler_Z': 'float64', 'Acc_X': 'float64', 'Acc_Y': 'float64', 'Acc_Z': 'float64', 'Gyr_X': 'float64', 'Gyr_Y': 'float64', 'Gyr_Z': 'float64'})
        df, _cleaning_report = clean_imu_outliers(df, acceleration_limit_g=ACCELERATION_ABERRANT_LIMIT_G)

        if sampleTimefineSynchro != 0:
            # slice the list from sampleTimefineSynchro
            synchroIndex = df[df['SampleTimeFine'] >= sampleTimefineSynchro].index[0]
            df = df[synchroIndex:].reset_index(drop=True)
        df = df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # adding the ms field, indicating how much ms has past since the beginning of the recording
        # we are using the SampleTimeFine field, which is a timestamp in microsecond

        # add 2^32 to the SampleTimeFine field when it is smaller than the previous one, because it means that the counter has overflowed
        initial_timeStamp = df['SampleTimeFine'][0]
        df.loc[df["SampleTimeFine"] < initial_timeStamp, 'SampleTimeFine'] += 4294967296

        initialSampleTimeFine = df['SampleTimeFine'][0]
        df['ms'] = (df['SampleTimeFine'] - initialSampleTimeFine) / 1000
        df['X_acc_derivative'] = df['Acc_X'].diff()
        df['Y_acc_derivative'] = df['Acc_Y'].diff()
        df['Z_acc_derivative'] = df['Acc_Z'].diff()
        df["Gyr_X_unfiltered"] = df["Gyr_X"].copy(deep=True)
        df["Gyr_X_smoothed"] = sp.ndimage.gaussian_filter1d(df["Gyr_X"], sigma=self.smoothing_sigma)
        df['X_gyr_derivative'] = df['Gyr_X_smoothed'].diff()
        df['Y_gyr_derivative'] = df['Gyr_Y'].diff()
        df['Z_gyr_derivative'] = df['Gyr_Z'].diff()
        df["X_gyr_second_derivative"] = df['X_gyr_derivative'].diff()

        # add markers when the value is crossing -0.2
        df['X_gyr_second_derivative_crossing'] = [False if x > self.detection_threshold else True for x in
                                                  df['X_gyr_second_derivative']]
        return df
    
    def initFromDataFrame(self, df: pd.DataFrame):
        """
        can be called as a constructor, provided that the dataframe correctly been preprocessed
        this function was meant to be a constructor overload. Things would be simpler if python was a decent programming language
        :param df: the dataframe containing the whole session
        """
        self.df = df
        self.jumps = gather_jumps(df, combination_gap_frames=self.combination_gap_frames)

    def plot(self):
        timestamps = [i.startTimestamp for i in self.jumps] + [i.endTimestamp for i in self.jumps]
        plot.plot_data(self.df, timestamps, str(self))
