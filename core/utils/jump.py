import numpy as np
import pandas as pd
import math

import constants
from constants import jumpType, jumpSuccess


class Jump:
    """
    Represents a single jump event within a skating session.

    This class encapsulates all relevant data and computations related to a jump,
    including its start and end frames, type, success status, duration, rotation,
    and associated sensor data. It provides methods to calculate rotation, resize
    the data frame for analysis, and export jump data to a CSV file.

    Attributes:
        start (int): The frame index where the jump starts.
        end (int): The frame index where the jump ends.
        type (jumpType): The type of the jump (e.g., Axel, Lutz). Defaults to jumpType.NONE.
        success (jumpSuccess): The success status of the jump. Defaults to jumpSuccess.NONE.
        combinate (bool): Indicates whether the jump is part of a combination.
        startTimestamp (float): The timestamp (in seconds) when the jump starts relative to the session start.
        endTimestamp (float): The timestamp (in seconds) when the jump ends relative to the session start.
        length (float): The duration of the jump in seconds.
        rotation (float): The absolute value of rotation in degrees around the vertical axis during the jump.
        df (pd.DataFrame): The resized dataframe containing sensor data for the jump.
        df_success (pd.DataFrame): Subset of `df` starting from frame index 120, potentially used for success analysis.
        df_type (pd.DataFrame): Subset of `df` up to frame index 240, potentially used for type analysis.
        max_rotation_speed (float): The maximum rotation speed recorded during the jump.
    """

    def __init__(
        self,
        start: int,
        end: int,
        df: pd.DataFrame,
        combinate: bool,
        jump_type: jumpType = jumpType.NONE,
        jump_success: jumpSuccess = jumpSuccess.NONE,
    ) -> None:
        """
        Initialize a Jump instance with specific parameters.

        Args:
            start (int): The frame index where the jump starts.
            end (int): The frame index where the jump ends.
            df (pd.DataFrame): The dataframe containing the session data where the jump occurs.
            combinate (bool): Indicates whether the jump is part of a combination.
            jump_type (jumpType, optional): The type of the jump. Defaults to jumpType.NONE.
            jump_success (jumpSuccess, optional): The success status of the jump. Defaults to jumpSuccess.NONE.

        Raises:
            ValueError: If `start` or `end` indices are out of bounds of the dataframe.
        """
        # Initialize basic attributes
        self.start = start
        self.end = end
        self.type = jump_type
        self.success = jump_success
        self.combinate = combinate

        # Calculate timestamps in seconds relative to the start of the session
        self.startTimestamp = (df['SampleTimeFine'][start] - df['SampleTimeFine'][0]) / 1000
        self.endTimestamp = (df['SampleTimeFine'][end] - df['SampleTimeFine'][0]) / 1000

        # Calculate the duration of the jump in seconds
        # 'ms' column is assumed to represent time in milliseconds
        self.length = round(np.longlong(df['ms'][end] - df['ms'][start]) / 1000, 3)

        # Calculate the total rotation during the jump
        self.rotation = self.calculate_rotation(df[self.start:self.end].copy().reset_index())

        # Resize the dataframe to focus on the jump period with additional frames before and after
        self.df = self.dynamic_resize(df)
        self.df["Combination"] = [int(self.combinate)] * len(self.df)

        # Split the dataframe into parts potentially used for different analyses
        self.df_success = self.df[120:]  # Frames after index 120
        self.df_type = self.df[:240]      # Frames up to index 240

        # Clean the original dataframe by replacing infinities and dropping NaNs in 'Gyr_X_unfiltered'
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["Gyr_X_unfiltered"], how="all", inplace=True)

        # Calculate the maximum rotation speed during the jump
        self.max_rotation_speed = round(df['Gyr_X_unfiltered'][start:end].abs().max() / 360, 1)

    def calculate_rotation(self, df: pd.DataFrame) -> float:
        """
        Calculate the absolute rotation in degrees around the vertical axis during the jump.

        The rotation is calculated based on gyroscope data ('Gyr_X'), assuming the initial frame
        is when the skater is standing still. The method integrates the gyroscope readings over time
        to determine the total rotation.

        Args:
            df (pd.DataFrame): The dataframe containing sensor data for the jump.

        Returns:
            float: The absolute value of the rotation in degrees.

        Notes:
            - Frames with infinite gyroscope readings or values exceeding 1e6 are excluded.
            - Rotation is computed by integrating angular velocity over time.
        """
        # Select relevant columns for rotation calculation
        df_rots = df[["SampleTimeFine", "Gyr_X"]]

        def is_invalid_reading(row):
            """
            Determine if a gyroscope reading is invalid.

            Args:
                row (pd.Series): A row from the dataframe.

            Returns:
                bool: True if the reading is invalid, False otherwise.
            """
            return math.isinf(row["Gyr_X"]) or np.abs(row["Gyr_X"]) > 1e6

        # Remove invalid gyroscope readings
        df_rots = df_rots.drop(df_rots[df_rots.apply(is_invalid_reading, axis=1)].index)
        n = len(df_rots)

        # Extract and normalize timestamps
        tps = df_rots['SampleTimeFine'].to_numpy()
        tps = tps - tps[0]  # Relative to the first timestamp
        difftps = np.diff(tps) / 1e6  # Convert microseconds to seconds

        # Extract gyroscope angular velocities
        vit = df_rots['Gyr_X'].to_numpy()[:-1]

        # Integrate angular velocity over time to get total rotation
        pos = np.nansum(vit * difftps)
        total_rotation_x = np.abs(pos / 360)  # Convert to degrees and take absolute value

        return total_rotation_x

    def dynamic_resize(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Normalize the jump data to a specific time frame by selecting frames around the jump.

        The method ensures that there are at least 120 frames (2 seconds) before the takeoff
        and 180 frames (3 seconds) after the landing to provide sufficient context for analysis.

        Args:
            df (pd.DataFrame, optional): The original dataframe containing session data.

        Returns:
            pd.DataFrame: A resized dataframe focused on the jump period with additional frames.
        """
        # Select 120 frames before the jump start and 180 frames after the jump end
        resampled_df = df[self.start - 120:self.start + 180].copy(deep=True)

        return resampled_df

    def generate_csv(self, path: str) -> None:
        """
        Export the jump data to a CSV file.

        The method saves the resized dataframe (`self.df`) containing sensor data and jump information
        to the specified file path.

        Args:
            path (str): The file path where the CSV file will be saved.
        """
        self.df.to_csv(path, index=False)
