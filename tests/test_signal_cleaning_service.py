import unittest

import pandas as pd

from synergie.services.signal_cleaning_service import clean_acceleration_outliers, clean_imu_outliers


class SignalCleaningServiceTests(unittest.TestCase):
    def test_clean_acceleration_outliers_interpolates_impossible_values(self):
        frame = pd.DataFrame(
            {
                "Acc_X": [1.0, 999.0, 3.0],
                "Acc_Y": [2.0, 4.0, 6.0],
                "Acc_Z": [-999.0, 5.0, 7.0],
            }
        )

        cleaned, report = clean_acceleration_outliers(frame, limit_g=32.0)

        self.assertEqual(report["replaced"], 2)
        self.assertEqual(cleaned.loc[1, "Acc_X"], 2.0)
        self.assertEqual(cleaned.loc[0, "Acc_Z"], 5.0)

    def test_clean_imu_outliers_interpolates_impossible_gyroscope_values(self):
        frame = pd.DataFrame(
            {
                "Acc_X": [1.0, 999.0, 3.0],
                "Gyr_X": [10.0, 1e34, 30.0],
                "Gyr_Y": [0.0, 5.0, 10.0],
            }
        )

        cleaned, report = clean_imu_outliers(frame, acceleration_limit_g=32.0, gyroscope_limit_dps=5000.0)

        self.assertEqual(report["replaced"], 2)
        self.assertEqual(report["acceleration"]["columns"]["Acc_X"], 1)
        self.assertEqual(report["gyroscope"]["columns"]["Gyr_X"], 1)
        self.assertEqual(cleaned.loc[1, "Acc_X"], 2.0)
        self.assertEqual(cleaned.loc[1, "Gyr_X"], 20.0)


if __name__ == "__main__":
    unittest.main()
