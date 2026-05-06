import unittest


try:
    import numpy as np
    import pandas as pd
    from core.data_treatment.data_generation.trainingSession import trainingSession

    HAS_NUMERIC_STACK = True
except Exception:
    HAS_NUMERIC_STACK = False


@unittest.skipUnless(HAS_NUMERIC_STACK, "numeric stack is not available")
class TrainingSessionTests(unittest.TestCase):
    def _build_dataframe(self):
        size = 600
        sample_time = np.arange(size, dtype=np.uint64) * 10000
        gyr_x = np.zeros(size, dtype=float)
        gyr_x[200:240] = -120
        gyr_x[240:280] = 120

        data = {
            "PacketCounter": np.arange(size, dtype=np.int64),
            "SampleTimeFine": sample_time,
            "Euler_X": np.zeros(size, dtype=float),
            "Euler_Y": np.zeros(size, dtype=float),
            "Euler_Z": np.zeros(size, dtype=float),
            "Acc_X": np.zeros(size, dtype=float),
            "Acc_Y": np.zeros(size, dtype=float),
            "Acc_Z": np.zeros(size, dtype=float),
            "Gyr_X": gyr_x,
            "Gyr_Y": np.zeros(size, dtype=float),
            "Gyr_Z": np.zeros(size, dtype=float),
        }
        return pd.DataFrame(data)

    def test_session_keeps_detection_parameters(self):
        session = trainingSession(
            self._build_dataframe(),
            detection_threshold=-0.4,
            smoothing_sigma=12,
            combination_gap_frames=90,
        )
        self.assertEqual(session.detection_threshold, -0.4)
        self.assertEqual(session.smoothing_sigma, 12)
        self.assertEqual(session.combination_gap_frames, 90)

    def test_preprocessing_creates_detection_columns(self):
        session = trainingSession(self._build_dataframe(), detection_threshold=-0.1, smoothing_sigma=6)
        self.assertIn("Gyr_X_smoothed", session.df.columns)
        self.assertIn("X_gyr_second_derivative", session.df.columns)
        self.assertIn("X_gyr_second_derivative_crossing", session.df.columns)

    def test_jump_rotation_exposes_signed_and_absolute_values(self):
        session = trainingSession(self._build_dataframe(), detection_threshold=-0.01, smoothing_sigma=2)
        self.assertGreaterEqual(len(session.jumps), 1)
        jump = session.jumps[0]
        self.assertAlmostEqual(jump.rotation, abs(jump.signed_rotation))
        self.assertIn(jump.rotation_direction, {"positive", "negative", "unknown"})


if __name__ == "__main__":
    unittest.main()
