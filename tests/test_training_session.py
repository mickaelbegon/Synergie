import unittest


try:
    import numpy as np
    import pandas as pd
    from core.data_treatment.data_generation.trainingSession import (
        consolidate_detector_intervals,
        detection_mask_from_second_derivative,
        detector_intervals,
        gather_jumps,
        trainingSession,
    )

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
            detection_derivative_polarity=1,
            consolidation_gap_frames=12,
        )
        self.assertEqual(session.detection_threshold, -0.4)
        self.assertEqual(session.smoothing_sigma, 12)
        self.assertEqual(session.combination_gap_frames, 90)
        self.assertEqual(session.detection_derivative_polarity, 1)
        self.assertEqual(session.consolidation_gap_frames, 12)

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

    def test_gather_jumps_pairs_begin_with_next_valid_end(self):
        frame = self._build_dataframe().iloc[:12].copy()
        frame["ms"] = (frame["SampleTimeFine"] - frame["SampleTimeFine"].iloc[0]) / 1000
        frame["Gyr_X_unfiltered"] = frame["Gyr_X"]
        frame["X_gyr_second_derivative_crossing"] = [
            True,
            False,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
        ]

        jumps = gather_jumps(frame, combination_gap_frames=2)

        self.assertEqual(len(jumps), 2)
        self.assertLess(jumps[0].start, jumps[0].end)
        self.assertEqual(jumps[0].start, 2)
        self.assertEqual(jumps[0].end, 4)
        self.assertEqual(jumps[1].start, 6)
        self.assertEqual(jumps[1].end, 9)

    def test_mirrored_rotation_polarity_matches_mirrored_signal(self):
        derivative = np.array([0.0, -0.1, -0.4, -0.1, 0.0, -0.5, 0.0])

        detected = detection_mask_from_second_derivative(
            derivative,
            detection_threshold=-0.3,
            derivative_polarity=-1,
        )
        mirrored_detected = detection_mask_from_second_derivative(
            -derivative,
            detection_threshold=-0.3,
            derivative_polarity=1,
        )

        np.testing.assert_array_equal(detected, mirrored_detected)
        self.assertEqual(detector_intervals(detected), detector_intervals(mirrored_detected))

    def test_legacy_mask_remains_sign_specific_until_mirrored_polarity_is_enabled(self):
        derivative = np.array([0.0, -0.4, 0.0])

        legacy = detection_mask_from_second_derivative(derivative, detection_threshold=-0.3)
        legacy_mirrored = detection_mask_from_second_derivative(-derivative, detection_threshold=-0.3)
        mirrored = detection_mask_from_second_derivative(
            -derivative,
            detection_threshold=-0.3,
            derivative_polarity=1,
        )

        self.assertTrue(legacy[1])
        self.assertFalse(legacy_mirrored[1])
        self.assertTrue(mirrored[1])

    def test_mirrored_session_uses_opposite_polarity_without_changing_candidates(self):
        frame = self._build_dataframe()
        mirrored_frame = frame.copy()
        mirrored_frame["Gyr_X"] = -mirrored_frame["Gyr_X"]

        legacy = trainingSession(frame, detection_threshold=-0.01, smoothing_sigma=2)
        mirrored = trainingSession(
            mirrored_frame,
            detection_threshold=-0.01,
            smoothing_sigma=2,
            detection_derivative_polarity=1,
        )

        self.assertEqual(
            [(jump.start, jump.detected_end) for jump in legacy.jumps],
            [(jump.start, jump.detected_end) for jump in mirrored.jumps],
        )

    def test_detector_interval_consolidation_is_explicit_and_preserves_separate_events_by_default(self):
        intervals = [(10, 14), (18, 22), (70, 74)]

        self.assertEqual(consolidate_detector_intervals(intervals), intervals)
        self.assertEqual(
            consolidate_detector_intervals(intervals, max_gap_frames=3),
            [(10, 22), (70, 74)],
        )

    def test_gather_jumps_merges_only_when_requested(self):
        frame = self._build_dataframe().iloc[:100].copy()
        frame["ms"] = (frame["SampleTimeFine"] - frame["SampleTimeFine"].iloc[0]) / 1000
        frame["Gyr_X_unfiltered"] = frame["Gyr_X"]
        frame["X_gyr_second_derivative_crossing"] = False
        frame.loc[10:14, "X_gyr_second_derivative_crossing"] = True
        frame.loc[18:22, "X_gyr_second_derivative_crossing"] = True

        separate = gather_jumps(frame, consolidation_gap_frames=0)
        consolidated = gather_jumps(frame, consolidation_gap_frames=3)

        self.assertEqual(len(separate), 2)
        self.assertEqual(len(consolidated), 1)
        self.assertEqual((consolidated[0].start, consolidated[0].detected_end), (9, 22))


if __name__ == "__main__":
    unittest.main()
