import unittest

import pandas as pd

from core.utils.jump import Jump, refine_landing_index_by_acceleration, rotation_start_index_for_turns


class JumpLandingRefinementTests(unittest.TestCase):
    def test_refines_landing_to_nearby_acceleration_peak(self):
        frame = pd.DataFrame(
            {
                "Acc_X": [1.0, 1.1, 1.0, 1.1, 1.0, 1.2, 1.0, 8.0, 1.1],
                "Acc_Y": [0.0] * 9,
                "Acc_Z": [0.0] * 9,
            }
        )

        refined = refine_landing_index_by_acceleration(frame, 2, 5, search_frames_before=2, search_frames_after=4)

        self.assertEqual(refined, 7)

    def test_keeps_detected_landing_when_no_clear_acceleration_peak_exists(self):
        frame = pd.DataFrame({"Acc_X": [1.0, 1.1, 1.0, 1.2, 1.1, 1.0, 1.1]})

        refined = refine_landing_index_by_acceleration(frame, 1, 4, search_frames_before=2, search_frames_after=2)

        self.assertEqual(refined, 4)

    def test_respects_next_takeoff_limit(self):
        frame = pd.DataFrame({"Acc_X": [1.0, 1.0, 1.1, 1.0, 1.2, 1.0, 7.5, 1.0]})

        refined = refine_landing_index_by_acceleration(
            frame,
            1,
            4,
            max_landing_index=5,
            search_frames_before=2,
            search_frames_after=4,
        )

        self.assertEqual(refined, 4)

    def test_jump_uses_refined_landing_for_timing_and_rotation(self):
        frame = pd.DataFrame(
            {
                "SampleTimeFine": [index * 100_000 for index in range(10)],
                "ms": [index * 100.0 for index in range(10)],
                "Gyr_X": [360.0] * 10,
                "Gyr_X_unfiltered": [360.0] * 10,
                "Acc_X": [1.0, 1.0, 1.1, 1.0, 1.2, 1.0, 1.0, 8.0, 1.0, 1.0],
                "Acc_Y": [0.0] * 10,
                "Acc_Z": [0.0] * 10,
            }
        )

        jump = Jump(2, 5, frame, False)

        self.assertEqual(jump.detected_end, 5)
        self.assertEqual(jump.end, 7)
        self.assertTrue(jump.landing_refined_by_acceleration)
        self.assertEqual(jump.endTimestamp, 700.0)
        self.assertGreater(jump.rotation, 0.2)

    def test_rotation_start_moves_to_first_sustained_pre_takeoff_rotation(self):
        frame = pd.DataFrame(
            {
                "Gyr_X": [0.0, 20.0, 30.0, 130.0, 135.0, 140.0, 220.0, 260.0, 280.0],
            }
        )

        start = rotation_start_index_for_turns(
            6,
            frame,
            9,
            max_search_frames_before_takeoff=5,
            min_speed_dps=120.0,
            peak_fraction=0.15,
            sustain_frames=3,
        )

        self.assertEqual(start, 3)

    def test_rotation_start_stays_at_takeoff_when_pre_rotation_is_not_sustained(self):
        frame = pd.DataFrame(
            {
                "Gyr_X": [0.0, 20.0, 150.0, 30.0, 40.0, 35.0, 220.0, 260.0, 280.0],
            }
        )

        start = rotation_start_index_for_turns(
            6,
            frame,
            9,
            max_search_frames_before_takeoff=5,
            min_speed_dps=120.0,
            peak_fraction=0.15,
            sustain_frames=3,
        )

        self.assertEqual(start, 6)


if __name__ == "__main__":
    unittest.main()
