import unittest

import pandas as pd

from synergie.services.sync_impact_service import detect_sync_impacts


class SyncImpactServiceTests(unittest.TestCase):
    def test_detect_sync_impacts_returns_strong_stable_acceleration_change(self):
        ms = list(range(0, 2100, 100))
        frame = pd.DataFrame(
            {
                "ms": ms,
                "Acc_X": [0 if value < 1200 else 30 for value in ms],
                "Acc_Y": [0 for _value in ms],
                "Acc_Z": [1 for _value in ms],
            }
        )

        impacts = detect_sync_impacts(frame, max_candidates=1)

        self.assertEqual(len(impacts), 1)
        self.assertEqual(impacts[0].ms, 1200)
        self.assertIn(impacts[0].confidence, {"medium", "high"})

    def test_detect_sync_impacts_rejects_candidate_before_one_second(self):
        ms = list(range(0, 1600, 100))
        frame = pd.DataFrame(
            {
                "ms": ms,
                "Acc_X": [0 if value < 500 else 30 for value in ms],
                "Acc_Y": [0 for _value in ms],
                "Acc_Z": [1 for _value in ms],
            }
        )

        self.assertEqual(detect_sync_impacts(frame, max_candidates=1), [])

    def test_detect_sync_impacts_rejects_unstable_context(self):
        ms = list(range(0, 2100, 100))
        acc_x = []
        for value in ms:
            if value < 1200:
                acc_x.append(20 if (value // 100) % 2 else 0)
            else:
                acc_x.append(60 if (value // 100) % 2 else 40)
        frame = pd.DataFrame(
            {
                "ms": ms,
                "Acc_X": acc_x,
                "Acc_Y": [0 for _value in ms],
                "Acc_Z": [1 for _value in ms],
            }
        )

        self.assertEqual(detect_sync_impacts(frame, max_candidates=1), [])

    def test_detect_sync_impacts_ignores_flat_signal(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 10, 20, 30],
                "Acc_X": [1, 1, 1, 1],
                "Acc_Y": [0, 0, 0, 0],
                "Acc_Z": [0, 0, 0, 0],
            }
        )

        self.assertEqual(detect_sync_impacts(frame), [])


if __name__ == "__main__":
    unittest.main()
