import unittest

import pandas as pd

from synergie.services.sync_impact_service import detect_sync_impacts


class SyncImpactServiceTests(unittest.TestCase):
    def test_detect_sync_impacts_returns_strong_early_acceleration_change(self):
        frame = pd.DataFrame(
            {
                "ms": [0, 10, 20, 30, 40, 50],
                "Acc_X": [0, 0, 0, 30, 30, 30],
                "Acc_Y": [0, 0, 0, 0, 0, 0],
                "Acc_Z": [1, 1, 1, 1, 1, 1],
            }
        )

        impacts = detect_sync_impacts(frame, max_candidates=1)

        self.assertEqual(len(impacts), 1)
        self.assertEqual(impacts[0].ms, 30)
        self.assertIn(impacts[0].confidence, {"medium", "high"})

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
