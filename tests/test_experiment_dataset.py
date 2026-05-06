import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd
    from synergie.experiments.dataset import load_classification_dataset

    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


@unittest.skipUnless(HAS_PANDAS, "pandas is not available")
class ExperimentDatasetTests(unittest.TestCase):
    def test_load_classification_dataset_for_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "annotated" / "total"
            jumps_dir = root / "annotated" / "session"
            dataset_dir.mkdir(parents=True)
            jumps_dir.mkdir(parents=True)

            jump_path = jumps_dir / "jump.csv"
            frame = pd.DataFrame(
                {
                    "Euler_X": [0.0] * 300,
                    "Euler_Y": [0.0] * 300,
                    "Euler_Z": [0.0] * 300,
                    "Gyr_X": [1.0] * 300,
                    "Gyr_Y": [0.0] * 300,
                    "Gyr_Z": [0.0] * 300,
                    "Acc_X": [0.0] * 300,
                    "Acc_Y": [0.0] * 300,
                    "Acc_Z": [0.0] * 300,
                    "Combination": [0] * 300,
                }
            )
            frame.to_csv(jump_path, index=False)

            pd.DataFrame(
                [{"path": str(jump_path), "videoTimeStamp": "00:01", "type": 1, "skater": 7, "success": 1, "rotations": 2.0}]
            ).to_csv(dataset_dir / "jumplist.csv", index=False)
            pd.DataFrame([{"skater": 7, "weight": 55, "height": 165}]).to_csv(dataset_dir / "skaterData.csv", index=False)

            dataset = load_classification_dataset("type", str(dataset_dir))

            self.assertEqual(dataset.n_samples, 1)
            self.assertEqual(dataset.n_channels, 10)
            self.assertEqual(dataset.sequence_length, 240)
            self.assertEqual(dataset.labels, [1])

    def test_load_classification_dataset_for_success_uses_shorter_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "annotated" / "total"
            jumps_dir = root / "annotated" / "session"
            dataset_dir.mkdir(parents=True)
            jumps_dir.mkdir(parents=True)

            jump_path = jumps_dir / "jump.csv"
            frame = pd.DataFrame(
                {
                    "Euler_X": [0.0] * 300,
                    "Euler_Y": [0.0] * 300,
                    "Euler_Z": [0.0] * 300,
                    "Gyr_X": [1.0] * 300,
                    "Gyr_Y": [0.0] * 300,
                    "Gyr_Z": [0.0] * 300,
                    "Acc_X": [0.0] * 300,
                    "Acc_Y": [0.0] * 300,
                    "Acc_Z": [0.0] * 300,
                    "Combination": [0] * 300,
                }
            )
            frame.to_csv(jump_path, index=False)

            pd.DataFrame(
                [{"path": str(jump_path), "videoTimeStamp": "00:01", "type": 1, "skater": 7, "success": 0, "rotations": 2.0}]
            ).to_csv(dataset_dir / "jumplist.csv", index=False)
            pd.DataFrame([{"skater": 7, "weight": 55, "height": 165}]).to_csv(dataset_dir / "skaterData.csv", index=False)

            dataset = load_classification_dataset("success", str(dataset_dir))

            self.assertEqual(dataset.n_samples, 1)
            self.assertEqual(dataset.sequence_length, 180)
            self.assertEqual(dataset.labels, [0])


if __name__ == "__main__":
    unittest.main()
