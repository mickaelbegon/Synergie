import os
import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd
    from synergie.experiments.dataset import load_classification_dataset
    from synergie.config import JUMP_WINDOW_FRAMES, SUCCESS_WINDOW_FRAMES, TYPE_WINDOW_FRAMES
    from synergie.services.hdf5_archive_service import export_hdf5_archive

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
                    "Euler_X": [0.0] * JUMP_WINDOW_FRAMES,
                    "Euler_Y": [0.0] * JUMP_WINDOW_FRAMES,
                    "Euler_Z": [0.0] * JUMP_WINDOW_FRAMES,
                    "Gyr_X": [1.0] * JUMP_WINDOW_FRAMES,
                    "Gyr_Y": [0.0] * JUMP_WINDOW_FRAMES,
                    "Gyr_Z": [0.0] * JUMP_WINDOW_FRAMES,
                    "Acc_X": [0.0] * JUMP_WINDOW_FRAMES,
                    "Acc_Y": [0.0] * JUMP_WINDOW_FRAMES,
                    "Acc_Z": [0.0] * JUMP_WINDOW_FRAMES,
                    "Combination": [0] * JUMP_WINDOW_FRAMES,
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
            self.assertEqual(dataset.sequence_length, TYPE_WINDOW_FRAMES)
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
                    "Euler_X": [0.0] * JUMP_WINDOW_FRAMES,
                    "Euler_Y": [0.0] * JUMP_WINDOW_FRAMES,
                    "Euler_Z": [0.0] * JUMP_WINDOW_FRAMES,
                    "Gyr_X": [1.0] * JUMP_WINDOW_FRAMES,
                    "Gyr_Y": [0.0] * JUMP_WINDOW_FRAMES,
                    "Gyr_Z": [0.0] * JUMP_WINDOW_FRAMES,
                    "Acc_X": [0.0] * JUMP_WINDOW_FRAMES,
                    "Acc_Y": [0.0] * JUMP_WINDOW_FRAMES,
                    "Acc_Z": [0.0] * JUMP_WINDOW_FRAMES,
                    "Combination": [0] * JUMP_WINDOW_FRAMES,
                }
            )
            frame.to_csv(jump_path, index=False)

            pd.DataFrame(
                [{"path": str(jump_path), "videoTimeStamp": "00:01", "type": 1, "skater": 7, "success": 0, "rotations": 2.0}]
            ).to_csv(dataset_dir / "jumplist.csv", index=False)
            pd.DataFrame([{"skater": 7, "weight": 55, "height": 165}]).to_csv(dataset_dir / "skaterData.csv", index=False)

            dataset = load_classification_dataset("success", str(dataset_dir))

            self.assertEqual(dataset.n_samples, 1)
            self.assertEqual(dataset.sequence_length, SUCCESS_WINDOW_FRAMES)
            self.assertEqual(dataset.labels, [0])

    def test_load_classification_dataset_reads_hdf5_after_segment_csv_move(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            os.chdir(root)
            try:
                dataset_dir = root / "data" / "annotated" / "total"
                jumps_dir = root / "data" / "annotated" / "session"
                dataset_dir.mkdir(parents=True)
                jumps_dir.mkdir(parents=True)
                jump_path = jumps_dir / "jump.csv"
                relative_jump_path = "data/annotated/session/jump.csv"
                frame = pd.DataFrame(
                    {
                        "Euler_X": [0.0] * JUMP_WINDOW_FRAMES,
                        "Euler_Y": [0.0] * JUMP_WINDOW_FRAMES,
                        "Euler_Z": [0.0] * JUMP_WINDOW_FRAMES,
                        "Gyr_X": [1.0] * JUMP_WINDOW_FRAMES,
                        "Gyr_Y": [0.0] * JUMP_WINDOW_FRAMES,
                        "Gyr_Z": [0.0] * JUMP_WINDOW_FRAMES,
                        "Acc_X": [0.0] * JUMP_WINDOW_FRAMES,
                        "Acc_Y": [0.0] * JUMP_WINDOW_FRAMES,
                        "Acc_Z": [0.0] * JUMP_WINDOW_FRAMES,
                        "Combination": [0] * JUMP_WINDOW_FRAMES,
                    }
                )
                frame.to_csv(jump_path, index=False)
                pd.DataFrame(
                    [{"path": relative_jump_path, "videoTimeStamp": "00:01", "type": 1, "skater": 7, "success": 1, "rotations": 2.0}]
                ).to_csv(dataset_dir / "jumplist.csv", index=False)
                pd.DataFrame([{"skater": 7, "weight": 55, "height": 165}]).to_csv(dataset_dir / "skaterData.csv", index=False)
                export_hdf5_archive(
                    "data/synergie_archive.h5",
                    manifest={
                        "format": "synergie-backup-manifest-v1",
                        "pending_annotation_files": [],
                        "training_dataset": {"trials": [{"path": relative_jump_path}]},
                    },
                )
                jump_path.unlink()

                dataset = load_classification_dataset("type", str(dataset_dir))
            finally:
                os.chdir(original_cwd)

        self.assertEqual(dataset.n_samples, 1)
        self.assertEqual(dataset.n_channels, 10)
        self.assertEqual(dataset.sequence_length, TYPE_WINDOW_FRAMES)
        self.assertEqual(dataset.labels, [1])


if __name__ == "__main__":
    unittest.main()
