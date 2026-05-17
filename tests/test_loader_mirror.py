import unittest

try:
    import numpy as np
    from core.model.training.loader import Loader

    HAS_TRAINING_STACK = True
except Exception:
    HAS_TRAINING_STACK = False


@unittest.skipUnless(HAS_TRAINING_STACK, "training stack is not available")
class LoaderMirrorTests(unittest.TestCase):
    def test_mirror_temporal_features_flips_rotation_axes_only(self):
        loader = Loader.__new__(Loader)
        fields = ["Euler_X", "Euler_Y", "Gyr_X", "Gyr_Y"]
        features = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [-1.0, -2.0, -3.0, -4.0],
            ]
        )

        mirrored = loader._mirror_temporal_features(features, fields)

        np.testing.assert_array_equal(mirrored[:, 0], -features[:, 0])
        np.testing.assert_array_equal(mirrored[:, 2], -features[:, 2])
        np.testing.assert_array_equal(mirrored[:, 1], features[:, 1])
        np.testing.assert_array_equal(mirrored[:, 3], features[:, 3])

    def test_compute_class_weight_uses_inverse_frequency(self):
        weights = Loader._compute_class_weight([0, 0, 0, 1])

        self.assertAlmostEqual(weights[0], 4 / (2 * 3))
        self.assertAlmostEqual(weights[1], 4 / (2 * 1))

    def test_can_use_stratify_requires_enough_examples_per_class(self):
        self.assertTrue(Loader._can_use_stratify([0, 0, 1, 1, 0, 1], train_ratio=0.8))
        self.assertFalse(Loader._can_use_stratify([0, 0, 0, 1], train_ratio=0.8))

    def test_lookup_skater_info_explains_unknown_athlete(self):
        import pandas as pd

        with self.assertRaisesRegex(ValueError, "Unknown skater '20250901_0910_10'"):
            Loader._lookup_skater_info(
                pd.DataFrame({"skater": ["10"], "weight": [71], "height": [174]}),
                "20250901_0910_10",
            )

    def test_lookup_skater_info_returns_neutral_scalars_when_disabled(self):
        import pandas as pd

        values = Loader._lookup_skater_info(
            pd.DataFrame({"skater": [], "weight": [], "height": []}),
            "unknown",
            use_scalar_features=False,
        )

        np.testing.assert_array_equal(values, np.zeros(2, dtype=np.float32))


class TrainerConfigurationTests(unittest.TestCase):
    def test_early_stopping_uses_requested_patience(self):
        from core.model.training.training import Trainer

        trainer = Trainer.__new__(Trainer)
        callback = trainer.early_stopping()

        self.assertEqual(callback.patience, 40)
        self.assertEqual(callback.monitor, "val_accuracy")


if __name__ == "__main__":
    unittest.main()
