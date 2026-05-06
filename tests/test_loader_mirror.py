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


if __name__ == "__main__":
    unittest.main()
