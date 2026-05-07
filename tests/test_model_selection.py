import unittest


try:
    from core.model import model
    from synergie.config import SUCCESS_WINDOW_FRAMES

    HAS_DL_STACK = True
except Exception:
    HAS_DL_STACK = False


@unittest.skipUnless(HAS_DL_STACK, "deep learning stack is not available")
class ModelSelectionTests(unittest.TestCase):
    def test_default_architecture_for_type(self):
        self.assertEqual(model.default_architecture("type"), "inceptiontime")

    def test_default_architecture_for_success(self):
        self.assertEqual(model.default_architecture("success"), "tcn")

    def test_invalid_model_combination_raises(self):
        with self.assertRaises(ValueError):
            model.build_model("success", "transformer")

    def test_success_models_expect_success_window_frames(self):
        tcn_model = model.build_model("success", "tcn")
        lstm_model = model.build_model("success", "lstm")

        self.assertEqual(tuple(tcn_model.inputs[0].shape[1:]), (SUCCESS_WINDOW_FRAMES, 10))
        self.assertEqual(tuple(lstm_model.inputs[0].shape[1:]), (SUCCESS_WINDOW_FRAMES, 10))


if __name__ == "__main__":
    unittest.main()
