import unittest


try:
    from core.model import model

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


if __name__ == "__main__":
    unittest.main()
