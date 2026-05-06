import unittest

try:
    import pandas as pd
    from synergie.services.jump_predictions import build_training_jump_payload

    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


@unittest.skipUnless(HAS_PANDAS, "pandas is not available")
class JumpPredictionPayloadTests(unittest.TestCase):
    def test_payload_rounds_standard_jump_rotations(self):
        predictions = pd.DataFrame(
            [
                {
                    "videoTimeStamp": "1:5",
                    "type": 1,
                    "success": 1,
                    "rotations": "1.7",
                    "rotation_speed": 2.3,
                    "length": 0.8,
                }
            ]
        )

        payload = build_training_jump_payload(predictions, "training-1")

        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["training_id"], "training-1")
        self.assertEqual(payload[0]["jump_rotations"], 2.0)
        self.assertEqual(payload[0]["jump_time"], "01:05")

    def test_payload_falls_back_to_unknown_rotations_when_needed(self):
        predictions = pd.DataFrame(
            [
                {
                    "videoTimeStamp": "0:9",
                    "type": 2,
                    "success": 0,
                    "rotations": "0.2",
                    "rotation_speed": 1.1,
                    "length": 0.5,
                }
            ]
        )

        payload = build_training_jump_payload(predictions, "training-2")

        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["jump_rotations"], 0.0)
        self.assertEqual(payload[0]["jump_time"], "00:09")


if __name__ == "__main__":
    unittest.main()
