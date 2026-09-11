import unittest

from core.utils.device_support import (
    is_valid_bluetooth_address,
    join_sensor_names,
    normalize_sample_time_fine,
)


class DeviceSupportTests(unittest.TestCase):
    def test_is_valid_bluetooth_address_accepts_common_formats(self):
        self.assertTrue(is_valid_bluetooth_address("AA:BB:CC:DD:EE:FF"))
        self.assertTrue(is_valid_bluetooth_address("AA-BB-CC-DD-EE-FF"))
        self.assertFalse(is_valid_bluetooth_address("AABBCCDDEEFF"))
        self.assertFalse(is_valid_bluetooth_address(""))

    def test_normalize_sample_time_fine_handles_wraparound(self):
        values = [4294967290, 3, 15]

        normalized = normalize_sample_time_fine(values)

        self.assertEqual(normalized, [0, 9, 21])

    def test_join_sensor_names_skips_empty_entries(self):
        result = join_sensor_names(["7", "", "  ", "10"])
        self.assertEqual(result, "7, 10")


if __name__ == "__main__":
    unittest.main()
