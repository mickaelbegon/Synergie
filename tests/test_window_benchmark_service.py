import unittest

from synergie.services.window_benchmark_service import benchmark_temporal_offsets, benchmark_temporal_windows


class WindowBenchmarkServiceTests(unittest.TestCase):
    def test_requires_no_explicit_windows_to_define_defaults(self):
        self.assertTrue(callable(benchmark_temporal_windows))

    def test_rejects_negative_offsets_without_longer_segments(self):
        with self.assertRaisesRegex(ValueError, "re-exported with more pre-takeoff context"):
            benchmark_temporal_offsets("type", "unused", "inceptiontime", offsets=[-40])
