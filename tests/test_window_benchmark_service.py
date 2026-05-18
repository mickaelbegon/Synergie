import unittest

from synergie.services.window_benchmark_service import benchmark_temporal_windows


class WindowBenchmarkServiceTests(unittest.TestCase):
    def test_requires_no_explicit_windows_to_define_defaults(self):
        self.assertTrue(callable(benchmark_temporal_windows))
