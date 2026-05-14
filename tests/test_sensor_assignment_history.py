import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from synergie.sensor_assignment_history import assignment_count, record_assignment, sort_skaters_for_sensor


@dataclass
class SkaterStub:
    skater_id: str
    skater_name: str


class SensorAssignmentHistoryTests(unittest.TestCase):
    def test_record_assignment_counts_by_sensor_and_skater(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"

            record_assignment("1", "alice", "Alice", path=history_path)
            record_assignment("1", "alice", "Alice", path=history_path)
            record_assignment("10", "alice", "Alice", path=history_path)

            self.assertEqual(assignment_count("1", "alice", path=history_path), 2)
            self.assertEqual(assignment_count("10", "alice", path=history_path), 1)

    def test_sort_skaters_for_sensor_keeps_frequent_skater_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            skaters = [SkaterStub("alice", "Alice"), SkaterStub("bob", "Bob")]
            record_assignment("1", "bob", "Bob", path=history_path)
            record_assignment("1", "bob", "Bob", path=history_path)

            sorted_skaters = sort_skaters_for_sensor(skaters, "1", path=history_path)

            self.assertEqual([skater.skater_id for skater in sorted_skaters], ["bob", "alice"])


if __name__ == "__main__":
    unittest.main()
