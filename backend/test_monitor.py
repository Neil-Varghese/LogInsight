"""Offline tests for monitor.py with a fake scorer (no Keras). Run: python -m unittest discover -s backend"""

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monitor

SAMPLE = Path(__file__).resolve().parent.parent / "test sets" / "sample_complete_sessions.log"


def fake_scorer(encoded):
    return [min(1.0, len(events) / 100) for events in encoded]  # longer block = higher "score"


class TestMonitor(unittest.TestCase):
    def setUp(self):
        folder = Path(tempfile.mkdtemp())
        self.log = folder / "live.log"
        self.log.write_text("")
        self.monitor = monitor.Monitor(self.log, folder / "m.sqlite3", scorer=fake_scorer)
        self.db = self.monitor._setup()
        self.lines = SAMPLE.read_text(encoding="utf-8").splitlines(keepends=True)

    def test_half_written_line_waits_for_its_newline(self):
        first, second = self.lines[0], self.lines[1]
        self.log.write_bytes((first + second.rstrip("\n")).encode())
        self.assertEqual(self.monitor._tick(self.db), 1)
        self.assertEqual(self.monitor.offset, len(first.encode()))
        self.log.write_bytes((first + second).encode())
        self.assertEqual(self.monitor._tick(self.db), 1)

    def test_window_keeps_last_events_but_counts_all(self):
        self.log.write_bytes(("".join(self.lines[:600])).encode())
        self.monitor._tick(self.db)
        longest = max(self.monitor.state.values(), key=lambda b: b["n"])
        self.assertLessEqual(len(longest["events"]), monitor.WINDOW)
        self.assertGreaterEqual(longest["n"], len(longest["events"]))

    def test_summary_threshold_and_restart_resume(self):
        self.log.write_bytes(("".join(self.lines[:600])).encode())
        self.monitor._tick(self.db)
        low, high = monitor.summary(self.db, 0.0), monitor.summary(self.db, 1.01)
        self.assertEqual(low["anomalous"], low["total"])
        self.assertEqual(high["anomalous"], 0)
        again = monitor.Monitor(self.log, self.monitor.db_path, scorer=fake_scorer)
        again._setup()
        self.assertEqual(again.offset, self.monitor.offset)
        self.assertEqual(len(again.state), len(self.monitor.state))
        self.assertEqual(again._tick(again._connect()), 0)  # nothing new, nothing re-read


if __name__ == "__main__":
    unittest.main()
