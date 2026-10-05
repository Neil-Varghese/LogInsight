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

    def test_traffic_adds_up_received_block_sizes(self):
        self.log.write_bytes(("".join(self.lines[:600])).encode())
        self.monitor._tick(self.db)
        expected = sum(int(m.group(1)) for l in self.lines[:600] if (m := monitor.TRANSFER.match(monitor.HEADER.match(l).group(3))))
        traffic = monitor.traffic_summary(self.db)
        self.assertGreater(expected, 0)
        self.assertEqual(traffic["total_bytes"], expected)
        self.assertEqual(sum(p["bytes"] for p in traffic["series"]), expected)

    def test_search_matches_whole_ip_and_time_window(self):
        self.log.write_bytes(("".join(self.lines[:600])).encode())
        self.monitor._tick(self.db)
        found = monitor.search_log(self.log, self.db, "10.250.19.102")
        self.assertGreater(found["total_lines"], 0)
        self.assertTrue(all("10.250.19.102" in l for l in found["lines"]))
        self.assertEqual(monitor.search_log(self.log, self.db, "10.250.19.10")["total_lines"], 0)  # prefix of another address
        first = monitor._epoch(*monitor.HEADER.match(self.lines[0]).groups()[:2])
        self.assertEqual(monitor.search_log(self.log, self.db, "10.250.19.102", end=first - 1)["total_lines"], 0)
        self.assertGreater(monitor.search_log(self.log, self.db, start=first, end=first)["total_lines"], 0)

    def test_ip_summary_counts_lines_and_flags_active(self):
        self.log.write_bytes(("".join(self.lines[:600])).encode())
        self.monitor._tick(self.db)
        found = monitor.ip_summary(self.db)
        self.assertEqual(found["ips"][0]["lines"], monitor.search_log(self.log, self.db, found["ips"][0]["ip"])["total_lines"])
        self.assertTrue(found["ips"][0]["active"])  # sorted newest first, so the top one is within the window of the newest line
        self.assertEqual(monitor.ip_summary(self.db, window=-1)["active_count"], 0)

    def test_perf_reports_after_a_busy_tick(self):
        self.assertEqual(self.monitor.perf()["ticks"], 0)
        self.log.write_bytes(("".join(self.lines[:200])).encode())
        self.monitor._tick(self.db)
        perf = self.monitor.perf()
        self.assertEqual(perf["ticks"], 1)
        self.assertGreater(perf["capacity_lines_per_s"], 0)

    def test_watch_switches_file_and_starts_from_zero(self):
        self.log.write_bytes(("".join(self.lines[:200])).encode())
        self.monitor._tick(self.db)
        other = self.log.parent / "other.log"
        self.monitor.watch(other)
        self.assertTrue(other.exists())
        self.assertEqual((self.monitor.log_path, self.monitor.offset, len(self.monitor.state)), (other, 0, 0))
        self.assertEqual(monitor.summary(self.db, 0.5)["total"], 0)

    def test_feed_streams_processed_lines_with_severity(self):
        self.log.write_bytes(("".join(self.lines[:200])).encode())
        self.monitor._tick(self.db)
        first = self.monitor.feed_since(-1)
        self.assertEqual(len(first["lines"]), 100)  # a new reader gets the latest 100
        self.assertEqual(sum(first["counts"].values()), 200)  # the counters cover every processed line
        self.assertTrue(all(l["sev"] == monitor.SEVERITY.get(l["level"], monitor.UNKNOWN_SEVERITY) for l in first["lines"]))
        self.assertEqual((monitor.SEVERITY["FATAL"], monitor.SEVERITY["ERROR"], monitor.SEVERITY["WARN"], monitor.SEVERITY["INFO"]), (2, 3, 4, 6))
        self.assertEqual(self.monitor.feed_since(first["latest"])["lines"], [])  # nothing new yet
        self.log.write_bytes(("".join(self.lines[:201])).encode())
        self.monitor._tick(self.db)
        self.assertEqual([l["seq"] for l in self.monitor.feed_since(first["latest"])["lines"]], [201])
        self.monitor.reset()
        again = self.monitor.feed_since(0)
        self.assertEqual((again["lines"], again["counts"], again["generation"]), ([], {}, first["generation"] + 1))


if __name__ == "__main__":
    unittest.main()
