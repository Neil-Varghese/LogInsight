"""Offline test for server.read_block_lines. Run: python -m unittest discover -s backend"""

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server
from server import read_block_lines


class TestReadBlockLines(unittest.TestCase):
    def test_exact_block_only_and_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "x.log"
            log.write_text("a blk_1 x\nb blk_12 y\nc blk_-1 z\nd blk_1\ne blk_1 q\n", encoding="utf-8")
            self.assertEqual(read_block_lines(log, "blk_1"), ["a blk_1 x", "d blk_1", "e blk_1 q"])
            self.assertEqual(read_block_lines(log, "blk_1", limit=2), ["a blk_1 x", "d blk_1"])
            self.assertEqual(read_block_lines(log, "blk_-1"), ["c blk_-1 z"])


class TestLastWatchedFile(unittest.TestCase):
    def test_restart_goes_back_to_the_file_the_results_came_from(self):
        with tempfile.TemporaryDirectory() as folder:
            cluster = Path(folder) / "cluster.log"
            cluster.write_text("x")
            db = Path(folder) / "m.sqlite3"
            saved, server.MONITOR.db_path = server.MONITOR.db_path, db
            try:
                self.assertIsNone(server.last_watched_file())  # fresh database
                connection = sqlite3.connect(db)
                connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute("INSERT INTO meta VALUES ('file', ?)", (str(cluster),))
                connection.commit()
                connection.close()
                self.assertEqual(server.last_watched_file(), cluster)
                cluster.unlink()
                self.assertIsNone(server.last_watched_file())  # file gone: fall back to the default
            finally:
                server.MONITOR.db_path = saved


if __name__ == "__main__":
    unittest.main()
