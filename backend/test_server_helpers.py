"""Offline test for server.read_block_lines. Run: python -m unittest discover -s backend"""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import read_block_lines


class TestReadBlockLines(unittest.TestCase):
    def test_exact_block_only_and_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "x.log"
            log.write_text("a blk_1 x\nb blk_12 y\nc blk_-1 z\nd blk_1\ne blk_1 q\n", encoding="utf-8")
            self.assertEqual(read_block_lines(log, "blk_1"), ["a blk_1 x", "d blk_1", "e blk_1 q"])
            self.assertEqual(read_block_lines(log, "blk_1", limit=2), ["a blk_1 x", "d blk_1"])
            self.assertEqual(read_block_lines(log, "blk_-1"), ["c blk_-1 z"])


if __name__ == "__main__":
    unittest.main()
