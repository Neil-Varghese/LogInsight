"""Tests for HDFS BlockId grouping."""

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from build_hdfs_blocks import build_hdfs_blocks, extract_block_id


class HDFSBlockBuilderTest(unittest.TestCase):
    def test_extract_block_id(self):
        self.assertEqual(extract_block_id("Receiving block blk_-123 src: /10.0.0.1"), "blk_-123")
        self.assertIsNone(extract_block_id("no block here"))

    def test_groups_events_and_preserves_line_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structured = root / "structured.csv"
            labels = root / "labels.csv"
            output = root / "blocks.jsonl.gz"

            with structured.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["LineId", "Content", "EventId"])
                writer.writeheader()
                writer.writerows([
                    {"LineId": "2", "Content": "block blk_-1 second", "EventId": "E2"},
                    {"LineId": "1", "Content": "block blk_-1 first", "EventId": "E1"},
                    {"LineId": "3", "Content": "block blk_2 only", "EventId": "E3"},
                ])
            with labels.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["BlockId", "Label"])
                writer.writeheader()
                writer.writerows([
                    {"BlockId": "blk_-1", "Label": "Normal"},
                    {"BlockId": "blk_2", "Label": "Anomaly"},
                ])

            summary = build_hdfs_blocks(structured, labels, output, show_progress=False)
            self.assertEqual(summary["block_count"], 2)
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(records[0]["BlockId"], "blk_-1")
            self.assertEqual(records[0]["Sequence"], ["E1", "E2"])
            self.assertEqual(records[1]["LabelId"], 1)


if __name__ == "__main__":
    unittest.main()
