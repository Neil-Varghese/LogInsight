import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_hdfs_block_evidence import build_hdfs_block_evidence


class HDFSBlockEvidenceTest(unittest.TestCase):
    def test_preserves_raw_and_parsed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            structured = root / "structured.csv"
            labels = root / "labels.csv"
            raw = root / "HDFS.log"
            output = root / "evidence.jsonl.gz"
            manifest = root / "manifest.json"
            fields = [
                "LineId", "Date", "Time", "Pid", "Level", "Component",
                "Content", "EventId", "EventTemplate", "ParameterList",
            ]
            rows = [
                ["2", "081109", "203518", "35", "INFO", "dfs.FSNamesystem",
                 "BLOCK* blk_1", "E2", "BLOCK* <*>", "['blk_1']"],
                ["1", "081109", "203518", "143", "INFO", "dfs.DataNode",
                 "Receiving block blk_1", "E1", "Receiving block <*>", "['blk_1']"],
            ]
            with structured.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            labels.write_text("BlockId,Label\nblk_1,Anomaly\n", encoding="utf-8")
            raw.write_text(
                "081109 203518 35 INFO dfs.FSNamesystem: BLOCK* blk_1\n"
                "081109 203518 143 INFO dfs.DataNode: Receiving block blk_1\n",
                encoding="utf-8",
            )

            summary = build_hdfs_block_evidence(
                structured, labels, output, raw, manifest, show_progress=False
            )
            self.assertEqual(summary["block_count"], 1)
            self.assertEqual(summary["evidence_event_count"], 2)
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                record = json.loads(next(handle))
            self.assertEqual(record["Sequence"], ["E1", "E2"])
            self.assertEqual([event["Position"] for event in record["Evidence"]], [0, 1])
            self.assertEqual(record["Evidence"][0]["RawLine"],
                             "081109 203518 143 INFO dfs.DataNode: Receiving block blk_1")
            self.assertEqual(record["Evidence"][0]["ParameterList"], ["blk_1"])
            self.assertEqual(record["Evidence"][1]["LineId"], 2)
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["schema_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
