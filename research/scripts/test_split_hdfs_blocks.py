import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from split_hdfs_blocks import allocate_counts, assign_splits


class SplitHDFSBlocksTest(unittest.TestCase):
    def test_allocate_counts_preserves_total(self):
        counts = allocate_counts(101, {"train": 0.8, "validation": 0.1, "test": 0.1})
        self.assertEqual(sum(counts.values()), 101)
        self.assertEqual(counts, {"train": 81, "validation": 10, "test": 10})

    def test_assignment_is_deterministic_disjoint_and_label_stratified(self):
        metadata = [
            {"BlockId": f"blk_{label}_{index}", "LabelId": label, "Label": "Anomaly" if label else "Normal"}
            for label in (0, 1)
            for index in range(10)
        ]
        first = assign_splits(metadata, seed=2026)
        second = assign_splits(metadata, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {record["BlockId"] for record in metadata})
        for label in (0, 1):
            ids = [record["BlockId"] for record in metadata if record["LabelId"] == label]
            self.assertEqual(sum(first[block_id] == "train" for block_id in ids), 8)
            self.assertEqual(sum(first[block_id] == "validation" for block_id in ids), 1)
            self.assertEqual(sum(first[block_id] == "test" for block_id in ids), 1)

    def test_duplicate_blocks_are_rejected(self):
        metadata = [
            {"BlockId": "blk_1", "LabelId": 0},
            {"BlockId": "blk_1", "LabelId": 0},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            assign_splits(metadata)


if __name__ == "__main__":
    unittest.main()
