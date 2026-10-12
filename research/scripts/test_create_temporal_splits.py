import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from create_temporal_splits import (  # noqa: E402
    assign_temporal_splits,
    bgl_record_interval,
    hdfs_timestamp,
    make_metadata,
)


class TemporalSplitTests(unittest.TestCase):
    def test_hdfs_timestamp_is_chronological(self):
        first = hdfs_timestamp("081109 203518 143 INFO dfs.Component: payload")
        second = hdfs_timestamp("081110 000001 143 INFO dfs.Component: payload")
        self.assertLess(first, second)

    def test_bgl_interval_is_positive(self):
        self.assertEqual(bgl_record_interval({"WindowStartEpoch": 10, "WindowEndEpochExclusive": 70}), (10, 70))

    def test_temporal_assignment_purges_crossing_intervals(self):
        metadata = [
            {"UnitId": f"u{i}", "Label": "Normal", "LabelId": 0, "FirstTimestamp": i, "EndTimestampExclusive": i + 1}
            for i in range(10)
        ]
        metadata[7]["EndTimestampExclusive"] = 9
        assignments, cutoffs = assign_temporal_splits(metadata)
        self.assertEqual(cutoffs, (8, 9))
        self.assertEqual(assignments["u0"], "train")
        self.assertEqual(assignments["u7"], "excluded_boundary_crossing")
        self.assertEqual(assignments["u8"], "validation")
        self.assertEqual(assignments["u9"], "test")

    def test_make_metadata_uses_prediction_unit(self):
        record = {
            "WindowId": "bgl_10",
            "WindowStartEpoch": 10,
            "WindowEndEpochExclusive": 70,
            "Label": "Anomaly",
            "LabelId": 1,
        }
        metadata = make_metadata("BGL", record)
        self.assertEqual(metadata["UnitId"], "bgl_10")
        self.assertEqual(metadata["FirstTimestamp"], 10)
        self.assertEqual(metadata["EndTimestampExclusive"], 70)


if __name__ == "__main__":
    unittest.main()
