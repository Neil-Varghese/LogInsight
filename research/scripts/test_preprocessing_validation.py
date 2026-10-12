import unittest

from preprocessing_validation import validate_bgl_windows, validate_hdfs_records


def hdfs_record(line_ids=(1, 2)):
    evidence = [
        {
            "Position": position,
            "LineId": line_id,
            "EventId": event_id,
            "EventTemplate": f"template {event_id}",
            "ParameterList": [],
            "RawLine": f"raw {event_id}",
        }
        for position, (line_id, event_id) in enumerate(zip(line_ids, ("E1", "E2")))
    ]
    return {
        "BlockId": "blk_1",
        "Sequence": [event["EventId"] for event in evidence],
        "Evidence": evidence,
        "Label": "Anomaly",
        "LabelId": 1,
        "FirstLineId": line_ids[0],
        "EventCount": len(evidence),
    }


class PreprocessingValidationTest(unittest.TestCase):
    def test_valid_hdfs_record(self):
        result = validate_hdfs_records([hdfs_record()])
        self.assertEqual(result, {"block_count": 1, "event_count": 2, "labelled_block_count": 1})

    def test_hdfs_rejects_out_of_order_evidence(self):
        with self.assertRaisesRegex(ValueError, "strictly chronological"):
            validate_hdfs_records([hdfs_record((2, 1))])

    def test_hdfs_rejects_sequence_length_mismatch(self):
        record = hdfs_record()
        record["EventCount"] = 1
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            validate_hdfs_records([record])

    def test_valid_bgl_windows(self):
        records = [
            {
                "WindowId": "bgl_0", "WindowStartEpoch": 0, "WindowEndEpochExclusive": 60,
                "LineStart": 1, "LineEnd": 2, "EventCount": 2, "Label": "Normal", "LabelId": 0,
            },
            {
                "WindowId": "bgl_60", "WindowStartEpoch": 60, "WindowEndEpochExclusive": 120,
                "LineStart": 3, "LineEnd": 3, "EventCount": 1, "Label": "Anomaly", "LabelId": 1,
            },
        ]
        result = validate_bgl_windows(records)
        self.assertEqual(result["window_count"], 2)
        self.assertEqual(result["line_count"], 3)

    def test_bgl_rejects_overlap(self):
        records = [
            {
                "WindowId": "bgl_0", "WindowStartEpoch": 0, "WindowEndEpochExclusive": 60,
                "LineStart": 1, "LineEnd": 2, "EventCount": 2, "Label": "Normal", "LabelId": 0,
            },
            {
                "WindowId": "bgl_60", "WindowStartEpoch": 60, "WindowEndEpochExclusive": 120,
                "LineStart": 2, "LineEnd": 3, "EventCount": 2, "Label": "Anomaly", "LabelId": 1,
            },
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_bgl_windows(records)


if __name__ == "__main__":
    unittest.main()
