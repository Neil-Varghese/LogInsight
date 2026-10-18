import unittest

try:
    from annotation_packet import build_annotation_packet, estimate_context_tokens, validate_annotation_export
except ImportError:  # pragma: no cover
    from research.scripts.annotation_packet import build_annotation_packet, estimate_context_tokens, validate_annotation_export


def _record():
    return {
        "dataset": "HDFS",
        "unit_id": "blk_-123456789",
        "detector_id": "loginsight_lstm",
        "condition": "unblinded",
        "events": [
            {
                "EventId": "e3df2680",
                "LineId": 9512780,
                "EventTemplate": "Received block <*> from <*>",
                "ParameterList": ["blk_-123456789", "/10.1.2.3:50010"],
                "RawLine": "081111 082100 24206 INFO dfs: Received block blk_-123456789 from /10.1.2.3:50010",
            },
            {
                "EventId": "d63ef163",
                "LineId": 10986891,
                "EventTemplate": "BLOCK* NameSystem.delete: <*>",
                "ParameterList": ["blk_-123456789"],
                "RawLine": "081111 110430 26 INFO dfs: BLOCK* NameSystem.delete: blk_-123456789",
            },
        ],
    }


class AnnotationPacketTests(unittest.TestCase):
    def test_packet_anonymizes_source_identifiers_and_provenance(self):
        packet = build_annotation_packet(_record())
        self.assertEqual(packet["packet_type"], "evidence_relevance")
        self.assertEqual(packet["context"]["context_status"], "complete")
        self.assertTrue(packet["context"]["annotation_eligible"])
        self.assertEqual([event["event_id"] for event in packet["events"]], ["E0001", "E0002"])
        self.assertEqual([event["line_id"] for event in packet["events"]], ["L0001", "L0002"])
        serialized = str(packet)
        self.assertNotIn("blk_-123456789", serialized)
        self.assertNotIn("10.1.2.3", serialized)
        self.assertNotIn("loginsight_lstm", serialized)
        self.assertNotIn("unblinded", serialized)
        self.assertIn("<BLOCK_ID>", serialized)
        self.assertIn("<HOST>", serialized)

    def test_nested_evidence_and_normal_comparison_are_supported(self):
        record = _record()
        record.pop("events")
        record["anomalous_block_evidence"] = {"events": _record()["events"]}
        record["normal_comparison_events"] = [_record()["events"][0]]
        packet = build_annotation_packet(record)
        self.assertEqual(len(packet["events"]), 2)
        self.assertIn("normal_comparison", packet)
        self.assertEqual(packet["normal_comparison"]["events"][0]["event_id"], "NE0001")

    def test_context_cap_records_omissions_and_marks_packet_ineligible(self):
        packet = build_annotation_packet(_record(), max_context_tokens=25)
        self.assertEqual(packet["context"]["context_status"], "capped")
        self.assertFalse(packet["context"]["annotation_eligible"])
        self.assertGreater(len(packet["context"]["omitted_event_positions"]), 0)
        self.assertEqual(packet["context"]["policy"], "deterministic_cap_record_omissions_never_silent_truncation")

    def test_token_estimate_is_deterministic(self):
        value = {"events": [{"raw_line": "one two three"}]}
        self.assertEqual(estimate_context_tokens(value), estimate_context_tokens(value))

    def test_completed_export_requires_anonymized_fields_and_labels(self):
        export = {
            "schema_version": "1.0",
            "guideline_version": "1.0",
            "packet_id": "PACKET-abc",
            "unit_alias": "UNIT-abc",
            "events": [
                {
                    "event_position": 0,
                    "event_id": "E0001",
                    "line_id": "L0001",
                    "relevance": "relevant",
                    "role": "failure_signal",
                    "reason": "The line records the failure.",
                }
            ],
            "contrastive_assessment": {
                "meaningful_difference": "uncertain",
                "difference_positions": [],
                "reason": "Not enough comparison evidence.",
            },
        }
        self.assertEqual(validate_annotation_export(export)["unit_alias"], "UNIT-abc")

    def test_export_rejects_forbidden_provenance_and_duplicate_positions(self):
        export = {
            "schema_version": "1.0",
            "guideline_version": "1.0",
            "packet_id": "PACKET-abc",
            "unit_alias": "UNIT-abc",
            "detector_id": "should-not-be-here",
            "events": [
                {"event_position": 0, "event_id": "E0001", "line_id": "L0001", "relevance": "uncertain", "role": None, "reason": None},
                {"event_position": 0, "event_id": "E0002", "line_id": "L0002", "relevance": "uncertain", "role": None, "reason": None},
            ],
            "contrastive_assessment": {"meaningful_difference": "no", "difference_positions": [], "reason": ""},
        }
        with self.assertRaisesRegex(ValueError, "duplicate event positions"):
            validate_annotation_export(export)

    def test_export_rejects_unanonymized_event_alias(self):
        export = {
            "schema_version": "1.0",
            "guideline_version": "1.0",
            "packet_id": "PACKET-abc",
            "unit_alias": "UNIT-abc",
            "events": [
                {"event_position": 0, "event_id": "e3df2680", "line_id": "L0001", "relevance": "uncertain", "role": None, "reason": None},
            ],
            "contrastive_assessment": {"meaningful_difference": "no", "difference_positions": [], "reason": ""},
        }
        with self.assertRaisesRegex(ValueError, "anonymized event alias"):
            validate_annotation_export(export)


if __name__ == "__main__":
    unittest.main()
