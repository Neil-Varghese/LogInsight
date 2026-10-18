import unittest

try:
    from annotation_blinding import build_id_memorization_probe, score_id_memorization_probe, validate_blinded_packet
    from annotation_packet import build_annotation_packet
except ImportError:  # pragma: no cover
    from research.scripts.annotation_blinding import build_id_memorization_probe, score_id_memorization_probe, validate_blinded_packet
    from research.scripts.annotation_packet import build_annotation_packet


def _source(unit_id, event_id):
    return {
        "dataset": "HDFS",
        "unit_id": unit_id,
        "events": [
            {
                "EventId": event_id,
                "LineId": 100,
                "EventTemplate": "Failure <*>",
                "ParameterList": ["blk_-private"],
                "RawLine": "Failure blk_-private from 10.1.2.3",
            },
            {
                "EventId": "other-event",
                "LineId": 101,
                "EventTemplate": "Cleanup",
                "ParameterList": [],
                "RawLine": "Cleanup complete",
            },
        ],
    }


def _packets():
    return [build_annotation_packet(_source(f"blk_-unit-{index}", f"event-{index}")) for index in range(3)]


class AnnotationBlindingTests(unittest.TestCase):
    def test_clean_packet_passes_blinding_validation(self):
        packet = validate_blinded_packet(_packets()[0])
        self.assertEqual(packet["packet_type"], "evidence_relevance")

    def test_blinding_rejects_condition_or_detector_fields(self):
        packet = _packets()[0]
        packet["condition"] = "unblinded"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_blinded_packet(packet)

        packet = _packets()[0]
        packet["detector_output"] = {"probability": 0.9}
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_blinded_packet(packet)

    def test_probe_uses_fresh_opaque_equal_length_views(self):
        probe = build_id_memorization_probe(_packets(), pair_count=6, seed=2026)
        self.assertEqual(probe["manifest"]["pair_count"], 6)
        self.assertEqual(probe["manifest"]["same_pairs"], 3)
        self.assertEqual(probe["manifest"]["different_pairs"], 3)
        for item in probe["items"]:
            self.assertIsNone(item["annotator_response"])
            self.assertEqual(len(item["left"]["events"]), len(item["right"]["events"]))
            self.assertNotIn("UNIT-", str(item))
            self.assertNotIn("blk_", str(item))
            self.assertNotIn("Failure", str(item))
            self.assertNotEqual(item["left"]["events"][0]["event_id"], item["right"]["events"][0]["event_id"])

    def test_probe_is_deterministic_for_same_seed(self):
        first = build_id_memorization_probe(_packets(), pair_count=4, seed=2026)
        second = build_id_memorization_probe(_packets(), pair_count=4, seed=2026)
        self.assertEqual(first, second)

    def test_probe_scores_responses_against_hidden_key(self):
        probe = build_id_memorization_probe(_packets(), pair_count=4, seed=2026)
        responses = [
            {"probe_id": item["probe_id"], "annotator_response": item["expected_relation"]}
            for item in probe["answer_key"]
        ]
        report = score_id_memorization_probe(responses, probe["answer_key"])
        self.assertEqual(report["answered"], 4)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertTrue(report["above_chance_indicator"])

    def test_probe_rejects_capped_packets(self):
        packet = build_annotation_packet(_source("blk_-unit", "event"), max_context_tokens=1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_id_memorization_probe([packet, _packets()[1]], pair_count=1)


if __name__ == "__main__":
    unittest.main()
