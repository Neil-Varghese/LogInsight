import json
import unittest
from pathlib import Path

try:
    from explanation_schema import validate_explanation_record
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.explanation_schema import validate_explanation_record


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "experiments" / "llm_explanation_schema.json"


def _answered():
    return {
        "schema_version": "1.0",
        "answer_status": "answered",
        "summary": "The block contains an anomalous event pattern.",
        "claims": [
            {
                "claim_id": "claim_1",
                "claim_type": "anomalous_pattern",
                "text": "The failure event is the strongest cited evidence.",
                "cited_event_ids": ["E1"],
                "cited_line_ids": [17],
                "confidence": 0.8,
                "abstained": False,
                "predicted_effect": "increase_anomaly_score",
            }
        ],
        "cited_event_ids": ["E1"],
        "cited_line_ids": [17],
        "overall_confidence": 0.8,
        "abstention": {"status": "none", "reason": None},
    }


class ExplanationSchemaTests(unittest.TestCase):
    def test_schema_file_is_versioned_and_declares_required_fields(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["$id"], "https://loginsight.local/schemas/llm-explanation.json")
        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0")
        self.assertIn("claims", schema["required"])
        self.assertIn("cited_event_ids", schema["required"])
        self.assertIn("cited_line_ids", schema["required"])
        self.assertIn("overall_confidence", schema["required"])
        self.assertIn("abstention", schema["required"])

    def test_valid_answered_record(self):
        self.assertEqual(validate_explanation_record(_answered())["answer_status"], "answered")

    def test_valid_full_abstention_record(self):
        record = _answered()
        record.update(
            {
                "answer_status": "fully_abstained",
                "summary": None,
                "claims": [],
                "cited_event_ids": [],
                "cited_line_ids": [],
                "overall_confidence": None,
                "abstention": {"status": "full", "reason": "The evidence is insufficient."},
            }
        )
        self.assertEqual(validate_explanation_record(record)["abstention"]["status"], "full")

    def test_rejects_unsupported_fields_and_uncited_claims(self):
        record = _answered()
        record["unexpected"] = True
        with self.assertRaises(ValueError):
            validate_explanation_record(record)
        record = _answered()
        record["claims"][0]["cited_event_ids"] = []
        record["claims"][0]["cited_line_ids"] = []
        record["cited_event_ids"] = []
        record["cited_line_ids"] = []
        with self.assertRaises(ValueError):
            validate_explanation_record(record)


if __name__ == "__main__":
    unittest.main()
