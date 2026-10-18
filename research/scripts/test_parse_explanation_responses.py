import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parse_explanation_responses import parse_jsonl, parse_raw_response  # noqa: E402


def _claim(*, citations=True):
    return {
        "claim_id": "c1",
        "claim_type": "anomalous_pattern",
        "text": "The block contains a repeated failure pattern.",
        "cited_event_ids": ["e1"] if citations else [],
        "cited_line_ids": [1] if citations else [],
        "confidence": 0.8,
        "abstained": False,
        "predicted_effect": "increase_anomaly_score",
    }


def _answer(*, citations=True):
    claim = _claim(citations=citations)
    return {
        "schema_version": "1.0",
        "answer_status": "answered",
        "summary": "A concise explanation.",
        "claims": [claim],
        "cited_event_ids": ["e1"] if citations else [],
        "cited_line_ids": [1] if citations else [],
        "overall_confidence": 0.8,
        "abstention": {"status": "none", "reason": None},
    }


class ParseExplanationResponsesTest(unittest.TestCase):
    def test_valid_and_fenced_responses_preserve_decoded_payload(self):
        payload = _answer()
        result = parse_raw_response(json.dumps(payload))
        self.assertEqual(result["parsing_status"], "valid")
        self.assertEqual(result["parsed_explanation"], payload)
        fenced = parse_raw_response("```json\n" + json.dumps(payload) + "\n```")
        self.assertEqual(fenced["parsing_status"], "valid")

    def test_invalid_json_is_classified(self):
        result = parse_raw_response('{"schema_version":')
        self.assertEqual(result["parsing_status"], "invalid_json")
        self.assertIsNone(result["parsed_explanation"])

    def test_missing_citations_are_classified_before_generic_schema_errors(self):
        self.assertEqual(parse_raw_response(json.dumps(_answer(citations=False)))["parsing_status"], "missing_citations")
        payload = _answer()
        del payload["cited_event_ids"]
        self.assertEqual(parse_raw_response(json.dumps(payload))["parsing_status"], "missing_citations")

    def test_malformed_claims_are_classified(self):
        payload = _answer()
        payload["claims"] = [{"claim_id": "c1"}]
        self.assertEqual(parse_raw_response(json.dumps(payload))["parsing_status"], "malformed_claims")
        payload["claims"] = "not a list"
        self.assertEqual(parse_raw_response(json.dumps(payload))["parsing_status"], "malformed_claims")

    def test_abstentions_are_valid_but_distinguishable(self):
        payload = {
            "schema_version": "1.0",
            "answer_status": "fully_abstained",
            "summary": None,
            "claims": [],
            "cited_event_ids": [],
            "cited_line_ids": [],
            "overall_confidence": None,
            "abstention": {"status": "full", "reason": "Insufficient evidence."},
        }
        self.assertEqual(parse_raw_response(json.dumps(payload))["parsing_status"], "valid_full_abstention")

    def test_non_object_json_and_upstream_failure_are_classified(self):
        self.assertEqual(parse_raw_response("[]")["parsing_status"], "malformed_payload")
        self.assertEqual(parse_raw_response(None, upstream_status="failed")["parsing_status"], "upstream_failed")
        self.assertEqual(parse_raw_response("", upstream_status="complete")["parsing_status"], "missing_response")

    def test_jsonl_parser_preserves_raw_record_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            input_path = tmp_path / "responses.jsonl"
            output_path = tmp_path / "parsed.jsonl"
            summary_path = tmp_path / "summary.json"
            raw = json.dumps(_answer())
            input_path.write_text(json.dumps({"record_id": "r1", "status": "complete", "raw_response": raw}) + "\n", encoding="utf-8")
            summary = parse_jsonl(input_path, output_path, summary_path)
            record = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(record["raw_response"], raw)
            self.assertEqual(record["record_id"], "r1")
            self.assertEqual(record["parsing_status"], "valid")
            self.assertEqual(summary["counts_by_status"], {"valid": 1})
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8"))["total_records"], 1)
