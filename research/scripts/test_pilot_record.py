import json
import tempfile
import unittest
from pathlib import Path

try:
    from pilot_record import audit_jsonl, prompt_messages, reproducibility_fields, validate_reproducibility_record
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.pilot_record import audit_jsonl, prompt_messages, reproducibility_fields, validate_reproducibility_record


class PilotRecordTests(unittest.TestCase):
    def _record(self, raw_response='{"answer_status":"fully_abstained"}'):
        record = {
            "record_id": "r1",
            "status": "complete",
            "step": "47A",
            "condition": "blinded",
            "repetition": 1,
            "model": {"model_name": "test"},
        }
        record.update(reproducibility_fields(
            user_prompt='{"anomalous_block_evidence": {}}',
            raw_response=raw_response,
            decoding={"temperature": 0.2, "top_p": 0.95, "max_output_tokens": 512, "do_sample": False},
        ))
        return record

    def test_complete_record_contains_reconstructible_prompt_and_response_hashes(self):
        record = self._record()
        validate_reproducibility_record(record)
        self.assertEqual(record["prompt_messages"], prompt_messages(record["exact_prompt"]))
        self.assertIsNotNone(record["raw_response_sha256"])

    def test_tampering_with_prompt_or_response_is_rejected(self):
        record = self._record()
        record["exact_prompt"] += " tampered"
        with self.assertRaises(ValueError):
            validate_reproducibility_record(record)
        record = self._record()
        record["raw_response"] = "changed"
        with self.assertRaises(ValueError):
            validate_reproducibility_record(record)

    def test_failed_request_can_be_audited_without_a_response(self):
        record = self._record(raw_response=None)
        record["status"] = "failed"
        validate_reproducibility_record(record)

    def test_jsonl_audit_reports_invalid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            valid = self._record()
            invalid = dict(valid)
            invalid["prompt_sha256"] = "bad"
            path.write_text(json.dumps(valid) + "\n" + json.dumps(invalid) + "\n", encoding="utf-8")
            self.assertEqual(audit_jsonl(path), {"record_count": 2, "invalid_record_count": 1})


if __name__ == "__main__":
    unittest.main()
