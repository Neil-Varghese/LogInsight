import gzip
import json
import tempfile
import unittest
from pathlib import Path

from validate_integrated_gradients import validate_attribution_file


def _record(*, completeness_error=0.001):
    return {
        "schema_version": "1.0",
        "method": "integrated_gradients",
        "target": "anomaly_probability",
        "unit_id": "blk_1",
        "label_id": 1,
        "probability": 0.8,
        "baseline_probability": 0.2,
        "output_delta": 0.6,
        "attribution_sum": 0.6 - completeness_error,
        "completeness_error": completeness_error,
        "sequence_length": 2,
        "padded_length": 4,
        "events": [
            {"input_position": 0, "original_sequence_index": None, "event_id": 0, "event_token": "__PAD__", "is_padding": True, "attribution": 0.0, "absolute_attribution": 0.0, "rank": None},
            {"input_position": 1, "original_sequence_index": None, "event_id": 0, "event_token": "__PAD__", "is_padding": True, "attribution": 0.0, "absolute_attribution": 0.0, "rank": None},
            {"input_position": 2, "original_sequence_index": 0, "event_id": 3, "event_token": "a", "is_padding": False, "attribution": 0.1, "absolute_attribution": 0.1, "rank": 2},
            {"input_position": 3, "original_sequence_index": 1, "event_id": 4, "event_token": "b", "is_padding": False, "attribution": 0.5, "absolute_attribution": 0.5, "rank": 1},
        ],
    }


class ValidateIntegratedGradientsTests(unittest.TestCase):
    def _write(self, directory, records, manifest=None):
        path = Path(directory) / "attributions.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        if manifest is not None:
            path.with_name("attributions.manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
        return path

    def test_valid_record_passes_alignment_and_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                [_record()],
                {"count": 1, "padding": "pre", "baseline": {"event_id": 0}, "max_completeness_error": 0.001},
            )
            summary = validate_attribution_file(path)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["records"], 1)

    def test_probability_arithmetic_and_alignment_failures_are_reported(self):
        record = _record()
        record["output_delta"] = 0.5
        record["events"][3]["input_position"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, [record])
            summary = validate_attribution_file(path)
        self.assertFalse(summary["valid"])
        joined = " ".join(error for item in summary["errors"] for error in item["errors"])
        self.assertIn("output_delta", joined)
        self.assertIn("input positions", joined)

    def test_completeness_threshold_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, [_record(completeness_error=0.2)])
            summary = validate_attribution_file(path, max_completeness_error=0.05)
        self.assertFalse(summary["valid"])
        self.assertGreater(summary["max_completeness_error"], summary["max_allowed_completeness_error"])


if __name__ == "__main__":
    unittest.main()
