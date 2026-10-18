import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    from verify_colab_transfer import compare_predictions, load_prediction_rows, sha256_file
except ImportError:  # pragma: no cover
    from research.scripts.verify_colab_transfer import compare_predictions, load_prediction_rows, sha256_file


class VerifyColabTransferTests(unittest.TestCase):
    def test_hash_and_prediction_loading_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"unit_id": "u1", "probability": 0.25, "architecture": "a", "label_id": 0}) + "\n")
            self.assertEqual(len(sha256_file(path)), 64)
            self.assertEqual(load_prediction_rows(path)["u1"]["probability"], 0.25)

    def test_comparison_requires_registered_tolerance(self):
        stored = {"u1": {"unit_id": "u1", "probability": 0.25, "architecture": "a", "label_id": 0}}
        metadata = [{"unit_id": "u1", "architecture": "a", "label_id": 0}]
        report = compare_predictions(stored, [0.2500005], metadata, tolerance=1e-6)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["outside_tolerance_count"], 0)

    def test_comparison_fails_for_probability_or_metadata_drift(self):
        stored = {"u1": {"unit_id": "u1", "probability": 0.25, "architecture": "a", "label_id": 0}}
        metadata = [{"unit_id": "u1", "architecture": "different", "label_id": 0}]
        report = compare_predictions(stored, [0.25001], metadata, tolerance=1e-6)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["outside_tolerance_count"], 1)
        self.assertEqual(report["metadata_mismatch_count"], 1)

    def test_comparison_rejects_unit_mismatch(self):
        stored = {"u1": {"unit_id": "u1", "probability": 0.25}}
        with self.assertRaises(ValueError):
            compare_predictions(stored, [0.25], [{"unit_id": "u2", "label_id": 0}], tolerance=1e-6)


if __name__ == "__main__":
    unittest.main()
