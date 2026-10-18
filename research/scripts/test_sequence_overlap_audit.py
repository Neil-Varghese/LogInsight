import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    from sequence_overlap_audit import build_audit
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.sequence_overlap_audit import build_audit


def _write(path: Path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _evidence(unit_id, sequence):
    return {"BlockId": unit_id, "Sequence": sequence}


def _prediction(unit_id, label, prediction, probability):
    return {
        "architecture": "test_detector",
        "seed": 13,
        "split": "test",
        "unit_id": unit_id,
        "label_id": label,
        "prediction": prediction,
        "probability": probability,
    }


class SequenceOverlapAuditTests(unittest.TestCase):
    def test_reports_overlap_novel_metrics_and_annotation_unique_sequences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl.gz"
            test = root / "test.jsonl.gz"
            pool = root / "pool.jsonl.gz"
            predictions = root / "predictions.jsonl.gz"
            output = root / "audit.json"
            _write(train, [_evidence("train_1", ["a", "b"]), _evidence("train_2", ["c"])])
            _write(test, [_evidence("test_1", ["a", "b"]), _evidence("test_2", ["d"]), _evidence("test_3", ["d"])])
            _write(pool, [
                {"annotation_item_id": "item_1", "unit_id": "test_1"},
                {"annotation_item_id": "item_2", "unit_id": "test_1"},
                {"annotation_item_id": "item_3", "unit_id": "test_2"},
                {"annotation_item_id": "item_4", "unit_id": "test_3"},
            ])
            _write(predictions, [
                _prediction("test_1", 0, 0, 0.1),
                _prediction("test_2", 1, 1, 0.9),
                _prediction("test_3", 0, 1, 0.8),
            ])
            manifest = build_audit(
                train_evidence=train,
                test_evidence=test,
                annotation_pool=pool,
                prediction_files=[predictions],
                output=output,
            )
            self.assertEqual(manifest["train"]["unique_sequence_count"], 2)
            self.assertEqual(manifest["test"]["shared_unique_sequence_count"], 1)
            self.assertEqual(manifest["test"]["units_with_novel_sequence"], 2)
            self.assertEqual(manifest["annotation_pool"]["unique_sequence_count"], 2)
            self.assertEqual(manifest["annotation_pool"]["annotation_item_count"], 4)
            self.assertEqual(manifest["annotation_pool"]["unique_unit_count"], 3)
            self.assertEqual(manifest["novel_sequence_metrics"][0]["novel_sequence_prediction_count"], 2)
            self.assertEqual(manifest["novel_sequence_metrics"][0]["f1"], 2 / 3)
            self.assertTrue(output.exists())

    def test_rejects_missing_annotation_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl.gz"
            test = root / "test.jsonl.gz"
            pool = root / "pool.jsonl.gz"
            _write(train, [_evidence("train_1", ["a"])])
            _write(test, [_evidence("test_1", ["b"])])
            _write(pool, [{"unit_id": "missing"}])
            with self.assertRaises(ValueError):
                build_audit(
                    train_evidence=train,
                    test_evidence=test,
                    annotation_pool=pool,
                    prediction_files=[],
                    output=root / "audit.json",
                )


if __name__ == "__main__":
    unittest.main()
