import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    from shortcut_baseline_audit import build_audit
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.shortcut_baseline_audit import build_audit


def _write(path: Path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _evidence(unit_id, sequence, label):
    return {"BlockId": unit_id, "Sequence": sequence, "LabelId": label}


def _prediction(unit_id, label, probability):
    return {
        "architecture": "test_detector",
        "seed": 13,
        "split": "test",
        "unit_id": unit_id,
        "label_id": label,
        "prediction": int(probability >= 0.5),
        "probability": probability,
        "threshold": 0.5,
    }


class ShortcutBaselineAuditTests(unittest.TestCase):
    def test_fits_training_only_baselines_and_compares_detector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl.gz"
            validation = root / "validation.jsonl.gz"
            test = root / "test.jsonl.gz"
            predictions = root / "predictions.jsonl.gz"
            output = root / "report.json"
            _write(train, [
                _evidence("train_1", ["normal"], 0),
                _evidence("train_2", ["failure", "failure"], 1),
                _evidence("train_3", ["normal", "normal"], 0),
                _evidence("train_4", ["failure"], 1),
            ])
            _write(validation, [_evidence("validation_1", ["normal"], 0), _evidence("validation_2", ["failure"], 1)])
            _write(test, [_evidence("test_1", ["normal"], 0), _evidence("test_2", ["failure"], 1)])
            _write(predictions, [_prediction("test_1", 0, 0.1), _prediction("test_2", 1, 0.9)])
            report = build_audit(
                train_evidence=train,
                validation_evidence=validation,
                test_evidence=test,
                prediction_files=[predictions],
                output=output,
            )
            names = [baseline["name"] for baseline in report["baselines"]]
            self.assertEqual(names, ["event_count_logistic_regression", "length_only_logistic_regression"])
            self.assertEqual(report["baselines"][0]["fit_split"], "train")
            self.assertEqual(report["detectors"][0]["metrics"]["test"]["f1"], 1.0)
            self.assertTrue(output.exists())

    def test_rejects_duplicate_units_in_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl.gz"
            validation = root / "validation.jsonl.gz"
            test = root / "test.jsonl.gz"
            _write(train, [_evidence("same", ["a"], 0), _evidence("same", ["b"], 1)])
            _write(validation, [_evidence("validation", ["a"], 0)])
            _write(test, [_evidence("test", ["b"], 1)])
            with self.assertRaises(ValueError):
                build_audit(
                    train_evidence=train,
                    validation_evidence=validation,
                    test_evidence=test,
                    prediction_files=[],
                    output=root / "report.json",
                )


if __name__ == "__main__":
    unittest.main()
