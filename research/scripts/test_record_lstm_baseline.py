import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from record_lstm_baseline import average_precision, record_baseline


class RecordLstmBaselineTests(unittest.TestCase):
    def test_average_precision_is_step_function_area(self):
        labels = np.asarray([1, 0, 1, 0])
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.1])
        self.assertAlmostEqual(average_precision(labels, probabilities), 5 / 6)

    def test_baseline_manifest_records_all_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            (run_dir / "calibrated").mkdir(parents=True)
            (run_dir / "thresholded").mkdir(parents=True)
            (run_dir / "run_manifest_seed_13.json").write_text(json.dumps({
                "architecture": "loginsight_lstm",
                "seed": 13,
                "checkpoints": {"best": "best.keras"},
            }), encoding="utf-8")
            (run_dir / "calibrated" / "calibration_manifest.json").write_text(json.dumps({
                "method": "temperature_scaling",
                "temperature": 1.2,
            }), encoding="utf-8")
            (run_dir / "thresholded" / "threshold_manifest.json").write_text(json.dumps({
                "selected_threshold": 0.4,
                "fit_split": "validation",
                "test_labels_used_for_selection": False,
            }), encoding="utf-8")
            records = [
                {"unit_id": "a", "label_id": 0, "probability": 0.1},
                {"unit_id": "b", "label_id": 0, "probability": 0.2},
                {"unit_id": "c", "label_id": 1, "probability": 0.8},
                {"unit_id": "d", "label_id": 1, "probability": 0.9},
            ]
            for split in ("train", "validation", "test"):
                with gzip.open(run_dir / f"predictions_{split}.jsonl.gz", "wt", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record) + "\n")
                with gzip.open(run_dir / "calibrated" / f"calibrated_predictions_{split}.jsonl.gz", "wt", encoding="utf-8") as handle:
                    for record in records:
                        output = dict(record)
                        output["calibrated_probability"] = record["probability"]
                        handle.write(json.dumps(output) + "\n")
                with gzip.open(run_dir / "thresholded" / f"thresholded_predictions_{split}.jsonl.gz", "wt", encoding="utf-8") as handle:
                    for record in records:
                        output = dict(record)
                        output["prediction"] = int(record["probability"] >= 0.4)
                        output["threshold"] = 0.4
                        handle.write(json.dumps(output) + "\n")
            output = Path(directory) / "baseline.json"
            result = record_baseline(run_dir, output)
            self.assertEqual(result["step"], 25)
            self.assertEqual(result["dataset"], "HDFS")
            self.assertIn("pr_auc", result["metrics_by_split"]["test"]["thresholded"])
            self.assertIn("ece", result["metrics_by_split"]["test"]["calibrated_probability"])
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
