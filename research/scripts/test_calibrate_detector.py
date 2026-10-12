import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from calibrate_detector import (
    calibrate_probabilities,
    calibration_metrics,
    fit_temperature,
    calibrate_files,
)


class CalibrateDetectorTests(unittest.TestCase):
    def test_temperature_scaling_preserves_order(self):
        probabilities = np.asarray([0.1, 0.4, 0.8, 0.95])
        calibrated = calibrate_probabilities(probabilities, temperature=2.0)
        self.assertTrue(np.all(np.diff(calibrated) > 0))
        self.assertGreater(calibrated[0], probabilities[0])
        self.assertLess(calibrated[-1], probabilities[-1])

    def test_temperature_fit_is_validation_only_and_positive(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        probabilities = np.asarray([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        temperature = fit_temperature(labels, probabilities)
        self.assertGreater(temperature, 0)
        before = calibration_metrics(labels, probabilities)
        after = calibration_metrics(labels, calibrate_probabilities(probabilities, temperature))
        self.assertLessEqual(after["nll"], before["nll"] + 1e-8)

    def test_fit_rejects_single_class_validation(self):
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            fit_temperature(np.asarray([0, 0]), np.asarray([0.2, 0.3]))

    def test_files_write_manifest_and_calibrated_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = {}
            records = [
                {"unit_id": "a", "label_id": 0, "probability": 0.1, "prediction": 0, "threshold": 0.5},
                {"unit_id": "b", "label_id": 0, "probability": 0.2, "prediction": 0, "threshold": 0.5},
                {"unit_id": "c", "label_id": 1, "probability": 0.8, "prediction": 1, "threshold": 0.5},
                {"unit_id": "d", "label_id": 1, "probability": 0.9, "prediction": 1, "threshold": 0.5},
            ]
            for split in ("train", "validation", "test"):
                path = directory / f"predictions_{split}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record) + "\n")
                paths[split] = path
            manifest = calibrate_files(
                validation_path=paths["validation"],
                split_paths=paths,
                output_dir=directory / "calibrated",
                show_progress=False,
            )
            self.assertEqual(manifest["fit_split"], "validation")
            output = directory / "calibrated" / "calibrated_predictions_test.jsonl.gz"
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                calibrated_record = json.loads(next(handle))
            self.assertIn("raw_probability", calibrated_record)
            self.assertIn("calibrated_probability", calibrated_record)
            self.assertNotIn("prediction", calibrated_record)
            self.assertTrue((directory / "calibrated" / "calibration_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
