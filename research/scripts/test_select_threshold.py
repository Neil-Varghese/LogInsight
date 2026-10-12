import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from select_threshold import (
    select_validation_threshold,
    threshold_candidates,
    select_and_apply_threshold,
)


class SelectThresholdTests(unittest.TestCase):
    def test_candidates_cover_distinct_prediction_sets(self):
        candidates = threshold_candidates(np.asarray([0.1, 0.4, 0.8]))
        self.assertIn(0.0, candidates)
        self.assertIn(0.1, candidates)
        self.assertIn(0.25, candidates)
        self.assertIn(1.0, candidates)

    def test_threshold_is_selected_from_validation_f1(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        probabilities = np.asarray([0.1, 0.2, 0.4, 0.45, 0.6, 0.7])
        threshold, metrics, candidates = select_validation_threshold(labels, probabilities)
        self.assertEqual(threshold, 0.45)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertGreater(len(candidates), len(probabilities))

    def test_selection_rejects_single_class_validation(self):
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            select_validation_threshold(np.asarray([0, 0]), np.asarray([0.1, 0.2]))

    def test_files_apply_frozen_validation_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = {}
            records_by_split = {
                "validation": [
                    {"unit_id": "v0", "label_id": 0, "probability": 0.1},
                    {"unit_id": "v1", "label_id": 0, "probability": 0.2},
                    {"unit_id": "v2", "label_id": 1, "probability": 0.7},
                    {"unit_id": "v3", "label_id": 1, "probability": 0.8},
                ],
                "train": [
                    {"unit_id": "t0", "label_id": 0, "probability": 0.3},
                    {"unit_id": "t1", "label_id": 1, "probability": 0.9},
                ],
                "test": [
                    {"unit_id": "x0", "label_id": 0, "probability": 0.4},
                    {"unit_id": "x1", "label_id": 1, "probability": 0.6},
                ],
            }
            for split, records in records_by_split.items():
                path = directory / f"predictions_{split}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record) + "\n")
                paths[split] = path
            manifest = select_and_apply_threshold(
                validation_path=paths["validation"],
                split_paths=paths,
                output_dir=directory / "thresholded",
                show_progress=False,
            )
            self.assertEqual(manifest["fit_split"], "validation")
            self.assertFalse(manifest["test_labels_used_for_selection"])
            self.assertAlmostEqual(manifest["selected_threshold"], 0.45)
            with gzip.open(directory / "thresholded" / "thresholded_predictions_test.jsonl.gz", "rt", encoding="utf-8") as handle:
                test_records = [json.loads(line) for line in handle]
            self.assertEqual([record["prediction"] for record in test_records], [0, 1])
            self.assertTrue(all(record["threshold_source"] == "validation" for record in test_records))


if __name__ == "__main__":
    unittest.main()
