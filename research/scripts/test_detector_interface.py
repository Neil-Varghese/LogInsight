import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from detector_interface import (
    DetectorContract,
    KerasDetectorAdapter,
    binary_classification_metrics,
    make_detector_batch,
    prediction_records,
    write_prediction_records,
)


class DetectorInterfaceTests(unittest.TestCase):
    def test_contract_and_batch_are_architecture_neutral(self):
        contract = DetectorContract("example_detector")
        batch = make_detector_batch(
            "test",
            np.asarray([[0, 1], [2, 0]], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
            [
                {"unit_id": "u1", "label_id": 0, "sequence_length": 1},
                {"unit_id": "u2", "label_id": 1, "sequence_length": 1},
            ],
        )
        records = prediction_records(contract, batch, np.asarray([0.1, 0.9]), 0.5, seed=13)
        self.assertEqual(records[0]["architecture"], "example_detector")
        self.assertEqual(records[1]["prediction"], 1)
        self.assertEqual(records[1]["split"], "test")
        self.assertEqual(records[0]["schema_version"], "1.0")

    def test_shared_metrics_match_existing_detector_protocol(self):
        metrics = binary_classification_metrics(
            np.asarray([0, 0, 0, 1], dtype=np.int32),
            np.asarray([0.1, 0.2, 0.8, 0.9]),
            threshold=0.5,
        )
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [0, 1]])
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_prediction_writer_round_trips_gzip_records(self):
        contract = DetectorContract("example_detector")
        batch = make_detector_batch(
            "validation",
            np.asarray([[1, 2]], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            [{"unit_id": "u1", "label_id": 1, "sequence_length": 2}],
        )
        records = prediction_records(contract, batch, np.asarray([0.75]), 0.5, seed=13)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl.gz"
            self.assertEqual(write_prediction_records(path, records), 1)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                loaded = json.loads(handle.readline())
        self.assertEqual(loaded["unit_id"], "u1")
        self.assertEqual(loaded["label_id"], 1)

    def test_invalid_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate unit_id"):
            make_detector_batch(
                "train",
                np.asarray([[1], [2]], dtype=np.int32),
                np.asarray([0, 1], dtype=np.int32),
                [{"unit_id": "same"}, {"unit_id": "same"}],
            )

    def test_keras_adapter_can_be_constructed_without_importing_tensorflow(self):
        adapter = KerasDetectorAdapter(DetectorContract("example"), lambda size, config: (size, config))
        self.assertEqual(adapter.build_model(4, "config"), (4, "config"))


if __name__ == "__main__":
    unittest.main()
