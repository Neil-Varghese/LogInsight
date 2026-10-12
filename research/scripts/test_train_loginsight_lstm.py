import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_loginsight_lstm import (
    TrainingConfig,
    assert_disjoint_splits,
    binary_metrics,
    class_weights,
    encode_and_pad,
    load_split,
)
from event_vocabulary import build_event_vocabulary


class TrainLogInsightLstmTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = build_event_vocabulary([["a", "b", "c"]])
        self.config = TrainingConfig(max_sequence_length=4)

    def test_encode_maps_unknown_and_pre_pads(self):
        encoded, original_length = encode_and_pad(
            ["a", "unseen", "b", "c", "a"],
            self.vocabulary,
            max_length=4,
            padding="pre",
            truncating="pre",
        )
        self.assertEqual(original_length, 5)
        self.assertEqual(encoded.tolist(), [self.vocabulary.unknown_id, self.vocabulary.token_to_id["b"], self.vocabulary.token_to_id["c"], self.vocabulary.token_to_id["a"]])

    def test_load_split_preserves_ids_lengths_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.jsonl.gz"
            records = [
                {"BlockId": "blk_1", "Sequence": ["a"], "Label": "Normal"},
                {"BlockId": "blk_2", "Sequence": ["b", "c"], "LabelId": 1},
            ]
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            inputs, labels, metadata = load_split(path, self.vocabulary, self.config, show_progress=False)
        self.assertEqual(inputs.shape, (2, 4))
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual([item["unit_id"] for item in metadata], ["blk_1", "blk_2"])
        self.assertEqual([item["sequence_length"] for item in metadata], [1, 2])

    def test_overlap_is_rejected(self):
        records = {"train": [{"unit_id": "same"}], "validation": [], "test": []}
        records["validation"].append({"unit_id": "same"})
        with self.assertRaisesRegex(ValueError, "both train and validation"):
            assert_disjoint_splits(records)

    def test_class_weights_and_metrics_are_deterministic(self):
        labels = np.asarray([0, 0, 0, 1], dtype=np.int32)
        self.assertEqual(class_weights(labels), {0: 2 / 3, 1: 2.0})
        metrics = binary_metrics(labels, np.asarray([0.1, 0.2, 0.8, 0.9]), threshold=0.5)
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [0, 1]])
        self.assertAlmostEqual(metrics["f1"], 0.6666666666666666)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(max_sequence_length=0).validate()


if __name__ == "__main__":
    unittest.main()
