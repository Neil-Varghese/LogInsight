import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample_annotation_pool import (  # noqa: E402
    confidence_bin,
    length_bin,
    normalize_candidate,
    stratified_sample,
)


def make_candidates():
    candidates = []
    architectures = ("loginsight_lstm", "logrobust_attention_bilstm")
    for index in range(240):
        dataset = "HDFS" if index % 2 == 0 else "BGL"
        label = "Normal" if index % 4 < 2 else "Anomaly"
        length = (index % 3) * 30 + 5
        confidence = (index % 3) / 2
        architecture = architectures[index % 2]
        candidates.append({
            "dataset": dataset,
            "unit_id": f"unit_{index}",
            "split": "test",
            "label": label,
            "sequence_length": length,
            "confidence": confidence,
            "architecture": architecture,
            "length_bin": length_bin(length),
            "confidence_bin": confidence_bin(confidence),
        })
    return candidates


class AnnotationPoolTests(unittest.TestCase):
    def test_bins_are_registered(self):
        self.assertEqual(length_bin(10), "short_0_10")
        self.assertEqual(length_bin(50), "medium_11_50")
        self.assertEqual(length_bin(51), "long_51_plus")
        self.assertEqual(confidence_bin(0.0), "low_0_33")
        self.assertEqual(confidence_bin(0.5), "medium_33_67")
        self.assertEqual(confidence_bin(1.0), "high_67_100")

    def test_normalize_requires_confidence_and_architecture(self):
        record = {
            "dataset": "HDFS",
            "unit_id": "blk_1",
            "split": "test",
            "label": "Normal",
            "sequence_length": 10,
        }
        with self.assertRaisesRegex(ValueError, "confidence, architecture"):
            normalize_candidate(record, 1, "test")

    def test_sampling_is_deterministic_and_exact(self):
        first = stratified_sample(make_candidates(), target_count=200, seed=2026)
        second = stratified_sample(make_candidates(), target_count=200, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 200)
        self.assertEqual(len({item["annotation_item_id"] for item in first}), 200)
        self.assertEqual({item["split"] for item in first}, {"test"})

    def test_sampling_rejects_out_of_range_target(self):
        with self.assertRaisesRegex(ValueError, "between 200 and 300"):
            stratified_sample(make_candidates(), target_count=100)


if __name__ == "__main__":
    unittest.main()
