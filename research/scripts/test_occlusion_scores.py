import unittest

import numpy as np

try:
    from detector_interface import DetectorInferenceCache
    from occlusion_scores import _original_sequence_index, build_occluded_inputs, compute_occlusion_batch
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.detector_interface import DetectorInferenceCache
    from research.scripts.occlusion_scores import (
        _original_sequence_index,
        build_occluded_inputs,
        compute_occlusion_batch,
    )


class _SumProbabilityModel:
    def predict(self, inputs, batch_size, verbose=0):
        return (np.sum(inputs, axis=1, dtype=np.float64) / 10.0).reshape(-1, 1)


class OcclusionScoresTests(unittest.TestCase):
    def test_pre_padding_deletes_each_active_event_and_repads(self):
        inputs = np.asarray([[0, 0, 4, 5, 6]], dtype=np.int32)
        occluded, references = build_occluded_inputs(inputs, pad_id=0, padding="pre")
        self.assertEqual([reference["input_position"] for reference in references], [2, 3, 4])
        self.assertEqual(occluded.tolist(), [[0, 0, 0, 5, 6], [0, 0, 0, 4, 6], [0, 0, 0, 4, 5]])

    def test_post_padding_preserves_event_order(self):
        inputs = np.asarray([[4, 5, 6, 0, 0]], dtype=np.int32)
        occluded, references = build_occluded_inputs(inputs, pad_id=0, padding="post")
        self.assertEqual([reference["event_id"] for reference in references], [4, 5, 6])
        self.assertEqual(occluded.tolist(), [[5, 6, 0, 0, 0], [4, 6, 0, 0, 0], [4, 5, 0, 0, 0]])

    def test_original_sequence_index_handles_truncation_and_padding(self):
        self.assertEqual(_original_sequence_index(2, 5, 4, "pre"), 3)
        self.assertEqual(_original_sequence_index(0, 2, 4, "post"), 0)

    def test_scores_use_original_minus_occluded_and_cache(self):
        inputs = np.asarray([[4, 5, 0, 0]], dtype=np.int32)
        cache = DetectorInferenceCache("test_detector", "test_model")
        original, references = compute_occlusion_batch(
            _SumProbabilityModel(),
            inputs,
            pad_id=0,
            padding="post",
            cache=cache,
            inference_batch_size=8,
        )
        self.assertAlmostEqual(float(original[0]), 0.9)
        self.assertEqual([reference["occluded_probability"] for reference in references], [0.5, 0.4])
        self.assertEqual([reference["occlusion_score"] for reference in references], [0.4, 0.5])
        self.assertEqual(cache.stats.model_rows, 3)


if __name__ == "__main__":
    unittest.main()
