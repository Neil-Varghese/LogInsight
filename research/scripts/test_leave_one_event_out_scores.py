import unittest

import numpy as np

try:
    from detector_interface import DetectorInferenceCache
    from leave_one_event_out_scores import (
        build_leave_one_event_out_inputs,
        compute_leave_one_event_out_batch,
    )
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.detector_interface import DetectorInferenceCache
    from research.scripts.leave_one_event_out_scores import (
        build_leave_one_event_out_inputs,
        compute_leave_one_event_out_batch,
    )


class _SumProbabilityModel:
    def predict(self, inputs, batch_size, verbose=0):
        return (np.sum(inputs, axis=1, dtype=np.float64) / 10.0).reshape(-1, 1)


class LeaveOneEventOutScoresTests(unittest.TestCase):
    def test_pre_padding_holds_out_each_event_and_repads(self):
        inputs = np.asarray([[0, 0, 4, 5, 6]], dtype=np.int32)
        held_out, references = build_leave_one_event_out_inputs(inputs, pad_id=0, padding="pre")
        self.assertEqual([reference["event_id"] for reference in references], [4, 5, 6])
        self.assertEqual(held_out.tolist(), [[0, 0, 0, 5, 6], [0, 0, 0, 4, 6], [0, 0, 0, 4, 5]])

    def test_post_padding_preserves_positions(self):
        inputs = np.asarray([[4, 5, 6, 0, 0]], dtype=np.int32)
        held_out, references = build_leave_one_event_out_inputs(inputs, pad_id=0, padding="post")
        self.assertEqual([reference["input_position"] for reference in references], [0, 1, 2])
        self.assertEqual(held_out.tolist(), [[5, 6, 0, 0, 0], [4, 6, 0, 0, 0], [4, 5, 0, 0, 0]])

    def test_scores_use_original_minus_held_out_and_cache(self):
        inputs = np.asarray([[4, 5, 0, 0]], dtype=np.int32)
        cache = DetectorInferenceCache("test_detector", "test_model")
        original, references = compute_leave_one_event_out_batch(
            _SumProbabilityModel(),
            inputs,
            pad_id=0,
            padding="post",
            cache=cache,
            inference_batch_size=8,
        )
        self.assertAlmostEqual(float(original[0]), 0.9)
        self.assertEqual([reference["leave_one_event_out_probability"] for reference in references], [0.5, 0.4])
        self.assertEqual([reference["leave_one_event_out_score"] for reference in references], [0.4, 0.5])
        self.assertEqual(cache.stats.model_rows, 3)


if __name__ == "__main__":
    unittest.main()
