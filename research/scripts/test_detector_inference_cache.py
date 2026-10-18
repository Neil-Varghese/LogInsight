import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector_interface import (  # noqa: E402
    DetectorInferenceCache,
    predict_probabilities_batched,
)


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, inputs, batch_size, verbose):
        self.calls.append((np.asarray(inputs).copy(), batch_size, verbose))
        return np.mean(inputs, axis=1, keepdims=True) / 10.0


class DetectorInferenceCacheTests(unittest.TestCase):
    def test_batched_prediction_preserves_order(self):
        model = FakeModel()
        inputs = np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.int32)
        probabilities = predict_probabilities_batched(model, inputs, batch_size=2)
        np.testing.assert_allclose(probabilities, [0.1, 0.2, 0.3])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual([call[0].shape[0] for call in model.calls], [2, 1])
        self.assertTrue(all(call[2] == 0 for call in model.calls))

    def test_cache_deduplicates_within_and_across_requests(self):
        model = FakeModel()
        cache = DetectorInferenceCache("loginsight_lstm", "checkpoint-seed-13")
        first = np.asarray([[1, 1], [2, 2], [1, 1]], dtype=np.int32)
        second = np.asarray([[2, 2], [3, 3]], dtype=np.int32)
        np.testing.assert_allclose(cache.predict(model, first, batch_size=2), [0.1, 0.2, 0.1])
        np.testing.assert_allclose(cache.predict(model, second, batch_size=2), [0.2, 0.3])
        self.assertEqual(cache.stats.requested_rows, 5)
        self.assertEqual(cache.stats.deduplicated_rows, 1)
        self.assertEqual(cache.stats.cache_hit_rows, 1)
        self.assertEqual(cache.stats.model_rows, 3)
        self.assertEqual(len(cache), 3)
        self.assertEqual(len(model.calls), 2)

    def test_cache_namespace_prevents_model_collisions(self):
        model = FakeModel()
        first = DetectorInferenceCache("loginsight_lstm", "seed-13")
        second = DetectorInferenceCache("loginsight_lstm", "seed-42")
        inputs = np.asarray([[1, 1]], dtype=np.int32)
        first.predict(model, inputs, batch_size=1)
        second.predict(model, inputs, batch_size=1)
        self.assertEqual(first.stats.model_rows, 1)
        self.assertEqual(second.stats.model_rows, 1)

    def test_empty_request_does_not_call_model(self):
        model = FakeModel()
        cache = DetectorInferenceCache("detector", "model")
        result = cache.predict(model, np.empty((0, 2), dtype=np.int32), batch_size=4)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
