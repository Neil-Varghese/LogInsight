import unittest

import numpy as np

from event_vocabulary import build_event_vocabulary
from integrated_gradients_lstm import (
    _manifest_path,
    integrated_gradients_batch,
    ranked_event_positions,
)


class IntegratedGradientsLstmTests(unittest.TestCase):
    def test_ranked_events_exclude_padding_and_preserve_alignment(self):
        vocabulary = build_event_vocabulary([["a", "b"]])
        row = np.asarray([0, 0, vocabulary.token_to_id["a"], vocabulary.token_to_id["b"]])
        attribution = np.asarray([0.4, -0.2, -0.1, 0.8])
        events = ranked_event_positions(row, attribution, vocabulary, sequence_length=2)
        self.assertEqual([event["rank"] for event in events], [None, None, 2, 1])
        self.assertEqual(events[2]["original_sequence_index"], 0)
        self.assertEqual(events[3]["original_sequence_index"], 1)
        self.assertTrue(events[0]["is_padding"])

    def test_manifest_path_is_stable(self):
        self.assertEqual(
            _manifest_path(__import__("pathlib").Path("out/seed_test.jsonl.gz")).as_posix(),
            "out/seed_test.manifest.json",
        )

    def test_integrated_gradients_completeness_for_research_lstm(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow is not installed")
        from train_loginsight_lstm import TrainingConfig, build_model

        vocabulary = build_event_vocabulary([["a", "b", "c"]])
        config = TrainingConfig(
            max_sequence_length=4,
            embedding_dim=4,
            first_lstm_units=4,
            second_lstm_units=3,
            dense_units=3,
        )
        model = build_model(vocabulary.size, config)
        inputs = np.asarray(
            [
                [0, vocabulary.token_to_id["a"], vocabulary.token_to_id["b"], vocabulary.token_to_id["c"]],
                [0, 0, vocabulary.token_to_id["b"], vocabulary.token_to_id["a"]],
            ],
            dtype=np.int32,
        )
        result = integrated_gradients_batch(model, inputs, integration_steps=8, baseline_token_id=0)
        self.assertEqual(result["attribution"].shape, inputs.shape)
        self.assertEqual(result["probability"].shape, (2,))
        self.assertTrue(np.all(np.isfinite(result["completeness_error"])))
        np.testing.assert_allclose(
            result["attribution_sum"],
            result["output_delta"],
            atol=2e-4,
            rtol=2e-4,
        )

    def test_logit_space_integrated_gradients_has_logit_completeness(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow is not installed")
        from train_loginsight_lstm import TrainingConfig, build_model

        vocabulary = build_event_vocabulary([["a", "b"]])
        config = TrainingConfig(max_sequence_length=3, embedding_dim=4, first_lstm_units=3, second_lstm_units=2, dense_units=2)
        model = build_model(vocabulary.size, config)
        inputs = np.asarray([[0, vocabulary.token_to_id["a"], vocabulary.token_to_id["b"]]], dtype=np.int32)
        result = integrated_gradients_batch(model, inputs, integration_steps=8, output_space="logit")
        self.assertEqual(result["output_space"], "logit")
        self.assertTrue(np.all(np.isfinite(result["output_delta"])))
        np.testing.assert_allclose(result["attribution_sum"], result["output_delta"], atol=3e-4, rtol=3e-4)


if __name__ == "__main__":
    unittest.main()
