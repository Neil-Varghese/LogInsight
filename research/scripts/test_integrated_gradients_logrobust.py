import unittest

import numpy as np

from event_vocabulary import build_event_vocabulary
from integrated_gradients_logrobust import integrated_gradients_batch


class IntegratedGradientsLogRobustTests(unittest.TestCase):
    def test_masked_attention_integrated_gradients_completeness(self):
        try:
            import tensorflow
        except ImportError:
            self.skipTest("TensorFlow is not installed")
        from train_logrobust_attention_bilstm import LogRobustTrainingConfig, build_model

        tensorflow.keras.utils.set_random_seed(2026)

        vocabulary = build_event_vocabulary([["a", "b", "c"]])
        config = LogRobustTrainingConfig(
            max_sequence_length=4,
            embedding_dim=8,
            bilstm_units=4,
            attention_units=6,
            dense_units=3,
            mask_zero=True,
            padding="post",
        )
        model = build_model(vocabulary.size, config)
        inputs = np.asarray(
            [
                [vocabulary.token_to_id["a"], vocabulary.token_to_id["b"], 0, 0],
                [vocabulary.token_to_id["c"], 0, 0, 0],
            ],
            dtype=np.int32,
        )
        result = integrated_gradients_batch(model, inputs, integration_steps=8, baseline_token_id=0)
        self.assertEqual(result["attribution"].shape, inputs.shape)
        self.assertTrue(np.all(np.isfinite(result["completeness_error"])))
        np.testing.assert_allclose(
            result["attribution_sum"],
            result["output_delta"],
            atol=3e-4,
            rtol=3e-4,
        )

    def test_logit_space_integrated_gradients_has_masked_completeness(self):
        try:
            import tensorflow
        except ImportError:
            self.skipTest("TensorFlow is not installed")
        from train_logrobust_attention_bilstm import LogRobustTrainingConfig, build_model

        tensorflow.keras.utils.set_random_seed(2026)

        vocabulary = build_event_vocabulary([["a", "b"]])
        config = LogRobustTrainingConfig(
            max_sequence_length=3,
            embedding_dim=6,
            bilstm_units=3,
            attention_units=4,
            dense_units=2,
            mask_zero=True,
            padding="post",
        )
        model = build_model(vocabulary.size, config)
        inputs = np.asarray([[vocabulary.token_to_id["a"], vocabulary.token_to_id["b"], 0]], dtype=np.int32)
        result = integrated_gradients_batch(model, inputs, integration_steps=64, output_space="logit")
        self.assertEqual(result["output_space"], "logit")
        np.testing.assert_allclose(result["attribution_sum"], result["output_delta"], atol=5e-4, rtol=5e-4)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            integrated_gradients_batch(
                object(),
                np.asarray([[1.0, 2.0]], dtype=np.float32),
                integration_steps=4,
            )


if __name__ == "__main__":
    unittest.main()
