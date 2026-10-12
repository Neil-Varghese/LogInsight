import importlib.util
import unittest

import numpy as np

from train_logrobust_attention_bilstm import (
    LogRobustTrainingConfig,
    build_model,
)


class TrainLogRobustAttentionBiLstmTests(unittest.TestCase):
    def test_configuration_accepts_attention_fields(self):
        config = LogRobustTrainingConfig.from_dict({
            "max_sequence_length": 8,
            "bilstm_units": 16,
            "attention_units": 12,
            "dense_units": 4,
        })
        self.assertEqual(config.max_sequence_length, 8)
        self.assertEqual(config.attention_units, 12)
        self.assertTrue(config.mask_zero)

    def test_configuration_rejects_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "Unknown LogRobust training fields"):
            LogRobustTrainingConfig.from_dict({"fasttext_path": "not-used"})

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_model_is_bidirectional_and_has_temporal_attention(self):
        config = LogRobustTrainingConfig(
            max_sequence_length=4,
            embedding_dim=8,
            bilstm_units=4,
            attention_units=6,
            dense_units=3,
        )
        model = build_model(vocabulary_size=10, config=config)
        self.assertEqual(model.name, "logrobust_attention_bilstm")
        self.assertEqual(model.get_layer("bilstm").forward_layer.units, 4)
        self.assertEqual(model.get_layer("bilstm").backward_layer.units, 4)
        self.assertEqual(model.get_layer("temporal_attention").attention_units, 6)
        outputs = model(np.asarray([[0, 1, 2, 3]], dtype=np.int32), training=False).numpy()
        self.assertEqual(outputs.shape, (1, 1))
        self.assertTrue(np.all((outputs >= 0.0) & (outputs <= 1.0)))


if __name__ == "__main__":
    unittest.main()
