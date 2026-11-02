import unittest

import numpy as np

try:
    from roar_retraining import replace_top_events, select_removed_positions
except ImportError:  # pragma: no cover
    from research.scripts.roar_retraining import replace_top_events, select_removed_positions


def _attribution(unit_id="u1"):
    return {
        "unit_id": unit_id,
        "padded_length": 6,
        "events": [
            {"input_position": 1, "event_id": 11, "score": 0.9},
            {"input_position": 3, "event_id": 13, "score": 0.7},
            {"input_position": 5, "event_id": 15, "score": 0.2},
        ],
    }


class RoarRetrainingTests(unittest.TestCase):
    def test_integrated_gradients_selects_highest_absolute_events(self):
        selected = select_removed_positions(
            _attribution(), fraction=0.5, strategy="integrated_gradients", seed=2026
        )
        self.assertEqual([item["input_position"] for item in selected], [1, 3])

    def test_zero_fraction_does_not_change_inputs(self):
        inputs = np.array([[0, 11, 0, 13, 0, 15]], dtype=np.int32)
        output, changes = replace_top_events(
            inputs,
            [{"unit_id": "u1"}],
            {"u1": _attribution()},
            fraction=0.0,
            strategy="integrated_gradients",
            neutral_id=99,
            seed=2026,
        )
        np.testing.assert_array_equal(output, inputs)
        self.assertEqual(changes[0]["removed_count"], 0)

    def test_replacement_preserves_length_and_uses_neutral_token(self):
        inputs = np.array([[0, 11, 0, 13, 0, 15]], dtype=np.int32)
        output, changes = replace_top_events(
            inputs,
            [{"unit_id": "u1"}],
            {"u1": _attribution()},
            fraction=0.5,
            strategy="integrated_gradients",
            neutral_id=99,
            seed=2026,
        )
        np.testing.assert_array_equal(output, np.array([[0, 99, 0, 99, 0, 15]], dtype=np.int32))
        self.assertEqual(output.shape, inputs.shape)
        self.assertEqual(changes[0]["removed_count"], 2)

    def test_random_selection_is_deterministic_and_equal_budget(self):
        first = select_removed_positions(
            _attribution(), fraction=0.5, strategy="random_event", seed=2026
        )
        second = select_removed_positions(
            _attribution(), fraction=0.5, strategy="random_event", seed=2026
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_event_mismatch_is_rejected(self):
        inputs = np.array([[0, 999, 0, 13, 0, 15]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "event mismatch"):
            replace_top_events(
                inputs,
                [{"unit_id": "u1"}],
                {"u1": _attribution()},
                fraction=0.5,
                strategy="integrated_gradients",
                neutral_id=99,
                seed=2026,
            )


if __name__ == "__main__":
    unittest.main()
