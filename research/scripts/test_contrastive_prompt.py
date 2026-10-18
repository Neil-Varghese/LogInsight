import unittest

try:
    from contrastive_prompt import (
        ATTRIBUTION_MODES,
        build_contrastive_prompt_context,
        registered_factor_cells,
        render_contrastive_prompt,
    )
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.contrastive_prompt import (
        ATTRIBUTION_MODES,
        build_contrastive_prompt_context,
        registered_factor_cells,
        render_contrastive_prompt,
    )


def _record(block_id):
    return {
        "BlockId": block_id,
        "Evidence": [
            {
                "Position": 0,
                "EventId": "E1",
                "LineId": 17,
                "EventTemplate": "Failure <*>",
                "ParameterList": ["x"],
                "RawLine": "raw failure line",
            }
        ],
    }


class ContrastivePromptTests(unittest.TestCase):
    def test_registered_factor_has_six_cells(self):
        cells = registered_factor_cells()
        self.assertEqual(len(cells), 6)
        self.assertEqual({cell["attribution"] for cell in cells}, set(ATTRIBUTION_MODES))
        self.assertEqual({cell["normal_comparison"] for cell in cells}, {True, False})

    def test_normal_comparison_and_none_attribution_cell(self):
        condition, payload = build_contrastive_prompt_context(
            _record("anomaly"),
            {"probability": 0.9},
            normal_comparison=True,
            attribution="none",
            normal_record=_record("normal"),
        )
        self.assertEqual(condition, "blinded")
        self.assertIn("normal_comparison_block", payload)
        self.assertNotIn("integrated_gradients", payload)

    def test_no_normal_integrated_gradients_cell(self):
        condition, payload = build_contrastive_prompt_context(
            _record("anomaly"),
            {"probability": 0.9},
            normal_comparison=False,
            attribution="integrated_gradients",
            integrated_gradients={"ranked_event_ids": ["E1"]},
        )
        self.assertEqual(condition, "unblinded_no_normal")
        self.assertNotIn("normal_comparison_block", payload)
        self.assertIn("integrated_gradients", payload)

    def test_no_normal_shuffled_cell(self):
        condition, payload = build_contrastive_prompt_context(
            _record("anomaly"),
            {"probability": 0.9},
            normal_comparison=False,
            attribution="shuffled_integrated_gradients",
            shuffled_integrated_gradients={"ranked_event_ids": ["E1"]},
        )
        self.assertEqual(condition, "shuffled_integrated_gradients_no_normal")
        self.assertNotIn("normal_comparison_block", payload)
        self.assertIn("shuffled_integrated_gradients", payload)

    def test_rejects_missing_factor_inputs(self):
        with self.assertRaises(ValueError):
            build_contrastive_prompt_context(
                _record("anomaly"), {"probability": 0.9}, normal_comparison=True, attribution="none"
            )
        with self.assertRaises(ValueError):
            build_contrastive_prompt_context(
                _record("anomaly"), {"probability": 0.9}, normal_comparison=False, attribution="integrated_gradients"
            )

    def test_rendered_prompt_is_deterministic_and_exposes_only_selected_attribution(self):
        first = render_contrastive_prompt(
            _record("anomaly"),
            {"probability": 0.9},
            normal_comparison=True,
            attribution="integrated_gradients",
            normal_record=_record("normal"),
            integrated_gradients={"ranked_event_ids": ["E1"]},
        )
        second = render_contrastive_prompt(
            _record("anomaly"),
            {"probability": 0.9},
            normal_comparison=True,
            attribution="integrated_gradients",
            normal_record=_record("normal"),
            integrated_gradients={"ranked_event_ids": ["E1"]},
        )
        self.assertEqual(first, second)
        self.assertIn("integrated_gradients", first[2])
        self.assertNotIn("occlusion", first[2].lower())
        self.assertNotIn("leave_one_event_out", first[2].lower())


if __name__ == "__main__":
    unittest.main()
