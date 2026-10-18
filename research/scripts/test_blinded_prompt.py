import unittest

try:
    from blinded_prompt import build_blinded_prompt_context, evidence_record_to_prompt_block, render_blinded_prompt
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.blinded_prompt import build_blinded_prompt_context, evidence_record_to_prompt_block, render_blinded_prompt


def _evidence_record(block_id):
    return {
        "BlockId": block_id,
        "LabelId": 1,
        "Sequence": ["E1"],
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


class BlindedPromptTests(unittest.TestCase):
    def test_context_contains_only_registered_blinded_fields(self):
        context = build_blinded_prompt_context(
            _evidence_record("anomaly"),
            _evidence_record("normal"),
            {"probability": 0.9, "prediction": 1},
        )
        self.assertEqual(set(context), {"anomalous_block_evidence", "normal_comparison_block", "detector_output"})
        self.assertNotIn("BlockId", json_keys(context["anomalous_block_evidence"]))

    def test_rendered_prompt_contains_no_attribution_fields(self):
        prompt = render_blinded_prompt(
            _evidence_record("anomaly"),
            _evidence_record("normal"),
            {"probability": 0.9, "prediction": 1},
        )
        lowered = prompt.lower()
        self.assertNotIn("integrated_gradients", lowered)
        self.assertNotIn("occlusion", lowered)
        self.assertNotIn("leave_one_event_out", lowered)

    def test_evidence_conversion_does_not_copy_unit_label(self):
        block = evidence_record_to_prompt_block(_evidence_record("anomaly"))
        self.assertEqual(list(block), ["events"])
        self.assertNotIn("LabelId", block)


def json_keys(value):
    return set(value) if isinstance(value, dict) else set()


if __name__ == "__main__":
    unittest.main()
