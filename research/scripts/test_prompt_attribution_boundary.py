import unittest

try:
    from prompt_attribution_boundary import (
        PromptAttributionBoundaryError,
        build_prompt_payload,
        render_prompt_payload,
    )
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.prompt_attribution_boundary import (
        PromptAttributionBoundaryError,
        build_prompt_payload,
        render_prompt_payload,
    )


class PromptAttributionBoundaryTests(unittest.TestCase):
    def _base_context(self):
        return {
            "anomalous_block_evidence": {"events": ["E1"]},
            "normal_comparison_block": {"events": ["E2"]},
            "detector_output": {"probability": 0.9},
        }

    def test_unblinded_allows_integrated_gradients_only(self):
        context = self._base_context()
        context["integrated_gradients"] = {"ranked_event_ids": ["E1"]}
        payload = build_prompt_payload("unblinded", context)
        self.assertIn("integrated_gradients", payload)
        self.assertNotIn("occlusion", payload)
        self.assertNotIn("leave_one_event_out", payload)

    def test_blinded_rejects_integrated_gradients(self):
        context = self._base_context()
        context["integrated_gradients"] = {"ranked_event_ids": ["E1"]}
        with self.assertRaises(PromptAttributionBoundaryError):
            build_prompt_payload("blinded", context)

    def test_rejects_nested_occlusion_scores(self):
        context = self._base_context()
        context["verification"] = {"events": [{"occlusion_score": 0.4}]}
        with self.assertRaises(PromptAttributionBoundaryError):
            build_prompt_payload("unblinded", context)

    def test_rejects_nested_leave_one_event_out_scores(self):
        context = self._base_context()
        context["verification"] = {
            "events": [{"leave_one_event_out_score": 0.4, "prompt_visible": False}]
        }
        with self.assertRaises(PromptAttributionBoundaryError):
            build_prompt_payload("unblinded", context)

    def test_rejects_verification_only_metadata_even_without_score_key(self):
        context = self._base_context()
        context["integrated_gradients"] = {"verification_only": True, "ranked_event_ids": ["E1"]}
        with self.assertRaises(PromptAttributionBoundaryError):
            build_prompt_payload("unblinded", context)

    def test_rendered_prompt_contains_only_registered_fields(self):
        rendered = render_prompt_payload("blinded", self._base_context())
        self.assertIn("anomalous_block_evidence", rendered)
        self.assertIn("detector_output", rendered)
        self.assertNotIn("integrated_gradients", rendered)
        self.assertNotIn("occlusion", rendered)
        self.assertNotIn("leave_one_event_out", rendered)


if __name__ == "__main__":
    unittest.main()
