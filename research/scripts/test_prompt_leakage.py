"""Step 49 tests for condition-specific prompt leakage."""

import json
import unittest

try:
    from blinded_prompt import render_blinded_prompt
    from contrastive_prompt import build_contrastive_prompt_context, registered_factor_cells, render_contrastive_prompt
    from no_detector_context_prompt import render_no_detector_context_prompt
    from prompt_attribution_boundary import PromptAttributionBoundaryError, build_prompt_payload, load_prompt_condition
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.blinded_prompt import render_blinded_prompt
    from research.scripts.contrastive_prompt import build_contrastive_prompt_context, registered_factor_cells, render_contrastive_prompt
    from research.scripts.no_detector_context_prompt import render_no_detector_context_prompt
    from research.scripts.prompt_attribution_boundary import PromptAttributionBoundaryError, build_prompt_payload, load_prompt_condition


CONDITION_IDS = (
    "blinded",
    "unblinded",
    "shuffled_integrated_gradients",
    "detector_only",
    "unblinded_no_normal",
    "shuffled_integrated_gradients_no_normal",
    "attribution_only",
    "random_event",
    "no_detector_context",
)


def _record(block_id):
    return {
        "BlockId": block_id,
        "LabelId": 1,
        "Evidence": [
            {
                "Position": 0,
                "EventId": "E1",
                "LineId": 17,
                "EventTemplate": "Failure <*>",
                "ParameterList": ["x"],
                "RawLine": "failure line",
            }
        ],
    }


def _value_for(field):
    return {"sentinel": field}


class PromptLeakageTests(unittest.TestCase):
    def test_registered_condition_matrix_is_complete_and_unambiguous(self):
        self.assertEqual(set(CONDITION_IDS), {load_prompt_condition(condition_id)["id"] for condition_id in CONDITION_IDS})
        for condition_id in CONDITION_IDS:
            condition = load_prompt_condition(condition_id)
            visible = set(condition["prompt_visible"])
            forbidden = set(condition["prompt_forbidden"])
            self.assertTrue(visible.isdisjoint(forbidden), condition_id)

    def test_each_condition_accepts_only_its_registered_root_fields(self):
        all_fields = {
            "anomalous_block_evidence",
            "normal_comparison_block",
            "detector_output",
            "integrated_gradients",
            "shuffled_integrated_gradients",
        }
        for condition_id in CONDITION_IDS:
            visible = set(load_prompt_condition(condition_id)["prompt_visible"])
            accepted = build_prompt_payload(condition_id, {field: _value_for(field) for field in visible})
            self.assertEqual(set(accepted), visible, condition_id)
            for field in sorted(all_fields - visible):
                with self.assertRaises(PromptAttributionBoundaryError, msg=f"{condition_id} accepted {field}"):
                    build_prompt_payload(condition_id, {field: _value_for(field)})

    def test_nested_verification_and_cross_condition_fields_are_rejected(self):
        for condition_id in CONDITION_IDS:
            context = {"anomalous_block_evidence": {"events": [{"occlusion_score": 0.4}]}}
            if "anomalous_block_evidence" not in load_prompt_condition(condition_id)["prompt_visible"]:
                continue
            with self.assertRaises(PromptAttributionBoundaryError):
                build_prompt_payload(condition_id, context)

        with self.assertRaises(PromptAttributionBoundaryError):
            build_prompt_payload("blinded", {"detector_output": {"integrated_gradients": {"ranked_event_ids": ["E1"]}}})
        with self.assertRaises(PromptAttributionBoundaryError):
            build_prompt_payload("unblinded", {"detector_output": {"normal_comparison_block": {"events": []}}})

    def test_six_factor_cells_render_exactly_their_registered_payload(self):
        for cell in registered_factor_cells():
            kwargs = {
                "normal_comparison": cell["normal_comparison"],
                "attribution": cell["attribution"],
                "normal_record": _record("normal") if cell["normal_comparison"] else None,
                "integrated_gradients": {"ranked_event_ids": ["E1"]} if cell["attribution"] == "integrated_gradients" else None,
                "shuffled_integrated_gradients": {"ranked_event_ids": ["E1"]} if cell["attribution"] == "shuffled_integrated_gradients" else None,
            }
            condition_id, payload = build_contrastive_prompt_context(_record("anomaly"), {"probability": 0.9}, **kwargs)
            self.assertEqual(condition_id, cell["condition_id"])
            self.assertEqual(set(payload), set(load_prompt_condition(condition_id)["prompt_visible"]))

    def test_rendered_prompts_do_not_contain_storage_or_verification_fields(self):
        blinded = render_blinded_prompt(_record("anomaly"), _record("normal"), {"probability": 0.9})
        no_detector = render_no_detector_context_prompt(_record("anomaly"))[2]
        contrastive = render_contrastive_prompt(
            _record("anomaly"),
            {"probability": 0.9},
            normal_comparison=True,
            attribution="integrated_gradients",
            normal_record=_record("normal"),
            integrated_gradients={"ranked_event_ids": ["E1"]},
        )[2]
        for prompt in (blinded, no_detector, contrastive):
            lowered = prompt.lower()
            self.assertNotIn("occlusion_score", lowered)
            self.assertNotIn("leave_one_event_out", lowered)
            self.assertNotIn("verification_only", lowered)
            self.assertNotIn('"blockid"', lowered)
            self.assertNotIn('"labelid"', lowered)

        payload = json.loads(blinded.split("INPUT_JSON:\n", 1)[1])
        self.assertEqual(set(payload), set(load_prompt_condition("blinded")["prompt_visible"]))


if __name__ == "__main__":
    unittest.main()
