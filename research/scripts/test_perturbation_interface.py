import unittest

import numpy as np

try:
    from perturbation_interface import (
        PerturbationBatch,
        apply_perturbation,
        apply_matched_random_perturbation,
        available_perturbation_strategies,
        delete_cited_events,
        register_perturbation_strategy,
        replace_with_normal_events,
        replace_with_unknown_token,
        resample_from_normal_events,
        mask_event_parameters,
        sample_matched_random_positions,
    )
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.perturbation_interface import (
        PerturbationBatch,
        apply_perturbation,
        apply_matched_random_perturbation,
        available_perturbation_strategies,
        delete_cited_events,
        register_perturbation_strategy,
        replace_with_normal_events,
        replace_with_unknown_token,
        resample_from_normal_events,
        mask_event_parameters,
        sample_matched_random_positions,
    )


class PerturbationInterfaceTests(unittest.TestCase):
    def test_registered_deletion_uses_common_contract(self):
        inputs = np.asarray([[0, 0, 4, 5, 6]], dtype=np.int32)
        result = apply_perturbation(
            "single_event_deletion",
            inputs,
            pad_id=0,
            padding="pre",
        )
        self.assertIn("single_event_deletion", available_perturbation_strategies())
        self.assertEqual(result.strategy, "single_event_deletion")
        self.assertEqual(result.inputs.tolist(), [[0, 0, 0, 5, 6], [0, 0, 0, 4, 6], [0, 0, 0, 4, 5]])
        self.assertEqual([reference["input_position"] for reference in result.references], [2, 3, 4])
        self.assertEqual(result.metadata["perturbed_count"], 3)

    def test_explicit_positions_and_metadata_are_preserved(self):
        inputs = np.asarray([[4, 5, 6, 0]], dtype=np.int32)
        result = apply_perturbation(
            "single_event_deletion",
            inputs,
            pad_id=0,
            padding="post",
            target_positions=[[1]],
            metadata={"unit_id": "block-1"},
        )
        self.assertEqual(result.inputs.tolist(), [[4, 6, 0, 0]])
        self.assertEqual(result.references[0]["event_id"], 5)
        self.assertEqual(result.metadata["unit_id"], "block-1")
        self.assertEqual(result.metadata["selection"], "explicit_positions")

    def test_deletion_removes_all_cited_events_in_one_output_per_unit(self):
        inputs = np.asarray([[0, 0, 4, 5, 6], [4, 5, 0, 0, 0]], dtype=np.int32)
        result = delete_cited_events(
            inputs,
            cited_positions=[[2, 4], [1]],
            pad_id=0,
            padding="pre",
        )
        self.assertIn("deletion", available_perturbation_strategies())
        self.assertEqual(result.inputs.tolist(), [[0, 0, 0, 0, 5], [0, 0, 0, 0, 4]])
        self.assertEqual(len(result.references), 2)
        self.assertEqual(result.references[0]["deleted_positions"], [2, 4])
        self.assertEqual(result.references[0]["deleted_event_ids"], [4, 6])
        self.assertTrue(result.metadata["one_output_per_input"])

    def test_deletion_with_no_citations_is_an_explicit_noop(self):
        inputs = np.asarray([[0, 4, 5, 0]], dtype=np.int32)
        result = delete_cited_events(inputs, cited_positions=[[]], pad_id=0, padding="post")
        self.assertEqual(result.inputs.tolist(), [[0, 4, 5, 0]])
        self.assertEqual(result.references[0]["deleted_positions"], [])
        self.assertEqual(result.references[0]["deleted_event_ids"], [])

    def test_deletion_requires_cited_positions(self):
        with self.assertRaisesRegex(ValueError, "requires target_positions"):
            apply_perturbation(
                "deletion",
                np.asarray([[1, 0]], dtype=np.int32),
                pad_id=0,
                padding="post",
            )

    def test_normal_replacement_aligns_by_active_position_and_preserves_width(self):
        anomalous = np.asarray([[0, 0, 4, 5, 6]], dtype=np.int32)
        normal = np.asarray([[0, 7, 8, 9, 0]], dtype=np.int32)
        result = replace_with_normal_events(
            anomalous,
            normal,
            cited_positions=[[2, 4]],
            pad_id=0,
            padding="pre",
        )
        self.assertIn("normal_block_event_replacement", available_perturbation_strategies())
        self.assertEqual(result.inputs.tolist(), [[0, 0, 7, 5, 9]])
        replacements = result.references[0]["replacements"]
        self.assertEqual([item["active_position"] for item in replacements], [0, 2])
        self.assertEqual([item["replacement_event_id"] for item in replacements], [7, 9])
        self.assertTrue(result.metadata["sequence_length_preserved"])
        self.assertTrue(result.metadata["chronology_preserved"])

    def test_normal_replacement_records_missing_counterparts_without_fallback(self):
        anomalous = np.asarray([[4, 5, 6, 0]], dtype=np.int32)
        normal = np.asarray([[7, 8, 0, 0]], dtype=np.int32)
        result = replace_with_normal_events(
            anomalous,
            normal,
            cited_positions=[[0, 2]],
            pad_id=0,
            padding="post",
        )
        self.assertEqual(result.inputs.tolist(), [[7, 5, 6, 0]])
        replacements = result.references[0]["replacements"]
        self.assertTrue(replacements[0]["replacement_available"])
        self.assertFalse(replacements[1]["replacement_available"])
        self.assertIsNone(replacements[1]["replacement_event_id"])
        self.assertEqual(result.metadata["replacement_unavailable_count"], 1)

    def test_normal_replacement_requires_comparison_rows(self):
        with self.assertRaisesRegex(ValueError, "one comparison row"):
            replace_with_normal_events(
                np.asarray([[1, 0], [2, 0]], dtype=np.int32),
                np.asarray([[3, 0]], dtype=np.int32),
                cited_positions=[[0], [0]],
                pad_id=0,
                padding="post",
            )

    def test_unknown_token_replacement_preserves_positions_and_width(self):
        inputs = np.asarray([[0, 0, 4, 5, 6], [4, 5, 0, 0, 0]], dtype=np.int32)
        result = replace_with_unknown_token(
            inputs,
            cited_positions=[[2, 4], [1]],
            unknown_id=9,
            pad_id=0,
            padding="pre",
        )
        self.assertIn("unknown_token_replacement", available_perturbation_strategies())
        self.assertEqual(result.inputs.tolist(), [[0, 0, 9, 5, 9], [4, 9, 0, 0, 0]])
        self.assertEqual(result.references[0]["replacements"][0]["original_event_id"], 4)
        self.assertEqual(result.references[0]["replacements"][0]["replacement_event_id"], 9)
        self.assertTrue(result.metadata["sequence_length_preserved"])

    def test_unknown_token_replacement_rejects_pad_token(self):
        with self.assertRaisesRegex(ValueError, "differ from pad_id"):
            replace_with_unknown_token(
                np.asarray([[1, 0]], dtype=np.int32),
                cited_positions=[[0]],
                unknown_id=0,
                pad_id=0,
                padding="post",
            )

    def test_normal_resampling_is_seeded_and_uses_only_normal_active_events(self):
        anomalous = np.asarray([[0, 0, 4, 5, 6]], dtype=np.int32)
        normal = np.asarray([[0, 7, 8, 9, 0]], dtype=np.int32)
        first = resample_from_normal_events(
            anomalous,
            normal,
            cited_positions=[[2, 4]],
            seed=2026,
            pad_id=0,
            padding="pre",
            metadata={"unit_id": "block-1"},
        )
        second = resample_from_normal_events(
            anomalous,
            normal,
            cited_positions=[[2, 4]],
            seed=2026,
            pad_id=0,
            padding="pre",
            metadata={"unit_id": "block-1"},
        )
        self.assertEqual(first.inputs.tolist(), second.inputs.tolist())
        self.assertIn("normal_block_event_resampling", available_perturbation_strategies())
        self.assertIn(first.inputs[0, 2], {7, 8, 9})
        self.assertIn(first.inputs[0, 4], {7, 8, 9})
        self.assertEqual(first.metadata["seed"], 2026)
        self.assertTrue(first.metadata["sequence_length_preserved"])

    def test_normal_resampling_records_with_replacement_when_pool_is_small(self):
        result = resample_from_normal_events(
            np.asarray([[4, 5, 6, 0]], dtype=np.int32),
            np.asarray([[7, 0, 0, 0]], dtype=np.int32),
            cited_positions=[[0, 1, 2]],
            seed=2026,
            pad_id=0,
            padding="post",
        )
        self.assertEqual(result.inputs.shape, (1, 4))
        self.assertEqual(result.metadata["rows_sampled_with_replacement"], 1)
        self.assertTrue(all(item["sampling_with_replacement"] for item in result.references[0]["replacements"]))

    def test_normal_resampling_leaves_targets_when_normal_pool_is_empty(self):
        result = resample_from_normal_events(
            np.asarray([[4, 5, 0]], dtype=np.int32),
            np.asarray([[0, 0, 0]], dtype=np.int32),
            cited_positions=[[0, 1]],
            seed=2026,
            pad_id=0,
            padding="post",
        )
        self.assertEqual(result.inputs.tolist(), [[4, 5, 0]])
        self.assertEqual(result.metadata["replacement_unavailable_count"], 2)
        self.assertFalse(result.references[0]["replacements"][0]["replacement_available"])

    def test_parameter_masking_preserves_template_and_sequence(self):
        inputs = np.asarray([[0, 4, 5, 0]], dtype=np.int32)
        evidence = [[
            {
                "Position": 0,
                "EventId": "E1",
                "EventTemplate": "Receiving block <*>",
                "ParameterList": ["blk-1"],
                "ParameterListRaw": "['blk-1']",
                "RawLine": "Receiving block blk-1",
            },
            {
                "Position": 1,
                "EventId": "E2",
                "EventTemplate": "Packet <*>",
                "ParameterList": ["7", "host-a"],
                "ParameterListRaw": "['7', 'host-a']",
                "RawLine": "Packet 7 host-a",
            },
        ]]
        result = mask_event_parameters(
            inputs,
            evidence,
            cited_positions=[[1]],
            pad_id=0,
            padding="post",
        )
        self.assertIn("parameter_masking", available_perturbation_strategies())
        self.assertEqual(result.inputs.tolist(), inputs.tolist())
        masked = result.evidence[0][0]
        untouched = result.evidence[0][1]
        self.assertEqual(masked["EventTemplate"], "Receiving block <*>")
        self.assertEqual(masked["ParameterList"], ["<MASKED_PARAMETER>"])
        self.assertEqual(masked["RawLine"], "Receiving block <MASKED_PARAMETER>")
        self.assertEqual(untouched["ParameterList"], ["7", "host-a"])
        self.assertEqual(result.metadata["masked_parameter_count"], 1)
        self.assertTrue(result.metadata["event_templates_preserved"])

    def test_parameter_masking_supports_lowercase_evidence_fields(self):
        result = mask_event_parameters(
            np.asarray([[4, 0]], dtype=np.int32),
            [[
                {
                    "position": 0,
                    "event_id": "E1",
                    "event_template": "Read <*>",
                    "parameter_list": ["path"],
                    "parameter_list_raw": "['path']",
                    "raw_line": "Read path",
                }
            ]],
            cited_positions=[[0]],
            pad_id=0,
            padding="post",
            mask_token="<PARAM>",
        )
        self.assertEqual(result.evidence[0][0]["parameter_list"], ["<PARAM>"])
        self.assertEqual(result.evidence[0][0]["raw_line"], "Read <PARAM>")

    def test_parameter_masking_requires_evidence(self):
        with self.assertRaisesRegex(ValueError, "requires event_evidence"):
            apply_perturbation(
                "parameter_masking",
                np.asarray([[1, 0]], dtype=np.int32),
                pad_id=0,
                padding="post",
                target_positions=[[0]],
            )

    def test_matched_random_deletion_preserves_event_budget_and_seed(self):
        inputs = np.asarray([[0, 0, 4, 5, 6]], dtype=np.int32)
        first = apply_matched_random_perturbation(
            "deletion",
            inputs,
            cited_positions=[[2, 4]],
            seed=2026,
            pad_id=0,
            padding="pre",
            metadata={"unit_id": "block-1"},
        )
        second = apply_matched_random_perturbation(
            "deletion",
            inputs,
            cited_positions=[[2, 4]],
            seed=2026,
            pad_id=0,
            padding="pre",
            metadata={"unit_id": "block-1"},
        )
        self.assertEqual(first.inputs.tolist(), second.inputs.tolist())
        self.assertEqual(len(first.references[0]["deleted_positions"]), 2)
        self.assertEqual(first.metadata["control"], "matched_random")
        self.assertEqual(first.metadata["matched_random_seed"], 2026)
        self.assertEqual(len(first.metadata["random_positions"][0]), 2)

    def test_matched_random_positions_are_active_and_budget_matched(self):
        inputs = np.asarray([[0, 4, 5, 0], [6, 0, 0, 0]], dtype=np.int32)
        positions, overlap = sample_matched_random_positions(
            inputs,
            [[1], [0]],
            pad_id=0,
            seed=2026,
            strategy="unknown_token_replacement",
        )
        self.assertEqual([len(row) for row in positions], [1, 1])
        self.assertIn(positions[0][0], {1, 2})
        self.assertEqual(positions[1], [0])
        self.assertGreaterEqual(overlap, 0)

    def test_matched_random_unknown_replacement_uses_same_strategy(self):
        result = apply_matched_random_perturbation(
            "unknown_token_replacement",
            np.asarray([[4, 5, 6, 0]], dtype=np.int32),
            cited_positions=[[0, 2]],
            seed=2026,
            pad_id=0,
            padding="post",
            unknown_id=9,
        )
        self.assertEqual(result.inputs.shape, (1, 4))
        self.assertEqual(len(result.references[0]["replacements"]), 2)
        self.assertTrue(all(item["replacement_event_id"] == 9 for item in result.references[0]["replacements"]))

    def test_common_metadata_records_length_vocabulary_chronology_and_ood(self):
        result = delete_cited_events(
            np.asarray([[0, 4, 5, 0]], dtype=np.int32),
            cited_positions=[[1]],
            pad_id=0,
            padding="post",
        )
        self.assertEqual(result.metadata["metadata_schema_version"], "1.0")
        self.assertEqual(result.metadata["sequence_length_change"], [-1])
        self.assertEqual(result.metadata["vocabulary_change"], [True])
        self.assertEqual(result.metadata["chronology_change"], [False])
        self.assertEqual(result.metadata["parser_status"], ["not_evaluated"])
        self.assertEqual(result.metadata["out_of_distribution_indicator"], [True])
        self.assertIn("sequence_length_change", result.metadata["out_of_distribution_reasons"][0])

    def test_metadata_marks_parameter_masking_as_unreparsed_evidence(self):
        result = mask_event_parameters(
            np.asarray([[4, 0]], dtype=np.int32),
            [[{"position": 0, "event_template": "Read <*>", "parameter_list": ["path"]}]],
            cited_positions=[[0]],
            pad_id=0,
            padding="post",
        )
        self.assertEqual(result.metadata["parser_status"], ["preserved_unreparsed"])
        self.assertEqual(result.metadata["parser_valid"], [None])
        self.assertEqual(result.metadata["out_of_distribution_indicator"], [False])

    def test_custom_strategy_uses_same_validation_and_cannot_change_width(self):
        strategy_name = "test_identity_strategy"

        def identity(context):
            return PerturbationBatch(
                inputs=context.inputs,
                references=tuple({"row_index": index} for index in range(context.inputs.shape[0])),
            )

        register_perturbation_strategy(strategy_name, identity)
        inputs = np.asarray([[1, 2, 0], [3, 0, 0]], dtype=np.int32)
        result = apply_perturbation(strategy_name, inputs, pad_id=0, padding="post")
        self.assertEqual(result.inputs.tolist(), inputs.tolist())
        self.assertEqual([reference["row_index"] for reference in result.references], [0, 1])

        def wrong_width(context):
            return PerturbationBatch(inputs=np.zeros((1, 1), dtype=np.int32), references=({"row_index": 0},))

        wrong_name = "test_wrong_width_strategy"
        register_perturbation_strategy(wrong_name, wrong_width)
        with self.assertRaisesRegex(ValueError, "changed sequence width"):
            apply_perturbation(wrong_name, inputs, pad_id=0, padding="post")

    def test_invalid_strategy_and_target_positions_fail_early(self):
        with self.assertRaisesRegex(ValueError, "unknown perturbation strategy"):
            apply_perturbation("missing", np.asarray([[1]], dtype=np.int32), pad_id=0, padding="post")
        with self.assertRaisesRegex(ValueError, "padding or out-of-range"):
            apply_perturbation(
                "single_event_deletion",
                np.asarray([[1, 0]], dtype=np.int32),
                pad_id=0,
                padding="post",
                target_positions=[[1]],
            )


if __name__ == "__main__":
    unittest.main()
