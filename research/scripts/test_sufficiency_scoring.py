import json
import tempfile
import unittest
from pathlib import Path

try:
    from sufficiency_scoring import score_multi_strategy_sufficiency, score_sufficiency_record
except ImportError:  # pragma: no cover
    from research.scripts.sufficiency_scoring import score_multi_strategy_sufficiency, score_sufficiency_record


class SufficiencyScoringTests(unittest.TestCase):
    def test_gap_and_normalized_gap_follow_registered_definition(self):
        result = score_sufficiency_record(
            {
                "claim_id": "claim-1",
                "strategy": "normal_block_event_replacement",
                "detector_score_original": 0.8,
                "detector_score_retained": 0.6,
            }
        )
        self.assertAlmostEqual(result["sufficiency_gap"], 0.2)
        self.assertAlmostEqual(result["normalized_sufficiency_gap"], 0.25)
        self.assertTrue(result["original_decision"])
        self.assertTrue(result["retained_decision"])
        self.assertEqual(result["sufficiency_conclusion"], "preserved")

    def test_requires_two_strategies_and_excludes_deletion_only(self):
        report = score_multi_strategy_sufficiency(
            [
                {
                    "claim_id": "deletion-only",
                    "strategy": "deletion",
                    "detector_score_original": 0.9,
                    "detector_score_retained": 0.8,
                },
                {
                    "claim_id": "multi",
                    "strategy": "deletion",
                    "detector_score_original": 0.9,
                    "detector_score_retained": 0.8,
                },
                {
                    "claim_id": "multi",
                    "strategy": "unknown_token_replacement",
                    "detector_score_original": 0.9,
                    "detector_score_retained": 0.85,
                },
            ]
        )
        metrics = report["sufficiency"]
        self.assertTrue(metrics["evaluated"])
        self.assertEqual(metrics["claims_tested"], 2)
        self.assertEqual(metrics["eligible_claims"], 1)
        self.assertEqual(metrics["deletion_only_claims_excluded"], 1)
        self.assertEqual(metrics["consistency_rate"], 1.0)
        self.assertFalse(report["claims"]["deletion-only"]["eligible"])
        self.assertTrue(report["claims"]["multi"]["eligible"])
        self.assertFalse(report["human_evidence_metrics_used"])

    def test_agreement_requires_same_conclusion_across_two_distinct_strategies(self):
        report = score_multi_strategy_sufficiency(
            [
                {
                    "claim_id": "agree",
                    "strategy": "normal_block_event_replacement",
                    "detector_score_original": 0.8,
                    "detector_score_retained": 0.7,
                },
                {
                    "claim_id": "agree",
                    "strategy": "normal_block_event_resampling",
                    "detector_score_original": 0.8,
                    "detector_score_retained": 0.6,
                },
                {
                    "claim_id": "disagree",
                    "strategy": "normal_block_event_replacement",
                    "detector_score_original": 0.8,
                    "detector_score_retained": 0.7,
                },
                {
                    "claim_id": "disagree",
                    "strategy": "normal_block_event_resampling",
                    "detector_score_original": 0.8,
                    "detector_score_retained": 0.3,
                },
            ]
        )
        self.assertTrue(report["claims"]["agree"]["agreement_across_at_least_two_strategies"])
        self.assertFalse(report["claims"]["disagree"]["agreement_across_at_least_two_strategies"])
        self.assertEqual(report["sufficiency"]["consistency_rate"], 0.5)

    def test_invalid_parser_result_cannot_make_claim_eligible(self):
        report = score_multi_strategy_sufficiency(
            [
                {
                    "claim_id": "claim-1",
                    "strategy": "deletion",
                    "detector_score_original": 0.9,
                    "detector_score_retained": 0.8,
                },
                {
                    "claim_id": "claim-1",
                    "strategy": "parameter_masking",
                    "detector_score_original": 0.9,
                    "detector_score_retained": 0.8,
                    "parser_valid": False,
                },
            ]
        )
        self.assertFalse(report["claims"]["claim-1"]["eligible"])
        self.assertFalse(report["sufficiency"]["evaluated"])

    def test_cli_reads_json_and_writes_report(self):
        try:
            from sufficiency_scoring import main
        except ImportError:  # pragma: no cover
            from research.scripts.sufficiency_scoring import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sufficiency.json"
            output = root / "report.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "claim_id": "claim-1",
                            "strategy": "unknown_token_replacement",
                            "detector_score_original": 0.7,
                            "detector_score_retained": 0.65,
                        },
                        {
                            "claim_id": "claim-1",
                            "strategy": "normal_block_event_resampling",
                            "detector_score_original": 0.7,
                            "detector_score_retained": 0.66,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(main(["--input", str(source), "--output", str(output)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["metric"], "sufficiency")
            self.assertEqual(report["sufficiency"]["eligible_claims"], 1)


if __name__ == "__main__":
    unittest.main()
