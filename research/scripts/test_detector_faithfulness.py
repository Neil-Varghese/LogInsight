import json
import tempfile
import unittest
from pathlib import Path

try:
    from detector_faithfulness import score_detector_faithfulness, score_perturbation_record
except ImportError:  # pragma: no cover
    from research.scripts.detector_faithfulness import score_detector_faithfulness, score_perturbation_record


class DetectorFaithfulnessTests(unittest.TestCase):
    def test_score_uses_after_minus_before_and_predicted_direction(self):
        result = score_perturbation_record(
            {
                "strategy": "deletion",
                "detector_score_before": 0.90,
                "detector_score_after": 0.20,
                "predicted_effect": "decrease_anomaly_score",
            }
        )
        self.assertAlmostEqual(result["detector_score_delta"], -0.70)
        self.assertEqual(result["observed_effect"], "decrease_anomaly_score")
        self.assertTrue(result["effect_agreement"])
        self.assertTrue(result["counterfactual_valid"])

    def test_multi_strategy_report_keeps_human_outcome_separate(self):
        report = score_detector_faithfulness(
            [
                {
                    "unit_id": "block-1",
                    "strategy": "deletion",
                    "detector_score_before": 0.9,
                    "detector_score_after": 0.2,
                    "predicted_effect": "decrease_anomaly_score",
                    "human_evidence_recovery": {"f1": 1.0},
                },
                {
                    "unit_id": "block-1",
                    "strategy": "normal_block_event_replacement",
                    "detector_score_before": 0.9,
                    "detector_score_after": 0.4,
                    "predicted_effect": "decrease_anomaly_score",
                    "evidence_f1": 0.0,
                },
                {
                    "unit_id": "block-1",
                    "strategy": "random_event_deletion",
                    "detector_score_before": 0.9,
                    "detector_score_after": 0.8,
                    "predicted_effect": "decrease_anomaly_score",
                },
            ]
        )
        metrics = report["detector_faithfulness"]
        self.assertTrue(metrics["evaluated"])
        self.assertEqual(metrics["predicted_direction_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_score_delta"], (-0.7 - 0.5 - 0.1) / 3)
        self.assertAlmostEqual(metrics["random_control_delta"], -0.1)
        self.assertFalse(report["human_evidence_metrics_used"])
        self.assertNotIn("human_evidence_recovery", report)
        self.assertEqual(set(report["by_strategy"]), {"deletion", "normal_block_event_replacement", "random_event_deletion"})

    def test_uncertain_effect_is_excluded_from_direction_accuracy(self):
        report = score_detector_faithfulness(
            [
                {
                    "strategy": "parameter_masking",
                    "detector_score_before": 0.5,
                    "detector_score_after": 0.6,
                    "predicted_effect": "uncertain",
                },
                {
                    "strategy": "parameter_masking",
                    "detector_score_before": 0.5,
                    "detector_score_after": 0.5,
                    "predicted_effect": "no_change",
                },
            ]
        )
        self.assertEqual(report["detector_faithfulness"]["predicted_direction_accuracy"], 1.0)
        self.assertEqual(report["detector_faithfulness"]["counterfactual_validity"], 1.0)

    def test_failed_or_invalid_parser_case_reduces_counterfactual_validity(self):
        report = score_detector_faithfulness(
            [
                {
                    "strategy": "deletion",
                    "detector_score_before": 0.8,
                    "detector_score_after": 0.2,
                    "predicted_effect": "decrease_anomaly_score",
                    "parser_valid": False,
                },
                {
                    "strategy": "deletion",
                    "detector_score_before": 0.8,
                    "detector_score_after": 0.2,
                    "predicted_effect": "decrease_anomaly_score",
                    "status": "completed",
                },
            ]
        )
        self.assertEqual(report["detector_faithfulness"]["counterfactual_validity"], 0.5)
        self.assertEqual(report["detector_faithfulness"]["predicted_direction_accuracy"], 1.0)

    def test_reported_delta_must_match_scores(self):
        with self.assertRaisesRegex(ValueError, "after - before"):
            score_perturbation_record(
                {
                    "strategy": "deletion",
                    "detector_score_before": 0.8,
                    "detector_score_after": 0.2,
                    "detector_score_delta": 0.6,
                }
            )

    def test_cli_input_and_output_are_json_artifacts(self):
        try:
            from detector_faithfulness import main
        except ImportError:  # pragma: no cover
            from research.scripts.detector_faithfulness import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "perturbations.json"
            output = root / "faithfulness.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "strategy": "unknown_token_replacement",
                            "detector_score_before": 0.7,
                            "detector_score_after": 0.5,
                            "predicted_effect": "decrease_anomaly_score",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(main(["--input", str(source), "--output", str(output)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["metric"], "detector_faithfulness")
            self.assertEqual(report["case_count"], 1)


if __name__ == "__main__":
    unittest.main()
