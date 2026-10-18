import json
import tempfile
import unittest
from pathlib import Path

try:
    from reference_floor import summarize_reference_levels
except ImportError:  # pragma: no cover
    from research.scripts.reference_floor import summarize_reference_levels


class ReferenceFloorTests(unittest.TestCase):
    def test_reports_attribution_reference_and_random_floor(self):
        report = summarize_reference_levels(
            [
                {"condition": "attribution_only", "architecture": "lstm", "strategy": "deletion", "top_k": 5, "metric_value": 0.8},
                {"condition": "attribution_only", "architecture": "lstm", "strategy": "deletion", "top_k": 5, "metric_value": 0.6},
                {"condition": "random_event", "architecture": "lstm", "strategy": "deletion", "top_k": 5, "metric_value": 0.2},
                {"condition": "random_event", "architecture": "lstm", "strategy": "deletion", "top_k": 5, "metric_value": 0.4},
                {"condition": "unblinded", "architecture": "lstm", "strategy": "deletion", "top_k": 5, "metric_value": 0.7},
            ]
        )
        self.assertTrue(report["evaluated"])
        self.assertFalse(report["attribution_only_is_upper_bound"])
        self.assertFalse(report["human_evidence_metrics_used"])
        self.assertEqual(report["strata_count"], 1)
        stratum = report["strata"][0]
        self.assertAlmostEqual(stratum["attribution_only_reference"]["mean"], 0.7)
        self.assertAlmostEqual(stratum["random_floor"]["mean"], 0.3)
        self.assertEqual(stratum["random_floor"]["count"], 2)
        self.assertAlmostEqual(stratum["conditions"]["unblinded"]["delta_vs_random_floor"], 0.4)
        self.assertAlmostEqual(stratum["conditions"]["unblinded"]["relative_to_random_floor"], 1.0)

    def test_keeps_detector_and_strategy_strata_separate(self):
        report = summarize_reference_levels(
            [
                {"condition": "attribution_only", "architecture": "lstm", "strategy": "deletion", "metric_value": 0.8},
                {"condition": "random_event", "architecture": "lstm", "strategy": "deletion", "metric_value": 0.2},
                {"condition": "attribution_only", "architecture": "bilstm", "strategy": "deletion", "metric_value": 0.7},
                {"condition": "random_event", "architecture": "bilstm", "strategy": "deletion", "metric_value": 0.1},
                {"condition": "attribution_only", "architecture": "lstm", "strategy": "resampling", "metric_value": 0.6},
                {"condition": "random_event", "architecture": "lstm", "strategy": "resampling", "metric_value": 0.3},
            ]
        )
        self.assertEqual(report["strata_count"], 3)
        strata = {(item["stratum"]["architecture"], item["stratum"]["strategy"]) for item in report["strata"]}
        self.assertEqual(strata, {("lstm", "deletion"), ("bilstm", "deletion"), ("lstm", "resampling")})

    def test_missing_reference_or_floor_is_explicit(self):
        report = summarize_reference_levels(
            [
                {"condition": "attribution_only", "metric_value": 0.5},
                {"condition": "unblinded", "metric_value": 0.6},
            ]
        )
        self.assertTrue(report["strata"][0]["reference_available"])
        self.assertFalse(report["strata"][0]["random_floor_available"])
        self.assertIsNone(report["strata"][0]["conditions"]["unblinded"]["delta_vs_random_floor"])

    def test_method_and_score_aliases_are_supported(self):
        report = summarize_reference_levels(
            [
                {"method": "attribution_only", "score": 0.8},
                {"method": "random_event", "score": 0.2},
            ]
        )
        self.assertEqual(report["strata"][0]["attribution_only_reference"]["mean"], 0.8)
        self.assertEqual(report["strata"][0]["random_floor"]["mean"], 0.2)

    def test_cli_writes_json_report(self):
        try:
            from reference_floor import main
        except ImportError:  # pragma: no cover
            from research.scripts.reference_floor import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.json"
            output = root / "report.json"
            source.write_text(
                json.dumps([
                    {"condition": "attribution_only", "metric_value": 0.8},
                    {"condition": "random_event", "metric_value": 0.2},
                ]),
                encoding="utf-8",
            )
            self.assertEqual(main(["--input", str(source), "--output", str(output)]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["metric"], "metric_value")
            self.assertEqual(payload["strata_count"], 1)


if __name__ == "__main__":
    unittest.main()
