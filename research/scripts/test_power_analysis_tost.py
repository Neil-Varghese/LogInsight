import json
import tempfile
import unittest
from pathlib import Path

from power_analysis_tost import build_report, required_pairs


class TostPowerAnalysisTests(unittest.TestCase):
    def test_proposal_default_matches_approximately_71_pairs(self):
        self.assertEqual(required_pairs(0.15), 71)

    def test_default_sd_sensitivity_table(self):
        report = build_report()
        self.assertEqual(report["step"], 61)
        self.assertEqual(
            [row["required_independent_pairs"] for row in report["results"]],
            [71, 126, 197, 283, 385],
        )

    def test_parameterization_changes_requirement(self):
        self.assertGreater(
            required_pairs(0.20, margin=0.04),
            required_pairs(0.20, margin=0.05),
        )

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            required_pairs(0.0)
        with self.assertRaises(ValueError):
            required_pairs(0.15, alpha=1.0)
        with self.assertRaises(ValueError):
            required_pairs(0.15, target_power=1.0)

    def test_report_can_be_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(build_report()), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["analysis"], "paired_tost_sample_size_planning")


if __name__ == "__main__":
    unittest.main()
