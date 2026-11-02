import json
import unittest
from pathlib import Path

try:
    from r7_audit import (
        _ledger_fixture,
        _validate_required_fields,
        audit_annotation_export,
        audit_ledger,
        audit_metrics,
        audit_prompt_boundary,
        audit_serialized_attributions,
    )
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.r7_audit import (
        _ledger_fixture,
        _validate_required_fields,
        audit_annotation_export,
        audit_ledger,
        audit_metrics,
        audit_prompt_boundary,
        audit_serialized_attributions,
    )


ROOT = Path(__file__).resolve().parents[1]


class R7AuditTests(unittest.TestCase):
    def test_golden_prompt_boundary_covers_all_registered_conditions(self):
        result = audit_prompt_boundary(ROOT / "experiments/experiment_config.json")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["condition_count"], 9)
        self.assertEqual(sum(item["status"] == "passed" for item in result["conditions"]), 8)

    def test_annotation_export_is_attribution_free(self):
        result = audit_annotation_export()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["event_count"], 2)

    def test_ledger_audit_checks_required_fields_and_rejection(self):
        result = audit_ledger(ROOT / "experiments/experiment_ledger_schema.json")
        self.assertEqual(result["status"], "passed")
        schema = json.loads((ROOT / "experiments/experiment_ledger_schema.json").read_text())
        record = _ledger_fixture()
        del record["quality"]
        with self.assertRaises(ValueError):
            _validate_required_fields(record, schema, schema)

    def test_independent_metrics_and_serialized_attribution_checks(self):
        metrics = audit_metrics(
            ROOT / "models/checkpoints/loginsight_lstm/thresholded/thresholded_predictions_test.jsonl.gz",
            ROOT / "results/tables/lstm_baseline_seed_13.json",
        )
        attribution = audit_serialized_attributions(
            ROOT / "models/attributions/loginsight_lstm/seed_13_roar_test_integrated_gradients.jsonl.gz",
            limit=2,
        )
        self.assertEqual(metrics["status"], "passed")
        self.assertEqual(attribution["status"], "passed")
        self.assertEqual(attribution["records_checked"], 2)


if __name__ == "__main__":
    unittest.main()
