import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    from audit_integrated_gradients import build_audit, compare_artifacts, distribution
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.audit_integrated_gradients import build_audit, compare_artifacts, distribution


def _write(path: Path, records, *, steps=64, target="anomaly_probability", randomize_model_seed=None):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    path.with_name(path.name[: -len(".jsonl.gz")] + ".manifest.json").write_text(
        json.dumps({
            "count": len(records),
            "integration_steps": steps,
            "target": target,
            "output_space": "logit" if target == "anomaly_logit" else "probability",
            "architecture": "test_architecture",
            "seed": 7,
            "split": "test",
            "randomize_model_seed": randomize_model_seed,
        }),
        encoding="utf-8",
    )


def _record(unit_id, values, baseline=0.1, error=0.001, target="anomaly_probability"):
    events = []
    for position, value in enumerate(values):
        events.append(
            {
                "input_position": position,
                "event_id": position + 1,
                "is_padding": False,
                "rank": sorted(range(len(values)), key=lambda i: (-abs(values[i]), i)).index(position) + 1,
                "attribution": value,
                "absolute_attribution": abs(value),
            }
        )
    return {
        "schema_version": "1.0",
        "method": "integrated_gradients",
        "target": target,
        "output_space": "logit" if target == "anomaly_logit" else "probability",
        "architecture": "test_architecture",
        "seed": 7,
        "split": "test",
        "unit_id": unit_id,
        "baseline_probability": baseline,
        "completeness_error": error,
        "integration_steps": 64,
        "events": events,
    }


class AuditIntegratedGradientsTests(unittest.TestCase):
    def test_distribution_and_artifact_summary_include_required_distributions(self):
        self.assertEqual(distribution([1.0, 2.0, 3.0])["median"], 2.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "probability.jsonl.gz"
            _write(path, [_record("a", [0.2, -0.1], baseline=0.2), _record("b", [0.3, -0.2], baseline=0.4)])
            report = build_audit(attribution_files=[path], output=root / "report.json")
        summary = report["probability_artifacts"][0]
        self.assertEqual(summary["records"], 2)
        self.assertIn("baseline_probability_distribution", summary)
        self.assertIn("completeness_error_distribution", summary)
        self.assertEqual(len(summary["sha256"]), 64)

    def test_step_stability_and_randomized_sanity_are_computed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trained = root / "trained.jsonl.gz"
            stable = root / "stable.jsonl.gz"
            randomized = root / "randomized.jsonl.gz"
            records = [_record("a", [0.9, 0.1]), _record("b", [-0.8, 0.2])]
            _write(trained, records, steps=16)
            _write(stable, [_record("a", [0.89, 0.11]), _record("b", [-0.79, 0.21])], steps=128)
            _write(
                randomized,
                [_record("a", [0.1, 0.9]), _record("b", [0.2, -0.8])],
                steps=128,
                randomize_model_seed=99,
            )
            report = build_audit(
                attribution_files=[trained],
                step_stability_pairs=[(trained, stable)],
                randomized_model_pairs=[(trained, randomized)],
            )
        self.assertEqual(len(report["step_count_stability"]), 1)
        self.assertEqual(report["randomized_model_sanity"]["status"], "pass")

    def test_coverage_requires_the_same_architectures_for_all_four_audits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probability = root / "probability.jsonl.gz"
            stable = root / "stable.jsonl.gz"
            randomized = root / "randomized.jsonl.gz"
            records = [_record("a", [0.9, 0.1])]
            _write(probability, records, steps=16)
            _write(stable, records, steps=64)
            _write(randomized, [_record("a", [0.1, 0.9])], randomize_model_seed=99)
            report = build_audit(
                attribution_files=[probability],
                step_stability_pairs=[(probability, stable)],
                randomized_model_pairs=[(probability, randomized)],
            )
        self.assertFalse(report["coverage"]["complete"])

    def test_logit_artifacts_are_kept_separate_and_units_must_match_for_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logit = root / "logit.jsonl.gz"
            probability = root / "probability.jsonl.gz"
            _write(logit, [_record("a", [0.2, 0.1], target="anomaly_logit")], target="anomaly_logit")
            _write(probability, [_record("different", [0.2, 0.1])])
            report = build_audit(attribution_files=[probability], logit_attribution_files=[logit])
            self.assertEqual(report["logit_artifacts"][0]["target"], ["anomaly_logit"])
            with self.assertRaises(ValueError):
                compare_artifacts(logit, probability)


if __name__ == "__main__":
    unittest.main()
