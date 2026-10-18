import json
import tempfile
import unittest
from pathlib import Path

try:
    from select_primary_explanation_seeds import select_primary_seeds
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.select_primary_explanation_seeds import select_primary_seeds


def _write_manifest(path: Path, architecture: str, seed: int, validation_f1: float, validation_loss: float, test_f1: float):
    path.write_text(
        json.dumps(
            {
                "architecture": architecture,
                "seed": seed,
                "checkpoints": {"best": f"/checkpoints/{architecture}/best_seed_{seed}.keras"},
                "metrics": {
                    "validation": {"f1": validation_f1, "loss": validation_loss, "precision": 0.9, "recall": 0.9},
                    "test": {"f1": test_f1},
                },
            }
        ),
        encoding="utf-8",
    )


class PrimaryExplanationSeedTests(unittest.TestCase):
    def test_selects_validation_best_and_keeps_other_seeds_secondary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            for seed, validation_f1, test_f1 in ((13, 0.91, 0.99), (42, 0.95, 0.80), (73, 0.93, 0.70)):
                path = root / f"run_manifest_seed_{seed}.json"
                _write_manifest(path, "lstm", seed, validation_f1, 0.1, test_f1)
                manifests.append(path)
            report = select_primary_seeds(manifests, root / "report.json")
            result = report["architectures"]["lstm"]
            self.assertEqual(result["primary_seed"], 42)
            self.assertEqual(result["secondary_attribution_stability_seeds"], [73, 13])
            self.assertFalse(report["selection_policy"]["test_metrics_used_for_selection"])

    def test_tie_breaks_by_validation_loss_then_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.json"
            second = root / "second.json"
            third = root / "third.json"
            _write_manifest(first, "lstm", 42, 0.95, 0.20, 0.1)
            _write_manifest(second, "lstm", 13, 0.95, 0.10, 0.1)
            _write_manifest(third, "lstm", 7, 0.95, 0.10, 0.1)
            report = select_primary_seeds([first, second, third], root / "report.json")
            self.assertEqual(report["architectures"]["lstm"]["primary_seed"], 7)

    def test_rejects_duplicate_architecture_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.json"
            second = root / "second.json"
            _write_manifest(first, "lstm", 13, 0.95, 0.1, 0.1)
            _write_manifest(second, "lstm", 13, 0.94, 0.2, 0.1)
            with self.assertRaises(ValueError):
                select_primary_seeds([first, second], root / "report.json")


if __name__ == "__main__":
    unittest.main()
