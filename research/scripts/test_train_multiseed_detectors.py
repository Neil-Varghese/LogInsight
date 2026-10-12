import json
import tempfile
import unittest
from pathlib import Path

from train_multiseed_detectors import (
    RunTarget,
    build_training_command,
    required_artifacts,
    run_multiseed,
    validate_run_artifacts,
)


class MultiSeedTrainingTests(unittest.TestCase):
    def test_command_uses_current_python_and_seed_override(self):
        command = build_training_command(RunTarget("loginsight_lstm", 42), Path("/repo"))
        self.assertEqual(command[-2:], ["--seed", "42"])
        self.assertTrue(command[1].endswith("train_loginsight_lstm.py"))

    def test_artifact_validation_requires_all_split_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "research/models/checkpoints/loginsight_lstm"
            output.mkdir(parents=True)
            for path in required_artifacts("loginsight_lstm", 13, root):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.name.startswith("run_manifest"):
                    path.write_text(json.dumps({
                        "architecture": "loginsight_lstm",
                        "seed": 13,
                        "split_counts": {"train": 2, "validation": 1, "test": 1},
                        "metrics": {"test": {"f1": 0.5}},
                    }))
                else:
                    path.write_text("artifact")
            summary = validate_run_artifacts(RunTarget("loginsight_lstm", 13), root)
            self.assertEqual(summary["seed"], 13)
            (output / "predictions_test.jsonl.gz").unlink()
            with self.assertRaisesRegex(ValueError, "missing artifacts"):
                validate_run_artifacts(RunTarget("loginsight_lstm", 13), root)

    def test_dry_run_reuses_existing_seed_and_plans_missing_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "research/models/checkpoints/loginsight_lstm"
            for path in required_artifacts("loginsight_lstm", 13, root):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.name.startswith("run_manifest"):
                    path.write_text(json.dumps({
                        "architecture": "loginsight_lstm",
                        "seed": 13,
                        "split_counts": {"train": 2, "validation": 1, "test": 1},
                        "metrics": {"test": {}},
                    }))
                else:
                    path.write_text("artifact")
            report = root / "report.json"
            payload = run_multiseed(
                root,
                ("loginsight_lstm",),
                (13, 42),
                report,
                dry_run=True,
            )
            self.assertEqual(payload["status"], "dry_run")
            self.assertEqual(payload["targets"]["loginsight_lstm:seed_13"]["status"], "planned_existing")
            self.assertEqual(payload["targets"]["loginsight_lstm:seed_42"]["status"], "planned_training")


if __name__ == "__main__":
    unittest.main()
