import json
import tempfile
import unittest
from pathlib import Path

try:
    from run_mistral_pilot import TransparentProgress
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.run_mistral_pilot import TransparentProgress


class MistralPilotProgressTests(unittest.TestCase):
    def test_progress_writes_machine_readable_status_and_failure_state(self):
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "status.json"
            reporter = TransparentProgress(
                status_path,
                output_path=Path(directory) / "generation.jsonl",
                model="test-model",
                max_items=None,
                conditions=("blinded",),
                repetitions=3,
            )
            reporter({"event": "preflight_failed", "error": "missing attribution", "status_counts": {"complete": 0, "skipped": 0, "failed": 0}})
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["step"], "47A")
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["error"], "missing attribution")
            self.assertIsNone(status["max_items"])


if __name__ == "__main__":
    unittest.main()
