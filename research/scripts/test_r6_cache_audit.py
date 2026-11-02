import json
import tempfile
import unittest
from pathlib import Path

try:
    from r6_cache_audit import run_audit
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.r6_cache_audit import run_audit


ROOT = Path(__file__).resolve().parents[2]


class R6CacheAuditTests(unittest.TestCase):
    def test_cache_audit_passes_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "r6.json"
            report = run_audit(ROOT, output)
            self.assertEqual(report["status"], "passed")
            self.assertTrue(output.exists())
            saved = json.loads(output.read_text())
            self.assertEqual(saved["step"], "R6")
            self.assertEqual(saved["checks"]["collision_probes"]["model_calls"], 4)


if __name__ == "__main__":
    unittest.main()
