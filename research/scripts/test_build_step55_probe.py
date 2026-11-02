import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    from build_step55_probe import build_probe
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.build_step55_probe import build_probe


class Step55ProbeBuildTests(unittest.TestCase):
    def test_builds_public_probe_and_separate_answer_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = root / "pool.jsonl.gz"
            evidence = root / "evidence.jsonl.gz"
            with gzip.open(pool, "wt", encoding="utf-8") as handle:
                for index in range(3):
                    handle.write(json.dumps({"unit_id": f"blk_{index}"}) + "\n")
            with gzip.open(evidence, "wt", encoding="utf-8") as handle:
                for index in range(3):
                    handle.write(json.dumps({
                        "BlockId": f"blk_{index}",
                        "Evidence": [{
                            "LineId": index,
                            "EventId": "e1",
                            "EventTemplate": "Failure <*>" if index else "Cleanup",
                            "ParameterList": ["secret"],
                            "RawLine": "Failure secret" if index else "Cleanup",
                        }],
                    }) + "\n")
            public = root / "probe.json"
            answer_key = root / "answer_key.json"
            result = build_probe(pool, evidence, public, answer_key, pair_count=4, seed=2026)
            self.assertEqual(result["manifest"]["pair_count"], 4)
            self.assertNotIn("answer_key", json.loads(public.read_text()))
            self.assertEqual(len(json.loads(answer_key.read_text())["answer_key"]), 4)


if __name__ == "__main__":
    unittest.main()
