import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_leakage import (  # noqa: E402
    check_parser_policy,
    check_retrieval_ledger,
    check_split_bundle,
)


class LeakageChecksTests(unittest.TestCase):
    def test_split_bundle_accepts_unique_matching_assignments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assignment = root / "assignments.jsonl.gz"
            with gzip.open(assignment, "wt", encoding="utf-8") as handle:
                for unit_id, split in (("a", "train"), ("b", "validation"), ("c", "test")):
                    handle.write(json.dumps({"UnitId": unit_id, "Split": split}) + "\n")
            paths = {}
            for split, unit_id in (("train", "a"), ("validation", "b"), ("test", "c")):
                path = root / f"{split}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    handle.write(json.dumps({"BlockId": unit_id}) + "\n")
                paths[split] = path
            result = check_split_bundle("test", assignment, paths, "BlockId", progress=False)
            self.assertEqual(result["assigned_count"], 3)
            self.assertEqual(result["duplicate_unit_count"], 0)

    def test_split_bundle_rejects_duplicate_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assignment = root / "assignments.jsonl.gz"
            with gzip.open(assignment, "wt", encoding="utf-8") as handle:
                for unit_id, split in (("a", "train"), ("b", "validation"), ("c", "test")):
                    handle.write(json.dumps({"UnitId": unit_id, "Split": split}) + "\n")
            paths = {}
            for split, unit_id in (("train", "a"), ("validation", "a"), ("test", "c")):
                path = root / f"{split}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    handle.write(json.dumps({"BlockId": unit_id}) + "\n")
                paths[split] = path
            with self.assertRaisesRegex(ValueError, "leakage"):
                check_split_bundle("test", assignment, paths, "BlockId", progress=False)

    def test_retrieval_ledger_requires_training_normal_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "retrieval.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"query_unit_id": "q", "normal_unit_id": "candidate", "source_split": "validation"}) + "\n")
            with self.assertRaisesRegex(ValueError, "leakage"):
                check_retrieval_ledger(path, {"candidate"}, progress=False)

    def test_parser_policy_requires_training_only(self):
        config = {
            "splits": {"fit_parser_and_vocabulary_on_train_only": True},
            "parser": {
                "parser_configuration_is_frozen_for_core_benchmark": True,
                "detector_vocabulary": {"fit_scope": "training_only"},
            },
        }
        self.assertEqual(check_parser_policy(config)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
