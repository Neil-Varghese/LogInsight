import gzip
import json
import tempfile
import unittest
from pathlib import Path

try:
    from training_only_retrieval import TrainingNormalRetriever, build_retrieval_ledger
except ImportError:  # pragma: no cover - package-style test discovery
    from research.scripts.training_only_retrieval import TrainingNormalRetriever, build_retrieval_ledger


def _write(path: Path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _record(unit_id, sequence, label):
    return {"BlockId": unit_id, "Sequence": sequence, "LabelId": label}


class TrainingOnlyRetrievalTests(unittest.TestCase):
    def test_indexes_only_normal_training_blocks_and_ranks_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl.gz"
            _write(train, [
                _record("normal_a", ["a", "b"], 0),
                _record("normal_b", ["c", "d"], 0),
                _record("anomaly", ["a", "b"], 1),
            ])
            retriever = TrainingNormalRetriever(train).fit()
            self.assertEqual(retriever.candidate_count, 2)
            result = retriever.retrieve(_record("query", ["a", "b"], 1))
            self.assertEqual(result[0]["candidate_unit_id"], "normal_a")
            self.assertGreater(result[0]["similarity"], 0.0)

    def test_zero_overlap_uses_documented_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl.gz"
            _write(train, [_record("normal_b", ["b"], 0), _record("normal_a", ["a"], 0)])
            result = TrainingNormalRetriever(train).fit().retrieve(_record("query", ["unknown"], 1))
            self.assertEqual(result[0]["candidate_unit_id"], "normal_a")
            self.assertTrue(result[0]["zero_similarity_fallback"])
            self.assertEqual(result[0]["similarity"], 0.0)

    def test_ledger_records_provenance_without_query_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl.gz"
            query = root / "test.jsonl.gz"
            output = root / "retrieval.jsonl.gz"
            _write(train, [_record("normal", ["a"], 0)])
            _write(query, [_record("query", ["a"], 1)])
            retriever = TrainingNormalRetriever(train).fit()
            manifest = build_retrieval_ledger(
                retriever=retriever,
                query_records=[_record("query", ["a"], 1)],
                output=output,
                query_split="test",
                query_path=query,
            )
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                record = json.loads(next(handle))
            self.assertFalse(record["provenance"]["query_labels_used_for_retrieval"])
            self.assertEqual(record["source_split"], "train")
            self.assertEqual(manifest["ledger_record_count"], 1)


if __name__ == "__main__":
    unittest.main()
