"""Retrieve normal contrastive blocks from the training split only.

The retriever fits an event-ID TF-IDF index on normal training blocks and
returns deterministic cosine-similarity matches. Query labels are never read,
and candidate labels are used only to enforce the normal-source restriction.
Retrieval provenance is written separately from the prompt payload.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


SCHEMA_VERSION = "1.0"
SIMILARITY_METHOD = "event_tfidf_cosine"


def _open_jsonl(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with _open_jsonl(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"record in {path} at line {line_number} must be an object")
            yield record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_id(record: dict[str, Any], path: Path) -> str:
    value = record.get("BlockId", record.get("unit_id"))
    if value in (None, ""):
        raise ValueError(f"{path} contains a record without BlockId/unit_id")
    return str(value)


def _sequence(record: dict[str, Any], path: Path) -> list[str]:
    sequence = record.get("Sequence")
    if not isinstance(sequence, list) or not sequence or any(not isinstance(event_id, str) or not event_id for event_id in sequence):
        raise ValueError(f"{path} contains a record with an invalid non-empty Sequence list")
    return sequence


def _label(record: dict[str, Any], path: Path) -> int:
    try:
        label = int(record["LabelId"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} contains a record without a valid LabelId") from error
    if label not in (0, 1):
        raise ValueError(f"{path} contains a non-binary LabelId")
    return label


class TrainingNormalRetriever:
    """A deterministic TF-IDF retriever fitted exclusively on normal train blocks."""

    def __init__(self, training_path: Path, *, source_split: str = "train") -> None:
        if source_split != "train":
            raise ValueError("normal contrastive retrieval is restricted to the training split")
        self.training_path = Path(training_path)
        self.source_split = source_split
        self._unit_ids: list[str] = []
        self._lengths: list[int] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._fitted = False

    def fit(self) -> "TrainingNormalRetriever":
        documents: list[str] = []
        seen_units: set[str] = set()
        for record in iter_records(self.training_path):
            if _label(record, self.training_path) != 0:
                continue
            unit_id = _unit_id(record, self.training_path)
            if unit_id in seen_units:
                raise ValueError(f"{self.training_path} contains duplicate normal unit {unit_id!r}")
            seen_units.add(unit_id)
            sequence = _sequence(record, self.training_path)
            self._unit_ids.append(unit_id)
            self._lengths.append(len(sequence))
            documents.append(" ".join(sequence))
        if not documents:
            raise ValueError(f"{self.training_path} contains no normal training blocks")
        self._vectorizer = TfidfVectorizer(
            lowercase=False,
            norm="l2",
            smooth_idf=True,
            sublinear_tf=True,
            tokenizer=str.split,
            token_pattern=None,
        )
        self._matrix = self._vectorizer.fit_transform(documents)
        self._fitted = True
        return self

    @property
    def candidate_count(self) -> int:
        return len(self._unit_ids)

    def retrieve(self, record: dict[str, Any], *, retrieval_k: int = 1) -> list[dict[str, Any]]:
        if not self._fitted or self._vectorizer is None or self._matrix is None:
            raise RuntimeError("fit() must be called before retrieve()")
        if retrieval_k < 1 or retrieval_k > self.candidate_count:
            raise ValueError("retrieval_k must be between 1 and the number of normal candidates")
        sequence = _sequence(record, self.training_path)
        query = self._vectorizer.transform([" ".join(sequence)])
        scores = np.asarray((self._matrix @ query.T).toarray()).reshape(-1)
        ranked = sorted(range(self.candidate_count), key=lambda index: (-float(scores[index]), self._unit_ids[index]))
        zero_overlap = bool(np.all(scores == 0.0))
        results = []
        for rank, index in enumerate(ranked[:retrieval_k], start=1):
            results.append(
                {
                    "candidate_rank": rank,
                    "candidate_unit_id": self._unit_ids[index],
                    "candidate_sequence_length": self._lengths[index],
                    "similarity": float(scores[index]),
                    "zero_similarity_fallback": zero_overlap,
                }
            )
        return results


def build_retrieval_ledger(
    *,
    retriever: TrainingNormalRetriever,
    query_records: Iterable[dict[str, Any]],
    output: Path,
    query_split: str,
    retrieval_k: int = 1,
    query_path: Path | None = None,
) -> dict[str, Any]:
    if query_split == "train":
        raise ValueError("query_split must identify a held-out split, not train")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_queries: set[str] = set()
    record_count = 0
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        for query_record in query_records:
            query_unit_id = _unit_id(query_record, retriever.training_path)
            if query_unit_id in seen_queries:
                raise ValueError(f"query records contain duplicate unit {query_unit_id!r}")
            seen_queries.add(query_unit_id)
            sequence = _sequence(query_record, retriever.training_path)
            for match in retriever.retrieve(query_record, retrieval_k=retrieval_k):
                ledger_record = {
                    "schema_version": SCHEMA_VERSION,
                    "query_split": query_split,
                    "query_unit_id": query_unit_id,
                    "query_sequence_length": len(sequence),
                    "retrieval_method": SIMILARITY_METHOD,
                    "source_split": retriever.source_split,
                    "candidate_rank": match["candidate_rank"],
                    "candidate_unit_id": match["candidate_unit_id"],
                    "candidate_sequence_length": match["candidate_sequence_length"],
                    "similarity": match["similarity"],
                    "zero_similarity_fallback": match["zero_similarity_fallback"],
                    "provenance": {
                        "source_evidence_path": str(retriever.training_path.resolve()),
                        "candidate_label_restriction": "LabelId == 0",
                        "query_labels_used_for_retrieval": False,
                        "attribution_used_for_retrieval": False,
                        "retrieval_k": retrieval_k,
                    },
                }
                handle.write(json.dumps(ledger_record, ensure_ascii=False, sort_keys=True) + "\n")
                record_count += 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "training_only_normal_retrieval",
        "status": "complete",
        "similarity_method": SIMILARITY_METHOD,
        "source_split": retriever.source_split,
        "source_evidence_path": str(retriever.training_path.resolve()),
        "source_evidence_sha256": sha256(retriever.training_path),
        "query_split": query_split,
        "query_evidence_path": str(query_path.resolve()) if query_path else None,
        "retrieval_k": retrieval_k,
        "normal_candidate_count": retriever.candidate_count,
        "ledger_record_count": record_count,
        "query_labels_used_for_retrieval": False,
        "attribution_used_for_retrieval": False,
        "tie_break": "highest cosine similarity, then lowest candidate_unit_id",
        "zero_similarity_fallback": "lowest candidate_unit_id",
        "ledger_path": str(output.resolve()),
        "ledger_sha256": sha256(output),
    }
    manifest_path = output.with_name(output.name.replace(".jsonl.gz", "_manifest.json"))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--query-evidence", type=Path, required=True)
    parser.add_argument("--query-split", choices=["validation", "test", "pilot"], default="test")
    parser.add_argument("--retrieval-k", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    retriever = TrainingNormalRetriever(args.training_evidence).fit()
    records: Iterable[dict[str, Any]] = iter_records(args.query_evidence)
    if args.max_queries is not None:
        if args.max_queries < 1:
            raise ValueError("--max-queries must be positive")
        records = (record for index, record in enumerate(records) if index < args.max_queries)
    manifest = build_retrieval_ledger(
        retriever=retriever,
        query_records=records,
        output=args.output,
        query_split=args.query_split,
        retrieval_k=args.retrieval_k,
        query_path=args.query_evidence,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
