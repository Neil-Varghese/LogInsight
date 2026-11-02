"""Build the blinded Step 55 ID-memorization probe from the real pool."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from annotation_blinding import build_id_memorization_probe, validate_blinded_packet
    from annotation_packet import build_annotation_packet
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.annotation_blinding import build_id_memorization_probe, validate_blinded_packet
    from research.scripts.annotation_packet import build_annotation_packet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected an object at {path}:{line_number}")
                records.append(value)
    return records


def build_probe(
    pool_path: Path,
    evidence_path: Path,
    public_output: Path,
    answer_key_output: Path,
    *,
    pair_count: int = 20,
    seed: int = 2026,
) -> dict[str, Any]:
    pool = _read_jsonl(pool_path)
    selected_units = list(dict.fromkeys(str(item["unit_id"]) for item in pool))
    evidence_by_unit = {}
    for record in _read_jsonl(evidence_path):
        unit_id = str(record.get("BlockId", ""))
        if unit_id in selected_units:
            evidence_by_unit[unit_id] = record
    missing = [unit_id for unit_id in selected_units if unit_id not in evidence_by_unit]
    if missing:
        raise ValueError(f"annotation evidence is missing {len(missing)} selected units; first={missing[:3]}")

    packets = []
    excluded_units = []
    for unit_id in selected_units:
        source = evidence_by_unit[unit_id]
        packet = build_annotation_packet({"dataset": "HDFS", "unit_id": unit_id, "events": source["Evidence"]})
        if not packet["context"]["annotation_eligible"]:
            excluded_units.append(unit_id)
            continue
        validate_blinded_packet(packet)
        packets.append(packet)
    if len(packets) < 2:
        raise ValueError("fewer than two selected units remain after context-cap eligibility filtering")
    probe = build_id_memorization_probe(packets, pair_count=pair_count, seed=seed)
    answer_key = {
        "schema_version": probe["schema_version"],
        "probe_type": probe["probe_type"],
        "seed": probe["seed"],
        "answer_key": probe.pop("answer_key"),
        "source_pool_sha256": _sha256(pool_path),
        "source_evidence_sha256": _sha256(evidence_path),
    }
    public = dict(probe)
    public["answer_key_file"] = str(answer_key_output)
    public["source_unit_count"] = len(selected_units)
    public["eligible_unit_count"] = len(packets)
    public["excluded_context_capped_unit_count"] = len(excluded_units)
    public["source_pool_sha256"] = answer_key["source_pool_sha256"]
    public["source_evidence_sha256"] = answer_key["source_evidence_sha256"]
    public_output.parent.mkdir(parents=True, exist_ok=True)
    answer_key_output.parent.mkdir(parents=True, exist_ok=True)
    public_output.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    answer_key_output.write_text(json.dumps(answer_key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return public


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-pool", type=Path, default=Path("research/experiments/pilot/annotation_pool.jsonl.gz"))
    parser.add_argument("--evidence", type=Path, default=Path("research/data/processed/hdfs_block_evidence_test.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/step55_id_memorization_probe.json"))
    parser.add_argument("--answer-key-output", type=Path, default=Path("research/results/tables/step55_id_memorization_answer_key.json"))
    parser.add_argument("--pair-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        result = build_probe(args.annotation_pool, args.evidence, args.output, args.answer_key_output, pair_count=args.pair_count, seed=args.seed)
    except Exception as error:
        print(f"Step 55 probe build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "output": str(args.output), "answer_key": str(args.answer_key_output), "pair_count": result["manifest"]["pair_count"], "source_unit_count": result["source_unit_count"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
