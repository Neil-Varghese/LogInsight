"""Build the public blinded packet bundle for Step 57 evidence labeling."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from annotation_blinding import validate_blinded_packet
    from annotation_packet import build_annotation_packet
except ImportError:  # pragma: no cover
    from research.scripts.annotation_blinding import validate_blinded_packet
    from research.scripts.annotation_packet import build_annotation_packet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def build_packets(pool_path: Path, evidence_path: Path, output_path: Path) -> dict[str, Any]:
    pool = read_jsonl(pool_path)
    selected_units = list(dict.fromkeys(str(item["unit_id"]) for item in pool))
    evidence_by_unit = {
        str(record.get("BlockId")): record
        for record in read_jsonl(evidence_path)
        if record.get("BlockId") is not None
    }
    missing = [unit_id for unit_id in selected_units if unit_id not in evidence_by_unit]
    if missing:
        raise ValueError(f"evidence is missing {len(missing)} selected units; first={missing[:3]}")

    packets: list[dict[str, Any]] = []
    excluded_count = 0
    for unit_id in selected_units:
        source = evidence_by_unit[unit_id]
        packet = build_annotation_packet(
            {"dataset": "HDFS", "unit_id": unit_id, "events": source["Evidence"]}
        )
        if not packet["context"]["annotation_eligible"]:
            excluded_count += 1
            continue
        validate_blinded_packet(packet)
        packets.append(packet)

    if not packets:
        raise ValueError("no annotation-eligible Step 57 packets were built")
    bundle = {
        "schema_version": "1.0",
        "guideline_version": "1.0",
        "packet_type": "evidence_relevance",
        "packets": packets,
        "manifest": {
            "packet_count": len(packets),
            "selected_unique_unit_count": len(selected_units),
            "excluded_context_capped_count": excluded_count,
            "source_pool_sha256": sha256(pool_path),
            "source_evidence_sha256": sha256(evidence_path),
            "source_architecture_metadata_exposed": False,
            "source_label_metadata_exposed": False,
            "source_detector_metadata_exposed": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-pool", type=Path, default=Path("research/experiments/pilot/annotation_pool.jsonl.gz"))
    parser.add_argument("--evidence", type=Path, default=Path("research/data/processed/hdfs_block_evidence_test.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/step57_annotation_packets.json"))
    args = parser.parse_args()
    try:
        bundle = build_packets(args.annotation_pool, args.evidence, args.output)
    except Exception as error:
        print(f"Step 57 packet build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "output": str(args.output), **bundle["manifest"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
