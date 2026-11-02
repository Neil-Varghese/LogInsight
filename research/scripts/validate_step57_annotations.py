"""Validate a completed Step 57 evidence-relevance annotation export."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from annotation_packet import validate_annotation_export
except ImportError:  # pragma: no cover
    from research.scripts.annotation_packet import validate_annotation_export


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(responses_path: Path, packet_bundle_path: Path, output_path: Path) -> dict[str, Any]:
    responses = load_json(responses_path)
    bundle = load_json(packet_bundle_path)
    if not isinstance(responses, list):
        raise ValueError("responses must be a JSON array")
    if not isinstance(bundle, dict) or not isinstance(bundle.get("packets"), list):
        raise ValueError("packet bundle must contain a packets array")

    expected = {str(packet["packet_id"]): packet for packet in bundle["packets"]}
    seen: set[str] = set()
    label_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    event_count = 0
    for record in responses:
        validated = validate_annotation_export(record)
        packet_id = str(validated["packet_id"])
        if packet_id in seen:
            raise ValueError(f"duplicate response packet: {packet_id}")
        if packet_id not in expected:
            raise ValueError(f"response packet is not in the public bundle: {packet_id}")
        expected_events = expected[packet_id]["events"]
        expected_by_position = {int(event["event_position"]): event for event in expected_events}
        actual_positions = {int(event["event_position"]) for event in validated["events"]}
        if actual_positions != set(expected_by_position):
            raise ValueError(f"event positions do not match packet {packet_id}")
        for event in validated["events"]:
            position = int(event["event_position"])
            expected_event = expected_by_position[position]
            if event["event_id"] != expected_event["event_id"] or event["line_id"] != expected_event["line_id"]:
                raise ValueError(f"event aliases do not match packet {packet_id} position {position}")
            label_counts[str(event["relevance"])] += 1
            if event.get("role") is not None:
                role_counts[str(event["role"])] += 1
        seen.add(packet_id)
        event_count += len(validated["events"])

    missing_packets = sorted(set(expected) - seen)
    if missing_packets:
        raise ValueError(f"annotation export is incomplete: {len(missing_packets)} packets missing")

    report = {
        "schema_version": "1.0",
        "status": "passed",
        "responses_path": str(responses_path),
        "packet_bundle_path": str(packet_bundle_path),
        "annotator_id": responses[0].get("annotator_id") if responses else None,
        "packet_count": len(seen),
        "expected_packet_count": len(expected),
        "event_count": event_count,
        "label_counts": dict(sorted(label_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--packet-bundle", type=Path, default=Path("research/results/tables/step57_annotation_packets.json"))
    parser.add_argument("--output", type=Path, default=Path("research/results/tables/step57_annotation_validation.json"))
    args = parser.parse_args()
    try:
        report = validate(args.responses, args.packet_bundle, args.output)
    except Exception as error:
        print(f"Step 57 annotation validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "output": str(args.output), "packet_count": report["packet_count"], "event_count": report["event_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
