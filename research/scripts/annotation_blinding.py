"""Validate annotation blinding and create an ID-memorization probe.

Step 55 treats blinding as an auditable boundary. Packets are checked for
condition, detector, attribution, source-provenance, and identifier leakage.
The memorization probe creates equal-length pairs with fresh aliases and no
log content, then scores an annotator's same/different answers against a
hidden answer key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from annotation_packet import _find_forbidden
    from seed_utils import make_numpy_rng
except ImportError:  # pragma: no cover
    from research.scripts.annotation_packet import _find_forbidden
    from research.scripts.seed_utils import make_numpy_rng


SCRIPT_SCHEMA_VERSION = "1.0"
DEFAULT_PROBE_SEED = 2026
RELATIONS = {"same", "different"}


def validate_blinded_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that an annotator-facing packet contains no study leakage."""

    if not isinstance(packet, Mapping):
        raise ValueError("blinded packet must be an object")
    required = {"schema_version", "packet_type", "guideline_version", "packet_id", "unit_alias", "events", "context"}
    missing = required - set(packet)
    if missing:
        raise ValueError(f"blinded packet missing fields: {sorted(missing)}")
    if packet["packet_type"] not in {"evidence_relevance", "claim_support"}:
        raise ValueError("packet_type is not a registered blinded packet type")
    if not isinstance(packet["events"], list):
        raise ValueError("blinded packet events must be a list")
    context = packet["context"]
    if not isinstance(context, Mapping):
        raise ValueError("blinded packet context must be an object")
    if context.get("annotation_eligible") is not True:
        raise ValueError("capped or incomplete packets cannot enter blinded annotation")
    if context.get("omitted_event_positions") or context.get("omitted_event_ids"):
        raise ValueError("blinded packet must not omit events")
    forbidden = _find_forbidden(packet)
    if forbidden:
        raise ValueError(f"blinded packet contains forbidden fields: {forbidden[:8]}")
    serialized = json.dumps(packet, ensure_ascii=False)
    for marker in ("blk_", "Integrated Gradients", "occlusion", "leave-one-out", "unblinded", "blinded", "random_event"):
        if marker.lower() in serialized.lower():
            raise ValueError(f"blinded packet contains forbidden marker: {marker}")
    for index, event in enumerate(packet["events"]):
        if not isinstance(event, Mapping):
            raise ValueError(f"events[{index}] must be an object")
        if not isinstance(event.get("event_position"), int):
            raise ValueError("blinded event position must be an integer")
        if not str(event.get("event_id", "")).startswith("E"):
            raise ValueError("blinded event IDs must be local aliases")
        if not str(event.get("line_id", "")).startswith("L"):
            raise ValueError("blinded line IDs must be local aliases")
    return dict(packet)


def _probe_alias(prefix: str, index: int, salt: int) -> str:
    digest = hashlib.sha256(f"{prefix}|{index}|{salt}".encode("utf-8")).hexdigest()[:8]
    return f"{prefix}{digest}"


def _alias_only_view(packet: Mapping[str, Any], *, view_index: int, salt: int, length: int) -> dict[str, Any]:
    return {
        "view_id": f"VIEW-{view_index:04d}",
        "events": [
            {
                "event_ordinal": index,
                "event_id": _probe_alias("P", index, salt),
                "line_id": _probe_alias("Q", index, salt),
            }
            for index in range(length)
        ],
    }


def build_id_memorization_probe(
    packets: Iterable[Mapping[str, Any]],
    *,
    pair_count: int | None = None,
    seed: int = DEFAULT_PROBE_SEED,
) -> dict[str, Any]:
    """Build a hidden-key same/different probe from blinded packets.

    Each displayed pair has equal event counts and fresh aliases. Source unit
    aliases, raw lines, templates, parameters, and labels are absent from the
    probe view. The answer key is returned separately so the coordinator can
    keep it hidden from annotators.
    """

    packet_list = [validate_blinded_packet(packet) for packet in packets]
    if len(packet_list) < 2:
        raise ValueError("the memorization probe requires at least two packets")
    if pair_count is None:
        pair_count = len(packet_list)
    if isinstance(pair_count, bool) or pair_count < 1:
        raise ValueError("pair_count must be positive")
    rng = make_numpy_rng(seed, "annotation_id_memorization_probe", len(packet_list), pair_count)
    items: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for probe_index in range(pair_count):
        same = bool(probe_index % 2 == 0)
        left_index = int(rng.integers(0, len(packet_list)))
        right_index = left_index if same else int(rng.integers(0, len(packet_list) - 1))
        if not same and right_index >= left_index:
            right_index += 1
        length = min(len(packet_list[left_index]["events"]), len(packet_list[right_index]["events"]))
        if length < 1:
            raise ValueError("probe packets must contain at least one event")
        left = _alias_only_view(packet_list[left_index], view_index=probe_index * 2, salt=int(rng.integers(0, 2**31)), length=length)
        right = _alias_only_view(packet_list[right_index], view_index=probe_index * 2 + 1, salt=int(rng.integers(0, 2**31)), length=length)
        probe_id = f"IDPROBE-{probe_index:04d}"
        items.append({"probe_id": probe_id, "left": left, "right": right, "annotator_response": None})
        answer_key.append({"probe_id": probe_id, "expected_relation": "same" if same else "different"})
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "probe_type": "annotation_id_memorization",
        "seed": seed,
        "items": items,
        "answer_key": answer_key,
        "manifest": {
            "pair_count": pair_count,
            "same_pairs": sum(item["expected_relation"] == "same" for item in answer_key),
            "different_pairs": sum(item["expected_relation"] == "different" for item in answer_key),
            "expected_chance_accuracy": 0.5,
            "source_identifiers_exposed": False,
            "raw_evidence_exposed": False,
            "labels_exposed": False,
        },
    }


def score_id_memorization_probe(
    responses: Iterable[Mapping[str, Any]],
    answer_key: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score completed same/different responses against the hidden key."""

    expected = {str(item.get("probe_id")): item.get("expected_relation") for item in answer_key}
    if not expected:
        raise ValueError("answer_key must contain at least one item")
    correct = 0
    answered = 0
    invalid = 0
    for response in responses:
        if not isinstance(response, Mapping):
            invalid += 1
            continue
        probe_id = str(response.get("probe_id"))
        relation = response.get("annotator_response", response.get("response"))
        if probe_id not in expected or relation not in RELATIONS:
            invalid += 1
            continue
        answered += 1
        correct += int(relation == expected[probe_id])
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "probe_type": "annotation_id_memorization",
        "expected_chance_accuracy": 0.5,
        "items": len(expected),
        "answered": answered,
        "correct": correct,
        "invalid_or_unmatched": invalid,
        "accuracy": correct / answered if answered else None,
        "above_chance_indicator": (correct / answered > 0.5) if answered else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True, help="JSON array of blinded packets")
    parser.add_argument("--output", type=Path, required=True, help="Probe JSON with hidden answer key")
    parser.add_argument("--pair-count", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_PROBE_SEED)
    args = parser.parse_args(argv)
    packets = json.loads(args.packets.read_text(encoding="utf-8"))
    if not isinstance(packets, list):
        raise ValueError("--packets must contain a JSON array")
    probe = build_id_memorization_probe(packets, pair_count=args.pair_count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
