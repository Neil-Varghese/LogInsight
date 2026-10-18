"""Build blinded annotation packets and validate evidence-label exports.

Step 54 keeps the annotator-facing artifact independent from detector and LLM
provenance. Source unit/event/line identifiers are replaced with stable local
aliases, sensitive host/path values are redacted, and a deterministic token
cap records every omitted occurrence instead of silently truncating context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


SCRIPT_SCHEMA_VERSION = "1.0"
GUIDELINE_VERSION = "1.0"
DEFAULT_CONTEXT_CAP_TOKENS = 6500
RELEVANCE_LABELS = {"relevant", "not_relevant", "uncertain"}
ROLES = {"failure_signal", "anomalous_transition", "meaningful_contrast", "recovery_or_consequence", "context_only"}
MEANINGFUL_DIFFERENCE = {"yes", "no", "uncertain"}

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_BLOCK_ID_RE = re.compile(r"\bblk_[A-Za-z0-9_-]+\b", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_PATH_RE = re.compile(r"/(?:[A-Za-z0-9_.-]+/){1,}[A-Za-z0-9_.-]+")
_HOST_RE = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+(?:local|internal|lan|example|com|org|net)\b", re.IGNORECASE)


def estimate_context_tokens(value: Any) -> int:
    """Return a deterministic, dependency-free token estimate for JSON data."""

    return len(_TOKEN_RE.findall(json.dumps(value, ensure_ascii=False, sort_keys=True)))


def anonymize_text(value: Any) -> Any:
    """Redact common host, path, block, UUID, and email identifiers."""

    if not isinstance(value, str):
        return value
    text = _BLOCK_ID_RE.sub("<BLOCK_ID>", value)
    text = _IP_RE.sub("<HOST>", text)
    text = _UUID_RE.sub("<UUID>", text)
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _PATH_RE.sub("<PATH>", text)
    text = _HOST_RE.sub("<HOST>", text)
    return text


def _source_value(event: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in event:
            return event[name]
    return None


def _alias(prefix: str, number: int) -> str:
    return f"{prefix}{number:04d}"


def _event_view(
    event: Mapping[str, Any],
    *,
    position: int,
    event_aliases: dict[str, str],
    line_aliases: dict[str, str],
    event_prefix: str = "E",
    line_prefix: str = "L",
) -> dict[str, Any]:
    source_event_id = str(_source_value(event, "event_id", "EventId") or f"position-{position}")
    source_line_id = str(_source_value(event, "line_id", "LineId") or f"position-{position}")
    if source_event_id not in event_aliases:
        event_aliases[source_event_id] = _alias(event_prefix, len(event_aliases) + 1)
    if source_line_id not in line_aliases:
        line_aliases[source_line_id] = _alias(line_prefix, len(line_aliases) + 1)
    parameters = _source_value(event, "parameter_list", "ParameterList")
    if parameters is None:
        parameters = []
    if not isinstance(parameters, list):
        raise ValueError("event parameter list must be a list")
    raw_line = _source_value(event, "raw_line", "RawLine")
    template = _source_value(event, "event_template", "EventTemplate")
    return {
        "event_position": position,
        "event_id": event_aliases[source_event_id],
        "line_id": line_aliases[source_line_id],
        "event_template": anonymize_text(str(template)) if template is not None else None,
        "parameter_list": [anonymize_text(str(value)) for value in parameters],
        "raw_line": anonymize_text(str(raw_line)) if raw_line is not None else None,
    }


def _record_events(record: Mapping[str, Any], *names: str) -> list[Mapping[str, Any]]:
    for name in names:
        value = record.get(name)
        if value is not None:
            if isinstance(value, Mapping) and "events" in value:
                value = value["events"]
            if not isinstance(value, list) or any(not isinstance(event, Mapping) for event in value):
                raise ValueError(f"{name} must be a list of event objects")
            return list(value)
    return []


def _packet_id(dataset: Any, unit_id: Any) -> str:
    digest = hashlib.sha256(f"{dataset}|{unit_id}".encode("utf-8")).hexdigest()[:12]
    return f"PACKET-{digest}"


def build_annotation_packet(
    record: Mapping[str, Any],
    *,
    max_context_tokens: int = DEFAULT_CONTEXT_CAP_TOKENS,
) -> dict[str, Any]:
    """Build one blinded evidence-relevance packet.

    Events are added in chronological order. If the deterministic estimate
    exceeds the cap, omitted positions and aliases are recorded and the packet
    is marked ``annotation_eligible: false`` for the primary analysis.
    """

    if not isinstance(record, Mapping):
        raise ValueError("annotation source record must be an object")
    if isinstance(max_context_tokens, bool) or max_context_tokens < 1:
        raise ValueError("max_context_tokens must be a positive integer")
    dataset = record.get("dataset", "unknown")
    unit_id = record.get("unit_id", record.get("BlockId", record.get("WindowId")))
    if unit_id is None:
        raise ValueError("annotation source record requires unit_id, BlockId, or WindowId")
    source_events = _record_events(record, "events", "evidence", "anomalous_block_evidence")
    event_aliases: dict[str, str] = {}
    line_aliases: dict[str, str] = {}
    mapped_events = [
        _event_view(event, position=index, event_aliases=event_aliases, line_aliases=line_aliases)
        for index, event in enumerate(source_events)
    ]
    normal_source = _record_events(record, "normal_events", "normal_comparison_events")
    normal_events = [
        _event_view(
            event,
            position=index,
            event_aliases={},
            line_aliases={},
            event_prefix="NE",
            line_prefix="NL",
        )
        for index, event in enumerate(normal_source)
    ]
    base = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "packet_type": "evidence_relevance",
        "guideline_version": GUIDELINE_VERSION,
        "packet_id": _packet_id(dataset, unit_id),
        "unit_alias": f"UNIT-{hashlib.sha256(str(unit_id).encode('utf-8')).hexdigest()[:10]}",
        "dataset": str(dataset),
        "events": [],
    }
    if normal_events:
        base["normal_comparison"] = {"events": []}

    omitted_positions: list[int] = []
    omitted_event_ids: list[str] = []
    for event in mapped_events:
        candidate = dict(base)
        candidate["events"] = [*base["events"], event]
        if normal_events:
            candidate["normal_comparison"] = {"events": normal_events}
        if estimate_context_tokens(candidate) <= max_context_tokens:
            base["events"].append(event)
        else:
            omitted_positions.append(int(event["event_position"]))
            omitted_event_ids.append(str(event["event_id"]))
    if normal_events:
        base["normal_comparison"] = {"events": normal_events}
        candidate = dict(base)
        candidate["normal_comparison"] = {"events": normal_events}
        if estimate_context_tokens(candidate) > max_context_tokens:
            base.pop("normal_comparison")

    base["context"] = {
        "max_context_tokens": max_context_tokens,
        "estimated_context_tokens": estimate_context_tokens(base),
        "context_status": "capped" if omitted_positions else "complete",
        "annotation_eligible": not omitted_positions and len(base["events"]) == len(mapped_events),
        "omitted_event_positions": omitted_positions,
        "omitted_event_ids": omitted_event_ids,
        "policy": "deterministic_cap_record_omissions_never_silent_truncation",
    }
    base["annotation_form"] = {
        "events_require_labels": True,
        "relevance_labels": sorted(RELEVANCE_LABELS),
        "roles": sorted(ROLES),
        "contrastive_assessment": bool(normal_events),
        "required_event_fields": ["event_position", "event_id", "line_id", "relevance", "role", "reason"],
    }
    return base


def _find_forbidden(value: Any, path: str = "") -> list[str]:
    forbidden_fragments = (
        "attribution", "occlusion", "leave_one", "detector_score", "probability",
        "threshold", "condition", "model", "checkpoint", "source_unit", "source_event",
        "source_line", "raw_block_id", "seed",
    )
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            child_path = f"{path}.{key}" if path else str(key)
            if any(fragment in key_text for fragment in forbidden_fragments):
                found.append(child_path)
            found.extend(_find_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden(child, f"{path}[{index}]"))
    return found


def validate_annotation_export(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a completed anonymized evidence-relevance export."""

    if not isinstance(record, Mapping):
        raise ValueError("annotation export must be an object")
    required = {"schema_version", "guideline_version", "packet_id", "unit_alias", "events", "contrastive_assessment"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"annotation export missing fields: {sorted(missing)}")
    if record["schema_version"] != SCRIPT_SCHEMA_VERSION or record["guideline_version"] != GUIDELINE_VERSION:
        raise ValueError("annotation export uses an unsupported schema or guideline version")
    if not isinstance(record["events"], list):
        raise ValueError("annotation export events must be a list")
    positions = []
    for index, event in enumerate(record["events"]):
        if not isinstance(event, Mapping):
            raise ValueError(f"events[{index}] must be an object")
        missing_event = {"event_position", "event_id", "line_id", "relevance", "role", "reason"} - set(event)
        if missing_event:
            raise ValueError(f"events[{index}] missing fields: {sorted(missing_event)}")
        if not isinstance(event["event_position"], int) or event["event_position"] < 0:
            raise ValueError("event_position must be a non-negative integer")
        if not re.fullmatch(r"(?:E|NE)\d{4}", str(event["event_id"])):
            raise ValueError("event_id must be an anonymized event alias")
        if not re.fullmatch(r"(?:L|NL)\d{4}", str(event["line_id"])):
            raise ValueError("line_id must be an anonymized line alias")
        if event["relevance"] not in RELEVANCE_LABELS:
            raise ValueError("event relevance label is not registered")
        if event["role"] is not None and event["role"] not in ROLES:
            raise ValueError("event role is not registered")
        if event["reason"] is not None and not isinstance(event["reason"], str):
            raise ValueError("event reason must be a string or null")
        positions.append(event["event_position"])
    if len(positions) != len(set(positions)):
        raise ValueError("annotation export contains duplicate event positions")
    contrastive = record["contrastive_assessment"]
    if not isinstance(contrastive, Mapping):
        raise ValueError("contrastive_assessment must be an object")
    if contrastive.get("meaningful_difference") not in MEANINGFUL_DIFFERENCE:
        raise ValueError("contrastive assessment label is not registered")
    if not isinstance(contrastive.get("difference_positions"), list):
        raise ValueError("difference_positions must be a list")
    forbidden = _find_forbidden(record)
    if forbidden:
        raise ValueError(f"annotation export contains forbidden provenance fields: {forbidden[:5]}")
    return dict(record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source JSON record")
    parser.add_argument("--output", type=Path, required=True, help="Blinded packet JSON")
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_CONTEXT_CAP_TOKENS)
    args = parser.parse_args(argv)
    record = json.loads(args.input.read_text(encoding="utf-8"))
    packet = build_annotation_packet(record, max_context_tokens=args.max_context_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
