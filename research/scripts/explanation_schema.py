"""Validation helpers for the Step 40 structured explanation payload.

This validates already-decoded JSON objects. It intentionally does not parse
raw LLM responses; malformed-response parsing belongs to Step 48.
"""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.0"
ANSWER_STATUSES = {"answered", "partially_abstained", "fully_abstained"}
ABSTENTION_STATUSES = {"none", "partial", "full"}
CLAIM_TYPES = {"anomalous_pattern", "contrastive_difference", "detector_behavior", "uncertainty"}
PREDICTED_EFFECTS = {"increase_anomaly_score", "decrease_anomaly_score", "no_change", "uncertain", None}
REQUIRED_FIELDS = {
    "schema_version",
    "answer_status",
    "summary",
    "claims",
    "cited_event_ids",
    "cited_line_ids",
    "overall_confidence",
    "abstention",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_confidence(value: Any, field: str) -> None:
    if value is not None and (not _is_number(value) or not 0 <= float(value) <= 1):
        raise ValueError(f"{field} must be null or a number in [0, 1]")


def _check_line_ids(values: Any, field: str) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    if len(values) != len({(type(value).__name__, value) for value in values}):
        raise ValueError(f"{field} must not contain duplicates")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)) or (isinstance(value, str) and not value):
            raise ValueError(f"{field} must contain only non-empty strings or integers")


def _check_event_ids(values: Any, field: str) -> None:
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field} must contain only non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must not contain duplicates")


def _check_claim(claim: Any, index: int, root_event_ids: set[str], root_line_ids: set[tuple[str, Any]]) -> None:
    field = f"claims[{index}]"
    if not isinstance(claim, dict):
        raise ValueError(f"{field} must be an object")
    required = {"claim_id", "claim_type", "text", "cited_event_ids", "cited_line_ids", "confidence", "abstained"}
    missing = required - set(claim)
    extra = set(claim) - required - {"predicted_effect"}
    if missing or extra:
        raise ValueError(f"{field} has missing={sorted(missing)} or unsupported={sorted(extra)} fields")
    if not isinstance(claim["claim_id"], str) or not claim["claim_id"]:
        raise ValueError(f"{field}.claim_id must be a non-empty string")
    if claim["claim_type"] not in CLAIM_TYPES:
        raise ValueError(f"{field}.claim_type is not registered")
    if not isinstance(claim["text"], str) or not claim["text"]:
        raise ValueError(f"{field}.text must be a non-empty string")
    _check_event_ids(claim["cited_event_ids"], f"{field}.cited_event_ids")
    _check_line_ids(claim["cited_line_ids"], f"{field}.cited_line_ids")
    _check_confidence(claim["confidence"], f"{field}.confidence")
    if not isinstance(claim["abstained"], bool):
        raise ValueError(f"{field}.abstained must be boolean")
    if claim.get("predicted_effect") not in PREDICTED_EFFECTS:
        raise ValueError(f"{field}.predicted_effect is not registered")
    if not claim["abstained"] and not (claim["cited_event_ids"] or claim["cited_line_ids"]):
        raise ValueError(f"{field} must cite an event or line unless abstained")
    if not set(claim["cited_event_ids"]).issubset(root_event_ids):
        raise ValueError(f"{field} cites an event missing from root cited_event_ids")
    if not {
        (type(value).__name__, value) for value in claim["cited_line_ids"]
    }.issubset(root_line_ids):
        raise ValueError(f"{field} cites a line missing from root cited_line_ids")


def validate_explanation_record(record: Any) -> dict[str, Any]:
    """Validate and return a structured explanation object.

    Raises ``ValueError`` with a field-oriented message when the object does
    not satisfy the schema and its cross-field citation/abstention rules.
    """

    if not isinstance(record, dict):
        raise ValueError("explanation must be an object")
    missing = REQUIRED_FIELDS - set(record)
    extra = set(record) - REQUIRED_FIELDS
    if missing or extra:
        raise ValueError(f"explanation has missing={sorted(missing)} or unsupported={sorted(extra)} fields")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if record["answer_status"] not in ANSWER_STATUSES:
        raise ValueError("answer_status is not registered")
    if record["summary"] is not None and not isinstance(record["summary"], str):
        raise ValueError("summary must be a string or null")
    if not isinstance(record["claims"], list):
        raise ValueError("claims must be a list")
    _check_event_ids(record["cited_event_ids"], "cited_event_ids")
    _check_line_ids(record["cited_line_ids"], "cited_line_ids")
    _check_confidence(record["overall_confidence"], "overall_confidence")
    abstention = record["abstention"]
    if not isinstance(abstention, dict) or set(abstention) != {"status", "reason"}:
        raise ValueError("abstention must contain exactly status and reason")
    if abstention["status"] not in ABSTENTION_STATUSES:
        raise ValueError("abstention.status is not registered")
    if abstention["reason"] is not None and not isinstance(abstention["reason"], str):
        raise ValueError("abstention.reason must be a string or null")
    expected_abstention = {
        "answered": "none",
        "partially_abstained": "partial",
        "fully_abstained": "full",
    }[record["answer_status"]]
    if abstention["status"] != expected_abstention:
        raise ValueError("answer_status and abstention.status disagree")
    if record["answer_status"] == "answered" and not record["claims"]:
        raise ValueError("answered explanations must contain at least one claim")
    if record["answer_status"] == "fully_abstained" and record["claims"]:
        raise ValueError("fully_abstained explanations must not contain claims")
    root_event_ids = set(record["cited_event_ids"])
    root_line_ids = {(type(value).__name__, value) for value in record["cited_line_ids"]}
    for index, claim in enumerate(record["claims"]):
        _check_claim(claim, index, root_event_ids, root_line_ids)
    return record
