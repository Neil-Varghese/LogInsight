"""Reproducibility fields and validation for Step 47 pilot records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


RECORD_SCHEMA_VERSION = "step50-1.0"
SYSTEM_PROMPT = "Return only one JSON object matching the requested explanation schema."


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def reproducibility_fields(
    *,
    user_prompt: str,
    raw_response: str | None,
    decoding: Mapping[str, Any],
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Return the complete prompt/response provenance for one request."""

    messages = messages or prompt_messages(user_prompt)
    fields: dict[str, Any] = {
        "exact_prompt": user_prompt,
        "prompt_messages": messages,
        "prompt_sha256": sha256_text(canonical_json(messages)),
        "decoding": dict(decoding),
        "raw_response": raw_response,
        "raw_response_sha256": sha256_text(raw_response) if raw_response is not None else None,
    }
    return fields


def validate_reproducibility_record(record: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if a completed pilot record is not auditable."""

    required = {
        "record_id",
        "status",
        "step",
        "condition",
        "repetition",
        "model",
        "exact_prompt",
        "prompt_messages",
        "prompt_sha256",
        "decoding",
        "raw_response",
        "raw_response_sha256",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"record is missing reproducibility fields: {missing}")
    if not isinstance(record["exact_prompt"], str) or not record["exact_prompt"]:
        raise ValueError("exact_prompt must be a non-empty string")
    messages = record["prompt_messages"]
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
        or messages[1].get("content") != record["exact_prompt"]
    ):
        raise ValueError("prompt_messages do not match exact_prompt")
    expected_prompt_hash = sha256_text(canonical_json(messages))
    if record["prompt_sha256"] != expected_prompt_hash:
        raise ValueError("prompt_sha256 does not match prompt_messages")
    raw_response = record["raw_response"]
    if raw_response is not None:
        if not isinstance(raw_response, str):
            raise ValueError("raw_response must be a string or null")
        if record["raw_response_sha256"] != sha256_text(raw_response):
            raise ValueError("raw_response_sha256 does not match raw_response")
    elif record["raw_response_sha256"] is not None:
        raise ValueError("raw_response_sha256 must be null when raw_response is null")
    if not isinstance(record["decoding"], dict):
        raise ValueError("decoding must be an object")


def audit_records(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    count = 0
    failures = 0
    for record in records:
        count += 1
        try:
            validate_reproducibility_record(record)
        except ValueError:
            failures += 1
    return {"record_count": count, "invalid_record_count": failures}


def audit_jsonl(path: Path) -> dict[str, int]:
    records: list[dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            records.append(value)
    return audit_records(records)
