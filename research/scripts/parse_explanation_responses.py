"""Parse and classify raw structured-explanation responses from Steps 47A/47B.

The pilot runners deliberately preserve the model's raw response and defer
interpretation to this module.  This keeps parsing deterministic and makes
invalid responses auditable rather than silently repairing them.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO

from explanation_schema import validate_explanation_record


PARSER_VERSION = "step48-1.0"
_FENCED_JSON = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.IGNORECASE | re.DOTALL)


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode, encoding="utf-8")


def _response_text(raw_response: Any) -> str | None:
    if raw_response is None:
        return None
    if not isinstance(raw_response, str):
        return None
    text = raw_response.strip()
    if not text:
        return None
    fenced = _FENCED_JSON.match(text)
    return fenced.group(1).strip() if fenced else text


def _looks_like_missing_citations(record: dict[str, Any]) -> bool:
    """Identify citation omissions before generic schema validation.

    Empty citation arrays are valid for a full abstention.  For claims that
    assert an answer, however, the registered schema requires event or line
    evidence.  Missing root/claim citation fields are also classified here so
    they are not hidden inside a generic schema error.
    """

    if "cited_event_ids" not in record or "cited_line_ids" not in record:
        return True
    claims = record.get("claims")
    if not isinstance(claims, list):
        return False
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        if "cited_event_ids" not in claim or "cited_line_ids" not in claim:
            return True
        if not claim.get("abstained", False) and not (
            claim.get("cited_event_ids") or claim.get("cited_line_ids")
        ):
            return True
    return False


def _looks_like_malformed_claims(record: dict[str, Any]) -> bool:
    claims = record.get("claims")
    if not isinstance(claims, list):
        return True
    required = {"claim_id", "claim_type", "text", "cited_event_ids", "cited_line_ids", "confidence", "abstained"}
    allowed = required | {"predicted_effect"}
    for claim in claims:
        if not isinstance(claim, dict):
            return True
        if not required.issubset(claim) or set(claim) - allowed:
            return True
        if not isinstance(claim.get("claim_id"), str) or not claim["claim_id"]:
            return True
        if not isinstance(claim.get("text"), str) or not claim["text"]:
            return True
        if claim.get("claim_type") not in {"anomalous_pattern", "contrastive_difference", "detector_behavior", "uncertainty"}:
            return True
        if not isinstance(claim.get("abstained"), bool):
            return True
    return False


def parse_raw_response(raw_response: Any, *, upstream_status: str | None = None) -> dict[str, Any]:
    """Parse one raw response and return classification fields.

    ``parsed_explanation`` is populated only for schema-valid responses.
    The original response is never modified or repaired by this function.
    """

    if upstream_status and upstream_status != "complete" and raw_response is None:
        return {"parsed_explanation": None, "parsing_status": "upstream_failed", "parsing_error": upstream_status}

    text = _response_text(raw_response)
    if text is None:
        return {"parsed_explanation": None, "parsing_status": "missing_response", "parsing_error": "raw_response is empty or not a string"}

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"parsed_explanation": None, "parsing_status": "invalid_json", "parsing_error": f"{exc.msg} at character {exc.pos}"}

    if not isinstance(decoded, dict):
        return {"parsed_explanation": None, "parsing_status": "malformed_payload", "parsing_error": "decoded response must be a JSON object"}
    if _looks_like_malformed_claims(decoded):
        return {"parsed_explanation": None, "parsing_status": "malformed_claims", "parsing_error": "claims do not match the registered claim structure"}
    if _looks_like_missing_citations(decoded):
        return {"parsed_explanation": None, "parsing_status": "missing_citations", "parsing_error": "required citation fields or evidence are missing"}

    try:
        validated = validate_explanation_record(decoded)
    except ValueError as exc:
        return {"parsed_explanation": None, "parsing_status": "schema_invalid", "parsing_error": str(exc)}

    answer_status = validated["answer_status"]
    status = {"partially_abstained": "valid_partial_abstention", "fully_abstained": "valid_full_abstention"}.get(answer_status, "valid")
    return {"parsed_explanation": validated, "parsing_status": status, "parsing_error": None}


def parse_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        output = copy.deepcopy(record)
        result = parse_raw_response(record.get("raw_response"), upstream_status=record.get("status"))
        output.update(result)
        output["parser_version"] = PARSER_VERSION
        raw_response = record.get("raw_response")
        output["raw_response_sha256"] = hashlib.sha256(raw_response.encode("utf-8")).hexdigest() if isinstance(raw_response, str) else None
        counts[result["parsing_status"]] += 1
        parsed_records.append(output)
    summary = {"parser_version": PARSER_VERSION, "total_records": len(parsed_records), "counts_by_status": dict(sorted(counts.items()))}
    return parsed_records, summary


def parse_jsonl(input_path: Path, output_path: Path, summary_path: Path | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with _open_text(input_path, "rt") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"input line {line_number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"input line {line_number} must contain a JSON object")
            records.append(record)
    parsed, summary = parse_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(output_path, "wt") as destination:
        for record in parsed:
            destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Step 47A/47B JSONL container file")
    parser.add_argument("--output", type=Path, required=True, help="Parsed JSONL output path")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path")
    args = parser.parse_args()
    summary = parse_jsonl(args.input, args.output, args.summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
