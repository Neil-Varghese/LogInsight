"""Validate Integrated Gradients artifacts for both detector architectures.

The validator is intentionally independent of TensorFlow. It checks the
serialized attribution contract produced by the Step 31 and Step 32 scripts:
probability/attribution arithmetic, completeness error, event-position
alignment, padding behavior, ranking, duplicate units, and manifest counts.
It fails by default when any record exceeds the configured completeness
tolerance so invalid artifacts cannot silently reach later prompt steps.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
METHOD = "integrated_gradients"
DEFAULT_MAX_COMPLETENESS_ERROR = 0.05
DEFAULT_NUMERIC_TOLERANCE = 1e-6


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def _manifest_path(attribution_path: Path) -> Path:
    name = attribution_path.name
    if name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")] + ".manifest.json"
    else:
        name = attribution_path.stem + ".manifest.json"
    return attribution_path.with_name(name)


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _close(left: Any, right: Any, tolerance: float) -> bool:
    return _is_finite(left) and _is_finite(right) and abs(float(left) - float(right)) <= tolerance


def _load_manifest(attribution_path: Path, manifest_path: Path | None) -> dict[str, Any] | None:
    path = manifest_path or _manifest_path(attribution_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _validate_events(
    record: dict[str, Any],
    pad_id: int,
    padding: str,
    errors: list[str],
) -> None:
    events = record.get("events")
    if not isinstance(events, list):
        errors.append("events must be a list")
        return
    padded_length = record.get("padded_length")
    sequence_length = record.get("sequence_length")
    if not isinstance(padded_length, int) or padded_length < 1:
        errors.append("padded_length must be a positive integer")
        return
    if not isinstance(sequence_length, int) or sequence_length < 1:
        errors.append("sequence_length must be a positive integer")
        return
    if len(events) != padded_length:
        errors.append(f"events length {len(events)} != padded_length {padded_length}")
        return

    retained_length = min(sequence_length, padded_length)
    pad_count = padded_length - retained_length if padding == "pre" else 0
    retained_start = max(sequence_length - padded_length, 0)
    active_positions: list[int] = []
    observed_positions: list[int] = []
    observed_ranks: list[int] = []
    for expected_position, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event {expected_position} is not an object")
            continue
        position = event.get("input_position")
        observed_positions.append(position)
        if position != expected_position:
            errors.append(f"event position {position!r} != expected {expected_position}")
        event_id = event.get("event_id")
        if not isinstance(event_id, int) or event_id < 0:
            errors.append(f"event {expected_position} has invalid event_id")
            continue
        is_padding = event.get("is_padding")
        expected_padding = event_id == pad_id
        if is_padding is not expected_padding:
            errors.append(f"event {expected_position} has inconsistent padding flag")
        attribution = event.get("attribution")
        absolute_attribution = event.get("absolute_attribution")
        if not _is_finite(attribution) or not _is_finite(absolute_attribution):
            errors.append(f"event {expected_position} has non-finite attribution")
        elif not _close(abs(float(attribution)), float(absolute_attribution), 1e-9):
            errors.append(f"event {expected_position} absolute attribution mismatch")
        rank = event.get("rank")
        if expected_padding:
            if rank is not None:
                errors.append(f"padding event {expected_position} must not have a rank")
            if event.get("original_sequence_index") is not None:
                errors.append(f"padding event {expected_position} must not have a sequence index")
        else:
            active_positions.append(expected_position)
            if not isinstance(rank, int) or rank < 1:
                errors.append(f"active event {expected_position} has invalid rank")
            else:
                observed_ranks.append(rank)
            expected_index = (
                retained_start + expected_position - pad_count
                if padding == "pre"
                else retained_start + expected_position
            )
            if event.get("original_sequence_index") != expected_index:
                errors.append(
                    f"event {expected_position} original_sequence_index "
                    f"{event.get('original_sequence_index')!r} != {expected_index}"
                )

    if observed_positions != list(range(padded_length)):
        errors.append("input positions are not unique and ordered")
    if len(active_positions) != retained_length:
        errors.append(f"active event count {len(active_positions)} != retained length {retained_length}")
    if sorted(observed_ranks) != list(range(1, len(active_positions) + 1)):
        errors.append("active event ranks are not a complete 1..N sequence")


def validate_record(
    record: dict[str, Any],
    *,
    pad_id: int,
    padding: str,
    max_completeness_error: float,
    numeric_tolerance: float,
) -> list[str]:
    """Return human-readable validation errors for one attribution record."""

    errors: list[str] = []
    required = (
        "schema_version",
        "method",
        "target",
        "unit_id",
        "label_id",
        "probability",
        "baseline_probability",
        "output_delta",
        "attribution_sum",
        "completeness_error",
        "events",
    )
    missing = [field for field in required if field not in record]
    if missing:
        return [f"missing fields: {', '.join(missing)}"]
    if record["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version {record['schema_version']!r} != {SCHEMA_VERSION!r}")
    if record["method"] != METHOD:
        errors.append(f"method {record['method']!r} != {METHOD!r}")
    if record["target"] != "anomaly_probability":
        errors.append("target must be anomaly_probability")
    if record.get("unit_id") in (None, ""):
        errors.append("unit_id must be non-empty")
    if record.get("label_id") not in (0, 1):
        errors.append("label_id must be binary")

    for field in ("probability", "baseline_probability", "output_delta", "attribution_sum", "completeness_error"):
        if not _is_finite(record.get(field)):
            errors.append(f"{field} must be finite")
    for field in ("probability", "baseline_probability"):
        if _is_finite(record.get(field)) and not 0 <= float(record[field]) <= 1:
            errors.append(f"{field} must be in [0, 1]")
    if _is_finite(record.get("probability")) and _is_finite(record.get("baseline_probability")):
        expected_delta = float(record["probability"]) - float(record["baseline_probability"])
        if not _close(record["output_delta"], expected_delta, numeric_tolerance):
            errors.append("output_delta does not match probability-baseline_probability")
    if _is_finite(record.get("attribution_sum")) and _is_finite(record.get("output_delta")):
        expected_error = abs(float(record["attribution_sum"]) - float(record["output_delta"]))
        if not _close(record["completeness_error"], expected_error, numeric_tolerance):
            errors.append("completeness_error does not match attribution_sum-output_delta")
        if expected_error > max_completeness_error:
            errors.append(
                f"completeness_error {expected_error:.8g} exceeds "
                f"threshold {max_completeness_error:.8g}"
            )
    _validate_events(record, pad_id, padding, errors)
    return errors


def validate_attribution_file(
    attribution_path: Path,
    *,
    manifest_path: Path | None = None,
    pad_id: int = 0,
    padding: str | None = None,
    max_completeness_error: float = DEFAULT_MAX_COMPLETENESS_ERROR,
    numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Validate one JSONL attribution artifact and return a summary."""

    if not attribution_path.exists():
        raise FileNotFoundError(attribution_path)
    if padding not in {None, "pre", "post"}:
        raise ValueError("padding must be pre, post, or omitted")
    manifest = _load_manifest(attribution_path, manifest_path)
    effective_padding = padding or (manifest or {}).get("padding", "pre")
    if effective_padding not in {"pre", "post"}:
        raise ValueError("manifest padding must be pre or post")
    effective_pad_id = int((manifest or {}).get("baseline", {}).get("event_id", pad_id))

    records = 0
    invalid_records = 0
    duplicate_units = 0
    seen_units: set[str] = set()
    total_completeness_error = 0.0
    max_observed_error = 0.0
    errors: list[dict[str, Any]] = []
    with _open_text(attribution_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_records += 1
                errors.append({"line": line_number, "errors": [f"invalid JSON: {exc.msg}"]})
                continue
            records += 1
            if not isinstance(record, dict):
                record_errors = ["record must be a JSON object"]
            else:
                unit_id = str(record.get("unit_id", ""))
                if unit_id in seen_units and unit_id:
                    duplicate_units += 1
                seen_units.add(unit_id)
                record_errors = validate_record(
                    record,
                    pad_id=effective_pad_id,
                    padding=effective_padding,
                    max_completeness_error=max_completeness_error,
                    numeric_tolerance=numeric_tolerance,
                )
                if _is_finite(record.get("completeness_error")):
                    error_value = float(record["completeness_error"])
                    total_completeness_error += error_value
                    max_observed_error = max(max_observed_error, error_value)
            if record_errors:
                invalid_records += 1
                if len(errors) < 50:
                    errors.append({"line": line_number, "unit_id": record.get("unit_id") if isinstance(record, dict) else None, "errors": record_errors})

    if duplicate_units:
        errors.append({"file": str(attribution_path), "errors": [f"duplicate unit IDs: {duplicate_units}"]})
    manifest_count = (manifest or {}).get("count")
    if manifest_count is not None and int(manifest_count) != records:
        errors.append({"file": str(attribution_path), "errors": [f"manifest count {manifest_count} != records {records}"]})
    manifest_max_error = (manifest or {}).get("max_completeness_error")
    if manifest_max_error is not None and _is_finite(manifest_max_error) and abs(float(manifest_max_error) - max_observed_error) > numeric_tolerance:
        errors.append({"file": str(attribution_path), "errors": ["manifest max_completeness_error does not match records"]})

    return {
        "path": str(attribution_path),
        "manifest": str(manifest_path or _manifest_path(attribution_path)) if manifest is not None else None,
        "padding": effective_padding,
        "pad_id": effective_pad_id,
        "records": records,
        "invalid_records": invalid_records,
        "duplicate_units": duplicate_units,
        "mean_completeness_error": total_completeness_error / max(records, 1),
        "max_completeness_error": max_observed_error,
        "max_allowed_completeness_error": max_completeness_error,
        "valid": not errors,
        "errors": errors,
    }


def validate_attribution_files(
    attribution_paths: Iterable[Path],
    **kwargs: Any,
) -> dict[str, Any]:
    summaries = [validate_attribution_file(path, **kwargs) for path in attribution_paths]
    return {
        "files": summaries,
        "file_count": len(summaries),
        "valid": all(summary["valid"] for summary in summaries),
        "records": sum(summary["records"] for summary in summaries),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-file", type=Path, action="append", required=True, help="JSONL(.gz) attribution file; repeat for both detectors")
    parser.add_argument("--manifest", type=Path, action="append", help="Optional manifest matching each attribution file")
    parser.add_argument("--padding", choices=("pre", "post"), help="Override padding when manifests do not record it")
    parser.add_argument("--pad-id", type=int, default=0)
    parser.add_argument("--max-completeness-error", type=float, default=DEFAULT_MAX_COMPLETENESS_ERROR)
    parser.add_argument("--numeric-tolerance", type=float, default=DEFAULT_NUMERIC_TOLERANCE)
    parser.add_argument("--allow-failures", action="store_true", help="Report failures but exit successfully")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_completeness_error < 0 or args.numeric_tolerance < 0:
        raise SystemExit("tolerances must be non-negative")
    manifests = args.manifest or []
    if manifests and len(manifests) != len(args.attribution_file):
        raise SystemExit("--manifest must be supplied once per --attribution-file")
    summaries = []
    for index, path in enumerate(args.attribution_file):
        manifest = manifests[index] if manifests else None
        summaries.append(
            validate_attribution_file(
                path,
                manifest_path=manifest,
                pad_id=args.pad_id,
                padding=args.padding,
                max_completeness_error=args.max_completeness_error,
                numeric_tolerance=args.numeric_tolerance,
            )
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "validator": "validate_integrated_gradients",
        "max_allowed_completeness_error": args.max_completeness_error,
        "numeric_tolerance": args.numeric_tolerance,
        "files": summaries,
        "file_count": len(summaries),
        "records": sum(summary["records"] for summary in summaries),
        "valid": all(summary["valid"] for summary in summaries),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
