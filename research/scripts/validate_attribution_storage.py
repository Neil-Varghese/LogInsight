"""Validate and register separate attribution-vector artifacts.

Integrated Gradients, event occlusion, and leave-one-event-out values must be
stored as separate method-specific artifacts. This script validates that each
input file contains exactly one registered method, that verification methods
carry their non-prompt-visible flags, that prediction units are unique within
each file, and that no path is reused for multiple methods. It writes only a
registry manifest with hashes and references; it never merges the vectors.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


SCRIPT_SCHEMA_VERSION = "1.0"
METHOD_POLICIES = {
    "integrated_gradients": {
        "prompt_visible": True,
        "verification_only": False,
        "score_keys": {"attribution", "absolute_attribution"},
    },
    "event_occlusion": {
        "prompt_visible": False,
        "verification_only": True,
        "score_keys": {"occlusion_score", "occluded_probability"},
    },
    "leave_one_event_out": {
        "prompt_visible": False,
        "verification_only": True,
        "score_keys": {"leave_one_event_out_score", "held_out_probability"},
    },
}


def _open_input(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    with _open_input(path) as handle:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_attribution_file(path: Path, expected_method: str) -> dict[str, Any]:
    """Validate one method-specific vector file and return its registry entry."""

    path = path.resolve()
    if expected_method not in METHOD_POLICIES:
        raise ValueError(f"unsupported attribution method: {expected_method!r}")
    if not path.exists():
        raise FileNotFoundError(path)
    policy = METHOD_POLICIES[expected_method]
    units: set[str] = set()
    count = 0
    event_count = 0
    records = iter_json_records(path)
    for record in records:
        count += 1
        if record.get("method") != expected_method:
            raise ValueError(
                f"{path} contains method {record.get('method')!r}; expected {expected_method!r}"
            )
        unit_id = record.get("unit_id")
        if unit_id in (None, ""):
            raise ValueError(f"{path} contains a record without unit_id")
        unit_key = str(unit_id)
        if unit_key in units:
            raise ValueError(f"{path} contains duplicate unit_id {unit_key!r}")
        units.add(unit_key)
        if policy["prompt_visible"]:
            if record.get("verification_only", False):
                raise ValueError(f"Integrated Gradients file {path} contains verification-only metadata")
            if "prompt_visible" in record and record["prompt_visible"] is False:
                raise ValueError(f"Integrated Gradients file {path} is marked prompt-invisible")
        else:
            if record.get("verification_only") is not True:
                raise ValueError(f"verification file {path} must set verification_only=true")
            if record.get("prompt_visible") is not False:
                raise ValueError(f"verification file {path} must set prompt_visible=false")
        events = record.get("events")
        if not isinstance(events, list):
            raise ValueError(f"{path} record {unit_key!r} must contain an events list")
        for event in events:
            if not isinstance(event, dict):
                raise ValueError(f"{path} record {unit_key!r} contains a non-object event")
            event_count += 1
            if any(key not in event for key in policy["score_keys"]):
                raise ValueError(f"{path} record {unit_key!r} lacks its {expected_method} score fields")
            if expected_method == "integrated_gradients":
                if any(key in event for key in ("occlusion_score", "leave_one_event_out_score", "verification_only")):
                    raise ValueError(f"Integrated Gradients file {path} contains verification fields")
    if count == 0:
        raise ValueError(f"attribution file {path} is empty")
    return {
        "method": expected_method,
        "path": str(path),
        "sha256": _sha256(path),
        "record_count": count,
        "unit_count": len(units),
        "event_count": event_count,
        "prompt_visible": policy["prompt_visible"],
        "verification_only": policy["verification_only"],
    }


def build_storage_manifest(
    *,
    integrated_gradients_files: list[Path],
    occlusion_files: list[Path],
    leave_one_event_out_files: list[Path],
    output: Path,
) -> dict[str, Any]:
    """Validate separate files and write a non-merging storage registry."""

    requested = [
        ("integrated_gradients", path) for path in integrated_gradients_files
    ] + [
        ("event_occlusion", path) for path in occlusion_files
    ] + [
        ("leave_one_event_out", path) for path in leave_one_event_out_files
    ]
    if not requested:
        raise ValueError("at least one attribution-vector file is required")
    paths = [str(path.resolve()) for _, path in requested]
    if len(paths) != len(set(paths)):
        raise ValueError("one attribution-vector file cannot be registered under multiple methods")
    entries = [validate_attribution_file(path, method) for method, path in requested]
    manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "registry": "separate_attribution_vectors",
        "storage_policy": "one_method_per_file_no_vector_merging",
        "prompt_visible_method": "integrated_gradients",
        "verification_only_methods": ["event_occlusion", "leave_one_event_out"],
        "entries": entries,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated-gradients-file", type=Path, action="append", default=[])
    parser.add_argument("--occlusion-file", type=Path, action="append", default=[])
    parser.add_argument("--leave-one-event-out-file", type=Path, action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/models/attributions/attribution_vector_store.manifest.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_storage_manifest(
        integrated_gradients_files=args.integrated_gradients_file,
        occlusion_files=args.occlusion_file,
        leave_one_event_out_files=args.leave_one_event_out_file,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
