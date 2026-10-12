"""Audit dataset, retrieval, and parser leakage before research experiments.

The checker is intentionally independent from model training.  It verifies:

* each prediction unit occurs in at most one split and split files agree with
  their assignment manifests;
* retrieval records use only normal candidates from the declared training
  split; and
* parser/vocabulary policies are training-only and BGL label prefixes are not
  present in parser-safe log input.

Retrieval and parser artifact audits are optional until those artifacts are
created.  The policy checks run immediately and the manifest records which
runtime checks were pending.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "validation", "test")
KNOWN_ASSIGNMENTS = set(SPLITS) | {"excluded_boundary_crossing"}
EPOCH_PATTERN = re.compile(r"^\d{9,12}$")


class ProgressBar:
    """Small dependency-free progress indicator."""

    def __init__(self, label: str, total: int | None = None, enabled: bool = True) -> None:
        self.label = label
        self.total = total
        self.enabled = enabled
        self.last_update = 0.0

    def update(self, current: int, suffix: str = "", force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_update < 0.15:
            return
        self.last_update = now
        width = 30
        if self.total is None:
            filled = (current // 10_000) % (width + 1)
            bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
            message = f"{self.label} [{bar}] records={current:,} {suffix}".rstrip()
        else:
            fraction = min(max(current, 0) / max(self.total, 1), 1.0)
            filled = int(width * fraction)
            bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
            message = f"{self.label} [{bar}] {fraction * 100:6.2f}% {suffix}".rstrip()
        print(f"\r{message}", end="", file=sys.stderr, flush=True)
        if self.total is not None and current >= self.total:
            print(file=sys.stderr)

    def finish(self, current: int, suffix: str = "") -> None:
        self.update(current, suffix=suffix, force=True)
        if self.total is None:
            print(file=sys.stderr)


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield record


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def check_split_bundle(
    name: str,
    assignment_path: Path,
    split_paths: dict[str, Path],
    id_field: str,
    progress: bool = True,
) -> dict[str, Any]:
    """Check assignments and split files for duplicate or misplaced units."""

    if set(split_paths) != set(SPLITS):
        raise ValueError(f"{name}: split paths must contain train/validation/test")
    assignments: dict[str, str] = {}
    assignment_duplicates: list[str] = []
    assignment_count = 0
    for record in iter_records(assignment_path):
        assignment_count += 1
        # Primary HDFS assignments retain the source field name (`BlockId`),
        # while temporal assignments use the generic `UnitId` field.
        unit_id = record.get("UnitId", record.get(id_field))
        split = record.get("Split")
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError(f"{name}: assignment has invalid UnitId")
        if split not in KNOWN_ASSIGNMENTS:
            raise ValueError(f"{name}: invalid assignment for {unit_id}: {split}")
        if unit_id in assignments:
            assignment_duplicates.append(unit_id)
        assignments[unit_id] = split

    seen_in_splits: set[str] = set()
    split_counts: Counter[str] = Counter()
    split_duplicates: list[str] = []
    misplaced: list[dict[str, str]] = []
    split_progress = ProgressBar(f"Checking {name} split files", total=None, enabled=progress)
    records_checked = 0
    for split, path in split_paths.items():
        for record in iter_records(path):
            records_checked += 1
            unit_id = record.get(id_field)
            if not isinstance(unit_id, str) or not unit_id:
                raise ValueError(f"{name}: split record has invalid {id_field}")
            if unit_id in seen_in_splits:
                split_duplicates.append(unit_id)
            seen_in_splits.add(unit_id)
            split_counts[split] += 1
            expected = assignments.get(unit_id)
            if expected != split:
                misplaced.append({"UnitId": unit_id, "expected": str(expected), "actual": split})
            if records_checked % 10_000 == 0:
                split_progress.update(records_checked, suffix=f"records={records_checked:,}")
    split_progress.finish(records_checked, suffix=f"records={records_checked:,}")

    expected_assigned = {unit_id for unit_id, split in assignments.items() if split in SPLITS}
    missing_from_splits = sorted(expected_assigned - seen_in_splits)
    unexpected_in_splits = sorted(seen_in_splits - expected_assigned)
    excluded_count = sum(split == "excluded_boundary_crossing" for split in assignments.values())
    errors = {
        "duplicate_assignment_ids": assignment_duplicates,
        "duplicate_split_ids": split_duplicates,
        "misplaced_records": misplaced,
        "missing_assigned_ids": missing_from_splits,
        "unexpected_split_ids": unexpected_in_splits,
    }
    if any(errors.values()):
        raise ValueError(f"{name}: leakage or assignment inconsistency detected: {errors}")
    return {
        "status": "PASS",
        "assignment_count": assignment_count,
        "assigned_count": len(seen_in_splits),
        "excluded_assignment_count": excluded_count,
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "duplicate_unit_count": 0,
        "misplaced_record_count": 0,
    }


def check_retrieval_policy(config: dict[str, Any]) -> dict[str, Any]:
    splits = config.get("splits", {})
    retrieval = config.get("normal_block_retrieval", {})
    if splits.get("retrieve_normal_comparisons_from_train_only") is not True:
        raise ValueError("Retrieval policy does not require training-only candidates")
    if retrieval.get("source_split") != "train":
        raise ValueError("normal_block_retrieval.source_split must be train")
    return {
        "status": "PASS",
        "enabled": bool(retrieval.get("enabled")),
        "declared_source_split": retrieval.get("source_split"),
        "runtime_ledger": "not_provided",
    }


def check_retrieval_ledger(
    path: Path,
    allowed_normal_ids: set[str],
    progress: bool = True,
) -> dict[str, Any]:
    """Validate optional retrieval records against training normal IDs."""

    count = 0
    failures: list[dict[str, Any]] = []
    for record in iter_records(path):
        count += 1
        candidate = record.get("normal_unit_id", record.get("normal_block_id"))
        source_split = record.get("source_split")
        query_id = record.get("query_unit_id", record.get("query_block_id"))
        candidate_label = record.get("normal_label", "Normal")
        reasons: list[str] = []
        if source_split != "train":
            reasons.append("candidate_not_from_train")
        if candidate not in allowed_normal_ids:
            reasons.append("candidate_not_in_training_normal_pool")
        if candidate_label != "Normal":
            reasons.append("candidate_is_not_normal")
        if query_id is not None and query_id == candidate:
            reasons.append("self_retrieval")
        if reasons:
            failures.append({"record_index": count, "reasons": reasons})
        if count % 10_000 == 0:
            ProgressBar("Checking retrieval ledger", total=None, enabled=progress).update(count)
    if failures:
        raise ValueError(f"Retrieval leakage detected: {failures[:10]}")
    return {
        "status": "PASS",
        "record_count": count,
        "training_normal_pool_count": len(allowed_normal_ids),
    }


def check_parser_policy(config: dict[str, Any]) -> dict[str, Any]:
    splits = config.get("splits", {})
    parser = config.get("parser", {})
    vocabulary = parser.get("detector_vocabulary", {})
    if splits.get("fit_parser_and_vocabulary_on_train_only") is not True:
        raise ValueError("Parser/vocabulary policy is not training-only")
    if vocabulary.get("fit_scope") != "training_only":
        raise ValueError("Detector vocabulary fit_scope must be training_only")
    if parser.get("parser_configuration_is_frozen_for_core_benchmark") is not True:
        raise ValueError("Parser configuration is not frozen for the core benchmark")
    return {
        "status": "PASS",
        "fit_scope": vocabulary.get("fit_scope"),
        "parser_configuration_frozen": parser.get("parser_configuration_is_frozen_for_core_benchmark"),
        "runtime_parser_manifest": "not_provided",
    }


def check_parser_manifest(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    fit_scope = payload.get("fit_scope")
    source_split = payload.get("source_split", payload.get("fit_split"))
    if fit_scope != "training_only" or source_split != "train":
        raise ValueError(f"Parser manifest indicates leakage: fit_scope={fit_scope}, source_split={source_split}")
    return {"status": "PASS", "fit_scope": fit_scope, "source_split": source_split, "path": str(path)}


def check_bgl_clean_input(
    manifest_path: Path,
    clean_path: Path,
    progress: bool = True,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    summary = manifest.get("summary", {})
    if summary.get("clean_output_contains_label_prefix") is not False:
        raise ValueError("BGL clean-output manifest does not prove label-prefix removal")
    line_count = 0
    invalid_first_tokens = 0
    with clean_path.open("r", encoding="utf-8") as handle:
        for line_count, line in enumerate(handle, start=1):
            fields = line.split(maxsplit=1)
            if not fields or not EPOCH_PATTERN.fullmatch(fields[0]):
                invalid_first_tokens += 1
                if invalid_first_tokens <= 5:
                    raise ValueError(f"BGL clean input has a non-epoch first token at line {line_count}")
            if line_count % 100_000 == 0:
                ProgressBar("Checking BGL parser-safe input", total=None, enabled=progress).update(line_count)
    return {
        "status": "PASS",
        "clean_line_count": line_count,
        "invalid_first_token_count": invalid_first_tokens,
        "label_prefix_manifest_flag": summary.get("clean_output_contains_label_prefix"),
        "audit_output_excluded_by_policy": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("research/experiments/experiment_config.json"))
    parser.add_argument("--manifest", type=Path, default=Path("research/data/manifests/leakage_check_manifest.json"))
    parser.add_argument("--retrieval-ledger", type=Path, help="Optional JSONL retrieval ledger to audit")
    parser.add_argument("--parser-manifest", type=Path, help="Optional parser/vocabulary fit manifest to audit")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    progress = not args.no_progress
    config = load_json(args.config)
    results: dict[str, Any] = {}

    split_root = Path("research")
    results["hdfs_primary_splits"] = check_split_bundle(
        "HDFS primary",
        split_root / "data/manifests/hdfs_block_assignments.jsonl.gz",
        {
            split: split_root / f"data/processed/hdfs_block_evidence_{split}.jsonl.gz"
            for split in SPLITS
        },
        "BlockId",
        progress,
    )
    results["hdfs_temporal_splits"] = check_split_bundle(
        "HDFS temporal",
        split_root / "data/manifests/hdfs_temporal_assignments.jsonl.gz",
        {
            split: split_root / f"data/processed/hdfs_block_evidence_temporal_{split}.jsonl.gz"
            for split in SPLITS
        },
        "BlockId",
        progress,
    )
    results["bgl_temporal_splits"] = check_split_bundle(
        "BGL temporal",
        split_root / "data/manifests/bgl_temporal_assignments.jsonl.gz",
        {
            split: split_root / f"data/processed/BGL_windows_temporal_{split}.jsonl.gz"
            for split in SPLITS
        },
        "WindowId",
        progress,
    )

    results["retrieval_policy"] = check_retrieval_policy(config)
    results["parser_policy"] = check_parser_policy(config)
    results["bgl_label_prefix"] = check_bgl_clean_input(
        split_root / "data/manifests/bgl_label_strip_manifest.json",
        split_root / "data/processed/BGL_clean.log",
        progress,
    )

    if args.retrieval_ledger:
        train_normal_ids = {
            record["UnitId"]
            for record in iter_records(split_root / "data/manifests/hdfs_block_assignments.jsonl.gz")
            if record.get("Split") == "train" and record.get("Label") == "Normal"
        }
        results["retrieval_runtime"] = check_retrieval_ledger(args.retrieval_ledger, train_normal_ids, progress)
    else:
        results["retrieval_runtime"] = {"status": "PENDING_ARTIFACT", "reason": "No retrieval ledger supplied"}

    if args.parser_manifest:
        results["parser_runtime"] = check_parser_manifest(args.parser_manifest)
    else:
        results["parser_runtime"] = {"status": "PENDING_ARTIFACT", "reason": "No parser manifest supplied"}

    manifest = {
        "manifest_version": "1.0.0",
        "status": "PASS_WITH_PENDING_RUNTIME_ARTIFACTS",
        "config": str(args.config),
        "checks": results,
        "pending_runtime_checks": [
            key for key in ("retrieval_runtime", "parser_runtime")
            if results[key]["status"] == "PENDING_ARTIFACT"
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
