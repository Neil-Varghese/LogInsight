"""Create strict chronological train/validation/test splits for HDFS and BGL.

The primary HDFS split is block-disjoint and label-stratified.  This script
creates a separate temporal sensitivity split: units are ordered by their
first timestamp, and a unit is retained only when its complete time interval
fits inside one temporal partition.  HDFS blocks that cross a partition
boundary are excluded rather than assigned to the wrong side of time.  BGL's
fixed windows are already non-overlapping, so they normally require no
exclusions.

The output manifest records the cutoffs, excluded units, label counts,
interval ordering, and SHA-256 hashes needed to reproduce the analysis.
"""

from __future__ import annotations

import argparse
import calendar
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "validation", "test")
DEFAULT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
HDFS_DATASET = "HDFS"
BGL_DATASET = "BGL"


class ProgressBar:
    """Dependency-free progress display for long local preprocessing runs."""

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
            # Unknown totals are expected during the first metadata scan.
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


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield record


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hdfs_timestamp(raw_line: str) -> int:
    """Parse HDFS's YYMMDD HHMMSS timestamp as a deterministic UTC second."""

    fields = raw_line.split(maxsplit=2)
    if len(fields) < 2:
        raise ValueError(f"HDFS RawLine has no timestamp: {raw_line!r}")
    try:
        parsed = datetime.strptime(f"{fields[0]} {fields[1]}", "%y%m%d %H%M%S")
    except ValueError as exc:
        raise ValueError(f"Invalid HDFS timestamp in RawLine: {raw_line!r}") from exc
    return calendar.timegm(parsed.replace(tzinfo=timezone.utc).timetuple())


def hdfs_record_interval(record: dict[str, Any]) -> tuple[int, int]:
    evidence = record.get("Evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"HDFS block has no Evidence: {record.get('BlockId')!r}")
    timestamps: list[int] = []
    for event in evidence:
        if not isinstance(event, dict) or not isinstance(event.get("RawLine"), str):
            raise ValueError(f"HDFS block has malformed RawLine: {record.get('BlockId')!r}")
        timestamps.append(hdfs_timestamp(event["RawLine"]))
    return min(timestamps), max(timestamps) + 1


def bgl_record_interval(record: dict[str, Any]) -> tuple[int, int]:
    try:
        start = int(record["WindowStartEpoch"])
        end = int(record["WindowEndEpochExclusive"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"BGL window has invalid interval: {record.get('WindowId')!r}") from exc
    if end <= start:
        raise ValueError(f"BGL window has non-positive duration: {record.get('WindowId')!r}")
    return start, end


def make_metadata(dataset: str, record: dict[str, Any]) -> dict[str, Any]:
    if dataset == HDFS_DATASET:
        unit_id = record.get("BlockId")
        start, end = hdfs_record_interval(record)
    elif dataset == BGL_DATASET:
        unit_id = record.get("WindowId")
        start, end = bgl_record_interval(record)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not isinstance(unit_id, str) or not unit_id:
        raise ValueError(f"Missing {dataset} prediction-unit ID")
    label_id = record.get("LabelId")
    if label_id not in (0, 1):
        raise ValueError(f"Invalid {dataset} LabelId for {unit_id!r}: {label_id!r}")
    return {
        "UnitId": unit_id,
        "Label": record.get("Label"),
        "LabelId": label_id,
        "FirstTimestamp": start,
        "EndTimestampExclusive": end,
    }


def choose_cutoffs(
    metadata: list[dict[str, Any]],
    fractions: dict[str, float] | None = None,
) -> tuple[int, int]:
    """Choose timestamp boundaries without splitting equal-start timestamp groups."""

    fractions = fractions or DEFAULT_FRACTIONS
    if set(fractions) != set(SPLITS) or any(value <= 0 for value in fractions.values()):
        raise ValueError("Temporal fractions must contain positive train/validation/test values")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Temporal fractions must sum to 1")
    ordered = sorted(metadata, key=lambda item: (item["FirstTimestamp"], item["UnitId"]))
    if len(ordered) < 3:
        raise ValueError("At least three temporal units are required")

    groups: list[tuple[int, int]] = []
    for item in ordered:
        timestamp = item["FirstTimestamp"]
        if groups and groups[-1][0] == timestamp:
            groups[-1] = (timestamp, groups[-1][1] + 1)
        else:
            groups.append((timestamp, 1))
    if len(groups) < 3:
        raise ValueError("At least three distinct timestamps are required for temporal splits")

    targets = [len(metadata) * fractions["train"], len(metadata) * (fractions["train"] + fractions["validation"])]
    candidates: list[tuple[int, int]] = []
    cumulative = 0
    for index in range(len(groups) - 1):
        cumulative += groups[index][1]
        candidates.append((groups[index + 1][0], cumulative))

    first = min(candidates, key=lambda item: abs(item[1] - targets[0]))
    second_candidates = [item for item in candidates if item[0] > first[0]]
    if not second_candidates:
        raise ValueError("Unable to choose distinct temporal boundaries")
    second = min(second_candidates, key=lambda item: abs(item[1] - targets[1]))
    return first[0], second[0]


def assign_temporal_splits(
    metadata: list[dict[str, Any]],
    fractions: dict[str, float] | None = None,
) -> tuple[dict[str, str], tuple[int, int]]:
    """Assign complete intervals to temporal splits; purge boundary-crossing units."""

    cutoff_one, cutoff_two = choose_cutoffs(metadata, fractions)
    assignments: dict[str, str] = {}
    for item in metadata:
        unit_id = item["UnitId"]
        start = item["FirstTimestamp"]
        end = item["EndTimestampExclusive"]
        if end <= cutoff_one:
            split = "train"
        elif start >= cutoff_one and end <= cutoff_two:
            split = "validation"
        elif start >= cutoff_two:
            split = "test"
        else:
            split = "excluded_boundary_crossing"
        assignments[unit_id] = split
    return assignments, (cutoff_one, cutoff_two)


def iso_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def summarize_temporal_assignment(
    metadata: list[dict[str, Any]],
    assignments: dict[str, str],
    cutoffs: tuple[int, int],
) -> dict[str, Any]:
    counts: Counter[str] = Counter(assignments.values())
    label_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    ranges: dict[str, dict[str, int | None]] = {
        split: {"min_start": None, "max_end_exclusive": None} for split in SPLITS
    }
    for item in metadata:
        split = assignments[item["UnitId"]]
        if split not in SPLITS:
            continue
        label_counts[split][str(item["Label"])] += 1
        current = ranges[split]
        current["min_start"] = item["FirstTimestamp"] if current["min_start"] is None else min(current["min_start"], item["FirstTimestamp"])
        current["max_end_exclusive"] = item["EndTimestampExclusive"] if current["max_end_exclusive"] is None else max(current["max_end_exclusive"], item["EndTimestampExclusive"])

    train_end = ranges["train"]["max_end_exclusive"]
    validation_start = ranges["validation"]["min_start"]
    validation_end = ranges["validation"]["max_end_exclusive"]
    test_start = ranges["test"]["min_start"]
    strict = (
        train_end is not None
        and validation_start is not None
        and validation_end is not None
        and test_start is not None
        and train_end <= validation_start
        and validation_end <= test_start
    )
    return {
        "source_unit_count": len(metadata),
        "assigned_unit_count": sum(counts[split] for split in SPLITS),
        "excluded_unit_count": counts["excluded_boundary_crossing"],
        "split_counts": {split: counts[split] for split in SPLITS},
        "split_label_counts": {
            split: dict(sorted(label_counts[split].items())) for split in SPLITS
        },
        "cutoffs": {
            "train_validation_epoch": cutoffs[0],
            "train_validation_iso": iso_timestamp(cutoffs[0]),
            "validation_test_epoch": cutoffs[1],
            "validation_test_iso": iso_timestamp(cutoffs[1]),
        },
        "split_time_ranges": {
            split: {
                "min_start_epoch": values["min_start"],
                "max_end_exclusive_epoch": values["max_end_exclusive"],
                "min_start_iso": iso_timestamp(values["min_start"]) if values["min_start"] is not None else None,
                "max_end_exclusive_iso": iso_timestamp(values["max_end_exclusive"]) if values["max_end_exclusive"] is not None else None,
            }
            for split, values in ranges.items()
        },
        "strict_temporal_disjoint": strict,
    }


def process_dataset(
    dataset: str,
    input_path: Path,
    output_dir: Path,
    manifest_dir: Path,
    fractions: dict[str, float],
    show_progress: bool,
    force: bool,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if dataset == HDFS_DATASET:
        stem = "hdfs_block_evidence"
        id_field = "BlockId"
    else:
        stem = "BGL_windows"
        id_field = "WindowId"
    split_paths = {
        split: output_dir / f"{stem}_temporal_{split}.jsonl.gz" for split in SPLITS
    }
    assignment_path = manifest_dir / f"{dataset.lower()}_temporal_assignments.jsonl.gz"
    paths_to_check = [assignment_path, *split_paths.values()]
    if not force:
        existing = [str(path) for path in paths_to_check if path.exists()]
        if existing:
            raise FileExistsError(f"Temporal outputs already exist: {existing}; use --force")

    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()
    scan = ProgressBar(f"Scanning {dataset} timestamps", total=None, enabled=show_progress)
    for index, record in enumerate(iter_jsonl(input_path), start=1):
        item = make_metadata(dataset, record)
        if item["UnitId"] in seen:
            raise ValueError(f"Duplicate {id_field} in source: {item['UnitId']}")
        seen.add(item["UnitId"])
        metadata.append(item)
        if index % 10_000 == 0:
            scan.update(index, suffix=f"units={index:,}")
    scan.finish(len(metadata), suffix=f"units={len(metadata):,}")

    assignments, cutoffs = assign_temporal_splits(metadata, fractions)
    summary = summarize_temporal_assignment(metadata, assignments, cutoffs)
    if not summary["strict_temporal_disjoint"]:
        raise ValueError(f"{dataset} temporal assignment is not strictly ordered")
    if any(summary["split_counts"][split] == 0 for split in SPLITS):
        raise ValueError(f"{dataset} temporal assignment produced an empty split")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    assignment_part = assignment_path.with_suffix(assignment_path.suffix + ".part")
    with gzip.open(assignment_part, "wt", encoding="utf-8", newline="\n") as handle:
        for item in sorted(metadata, key=lambda value: (value["FirstTimestamp"], value["UnitId"])):
            assignment = dict(item)
            assignment["Split"] = assignments[item["UnitId"]]
            handle.write(json.dumps(assignment, separators=(",", ":")) + "\n")
    assignment_part.replace(assignment_path)

    parts = {split: path.with_suffix(path.suffix + ".part") for split, path in split_paths.items()}
    handles: dict[str, Any] = {}
    write_progress = ProgressBar(f"Writing {dataset} temporal splits", total=len(metadata), enabled=show_progress)
    try:
        for split, part in parts.items():
            handles[split] = gzip.open(part, "wt", encoding="utf-8", newline="\n")
        for index, record in enumerate(iter_jsonl(input_path), start=1):
            unit_id = record[id_field]
            split = assignments[unit_id]
            if split in SPLITS:
                handles[split].write(json.dumps(record, separators=(",", ":")) + "\n")
            if index % 10_000 == 0 or index == len(metadata):
                write_progress.update(index, suffix=f"source_units={index:,}")
        for handle in handles.values():
            handle.close()
        for split, path in split_paths.items():
            parts[split].replace(path)
    except Exception:
        for handle in handles.values():
            handle.close()
        for part in parts.values():
            part.unlink(missing_ok=True)
        assignment_path.unlink(missing_ok=True)
        raise

    summary.update(
        {
            "dataset": dataset,
            "input": str(input_path),
            "prediction_unit": id_field,
            "timestamp_policy": (
                "min/max Evidence.RawLine HDFS timestamps; boundary-crossing blocks purged"
                if dataset == HDFS_DATASET
                else "WindowStartEpoch/WindowEndEpochExclusive; boundary-crossing windows purged"
            ),
            "fractions": fractions,
            "boundary_policy": "strict_interval_containment_with_boundary_purge",
            "assignment_path": str(assignment_path),
            "assignment_sha256": file_hash(assignment_path),
            "split_paths": {split: str(path) for split, path in split_paths.items()},
            "split_sha256": {split: file_hash(path) for split, path in split_paths.items()},
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", HDFS_DATASET, BGL_DATASET), default="all")
    parser.add_argument("--hdfs-input", type=Path, default=Path("research/data/processed/hdfs_block_evidence.jsonl.gz"))
    parser.add_argument("--bgl-input", type=Path, default=Path("research/data/processed/BGL_windows.jsonl.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/data/processed"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("research/data/manifests"))
    parser.add_argument("--manifest", type=Path, default=Path("research/data/manifests/temporal_split_manifest.json"))
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace existing temporal artifacts")
    args = parser.parse_args()

    datasets = (HDFS_DATASET, BGL_DATASET) if args.dataset == "all" else (args.dataset,)
    summaries: dict[str, Any] = {}
    for dataset in datasets:
        input_path = args.hdfs_input if dataset == HDFS_DATASET else args.bgl_input
        summaries[dataset] = process_dataset(
            dataset=dataset,
            input_path=input_path,
            output_dir=args.output_dir,
            manifest_dir=args.manifest_dir,
            fractions=DEFAULT_FRACTIONS,
            show_progress=not args.no_progress,
            force=args.force,
        )

    manifest = {
        "manifest_version": "1.0.0",
        "strategy": "strict_temporal_split_with_boundary_purge",
        "fractions": DEFAULT_FRACTIONS,
        "datasets": summaries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
