"""Create deterministic, label-stratified, BlockId-disjoint HDFS splits.

The source evidence file is read twice: once to assign every BlockId and once
to stream the complete evidence record into exactly one split file. Assignment
is based only on BlockId, label, and the registered seed; chronological record
order is preserved inside every output split.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEED = 2026
DEFAULT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}


class ProgressBar:
    """Small dependency-free record progress bar."""

    def __init__(self, label: str, total: int, enabled: bool = True) -> None:
        self.label = label
        self.total = max(total, 1)
        self.enabled = enabled
        self.last_update = 0.0

    def update(self, current: int, suffix: str = "") -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if current < self.total and now - self.last_update < 0.15:
            return
        self.last_update = now
        fraction = min(max(current, 0) / self.total, 1.0)
        width = 30
        filled = int(width * fraction)
        bar = "=" * filled + ">" + " " * max(width - filled - 1, 0)
        print(
            f"\r{self.label} [{bar}] {fraction * 100:6.2f}% {suffix}",
            end="",
            file=sys.stderr,
            flush=True,
        )
        if current >= self.total:
            print(file=sys.stderr)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            yield record


def allocate_counts(count: int, fractions: dict[str, float]) -> dict[str, int]:
    """Allocate an integer count with largest-remainder rounding."""

    if count < 0 or any(value < 0 for value in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Split fractions must be non-negative and sum to 1")
    names = ["train", "validation", "test"]
    raw = {name: count * fractions[name] for name in names}
    allocated = {name: int(raw[name]) for name in names}
    remaining = count - sum(allocated.values())
    order = sorted(names, key=lambda name: (raw[name] - allocated[name], -names.index(name)), reverse=True)
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def assign_splits(
    metadata: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
    fractions: dict[str, float] | None = None,
) -> dict[str, str]:
    """Assign each BlockId once, stratified by LabelId and deterministically."""

    fractions = fractions or DEFAULT_FRACTIONS
    by_label: dict[int, list[str]] = defaultdict(list)
    for record in metadata:
        block_id = record["BlockId"]
        label_id = record["LabelId"]
        if label_id not in (0, 1):
            raise ValueError(f"Invalid HDFS LabelId for {block_id}: {label_id}")
        by_label[label_id].append(block_id)
    assignments: dict[str, str] = {}
    for label_id, block_ids in sorted(by_label.items()):
        unique_ids = sorted(block_ids)
        if len(unique_ids) != len(set(unique_ids)):
            raise ValueError(f"Duplicate BlockId in label stratum {label_id}")
        rng = random.Random(seed + label_id)
        rng.shuffle(unique_ids)
        counts = allocate_counts(len(unique_ids), fractions)
        cursor = 0
        for split in ("train", "validation", "test"):
            for block_id in unique_ids[cursor:cursor + counts[split]]:
                assignments[block_id] = split
            cursor += counts[split]
    if len(assignments) != len(metadata):
        raise ValueError("Not every HDFS BlockId received exactly one split")
    return assignments


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def split_hdfs_blocks(
    input_path: Path,
    assignment_path: Path,
    output_dir: Path,
    manifest_path: Path,
    seed: int = DEFAULT_SEED,
    fractions: dict[str, float] | None = None,
    show_progress: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Create assignment, split evidence files, and a reproducibility manifest."""

    fractions = fractions or DEFAULT_FRACTIONS
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if assignment_path.exists() and not force:
        raise FileExistsError(f"Assignment output already exists: {assignment_path}; use --force")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        split: output_dir / f"hdfs_block_evidence_{split}.jsonl.gz"
        for split in ("train", "validation", "test")
    }
    if not force:
        existing = [str(path) for path in split_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Split outputs already exist: {existing}; use --force")

    expected_blocks = None
    metadata: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in iter_jsonl(input_path):
        block_id = record.get("BlockId")
        if not isinstance(block_id, str) or not block_id:
            raise ValueError("HDFS evidence contains an invalid BlockId")
        if block_id in seen:
            raise ValueError(f"Duplicate BlockId in source evidence: {block_id}")
        seen.add(block_id)
        if record.get("LabelId") not in (0, 1):
            raise ValueError(f"Missing or invalid LabelId for {block_id}")
        metadata.append({
            "BlockId": block_id,
            "Label": record.get("Label"),
            "LabelId": record["LabelId"],
            "FirstLineId": record.get("FirstLineId"),
            "EventCount": record.get("EventCount"),
        })
    expected_blocks = len(metadata)
    assignments = assign_splits(metadata, seed, fractions)

    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_part = assignment_path.with_suffix(assignment_path.suffix + ".part")
    with gzip.open(assignment_part, "wt", encoding="utf-8", newline="\n") as handle:
        for record in sorted(metadata, key=lambda item: item["FirstLineId"]):
            assignment = dict(record)
            assignment["Split"] = assignments[record["BlockId"]]
            handle.write(json.dumps(assignment, separators=(",", ":")) + "\n")
    assignment_part.replace(assignment_path)

    counts: Counter[str] = Counter(assignments.values())
    label_counts: dict[str, Counter[str]] = {split: Counter() for split in split_paths}
    write_progress = ProgressBar("Writing HDFS split files", expected_blocks, enabled=show_progress)
    handles = {}
    parts = {}
    try:
        for split, path in split_paths.items():
            part = path.with_suffix(path.suffix + ".part")
            parts[split] = part
            handles[split] = gzip.open(part, "wt", encoding="utf-8", newline="\n")
        for index, record in enumerate(iter_jsonl(input_path), start=1):
            split = assignments[record["BlockId"]]
            handles[split].write(json.dumps(record, separators=(",", ":")) + "\n")
            label_counts[split][str(record["Label"])] += 1
            if index % 10_000 == 0 or index == expected_blocks:
                write_progress.update(index, suffix=f"blocks={index:,}")
        for handle in handles.values():
            handle.close()
        for split, path in split_paths.items():
            parts[split].replace(path)
    except Exception:
        for handle in handles.values():
            handle.close()
        for part in parts.values():
            part.unlink(missing_ok=True)
        raise

    summary: dict[str, Any] = {
        "input": str(input_path),
        "assignment_path": str(assignment_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "fractions": fractions,
        "block_count": expected_blocks,
        "split_counts": {split: counts[split] for split in ("train", "validation", "test")},
        "split_label_counts": {
            split: dict(sorted(label_counts[split].items()))
            for split in ("train", "validation", "test")
        },
        "split_paths": {split: str(path) for split, path in split_paths.items()},
        "block_disjoint": True,
        "assignment_sha256": file_hash(assignment_path),
        "split_sha256": {split: file_hash(path) for split, path in split_paths.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/data/processed/hdfs_block_evidence.jsonl.gz"),
    )
    parser.add_argument(
        "--assignment",
        type=Path,
        default=Path("research/data/manifests/hdfs_block_assignments.jsonl.gz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/data/processed"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/data/manifests/hdfs_split_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace existing split artifacts")
    args = parser.parse_args()
    summary = split_hdfs_blocks(
        args.input,
        args.assignment,
        args.output_dir,
        args.manifest,
        seed=args.seed,
        show_progress=not args.no_progress,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
