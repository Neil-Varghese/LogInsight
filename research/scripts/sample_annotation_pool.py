"""Sample a reproducible, stratified annotation pool.

The input is detector-produced candidate metadata.  Each candidate must
contain a dataset, prediction-unit ID, held-out split, gold label, sequence
length, detector confidence, and detector architecture.  The sampler refuses
missing confidence or architecture values because substituting placeholders
would invalidate the planned stratification.

An annotation item is a dataset/unit/architecture tuple.  The same prediction
unit may therefore appear once per detector architecture, which lets the
annotation study compare architectures while the manifest also reports the
number of unique blocks/windows.
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

try:
    from seed_utils import derive_seed
except ImportError:  # pragma: no cover - supports package-style imports
    from research.scripts.seed_utils import derive_seed


DATASETS = {"HDFS", "BGL"}
SPLIT_DEFAULT = "test"
MIN_POOL_SIZE = 200
MAX_POOL_SIZE = 300
LENGTH_BINS = ("short_0_10", "medium_11_50", "long_51_plus")
CONFIDENCE_BINS = ("low_0_33", "medium_33_67", "high_67_100")


class ProgressBar:
    """Dependency-free progress display."""

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
            filled = (current // 100) % (width + 1)
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


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def length_bin(sequence_length: int) -> str:
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if sequence_length <= 10:
        return "short_0_10"
    if sequence_length <= 50:
        return "medium_11_50"
    return "long_51_plus"


def confidence_bin(confidence: float) -> str:
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if confidence < 1 / 3:
        return "low_0_33"
    if confidence < 2 / 3:
        return "medium_33_67"
    return "high_67_100"


def normalize_candidate(record: dict[str, Any], line_number: int, eligible_split: str) -> dict[str, Any]:
    dataset = record.get("dataset", record.get("Dataset"))
    unit_id = record.get("unit_id", record.get("UnitId", record.get("BlockId", record.get("WindowId"))))
    split = record.get("split", record.get("source_split", record.get("Split")))
    label = record.get("label", record.get("Label"))
    sequence_length = record.get("sequence_length", record.get("length", record.get("EventCount")))
    confidence = record.get("confidence", record.get("detector_confidence"))
    architecture = record.get("architecture", record.get("Architecture"))
    missing = [
        name for name, value in {
            "dataset": dataset,
            "unit_id": unit_id,
            "split": split,
            "label": label,
            "sequence_length": sequence_length,
            "confidence": confidence,
            "architecture": architecture,
        }.items()
        if value is None or value == ""
    ]
    if missing:
        raise ValueError(f"Candidate line {line_number} is missing: {', '.join(missing)}")
    if dataset not in DATASETS:
        raise ValueError(f"Candidate line {line_number} has unsupported dataset: {dataset!r}")
    if split != eligible_split:
        raise ValueError(
            f"Candidate line {line_number} uses split {split!r}; expected eligible split {eligible_split!r}"
        )
    if label not in ("Normal", "Anomaly"):
        raise ValueError(f"Candidate line {line_number} has invalid label: {label!r}")
    try:
        sequence_length = int(sequence_length)
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Candidate line {line_number} has invalid length/confidence") from exc
    if sequence_length < 1:
        raise ValueError(f"Candidate line {line_number} has non-positive sequence length")
    if not 0 <= confidence <= 1:
        raise ValueError(f"Candidate line {line_number} confidence must be between 0 and 1")
    unit_id = str(unit_id)
    architecture = str(architecture)
    return {
        "dataset": dataset,
        "unit_id": unit_id,
        "split": split,
        "label": label,
        "sequence_length": sequence_length,
        "confidence": confidence,
        "architecture": architecture,
        "length_bin": length_bin(sequence_length),
        "confidence_bin": confidence_bin(confidence),
    }


def load_candidates(path: Path, eligible_split: str, show_progress: bool = True) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    progress = ProgressBar("Reading annotation candidates", total=None, enabled=show_progress)
    for line_number, record in enumerate(iter_records(path), start=1):
        candidate = normalize_candidate(record, line_number, eligible_split)
        key = (candidate["dataset"], candidate["unit_id"], candidate["architecture"])
        if key in seen:
            raise ValueError(f"Duplicate dataset/unit/architecture candidate: {key}")
        seen.add(key)
        candidates.append(candidate)
        if line_number % 100 == 0:
            progress.update(line_number, suffix=f"candidates={line_number:,}")
    progress.finish(len(candidates), suffix=f"candidates={len(candidates):,}")
    return candidates


def _allocate_quotas(capacities: dict[tuple[str, ...], int], target: int) -> dict[tuple[str, ...], int]:
    if target < len(capacities):
        raise ValueError(
            f"Target pool size {target} is smaller than the {len(capacities)} required strata; "
            "increase the target or reduce stratification factors"
        )
    if target > sum(capacities.values()):
        raise ValueError("Target pool size exceeds the available eligible candidates")
    quotas = {key: 1 for key in capacities}
    remaining = target - len(quotas)
    remaining_capacity = {key: capacity - 1 for key, capacity in capacities.items()}
    total_remaining_capacity = sum(remaining_capacity.values())
    if remaining == 0 or total_remaining_capacity == 0:
        return quotas
    raw = {
        key: remaining * capacity / total_remaining_capacity
        for key, capacity in remaining_capacity.items()
    }
    for key, value in raw.items():
        quotas[key] += min(remaining_capacity[key], int(value))
    allocated = sum(quotas.values())
    leftover = target - allocated
    order = sorted(
        capacities,
        key=lambda key: (raw[key] - int(raw[key]), remaining_capacity[key], key),
        reverse=True,
    )
    for key in order:
        if leftover == 0:
            break
        if quotas[key] < capacities[key]:
            quotas[key] += 1
            leftover -= 1
    if leftover:
        raise ValueError("Unable to allocate the requested pool size across strata")
    return quotas


def stratified_sample(
    candidates: list[dict[str, Any]],
    target_count: int = 240,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    if not MIN_POOL_SIZE <= target_count <= MAX_POOL_SIZE:
        raise ValueError(f"target_count must be between {MIN_POOL_SIZE} and {MAX_POOL_SIZE}")
    if not candidates:
        raise ValueError("No eligible annotation candidates were supplied")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        key = (
            candidate["dataset"],
            candidate["label"],
            candidate["length_bin"],
            candidate["confidence_bin"],
            candidate["architecture"],
        )
        groups[key].append(candidate)
    quotas = _allocate_quotas({key: len(items) for key, items in groups.items()}, target_count)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        items = list(groups[key])
        random.Random(derive_seed(seed, "annotation_pool", *key)).shuffle(items)
        selected.extend(items[:quotas[key]])
    selected.sort(key=lambda item: (item["dataset"], item["unit_id"], item["architecture"]))
    for index, item in enumerate(selected, start=1):
        item["pool_index"] = index
        item["annotation_item_id"] = f"{item['dataset']}:{item['unit_id']}:{item['architecture']}"
        item["sampling_seed"] = seed
        item["sampling_stratum"] = (
            f"{item['dataset']}|{item['label']}|{item['length_bin']}|"
            f"{item['confidence_bin']}|{item['architecture']}"
        )
    return selected


def build_manifest(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    input_path: Path,
    target_count: int,
    seed: int,
    eligible_split: str,
) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(item[field] for item in selected).items()))

    unique_units = {(item["dataset"], item["unit_id"]) for item in selected}
    return {
        "manifest_version": "1.0.0",
        "input": str(input_path),
        "input_sha256": file_hash(input_path),
        "eligible_split": eligible_split,
        "target_item_count": target_count,
        "selected_item_count": len(selected),
        "selected_unique_prediction_unit_count": len(unique_units),
        "candidate_item_count": len(candidates),
        "sampling_seed": seed,
        "sampling_unit": "dataset_prediction_unit_architecture_item",
        "length_bin_rule": {
            "short_0_10": "sequence_length <= 10",
            "medium_11_50": "11 <= sequence_length <= 50",
            "long_51_plus": "sequence_length >= 51",
        },
        "confidence_bin_rule": {
            "low_0_33": "0 <= confidence < 1/3",
            "medium_33_67": "1/3 <= confidence < 2/3",
            "high_67_100": "2/3 <= confidence <= 1",
        },
        "selected_counts": {
            "dataset": counts("dataset"),
            "label": counts("label"),
            "length_bin": counts("length_bin"),
            "confidence_bin": counts("confidence_bin"),
            "architecture": counts("architecture"),
            "stratum": dict(sorted(Counter(item["sampling_stratum"] for item in selected).items())),
        },
    }


def write_pool(path: Path, selected: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(part, "wt", encoding="utf-8", newline="\n") as handle:
        for item in selected:
            handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
    part.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("research/experiments/pilot/annotation_pool.jsonl.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("research/experiments/pilot/annotation_pool_manifest.json"))
    parser.add_argument("--target-count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eligible-split", default=SPLIT_DEFAULT)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace an existing pool and manifest")
    args = parser.parse_args()
    if not args.candidate_metadata.exists():
        raise FileNotFoundError(args.candidate_metadata)
    if not args.force and (args.output.exists() or args.manifest.exists()):
        raise FileExistsError("Annotation pool output already exists; use --force to replace it")

    candidates = load_candidates(args.candidate_metadata, args.eligible_split, not args.no_progress)
    selected = stratified_sample(candidates, args.target_count, args.seed)
    manifest = build_manifest(
        candidates,
        selected,
        args.candidate_metadata,
        args.target_count,
        args.seed,
        args.eligible_split,
    )
    write_pool(args.output, selected)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
